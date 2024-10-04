import functools
import gin
import torch
import torch.nn.functional as F
from misc.defs import *


@gin.configurable
def level_sample(i_level, rays, t_vals, weights, n_sample=64,
                 resample_padding=0.01,
                 perturb=False,
                 lindisp=False,
                 cast_cone=True,
                 stop_level_grad=True):
    if i_level == 0:
        t_vals, samples = sample_along_rays(
            rays.origins,
            rays.directions,
            rays.radii,
            n_sample,
            rays.near,
            rays.far,
            perturb,
            lindisp,
            cast_cone,
        )
    else:
        t_vals, samples = resample_along_rays(
            rays.origins,
            rays.directions,
            rays.radii,
            t_vals,
            weights,
            perturb,
            cast_cone,
            stop_level_grad,
            resample_padding=resample_padding,
        )
    return t_vals, samples


@gin.configurable
def density_process(raw_rgb, raw_density, means, nerf, t_vals, rays_d,
                    rgb_activation=F.sigmoid,
                    density_bias=-1.,
                    density_activation=F.softmax,
                    rgb_padding=0.001,
                    raw_noise_std=0.,
                    white_bkgd=False):
    if raw_noise_std > 0:
        raw_density += raw_noise_std * torch.randn_like(raw_density)

    # Volumetric rendering.
    rgb = rgb_activation(raw_rgb)
    rgb = rgb * (1 + 2 * rgb_padding) - rgb_padding
    density = density_activation(raw_density + density_bias)

    t_mids = 0.5 * (t_vals[..., :-1] + t_vals[..., 1:])
    t_dists = t_vals[..., 1:] - t_vals[..., :-1]
    delta = t_dists * torch.linalg.norm(rays_d[..., None, :], dim=-1)
    # Note that we're quietly turning density from [..., 0] to [...].
    density_delta = density[..., 0] * delta

    alpha = 1 - torch.exp(-density_delta)
    trans = torch.exp(-torch.cat([
        torch.zeros_like(density_delta[..., :1]),
        torch.cumsum(density_delta[..., :-1], dim=-1)
    ], dim=-1))
    weights = alpha * trans

    comp_rgb = (weights[..., None] * rgb).sum(dim=-2)
    acc = weights.sum(dim=-1)
    distance = (weights * t_mids).sum(dim=-1) / acc
    distance = torch.clip(torch.nan_to_num(distance, torch.inf), t_vals[:, 0], t_vals[:, -1])
    if white_bkgd:
        comp_rgb = comp_rgb + (1. - acc[..., None])

    return {
            "rgb": comp_rgb,
            "dist": distance,
            "acc": acc,
            "weights": weights,
            "sim_or_grad": torch.ones_like(alpha),
        }


@gin.configurable
def similarity_process(raw_rgb, raw_density, means, nerf, t_vals, rays_d,
                       raw_noise_std=0.,
                       white_bkgd=False,
                       mode='sim'):

    def cos_sim(u, v):
        # return (u * v).sum(-1) / (torch.norm(u, dim=-1) + 1.) / (torch.norm(v, dim=-1) + 1.)
        return (u * v).sum(-1) / (torch.norm(u, dim=-1) + 1e-3) / (torch.norm(v, dim=-1) + 1e-3)

    def sim_to_alpha(sim):
        # return F.relu(1. - F.relu(2 * sim))
        return F.relu(1. - F.relu(sim + 0.5))

    raw_density = raw_density.squeeze()
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw) * dists)

    dists = t_vals[..., 1:] - t_vals[..., :-1]
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw_rgb)  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw_density.shape, device=raw_density.device) * raw_noise_std

    if 'sim' in mode:
        # raw_density = noise + raw_density
        if len(raw_density.shape) == 2:
            raw_density = raw_density[:, :, None]
        a_sig = raw_density[:, :-1]
        b_sig = raw_density[:, 1:]

        sim = cos_sim(a_sig, b_sig)
        sim = torch.cat([sim, sim[:, -1:]], 1)              # padding
        sig = sim_to_alpha(sim).unsqueeze(-1)

        alpha = sig[..., 0]

        rgb = (rgb[:, 1:] + rgb[:, :-1]) / 2
        rgb = torch.cat([rgb, rgb[:, -1:]], 1)

    elif 'sdf' in mode:
        batch_size = means.size(0)
        n_samples = means.size(1)
        # Section length
        sdf = raw_density
        gradients = nerf.grad(means)
        inv_s = nerf.dev(means)
        inv_s = inv_s.expand(batch_size, n_samples)

        dirs = rays_d[:, None, :].expand(means.shape)
        true_cos = (dirs * gradients).sum(-1)

        # auto annealing
        if not hasattr(similarity_process, "__cos_anneal_ratio"):
            similarity_process.__cos_anneal_ratio = 0
        cos_anneal_ratio = similarity_process.__cos_anneal_ratio
        similarity_process.__cos_anneal_ratio = min(cos_anneal_ratio + 0.0001, 1)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)

        pts_norm = torch.linalg.norm(means, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        sphere_radius = nerf.radius()
        inside_sphere = (pts_norm < sphere_radius).float().detach()
        relax_inside_sphere = (pts_norm < sphere_radius * 1.2).float().detach()

        # there is no background model, so we directly discard the density outside the sphere
        alpha = alpha * inside_sphere

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)
        sim = gradient_error
    else:
        alpha = raw2alpha(raw_density + noise, dists)  # [N_rays, N_samples]
        sim = torch.ones_like(alpha)

    Ts = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=dists.device), 1.-alpha + 1e-10], -1), -1)
    weights = alpha * Ts[:, :-1]

    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    mid_z = (t_vals[:, 1:] + t_vals[:, :-1]) / 2
    depth_map = torch.sum(weights * mid_z, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map, device=dists.device), depth_map / (torch.sum(weights, -1) + 1e-8))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[...,None])

    return {
            "rgb": rgb_map,
            "dist": depth_map,
            "acc": acc_map,
            "weights": weights,
            "sim_or_grad": sim,
        }


@gin.configurable
def render_mip(rays, nerf, n_levels=2, mode='mip', is_eval=False):
    processing = density_process if mode == 'mip' else functools.partial(similarity_process, mode=mode)
    sampling = level_sample
    if is_eval:
        processing = functools.partial(processing, raw_noise_std=0.)
        sampling = functools.partial(sampling, perturb=False)

    ret = []
    t_vals = None
    weights = None
    for i_level in range(n_levels):
        t_vals, samples = sampling(i_level, rays, t_vals, weights)

        if isinstance(nerf, IMip):
            raw_rgb, raw_density = nerf.color_and_density_of_gaussian(samples[0], samples[1], rays.viewdirs)
        else:
            raw_rgb, raw_density = nerf(samples[0], dirs=rays.viewdirs)

        ret_i = processing(raw_rgb, raw_density, samples[0], nerf, t_vals, rays.directions)
        ret_i["means"] = samples[0]
        weights = ret_i["weights"]

        ret.append(ret_i)

    return ret


""" Sampler """


def lift_gaussian(d, t_mean, t_var, r_var, diag):
    """Lift a Gaussian defined along a ray to 3D coordinates."""
    mean = d[..., None, :] * t_mean[..., None]

    mag = torch.sum(d**2, dim=-1, keepdims=True)
    d_mag_sq = torch.maximum(mag, torch.ones_like(mag) * 1e-10)

    if diag:
        d_outer_diag = d**2
        null_outer_diag = 1 - d_outer_diag / d_mag_sq
        t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
        xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
        cov_diag = t_cov_diag + xy_cov_diag
        return mean, cov_diag
    else:
        d_outer = d[..., :, None] * d[..., None, :]
        eye = torch.eye(d.shape[-1], device=d.device)
        null_outer = eye - d[..., :, None] * (d / d_mag_sq)[..., None, :]
        t_cov = t_var[..., None, None] * d_outer[..., None, :, :]
        xy_cov = r_var[..., None, None] * null_outer[..., None, :, :]
        cov = t_cov + xy_cov
        return mean, cov


def conical_frustum_to_gaussian(d, t0, t1, base_radius, diag, stable=True):
    """Approximate a conical frustum as a Gaussian distribution (mean+cov).

    Assumes the ray is originating from the origin, and base_radius is the
    radius at dist=1. Doesn't assume `d` is normalized.

    Args:
        d: jnp.float32 3-vector, the axis of the cone
        t0: float, the starting distance of the frustum.
        t1: float, the ending distance of the frustum.
        base_radius: float, the scale of the radius as a function of distance.
        diag: boolean, whether or the Gaussian will be diagonal or full-covariance.
        stable: boolean, whether or not to use the stable computation described in
          the paper (setting this to False will cause catastrophic failure).

    Returns:
        a Gaussian (mean and covariance).
    """
    if stable:
        mu = (t0 + t1) / 2
        hw = (t1 - t0) / 2
        t_mean = mu + (2 * mu * hw**2) / (3 * mu**2 + hw**2)
        t_var = (hw**2) / 3 - (4 / 15) * ((hw**4 * (12 * mu**2 - hw**2)) /
                                          (3 * mu**2 + hw**2)**2)
        r_var = base_radius**2 * ((mu**2) / 4 + (5 / 12) * hw**2 - 4 / 15 *
                                  (hw**4) / (3 * mu**2 + hw**2))
    else:
        t_mean = (3 * (t1**4 - t0**4)) / (4 * (t1**3 - t0**3))
        r_var = base_radius**2 * (3 / 20 * (t1**5 - t0**5) / (t1**3 - t0**3))
        t_mosq = 3 / 5 * (t1**5 - t0**5) / (t1**3 - t0**3)
        t_var = t_mosq - t_mean**2
    return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cast_rays(t_vals, origins, directions, radii, diag=False):
    """Cast rays (cone- or cylinder-shaped) and featurize sections of it.

    Args:
        t_vals: float array, the "fencepost" distances along the ray.
        origins: float array, the ray origin coordinates.
        directions: float array, the ray direction vectors.
        radii: float array, the radii (base radii for cones) of the rays.
        ray_shape: string, the shape of the ray, must be 'cone' or 'cylinder'.
        diag: boolean, whether or not the covariance matrices should be diagonal.

    Returns:
        a tuple of arrays of means and covariances.
    """
    t0 = t_vals[..., :-1]
    t1 = t_vals[..., 1:]
    means, covs = conical_frustum_to_gaussian(directions, t0, t1, radii, diag)
    means = means + origins[..., None, :]
    return means, covs


def sample_along_rays(origins, directions, radii, num_samples, near, far, randomized, lindisp, cast_cone=True):
    """Stratified sampling along the rays.

    Args:
        origins: jnp.ndarray(float32), [batch_size, 3], ray origins.
        directions: jnp.ndarray(float32), [batch_size, 3], ray directions.
        radii: jnp.ndarray(float32), [batch_size, 3], ray radii.
        num_samples: int.
        near: jnp.ndarray, [batch_size, 1], near clip.
        far: jnp.ndarray, [batch_size, 1], far clip.
        randomized: bool, use randomized stratified sampling.
        lindisp: bool, sampling linearly in disparity rather than depth.
        ray_shape: string, which shape ray to assume.

    Returns:
        t_vals: jnp.ndarray, [batch_size, num_samples], sampled z values.
        means: jnp.ndarray, [batch_size, num_samples, 3], sampled means.
        covs: jnp.ndarray, [batch_size, num_samples, 3, 3], sampled covariances.
    """
    batch_size = origins.shape[0]

    t_vals = torch.linspace(0., 1., num_samples + 1, device=origins.device)
    if lindisp:
        t_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
    else:
        t_vals = near * (1. - t_vals) + far * t_vals

    if randomized:
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], -1)
        lower = torch.cat([t_vals[..., :1], mids], -1)
        t_rand = torch.rand(batch_size, num_samples + 1, device=origins.device)
        t_vals = lower + (upper - lower) * t_rand
    else:
        # Broadcast t_vals to make the returned shape consistent.
        t_vals = t_vals.expand(batch_size, num_samples + 1)
    if not cast_cone:
        return t_vals
    means, covs = cast_rays(t_vals, origins, directions, radii)
    return t_vals, (means, covs)


def sorted_piecewise_constant_pdf(bins, weights, num_samples, randomized):
    """Piecewise-Constant PDF sampling from sorted bins.

    Args:
        key: jnp.ndarray(float32), [2,], random number generator.
        bins: jnp.ndarray(float32), [batch_size, num_bins + 1].
        weights: jnp.ndarray(float32), [batch_size, num_bins].
        num_samples: int, the number of samples.
        randomized: bool, use randomized samples.

    Returns:
        t_samples: jnp.ndarray(float32), [batch_size, num_samples].
    """
    # Pad each weight vector (only if necessary) to bring its sum to `eps`. This
    # avoids NaNs when the input is zeros or small, but has no effect otherwise.
    device = bins.device
    eps = 1e-5
    weight_sum = torch.sum(weights, dim=-1, keepdims=True)
    padding = torch.maximum(eps - weight_sum, torch.zeros_like(weight_sum))
    weights += padding / weights.shape[-1]
    weight_sum += padding

    # Compute the PDF and CDF for each weight vector, while ensuring that the CDF
    # starts with exactly 0 and ends with exactly 1.
    pdf = weights / weight_sum
    cdf = torch.minimum(torch.ones_like(pdf[..., :-1]), torch.cumsum(pdf[..., :-1], dim=-1))

    pad_shape = list(cdf.shape[:-1]) + [1]
    cdf = torch.cat([
        torch.zeros(*pad_shape, device=device), cdf,
        torch.ones(*pad_shape, device=device)
    ], dim=-1)

    full_shape = list(cdf.shape[:-1]) + [num_samples]
    # Draw uniform samples.
    if randomized:
        eps = 1e-8
        s = 1 / num_samples
        u = torch.arange(num_samples, device=device) * s
        u = u + torch.rand(*full_shape, device=device) * (s - eps)
        # `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u = torch.minimum(u, torch.ones_like(u) - eps)
    else:
        # Match the behavior of jax.random.uniform() by spanning [0, 1-eps].
        u = torch.linspace(0., 1. - eps, num_samples, device=device)
        u = u.expand(*full_shape)

    # Identify the location in `cdf` that corresponds to a random sample.
    # The final `True` index in `mask` will be the start of the sampled interval.
    mask = cdf[..., :, None] <= u[..., None, :]

    def find_interval(x):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0 = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), -2)[0]
        x1 = torch.min(torch.where(torch.logical_not(mask), x[..., None], x[..., -1:, None]), -2)[0]
        return x0, x1

    bins_g0, bins_g1 = find_interval(bins)
    cdf_g0, cdf_g1 = find_interval(cdf)

    t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
    samples = bins_g0 + t * (bins_g1 - bins_g0)
    return samples


def resample_along_rays(origins, directions, radii, t_vals, weights, randomized, cast_cone, stop_grad, resample_padding):
    """Resampling.

    Args:
    origins: jnp.ndarray(float32), [batch_size, 3], ray origins.
    directions: jnp.ndarray(float32), [batch_size, 3], ray directions.
    radii: jnp.ndarray(float32), [batch_size, 3], ray radii.
    t_vals: jnp.ndarray(float32), [batch_size, num_samples+1].
    weights: jnp.array(float32), weights for t_vals
    randomized: bool, use randomized samples.
    ray_shape: string, which kind of shape to assume for the ray.
    stop_grad: bool, whether or not to backprop through sampling.
    resample_padding: float, added to the weights before normalizing.

    Returns:
    t_vals: jnp.ndarray(float32), [batch_size, num_samples+1].
    points: jnp.ndarray(float32), [batch_size, num_samples, 3].
    """
    # Do a blurpool.
    weights_pad = torch.cat([
        weights[..., :1],
        weights,
        weights[..., -1:],
    ], dim=-1)
    weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
    weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

    # Add in a constant (the sampling function will renormalize the PDF).
    weights = weights_blur + resample_padding

    new_t_vals = sorted_piecewise_constant_pdf(
        t_vals,
        weights,
        t_vals.shape[-1],
        randomized,
    )
    if stop_grad:
        new_t_vals = new_t_vals.detach()
    if not cast_cone:
        return new_t_vals

    means, covs = cast_rays(new_t_vals, origins, directions, radii)
    return new_t_vals, (means, covs)


