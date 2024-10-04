import torch


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

