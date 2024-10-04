import torch
import torch.nn.functional as F
import collections

Rays = collections.namedtuple(
    'Rays',
    ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far'))


class IComp:

    def radius(self) -> float:
        raise NotImplementedError

    def background(self, x, dirs):
        raise NotImplementedError


class ISDF(IComp):

    def sdf(self, x):
        raise NotImplementedError

    def sdf_and_feat(self, x):
        raise NotImplementedError

    def color(self, x, gradients, dirs, feature_vector):
        raise NotImplementedError

    def grad(self, x):
        raise NotImplementedError

    def dev(self, x):
        raise NotImplementedError


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples, device=bins.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=bins.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def up_sample(rays_o, rays_d, z_vals, sdf, n_importance, inv_s, sphere_radius=1.0):
    """
    Up sampling give a fixed inv_s
    """
    batch_size, n_samples = z_vals.shape
    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
    radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
    inside_sphere = (radius[:, :-1] < sphere_radius) | (radius[:, 1:] < sphere_radius)
    sdf = sdf.reshape(batch_size, n_samples)
    prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
    prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
    mid_sdf = (prev_sdf + next_sdf) * 0.5
    cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

    # ----------------------------------------------------------------------------------------------------------
    # Use min value of [ cos, prev_cos ]
    # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
    # robust when meeting situations like below:
    #
    # SDF
    # ^
    # |\          -----x----...
    # | \        /
    # |  x      x
    # |---\----/-------------> 0 level
    # |    \  /
    # |     \/
    # |
    # ----------------------------------------------------------------------------------------------------------
    prev_cos_val = torch.cat([torch.zeros([batch_size, 1], device=rays_o.device), cos_val[:, :-1]], dim=-1)
    cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
    cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
    cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

    dist = (next_z_vals - prev_z_vals)
    prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
    next_esti_sdf = mid_sdf + cos_val * dist * 0.5
    prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
    next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
    alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones([batch_size, 1], device=rays_o.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]

    z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
    return z_samples


def cat_z_vals(model: ISDF, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
    batch_size, n_samples = z_vals.shape
    _, n_importance = new_z_vals.shape
    pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
    z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
    z_vals, index = torch.sort(z_vals, dim=-1)

    if not last:
        new_sdf = model.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
        sdf = torch.cat([sdf, new_sdf], dim=-1)
        xx = torch.arange(batch_size, device=rays_o.device)[:, None].expand(batch_size,
                                                                            n_samples + n_importance).reshape(-1)
        index = index.reshape(-1)
        sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

    return z_vals, sdf


def render_core_outside(rays_o, rays_d, z_vals, sample_dist, model: IComp, background_rgb=None):
    """
    Render background
    """
    batch_size, n_samples = z_vals.shape

    # Section length
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape).to(rays_o.device)], -1)
    mid_z_vals = z_vals + dists * 0.5

    # Section midpoints
    pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

    dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
    pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)  # batch_size, n_samples, 4

    dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

    pts = pts.reshape(-1, 3 + 1)  # 1 = int(self.n_outside > 0)
    dirs = dirs.reshape(-1, 3)

    density, sampled_color = model.background(pts, dirs)
    alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)
    alpha = alpha.reshape(batch_size, n_samples)
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones([batch_size, 1], device=rays_o.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]
    sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
    color = (weights[:, :, None] * sampled_color).sum(dim=1)
    if background_rgb is not None:
        color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

    return {
        'color': color,
        'sampled_color': sampled_color,
        'alpha': alpha,
        'weights': weights,
    }


def render_core(rays_o,
                rays_d,
                z_vals,
                sample_dist,
                model: ISDF,
                background_alpha=None,
                background_sampled_color=None,
                background_rgb=None,
                cos_anneal_ratio=0.0):
    batch_size, n_samples = z_vals.shape

    # Section length
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape).to(rays_o.device)], -1)
    mid_z_vals = z_vals + dists * 0.5

    # Section midpoints
    pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
    dirs = rays_d[:, None, :].expand(pts.shape)

    pts = pts.reshape(-1, 3)
    dirs = dirs.reshape(-1, 3)

    sdf, feature_vector = model.sdf_and_feat(pts)

    gradients = model.grad(pts).squeeze()
    sampled_color = model.color(pts, gradients, dirs, feature_vector).reshape(batch_size, n_samples, 3)

    inv_s = model.dev(torch.zeros([1, 3], device=rays_o.device))[:, :1].clip(1e-6, 1e6)  # Single parameter
    inv_s = inv_s.expand(batch_size * n_samples, 1)

    sdf_bn = sdf.view(batch_size, n_samples, 1)

    # Estimate signed distances at section points
    estimated_next_sdf = torch.cat([sdf_bn[:, 1:], sdf_bn[:, -1:]], 1).view(-1, 1)
    estimated_prev_sdf = torch.cat([sdf_bn[:, :-1], sdf_bn[:, -1:]], 1).view(-1, 1)

    prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
    next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

    p = prev_cdf - next_cdf
    c = prev_cdf

    alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

    pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
    sphere_radius = model.radius()
    inside_sphere = (pts_norm < sphere_radius).float().detach()
    relax_inside_sphere = (pts_norm < sphere_radius * 1.2).float().detach()

    # Render with background
    if background_alpha is not None:
        alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
        alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
        sampled_color = sampled_color * inside_sphere[:, :, None] + \
                        background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
        sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)
    else:
        # n_outside is 0, but we still need to discard the density outside the sphere
        alpha = alpha * inside_sphere

    weights = alpha * torch.cumprod(
        torch.cat([torch.ones([batch_size, 1], device=rays_o.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]
    weights_sum = weights.sum(dim=-1, keepdim=True)

    color = (sampled_color * weights[:, :, None]).sum(dim=1)
    if background_rgb is not None:  # Fixed background, usually black
        color = color + background_rgb * (1.0 - weights_sum)

    # Eikonal loss
    gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                        dim=-1) - 1.0) ** 2
    gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

    return {
        'color': color,
        'sdf': sdf,
        'dists': dists,
        'gradients': gradients.reshape(batch_size, n_samples, 3),
        's_val': 1.0 / inv_s,
        'mid_z_vals': mid_z_vals,
        'weights': weights,
        'cdf': c.reshape(batch_size, n_samples),
        'gradient_error': gradient_error,
        'inside_sphere': inside_sphere
    }


def render_neus(rays: Rays, model: ISDF,
                cos_anneal_ratio,
                n_samples=64,
                n_importance=64,
                n_outside=32,
                up_sample_steps=4,
                white_bkgd=True,
                lindisp=False,
                perturb=1.0,
                is_eval=False):
    if is_eval:
        perturb = 0

    rays_o, rays_d, near, far = rays.origins, rays.directions, rays.near, rays.far
    batch_size = len(rays_o)
    sample_dist = 2.0 / n_samples  # Assuming the region of interest is a unit sphere
    z_vals = torch.linspace(0.0, 1.0, n_samples, device=rays_o.device)[None, :]
    if lindisp:
        z_vals = 1. / (1. / near * (1. - z_vals) + 1. / far * z_vals)
    else:
        z_vals = near + (far - near) * z_vals

    z_vals_outside = None
    if n_outside > 0:
        z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (n_outside + 1.0), n_outside, device=rays_o.device)

    background_rgb = None
    if white_bkgd:
        background_rgb = torch.ones([1, 3], device=rays_o.device)

    if perturb > 0:
        t_rand = (torch.rand([batch_size, 1], device=rays_o.device) - 0.5)
        z_vals = z_vals + t_rand * 2.0 / n_samples

        if n_outside > 0:
            mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
            upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
            lower = torch.cat([z_vals_outside[..., :1], mids], -1)
            t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]], device=rays_o.device)
            z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

    if n_outside > 0:
        z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / n_samples

    background_alpha = None
    background_sampled_color = None

    # Up sample
    if n_importance > 0:
        with torch.no_grad():
            pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
            sdf = model.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_samples)

            for i in range(up_sample_steps):
                new_z_vals = up_sample(rays_o,
                                       rays_d,
                                       z_vals,
                                       sdf,
                                       n_importance // up_sample_steps,
                                       64 * 2 ** i,
                                       model.radius())
                z_vals, sdf = cat_z_vals(model,
                                         rays_o,
                                         rays_d,
                                         z_vals,
                                         new_z_vals,
                                         sdf,
                                         last=(i + 1 == up_sample_steps))

        n_samples = n_samples + n_importance

    # Background model
    if n_outside > 0:
        z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
        z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
        ret_outside = render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, model)

        background_sampled_color = ret_outside['sampled_color']
        background_alpha = ret_outside['alpha']

    # Render core
    ret_fine = render_core(rays_o,
                           rays_d,
                           z_vals,
                           sample_dist,
                           model,
                           background_rgb=background_rgb,
                           background_alpha=background_alpha,
                           background_sampled_color=background_sampled_color,
                           cos_anneal_ratio=cos_anneal_ratio)

    color_fine = ret_fine['color']
    weights = ret_fine['weights']
    acc = weights.sum(dim=-1)

    gradients = ret_fine['gradients']
    normal = (weights[..., None] * gradients).sum(-2)
    normal = normal / (torch.linalg.norm(normal, dim=-1, keepdim=True) + 0.0001)
    normal[acc > 0.8] = 1.0

    z_mids = ret_fine['mid_z_vals']
    distance = (weights[..., :n_samples + n_importance] * z_mids).sum(dim=-1) / acc
    distance = torch.clip(torch.nan_to_num(distance, torch.inf), near.squeeze(), far.squeeze())

    return {
        'rgb': color_fine,
        'dist': distance,
        "acc": acc,
        'grad_error': ret_fine['gradient_error'],
        'grad': normal,
        'weights': weights,
    }


def wrap_renderer(my_sdf_model, color_fn, model_input, near=1.0, far=6.0, is_eval=False):
    rays_o, rays_d = model_input['points'].view(-1, 3), model_input['dirs'].view(-1, 3)

    rays_o = rays_o * 2.0

    ones = torch.ones_like(rays_o[..., :1])
    zeros_rgb = torch.zeros_like(rays_o)
    rays = Rays(rays_o, rays_d, rays_d, ones * 0.001, ones, ones * near, ones * far)

    neus = my_sdf_model.implicit_network.neus_model

    def color(x, gradients, dirs, feature_vector):
        return color_fn(x * 0.5)

    neus.color = color

    neus.eval()

    if is_eval:
        with torch.no_grad():
            ret = render_neus(rays, neus, 1.0, n_samples=32, n_importance=32, n_outside=0, up_sample_steps=2)
    else:
        ret = render_neus(rays, neus, 1.0, n_samples=32, n_importance=32, n_outside=0, up_sample_steps=2)

    return_obj = {
        'points': rays_o,
        'sdf_output': ones,
        'network_object_mask': ones,
        'object_mask': ones,
    }

    return_obj.update({
        'bg_rgb': zeros_rgb,
        'sg_rgb': ret['rgb'],
        'indir_rgb': zeros_rgb,
        'sg_diffuse_rgb': ret['dist'][..., None].expand(-1, 3),
        'sg_specular_rgb': ret['acc'][..., None].expand(-1, 3),
        'indir_diffuse_rgb': zeros_rgb,
        'indir_specular_rgb': zeros_rgb,
        'normals': ret['grad'],
        'diffuse_albedo': ret['rgb'],
        'roughness': zeros_rgb,
        'surface_mask': ret['acc'] > 0.8,
        'vis_shadow': zeros_rgb,
        'random_xi_roughness': zeros_rgb,
        'random_xi_diffuse_albedo': zeros_rgb,
        'pe_rgb': zeros_rgb,
    })

    return return_obj
