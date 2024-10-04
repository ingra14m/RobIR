from dev.mip.sampler import *
from dev.misc_old.utils import *


class NeuSRenderModel:

    def __init__(self, neus_model, renderer, global_step=0, anneal_end=50000.0):
        self.neus_model = neus_model
        self.renderer = renderer
        self.global_step = global_step
        self.anneal_end = anneal_end

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            self.global_step += 1
            return np.min([1.0, self.global_step / self.anneal_end])

    def render(self, rays: Rays, white_bkgd=True):
        background_rgb = None
        if white_bkgd:
            background_rgb = torch.ones([1, 3])
        render_out = self.renderer.render(rays.origins, rays.directions, rays.near, rays.far,
                                          background_rgb=background_rgb,
                                          cos_anneal_ratio=self.get_cos_anneal_ratio())

        # weights = weights[..., :mid_z_vals.shape[-1]]
        # distance = (weights * mid_z_vals).sum(dim=-1) / render_out['weight_sum'].unsqueeze(-1)
        # distance = torch.clip(torch.nan_to_num(distance, torch.inf), rays.near, rays.far)
        distance = torch.zeros_like(render_out['color_fine'])
        return {
            "rgb": render_out['color_fine'],
            "dist": distance,
            "acc": render_out['weight_sum'].unsqueeze(-1),
            "weights": render_out['weights'],
            "sim_or_grad": render_out['gradient_error'],
            "means": render_out['mid_z_vals'],
        }


@gin.configurable("Render")
def volume_rendering(rays, nerf,
                     n_sample=64,
                     n_resample=128,
                     n_levels=2,
                     perturb=True,
                     white_bkgd=True,
                     use_mip=True,
                     density_proc=False,
                     rgb_activation=F.sigmoid,
                     density_bias=-1.,
                     density_activation=F.softmax,
                     rgb_padding=0.001,
                     raw_noise_std=0.,
                     lindisp=False,
                     cast_cone=True,
                     stop_level_grad=True,
                     resample_padding=0.01,
                     use_neus_model=False,
                     is_eval=False,
                     ):
    if is_eval:
        perturb = False
        raw_noise_std = 0.

    if use_neus_model:
        if not hasattr(volume_rendering, "__neus_render_model"):
            from dev.neus.neus_renderer import NeuSRenderer
            renderer = NeuSRenderer(nerf.nerf_outside,
                                    nerf.sdf_network,
                                    nerf.deviation_network,
                                    nerf.color_network)
            volume_rendering.__neus_render_model = NeuSRenderModel(nerf, renderer)
        render_model = volume_rendering.__neus_render_model

        return [render_model.render(rays, white_bkgd)]

    ret = []
    for i_level in range(n_levels):
        if i_level == 0:
            # Stratified sampling along rays
            # t_vals, samples = naive_sample_along_rays(
            #     rays.origins,
            #     rays.directions,
            #     n_sample,
            #     rays.near,
            #     rays.far,
            #     perturb,
            #     lindisp,
            # )
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
            # t_vals, samples = naive_resample_along_rays(
            #     rays.origins,
            #     rays.directions,
            #     t_vals,
            #     weights,
            #     perturb,
            # )
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

        if use_mip:
            raw_rgb, raw_density = nerf(samples[0], covs=samples[1], dirs=rays.viewdirs)
        else:
            raw_rgb, raw_density = nerf(samples[0], dirs=rays.viewdirs)

        if density_proc:
            comp_rgb, _, acc, weights, distance, sim_or_grad = raw2outputs(raw_rgb,
                                                                          raw_density,
                                                                          samples[0],
                                                                          nerf,
                                                                          t_vals,
                                                                          rays.directions,
                                                                          raw_noise_std,
                                                                          white_bkgd)
            ret.append({
                "rgb": comp_rgb,
                "dist": distance,
                "acc": acc,
                "weights": weights,
                "sim_or_grad": sim_or_grad,
                "means": samples[0],
            })
            continue

        # Add noise to regularize the density predictions if needed.
        if perturb and (raw_noise_std > 0):
            raw_density += raw_noise_std * torch.randn_like(raw_density)

        # Volumetric rendering.
        rgb = rgb_activation(raw_rgb)
        rgb = rgb * (1 + 2 * rgb_padding) - rgb_padding
        density = density_activation(raw_density + density_bias)

        dirs = rays.directions
        t_mids = 0.5 * (t_vals[..., :-1] + t_vals[..., 1:])
        t_dists = t_vals[..., 1:] - t_vals[..., :-1]
        delta = t_dists * torch.linalg.norm(dirs[..., None, :], dim=-1)
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

        if nan(weights):
            print(nan(alpha))

        ret.append({
            "rgb": comp_rgb,
            "dist": distance,
            "acc": acc,
            "weights": weights,
            "sim_or_grad": torch.ones_like(alpha),
            "means": samples[0],
        })

    return ret


@gin.configurable("DensityProcess")
def raw2outputs(raw_rgb, raw_density, means, nerf, z_vals, rays_d, raw_noise_std=0., white_bkgd=False, use_similarity=True, use_sdf=False):

    def cos_sim(u, v):
        # return (u * v).sum(-1) / (torch.norm(u, dim=-1) + 1.) / (torch.norm(v, dim=-1) + 1.)
        return (u * v).sum(-1) / (torch.norm(u, dim=-1) + 1e-3) / (torch.norm(v, dim=-1) + 1e-3)

    def sim_to_alpha(sim):
        # return F.relu(1. - F.relu(2 * sim))
        return F.relu(1. - F.relu(sim + 0.5))

    raw_density = raw_density.squeeze()
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw_rgb)  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw_density.shape, device=raw_density.device) * raw_noise_std

    if use_similarity:
        # raw_density = noise + raw_density
        if len(raw_density.shape) == 2:
            raw_density = raw_density[:, :, None]
        a_sig = raw_density[:, :-1]
        b_sig = raw_density[:, 1:]

        sim = cos_sim(a_sig, b_sig)
        sig = sim_to_alpha(sim).unsqueeze(-1)

        # padding
        sig = torch.cat([sig, sig[:, -1:]], 1)
        alpha = sig[..., 0]

        rgb = (rgb[:, 1:] + rgb[:, :-1]) / 2
        rgb = torch.cat([rgb, rgb[:, -1:]], 1)
    elif use_sdf:
        batch_size = means.size(0)
        n_samples = means.size(1)
        # Section length
        sdf = raw_density
        gradients = nerf.gradients(means)
        inv_s = nerf.variance
        inv_s = inv_s.expand(batch_size, n_samples)

        dirs = rays_d[:, None, :].expand(means.shape)
        true_cos = (dirs * gradients).sum(-1)

        # auto annealing
        if not hasattr(raw2outputs, "__cos_anneal_ratio"):
            raw2outputs.__cos_anneal_ratio = 0
        cos_anneal_ratio = raw2outputs.__cos_anneal_ratio
        raw2outputs.__cos_anneal_ratio = min(cos_anneal_ratio + 0.001, 1)

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
        sim = gradients
    else:
        alpha = raw2alpha(raw_density + noise, dists)  # [N_rays, N_samples]
        sim = torch.ones_like(alpha)

    Ts = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=dists.device), 1.-alpha + 1e-10], -1), -1)
    weights = alpha * Ts[:, :-1]

    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    mid_z = (z_vals[:, 1:] + z_vals[:, :-1]) / 2
    depth_map = torch.sum(weights * mid_z, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map, device=dists.device), depth_map / (torch.sum(weights, -1) + 1e-8))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map, sim


