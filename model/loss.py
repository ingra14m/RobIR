import torch
from torch import nn
from torch.nn import functional as F
from model.embedder import get_embedder


class InvLoss(nn.Module):
    def __init__(self, idr_rgb_weight, eikonal_weight, mask_weight, alpha,
                 sg_rgb_weight, kl_weight, latent_smooth_weight,
                 brdf_multires=10, loss_type='L1'):
        super().__init__()
        self.idr_rgb_weight = idr_rgb_weight
        self.eikonal_weight = eikonal_weight
        self.mask_weight = mask_weight
        self.alpha = alpha

        self.sg_rgb_weight = sg_rgb_weight
        self.kl_weight = kl_weight
        self.latent_smooth_weight = latent_smooth_weight
        self.brdf_multires = brdf_multires

        if loss_type == 'L1':
            print('Using L1 loss for comparing images!')
            self.img_loss = nn.L1Loss(reduction='sum')
        elif loss_type == 'L2':
            print('Using L2 loss for comparing images!')
            self.img_loss = nn.MSELoss(reduction='sum')
        else:
            raise Exception('Unknown loss_type!')

    def get_rgb_loss(self, rgb_values, rgb_gt, network_object_mask, object_mask):
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        rgb_values = rgb_values[network_object_mask & object_mask]
        rgb_gt = rgb_gt.reshape(-1, 3)[network_object_mask & object_mask]

        # rgb_values = rgb_values[1:] - rgb_values[:-1]
        # rgb_gt = rgb_gt[1:] - rgb_gt[:-1]

        rgb_loss = self.img_loss(rgb_values, rgb_gt) / float(object_mask.shape[0])
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        if grad_theta.shape[0] == 0:
            return torch.tensor(0.0).cuda().float()

        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_mask_loss(self, sdf_output, network_object_mask, object_mask):
        mask = ~(network_object_mask & object_mask)
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        sdf_pred = -self.alpha * sdf_output[mask]
        gt = object_mask[mask].float()
        mask_loss = (1 / self.alpha) * F.binary_cross_entropy_with_logits(
            sdf_pred.squeeze(), gt, reduction='sum') / float(object_mask.shape[0])
        return mask_loss

    def get_latent_smooth_loss(self, model_outputs):
        d_diff = model_outputs['diffuse_albedo']
        d_rough = model_outputs['roughness'][..., 0]
        d_xi_diff = model_outputs['random_xi_diffuse_albedo']
        d_xi_rough = model_outputs['random_xi_roughness'][..., 0]
        loss = nn.L1Loss()(d_diff, d_xi_diff) + nn.L1Loss()(d_rough, d_xi_rough) * 0.2  # trick 0.2 for roughness
        return loss

    def get_normal_loss(self, model_outputs):
        surface_mask = model_outputs['surface_mask']
        d_normal = model_outputs['normal_map'][surface_mask]
        normal = model_outputs['normals'][surface_mask]
        return nn.MSELoss()(d_normal, normal)

    def kl_divergence(self, rho, rho_hat):
        rho_hat = torch.mean(torch.sigmoid(rho_hat), 0)
        rho = torch.tensor([rho] * len(rho_hat)).cuda()
        return torch.mean(
            rho * torch.log(rho / (rho_hat + 1e-4)) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat + 1e-4)))

    def get_kl_loss(self, model, points, train_spec):
        loss = 0
        embed_fn, _ = get_embedder(self.brdf_multires)

        if train_spec:
            values = model.brdf_embed_fn(points)  # pe
            values = model.spec_brdf_encoder_layer.encode(values)
            loss += self.kl_divergence(0.05, values)

        else:
            values = model.brdf_embed_fn(points)
            values = model.brdf_encoder_layer.encode(values)
            loss += self.kl_divergence(0.05, values)

        return loss

    def forward(self, model_outputs, ground_truth, mat_model=None, train_idr=False, train_spec=False, hdr_fn=None):
        rgb_gt = ground_truth['rgb'].cuda()
        network_object_mask = model_outputs['network_object_mask']
        object_mask = model_outputs['object_mask']

        pred_rgb = model_outputs['sg_rgb'] + model_outputs['indir_rgb']

        # hdr -> ldr
        if hdr_fn is not None:
            pred_rgb = hdr_fn(pred_rgb)
        else:
            pred_rgb = pred_rgb / (pred_rgb + 1)  # experimental formular

        sg_rgb_loss = self.get_rgb_loss(pred_rgb, rgb_gt, network_object_mask, object_mask)

        latent_smooth_loss = self.get_latent_smooth_loss(model_outputs)  # train roughness and albedo
        kl_loss = self.get_kl_loss(mat_model, model_outputs['points'][network_object_mask], train_spec)  # train brdf
        normal_loss = self.get_normal_loss(model_outputs)

        loss = self.sg_rgb_weight * sg_rgb_loss

        output = {
            'sg_rgb_loss': sg_rgb_loss,
            'kl_loss': self.kl_weight * kl_loss,
            'latent_smooth_loss': self.latent_smooth_weight * latent_smooth_loss,
            'normal_loss': normal_loss,
            'loss': loss}

        return output


def query_indir_illum(lgtSGs, sample_dirs):
    nsamp = sample_dirs.shape[1]
    nlobe = lgtSGs.shape[1]
    lgtSGs = lgtSGs.unsqueeze(-3).expand(-1, nsamp, -1, -1)
    sample_dirs = sample_dirs.unsqueeze(-2).expand(-1, -1, nlobe, -1)

    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True))
    lgtSGLambdas = lgtSGs[..., 3:4]
    lgtSGMus = lgtSGs[..., -3:]  # positive values

    pred_radiance = lgtSGMus * torch.exp(
        lgtSGLambdas * (torch.sum(sample_dirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
    pred_radiance = torch.sum(pred_radiance, dim=2)
    return pred_radiance


class IllumLoss(nn.Module):
    def __init__(self, loss_type='L1'):
        super().__init__()
        if loss_type == 'L1':
            print('Using L1 loss for comparing radiance!')
            self.rgb_loss = nn.L1Loss(reduction='mean')
        elif loss_type == 'L2':
            print('Using L2 loss for comparing radiance!')
            self.rgb_loss = nn.MSELoss(reduction='mean')
        else:
            raise Exception('Unknown loss_type!')

    def forward(self, model_outputs, trace_outputs, anneal_t):
        # compute indirect illumination loss
        indir_mask = trace_outputs["indir_mask"]
        points_mask = model_outputs["network_object_mask"]
        lgtSGs = model_outputs["indirect_sgs"][points_mask]

        # compute radiance loss
        gt_radiance = trace_outputs['trace_radiance'][indir_mask] + anneal_t

        sample_dirs = trace_outputs['sample_dirs']
        pred_radiance = query_indir_illum(lgtSGs, sample_dirs)  # compute radiance by sg and dir

        radiance_loss = self.rgb_loss(gt_radiance, pred_radiance[indir_mask[points_mask]])
        int_env = model_outputs['indir_integral'][points_mask]
        gt_int_env = trace_outputs['gt_integral'][points_mask]
        radiance_loss = radiance_loss + self.rgb_loss(gt_int_env, int_env)

        # compute visibility loss
        gt_vis = (~trace_outputs['gt_vis'][points_mask]).long().reshape(-1)
        pred_vis = trace_outputs['pred_vis'][points_mask].reshape(-1, 2)

        visibility_loss = nn.CrossEntropyLoss()(pred_vis, gt_vis)

        return radiance_loss, visibility_loss
