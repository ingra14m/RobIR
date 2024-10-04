import os
import sys
from datetime import datetime

import imageio
import numpy as np
import torch
from pyhocon import ConfigFactory
from tensorboardX import SummaryWriter

import utils.general as utils
import utils.plots as plt
from model.sg_render import compute_envmap
from training.tex_module import TexSpaceSampler
import itertools
from model.octree_tracing import OctreeVisModel

mse2psnr = lambda x: -10. * np.log(x + 1e-8) / np.log(10.)


class ClusteredAlbedoTrainRunner:
    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.exps_folder_name = kwargs['exps_folder_name']
        self.batch_size = kwargs['batch_size']
        self.nepochs = self.conf.get_int('train.sg_epoch')
        self.max_niters = kwargs['max_niters']
        self.index = kwargs['index']
        self.chunk = kwargs['chunk']

        self.expname = 'CESR-' + kwargs['expname']

        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join('./', kwargs['exps_folder_name'], self.expname)):
                timestamps = os.listdir(os.path.join('./', kwargs['exps_folder_name'], self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        utils.mkdir_ifnotexists(os.path.join('./', self.exps_folder_name))
        self.expdir = os.path.join('./', self.exps_folder_name, self.expname)
        utils.mkdir_ifnotexists(self.expdir)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.sg_optimizer_params_subdir = "SGOptimizerParameters"
        self.sg_scheduler_params_subdir = "SGSchedulerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.sg_optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.sg_scheduler_params_subdir))

        print('Write tensorboard to: ', os.path.join(self.expdir, self.timestamp))
        self.writer = SummaryWriter(os.path.join(self.expdir, self.timestamp))

        os.system(
            """cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')
        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(
            kwargs['data_split_dir'], kwargs['frame_skip'], split='train')
        print('Finish loading data ...')

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn
                                                            )

        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                           batch_size=1,
                                                           shuffle=False,
                                                           collate_fn=self.train_dataset.collate_fn
                                                           )
        self.plot_only = kwargs['plot_only']

        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=self.conf.get_config('model'))
        if torch.cuda.is_available():
            self.model.cuda()

        from model.sg_envmap_material import SparseAE
        from model.neus_model import SDFNetwork
        import torch.nn.functional as F
        from model.embedder import get_embedder
        self.shadow_embed, in_dim = get_embedder(10)
        self.shadow_net = SDFNetwork(in_dim + 128, 2, 512, 8, [4], 0)
        self.shadow_net.cuda()
        self.normal_net = SDFNetwork(in_dim, 3, 512, 8, [4], 0)
        self.normal_net.cuda()

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))
        lr = self.conf.get_float('train.sg_learning_rate')
        self.c_optimizer = torch.optim.Adam([{"lr": lr, "params": self.model.gamma.parameters()},
                                             {"lr": lr, "params": self.shadow_net.parameters()},
                                             {"lr": lr, "params": self.normal_net.parameters()},
                                             {"lr": lr, "params": self.model.envmap_material_network.parameters()}])
        self.c_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.c_optimizer,
                                                                self.conf.get_list('train.sg_sched_milestones',
                                                                                   default=[]),
                                                                gamma=self.conf.get_float('train.sg_sched_factor',
                                                                                          default=0.0))

        self.start_epoch = 0
        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

            print('Loading pretrained model: ', os.path.join(
                old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth"))

            no_discard = self.conf.get_int("train.dropout_iter") == -1

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            model_state = saved_model_state["model_state_dict"]
            for key in list(filter(
                    lambda x: "cluster" in x or ("spec_brdf" in x and not no_discard),
                    model_state)):
                del model_state[key]
            self.model.load_state_dict(model_state, strict=False)
            self.start_epoch = saved_model_state['epoch']

            saved_shadow_state = torch.load(
                os.path.join(old_checkpnts_dir, str(kwargs['checkpoint']) + "-shadow.pth"))
            self.shadow_net.load_state_dict(saved_shadow_state["model_state_dict"])

            saved_normal_state = torch.load(
                os.path.join(old_checkpnts_dir, str(kwargs['checkpoint']) + "-normal.pth"))
            self.normal_net.load_state_dict(saved_normal_state["model_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.sg_optimizer_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.c_optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.sg_scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.c_scheduler.load_state_dict(data["scheduler_state_dict"])
            self.no_normal = False
        else:
            self.no_normal = False
            mat_dir = os.path.join('./', kwargs['exps_folder_name'], 'Norm-' + kwargs['expname'])
            if os.path.exists(mat_dir):
                timestamps = os.listdir(mat_dir)
                if len(timestamps) < 1:
                    self.no_normal = True
                else:
                    timestamp = sorted(timestamps)[-1]  # using the newest training result
            else:
                print('No Norm_model pretrain, please train it first!')
                self.no_normal = True

            mat_dir = os.path.join('./', kwargs['exps_folder_name'], 'PBR-' + kwargs['expname'])
            if os.path.exists(mat_dir):
                timestamps = os.listdir(mat_dir)
                timestamp = sorted(timestamps)[-1]  # using the newest training result
            else:
                print('No Mat_model pretrain, please train it first!')
                exit(0)

            old_checkpnts_dir = os.path.join(mat_dir, timestamp, 'checkpoints')

            pth_path = os.path.join(old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth")
            print('Loading pretrained model: ', pth_path)

            no_discard = self.conf.get_int("train.dropout_iter") == -1
            softplus_act = self.conf.get_int("train.dropout_iter") == -2
            if softplus_act:
                self.model.envmap_material_network.spec_brdf_encoder_layer.lc_act = F.softplus

            saved_model_state = torch.load(pth_path)
            model_state = saved_model_state["model_state_dict"]
            for key in list(filter(
                    lambda x: "cluster" in x or ("spec_brdf" in x and not no_discard),
                    model_state)):
                del model_state[key]
            self.model.load_state_dict(model_state, strict=False)

            if not self.no_normal:  # Load the pth model of the Norm Stage
                old_checkpnts_dir = os.path.join(mat_dir, timestamp, 'checkpoints')

                pth_path = os.path.join(old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth")
                print('Loading pretrained model: ', pth_path)

                saved_model_state = torch.load(pth_path)
                model_state = saved_model_state["model_state_dict"]
                for key in list(filter(lambda x: "normal_decoder_layer" not in x, model_state)):
                    del model_state[key]
                self.model.load_state_dict(model_state, strict=False)

            if self.conf.get_bool('train.ces_pretrained_light'):
                ces_dir = os.path.join('./', kwargs['exps_folder_name'], 'DIR-' + kwargs['expname'])
                if os.path.exists(ces_dir):
                    timestamps = os.listdir(ces_dir)
                    timestamp = sorted(timestamps)[-1]  # using the newest training result
                else:
                    print('No Mat_model pretrain, please train it first!')
                    exit(0)

                old_checkpnts_dir = os.path.join(ces_dir, timestamp, 'checkpoints')

                pth_path = os.path.join(old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth")
                print('Loading pretrained model: ', pth_path)

                saved_model_state = torch.load(pth_path)
                model_state = saved_model_state["model_state_dict"]
                for key in list(filter(lambda x: "cluster" in x, model_state)):
                    del model_state[key]
                self.model.load_state_dict(model_state, strict=False)
            else:  # Load the latest pth model of Vis Stage
                illum_dir = os.path.join('./', kwargs['exps_folder_name'], 'Vis-' + kwargs['expname'])
                if os.path.exists(illum_dir):
                    timestamps = os.listdir(illum_dir)
                    timestamp = sorted(timestamps)[-1]  # using the newest training result
                else:
                    print('No illum_model pretrain, please train it first!')
                    exit(0)

                # reload pretrain geometry model & indirect illumination model
                illum_path = os.path.join(illum_dir, timestamp) + '/checkpoints/ModelParameters/latest.pth'
                print('Reloading indirect illumination from: ', illum_path)
                model = torch.load(illum_path)['model_state_dict']

                if not self.conf.get_bool('model.use_neus'):
                    geometry = {k.split('network.')[1]: v for k, v in model.items() if 'implicit_network' in k}
                    radiance = {k.split('network.')[1]: v for k, v in model.items() if 'rendering_network' in k}
                    self.model.implicit_network.load_state_dict(geometry)
                    self.model.rendering_network.load_state_dict(radiance)

                incident_radiance = {k.split('network.')[1]: v for k, v in model.items() if
                                     'indirect_illum_network' in k}
                visibility = {k.split('network.')[1]: v for k, v in model.items() if 'visibility_network' in k}
                self.model.indirect_illum_network.load_state_dict(incident_radiance)
                self.model.visibility_network.load_state_dict(visibility)

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.ckpt_freq = self.conf.get_int('train.ckpt_freq')

        self.tex_space_sampler = TexSpaceSampler(self)
        self.white_light = self.conf.get_bool('train.white_light')

    def save_checkpoints(self, epoch):
        torch.save({"epoch": epoch, "model_state_dict": self.shadow_net.state_dict()},
                   os.path.join(self.checkpoints_path, str(epoch) + "-shadow.pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.shadow_net.state_dict()},
            os.path.join(self.checkpoints_path, "latest-shadow.pth"))

        torch.save({"epoch": epoch, "model_state_dict": self.normal_net.state_dict()},
                   os.path.join(self.checkpoints_path, str(epoch) + "-normal.pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.normal_net.state_dict()},
            os.path.join(self.checkpoints_path, "latest-normal.pth"))

        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.c_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.sg_optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.c_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.sg_optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.c_scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.sg_scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.c_scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.sg_scheduler_params_subdir, "latest.pth"))

    def plot_to_disk(self, from_light=False, chunk=1024):
        self.model.eval()
        self.is_training = False
        sampling_idx = self.train_dataset.sampling_idx
        self.train_dataset.change_sampling_idx(-1)
        # indices, model_input, ground_truth = next(
        #     itertools.islice(self.plot_dataloader, self.index, None))
        for idx, item in enumerate(self.plot_dataloader):
            indices, model_input, ground_truth = item
            for key in model_input.keys():
                model_input[key] = model_input[key].cuda()

            split = utils.split_input(model_input, self.total_pixels, n_pixels=chunk)

            if from_light:
                rays_o, rays_d = self.model.dir_light.orthogonal_rays(800, 0.6)
                input_dict = {
                    'points': rays_o[None],
                    'dirs': rays_d[None]
                }
                split = utils.split_gt(input_dict, self.total_pixels, *input_dict.keys(), device='cuda')

            res = []
            for i, s in enumerate(split):
                s['hdr_shift'] = self.model.gamma.hdr_shift.as_input().expand(s["uv"].shape[1], 1)
                out = self.model(s, trainstage="Material", lin_diff=False, fun_spec=False, train_spec=self.train_spec)
                trace_outputs = self.model.trace_radiance(out, nsamp=8)
                _, pred_vis = torch.max(trace_outputs["pred_vis"].detach(), dim=-1)
                pred_vis = torch.mean(pred_vis.float(), axis=1)
                model_outputs = out
                indir_rgb = model_outputs['indir_rgb']
                roughness = model_outputs['roughness'][..., 0:1]  # monochrome assumption
                diffuse_albedo = model_outputs["diffuse_albedo"]

                normals = model_outputs['normals' if self.no_normal else 'normal_map'].detach()

                sg_rgb = model_outputs['sg_rgb']
                pred_rgb = sg_rgb + indir_rgb

                # hdr -> ldr
                sg_rgb = self.model.gamma.hdr_shift.hdr2ldr(sg_rgb)

                res.append({
                    'normals': normals.detach(),
                    'pred_vis': pred_vis.detach(),
                    'roughness': roughness.detach().expand(diffuse_albedo.shape),
                    'diffuse_albedo': diffuse_albedo.detach(),
                    'indir_rgb': indir_rgb.detach(),
                    'sg_rgb': sg_rgb.detach(),
                    'pred_rgb': pred_rgb.detach(),
                    'vis_shadow': model_outputs['vis_shadow'].detach(),
                })

            batch_size = ground_truth['rgb'].shape[0]
            model_outputs = utils.merge_output(res, self.total_pixels, batch_size)

            plt.plot_cesr(
                model_outputs,
                ground_truth['rgb'],
                self.plots_dir,
                self.cur_iter,
                self.img_res,
                index=idx,
            )

        # log environment map
        lgtSGs = self.model.envmap_material_network.get_light()
        envmap = compute_envmap(lgtSGs=lgtSGs,
                                H=256, W=512, upper_hemi=self.model.envmap_material_network.upper_hemi)
        envmap = envmap.cpu().numpy()
        imageio.imwrite(os.path.join(self.plots_dir, 'envmap1_{}.png'.format(self.cur_iter)), envmap)
        imageio.imwrite(os.path.join(self.plots_dir, 'envmap1_{}.exr'.format(self.cur_iter)), envmap)

        # comp_dir = os.path.join(self.plots_dir, f'comp_{self.cur_iter}')
        # os.makedirs(comp_dir, exist_ok=True)
        # for k in model_outputs:
        #     if "mask" not in k:
        #         imageio.imwrite(os.path.join(comp_dir, k + '.exr'), model_outputs[k].view(800, 800, 3).cpu().numpy())
        # imageio.imwrite(os.path.join(comp_dir, 'gt.exr'), ground_truth['rgb'].view(800, 800, 3).cpu().numpy())

        lobes = lgtSGs[..., :3]
        mus = torch.relu(lgtSGs[..., -3:]).norm(dim=-1, keepdim=True)
        lobe = (lobes * mus).sum(-2) / mus.sum(-2)
        print("[mean lobe]", lobe)

        self.model.train()
        self.is_training = True
        self.train_dataset.sampling_idx = sampling_idx

    def pbr_step(self, model_outputs, ground_truth):
        loss = 0.

        pbar_loss_dict = {}

        if self.cur_iter > 500:
            loss_output = self.loss(model_outputs, ground_truth, mat_model=self.model.envmap_material_network,
                                    train_idr=False, train_spec=self.train_spec,
                                    hdr_fn=self.model.gamma.hdr_shift.hdr2ldr)

            sg_rgb_loss = loss_output["loss"]
            loss = loss + sg_rgb_loss
            pbar_loss_dict['rgb_loss'] = sg_rgb_loss
            pbar_loss_dict["psnr"] = mse2psnr(loss_output['sg_rgb_loss'].item())

            smooth_w = self.conf.get_float('train.explore_smooth')
            kl_w = self.conf.get_float('train.explore_kl')

            if self.prefit_option() == "project":
                smooth_w = self.conf.get_float('train.proj_smooth')
                kl_w = self.conf.get_float('train.proj_kl')

            kl_loss = loss_output["kl_loss"] * kl_w
            smooth_loss = loss_output["latent_smooth_loss"] * smooth_w

            loss = loss + kl_loss + smooth_loss
            pbar_loss_dict["kl_loss"] = kl_loss
            pbar_loss_dict["smooth_loss"] = smooth_loss

            pbar_loss_dict["spec_refl"] = self.model.envmap_material_network.specular_reflectance.detach().item()
            pbar_loss_dict["hdr_shift"] = self.model.gamma.hdr_shift.as_input().detach().item()

            # surface_mask = model_outputs["surface_mask"]
            # points_select = model_outputs["points"][surface_mask][:64]
            # cov_loss = self.cov_loss(points_select) * 64 / surface_mask.sum()
            # loss = loss + cov_loss
            # pbar_loss_dict["cov_loss"] = cov_loss

        # reg_loss = self.shadow_net.kl_smooth_loss(model_outputs["points"][model_outputs["surface_mask"]], 1.0, 1.0)
        # loss = loss + reg_loss
        # pbar_loss_dict["reg_loss"] = reg_loss

        sv_loss = model_outputs["gradient_error"]
        loss = loss + sv_loss
        pbar_loss_dict["sv_loss"] = sv_loss

        return loss, pbar_loss_dict

    def cov_loss(self, points):
        nsamp = 32
        points = points[None].expand(nsamp, -1, 3).reshape(-1, 3)
        points = points + torch.randn_like(points) * 0.01

        shadow_embed = self.shadow_embed(points.detach())[:, None, :].expand(-1, 128, -1)
        labels = torch.eye(128, device=shadow_embed.device)[None].expand(shadow_embed.shape[0], -1, -1)
        shadow_embed = torch.cat([shadow_embed, labels], -1)
        diffuse_vis = torch.sigmoid(self.shadow_net(shadow_embed.reshape(-1, shadow_embed.shape[-1])))

        sg_envmap_material = self.model.envmap_material_network(points, train_spec=self.train_spec)
        diffuse_albedo = sg_envmap_material['sg_diffuse_albedo']

        light_vis = diffuse_vis.reshape(-1, 128, 1).expand(-1, 128, 3)
        lgtSGs = sg_envmap_material['sg_lgtSGs'][None, :, -3:]
        vis_shadow = (light_vis * lgtSGs).sum(1) / torch.clamp(lgtSGs.sum(1), 1e-4)

        v = vis_shadow.reshape(nsamp, -1, 3)
        a = diffuse_albedo.reshape(nsamp, -1, 3)

        cov = ((a * v).mean(0) - a.mean(0) * v.mean(0))  # Cov: E(X*Y) - E(X)*E(Y)
        cov = (cov + 1e-4) / torch.sqrt(a.var(0) * v.var(0) + 1e-4)  # Rel: Cov(X, Y) / v(D(X)*D(Y))
        cov = cov.mean()
        return cov.abs() * 0.01

    def white_loss(self, lgtSGs):
        lgt = torch.abs(lgtSGs[..., -3:])
        mu = lgt.norm(dim=-1, keepdim=True) + 1e-4
        return (lgt / mu).var(-1).mean() * 0.01

    def get_sg_render(self, points, view_dirs, indir_lgtSGs, albedo_ratio=None,
                      fun_spec=False, lin_diff=False, train_spec=False, indir_integral=None, **kwargs):
        from model.sg_render import render_with_all_sg
        view_dirs = view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-6)
        normals = self.model.get_idr_render(points, view_dirs, normal_only=True)
        normals = normals / torch.clamp(torch.norm(normals, dim=-1, keepdim=True), 1e-4)

        ret = {'normals': normals, }

        assert train_spec == self.train_spec

        # sg renderer
        sg_envmap_material = self.model.envmap_material_network(points,
                                                                train_spec=train_spec)

        lgtSGs = sg_envmap_material['sg_lgtSGs']

        indir_lgtSGs = torch.cat([indir_lgtSGs[..., :4], indir_lgtSGs[..., 4:]], -1)
        indir_integral = indir_integral * 2 * np.pi

        diffuse_albedo = sg_envmap_material['sg_diffuse_albedo']
        roughness = sg_envmap_material['sg_roughness']
        metallic = sg_envmap_material['sg_metallic']
        normal_map = sg_envmap_material['sg_normal_map']

        normal_map = normal_map.detach()

        shadow_embed = self.shadow_embed(points.detach())[:, None, :].expand(-1, 128, -1)
        labels = torch.eye(128, device=shadow_embed.device)[None].expand(shadow_embed.shape[0], -1, -1)
        shadow_embed = torch.cat([shadow_embed, labels], -1)

        if not self.is_training:
            with torch.no_grad():
                diffuse_vis = self.shadow_net(shadow_embed.reshape(-1, shadow_embed.shape[-1]))
                normal_new = self.normal_net(self.shadow_embed(points.detach()))
        else:
            diffuse_vis = self.shadow_net(shadow_embed.reshape(-1, shadow_embed.shape[-1]))
            normal_new = self.normal_net(self.shadow_embed(points.detach()))
        normal_new = normal_new / torch.clamp(normal_new.norm(dim=-1, keepdim=True), 1e-4)
        diffuse_vis = torch.softmax(diffuse_vis, -1)[..., 1]

        sg_ret = render_with_all_sg(points=points.detach(),
                                    normal=normal_new if self.cur_iter > 1000 else normal_map,
                                    viewdirs=view_dirs,
                                    lgtSGs=lgtSGs,
                                    indir_integral=indir_integral,
                                    specular_reflectance=sg_envmap_material['sg_specular_reflectance'].abs(),
                                    roughness=roughness,
                                    diffuse_albedo=diffuse_albedo,
                                    indir_lgtSGs=indir_lgtSGs,
                                    VisModel=self.model.visibility_network,
                                    fun_spec=False,
                                    lin_diff=True,
                                    testing=not self.is_training,
                                    metallic=None,
                                    diffuse_vis=diffuse_vis,
                                    prefit=self.prefit_option(),
                                    argmax_vis=self.conf.get_bool("train.argmax_vis"))
        sg_ret["sg_rgb"] = sg_ret["sg_diffuse_rgb"] * diffuse_albedo / np.pi + sg_ret["sg_specular_rgb"]
        sg_ret["indir_rgb"] = sg_ret["indir_diffuse_rgb"] * diffuse_albedo / np.pi + sg_ret["indir_specular_rgb"]

        supervise = sg_ret['supervise']

        if self.white_light and self.prefit_option() != "warmup":
            supervise = supervise + self.white_loss(lgtSGs)

        supervise = supervise + ((normal_map - normal_new) ** 2).mean()

        ret.update(sg_ret)
        # ret["vis_shadow"] = diffuse_lgt.detach()
        ret.update({'diffuse_albedo': diffuse_albedo,
                    'roughness': roughness,
                    'metallic': metallic,
                    'normal_map': normal_new,
                    'gradient_error': supervise,
                    'random_xi_roughness': sg_envmap_material['random_xi_roughness'],
                    'random_xi_metallic': sg_envmap_material['random_xi_metallic'],
                    'random_xi_diffuse_albedo': sg_envmap_material['random_xi_diffuse_albedo']})

        return ret

    def is_explore_step(self):
        explore_rate = self.conf.get_int('train.explore_iter')
        project_rate = self.conf.get_int('train.proj_iter')
        if self.cur_iter > 500:
            if self.cur_iter % (explore_rate + project_rate) >= project_rate:
                return True
        return False

    def prefit_option(self):
        if not self.is_explore_step():
            if self.cur_iter <= 500:
                return "warmup"
            return "project"
        return "explore"

    def run(self):
        trace_vis = False
        # render_path = True
        #
        # if render_path:
        #     self.index = 0

        extra_epoch = 1

        print("training...")
        self.cur_iter = self.start_epoch * len(self.train_dataloader)

        if self.cur_iter > 0:
            print("[SL]", "use octree vis")
            # trace_vis = True
            extra_epoch = self.start_epoch + 1

        if hasattr(self.model.ray_tracer, "generate"):
            self.model.ray_tracer.generate(lambda x: self.model.implicit_network(x)[:, 0], self.tex_space_sampler)
        if hasattr(self.model, "octree_ray_tracer"):
            self.model.octree_ray_tracer.generate(lambda x: self.model.implicit_network(x)[:, 0],
                                                  self.tex_space_sampler)

        if trace_vis:
            self.model.visibility_network = OctreeVisModel(self.model.octree_ray_tracer)

        # self.model.gamma.hdr_shift.fit_data(self.train_dataset)
        self.model.get_sg_render = self.get_sg_render
        self.train_spec = True
        self.is_training = True

        if self.plot_only:
            self.save_checkpoints(10000)
            self.plot_to_disk(chunk=self.chunk)
            return

        dropout_iter = self.conf.get_int("train.dropout_iter")

        for epoch in range(self.start_epoch, self.nepochs + extra_epoch):
            self.train_dataset.change_sampling_idx(self.num_pixels)

            if self.cur_iter > self.max_niters:
                self.save_checkpoints(epoch)
                self.plot_to_disk(chunk=self.chunk)
                print('Training has reached max number of iterations: {}; exiting...'.format(self.cur_iter))
                exit(0)

            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                if self.cur_iter % self.ckpt_freq == 0:
                    self.save_checkpoints(epoch)

                if self.cur_iter % self.plot_freq == 0 and self.cur_iter > 0:  # don't plot iterator 0
                    self.plot_to_disk(chunk=self.chunk)

                for key in model_input.keys():
                    model_input[key] = model_input[key].cuda()

                model_input['hdr_shift'] = self.model.gamma.hdr_shift.as_input().expand(model_input["uv"].shape[1], 1)
                model_outputs = self.model(model_input, trainstage="Material", fun_spec=False, lin_diff=False,
                                           train_spec=self.train_spec)

                loss, pbar_loss_dict = self.pbr_step(model_outputs, ground_truth)

                self.c_optimizer.zero_grad()
                loss.backward()
                self.c_optimizer.step()

                if self.cur_iter % 50 == 0:
                    post_fix = ", ".join(
                        map(lambda it: f"{it[0]}={it[1].item() if isinstance(it[1], torch.Tensor) else it[1]}",
                            pbar_loss_dict.items()))
                    print('{0} [{1}] ({2}/{3}): loss = {4}, sg_lr = {5}, '
                          .format(self.expname, epoch, indices, self.n_batches,
                                  loss.item(), self.c_scheduler.get_last_lr()[0]) + post_fix)

                self.cur_iter += 1
                self.c_scheduler.step()

                if dropout_iter > 0 and self.cur_iter % dropout_iter == 0:
                    net = self.model.envmap_material_network.spec_brdf_encoder_layer
                    net.var = (torch.rand_like(net.var) > 0.8).float()
