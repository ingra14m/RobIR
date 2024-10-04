import os
import sys
from datetime import datetime

import itertools
import imageio
import numpy as np
import torch
from pyhocon import ConfigFactory
from tensorboardX import SummaryWriter

import utils.general as utils
import utils.plots as plt
from model.sg_render import compute_envmap
from training.tex_module import TexSpaceSampler

mse2psnr = lambda x: -10. * np.log(x + 1e-8) / np.log(10.)


class NormalTrainRunner:
    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.exps_folder_name = kwargs['exps_folder_name']
        self.batch_size = kwargs['batch_size']
        self.nepochs = self.conf.get_int('train.sg_epoch')
        self.max_niters = kwargs['max_niters']
        self.index = kwargs['index']
        self.plot_only = kwargs['plot_only']

        self.expname = 'Norm-' + kwargs['expname']

        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':  # continue the training process
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
        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(kwargs['data_split_dir'],
                                                                                          kwargs['frame_skip'],
                                                                                          split='train')
        print('Finish loading data ...')

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn)

        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                           batch_size=1,
                                                           shuffle=False,
                                                           collate_fn=self.train_dataset.collate_fn)

        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=self.conf.get_config('model'))
        if torch.cuda.is_available():
            self.model.cuda()

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))
        lr = self.conf.get_float('train.sg_learning_rate')
        self.c_optimizer = torch.optim.Adam([{"lr": lr, "params": self.model.gamma.parameters()},
                                             {"lr": lr, "params": self.model.implicit_network.parameters()},
                                             {"lr": lr, "params": self.model.envmap_material_network.parameters()},
                                             ])
        self.c_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.c_optimizer,
                                                                self.conf.get_list('train.sg_sched_milestones',
                                                                                   default=[]),
                                                                gamma=self.conf.get_float('train.sg_sched_factor',
                                                                                          default=0.0))

        self.start_epoch = 0
        if is_continue:  # continue training
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

            print('Loading pretrained model: ',
                  os.path.join(old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth"))

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = saved_model_state['epoch']

            data = torch.load(os.path.join(old_checkpnts_dir,
                                           self.sg_optimizer_params_subdir,
                                           str(kwargs['checkpoint']) + ".pth"))
            self.c_optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(os.path.join(old_checkpnts_dir,
                                           self.sg_scheduler_params_subdir,
                                           str(kwargs['checkpoint']) + ".pth"))
            self.c_scheduler.load_state_dict(data["scheduler_state_dict"])

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.ckpt_freq = self.conf.get_int('train.ckpt_freq')

        self.tex_space_sampler = TexSpaceSampler(self)
        self.minimum_mem = True

    def save_checkpoints(self, epoch):
        torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict()},
                   os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict()},
                   os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))
        torch.save({"epoch": epoch, "optimizer_state_dict": self.c_optimizer.state_dict()},
                   os.path.join(self.checkpoints_path, self.sg_optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save({"epoch": epoch, "optimizer_state_dict": self.c_optimizer.state_dict()},
                   os.path.join(self.checkpoints_path, self.sg_optimizer_params_subdir, "latest.pth"))
        torch.save({"epoch": epoch, "scheduler_state_dict": self.c_scheduler.state_dict()},
                   os.path.join(self.checkpoints_path, self.sg_scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save({"epoch": epoch, "scheduler_state_dict": self.c_scheduler.state_dict()},
                   os.path.join(self.checkpoints_path, self.sg_scheduler_params_subdir, "latest.pth"))

    def plot_to_disk(self, from_light=False):
        self.model.eval()
        self.is_training = False
        if not self.plot_only:
            sampling_idx = self.train_dataset.sampling_idx
            self.train_dataset.change_sampling_idx(-1)
        indices, model_input, ground_truth = next(
            itertools.islice(self.plot_dataloader, self.index, None))  # explicitly set the index

        for key in model_input.keys():
            model_input[key] = model_input[key].cuda()

        split = utils.split_input(model_input, self.total_pixels)

        if from_light:
            rays_o, rays_d = self.model.dir_light.orthogonal_rays(800, 0.6)
            input_dict = {'points': rays_o[None], 'dirs': rays_d[None]}
            split = utils.split_gt(input_dict, self.total_pixels, *input_dict.keys(), device='cuda')

        res = []
        for i, s in enumerate(split):
            out = self.model(s,
                             trainstage="Material",
                             lin_diff=False,
                             fun_spec=False,
                             train_spec=self.train_spec)

            model_outputs = out
            points = out['points'].detach()
            object_mask = out['surface_mask']
            ray_dirs = out['ray_dirs']
            # sample_normals = s['normals']  # get from the mesh in the neus stage(can be processed by ourselves)
            # mask = s['object_mask']

            normals = model_outputs['normal_map'].detach()

            neus_normals = torch.ones_like(normals)

            with torch.no_grad():
                pnts = points[object_mask]
                dirs = ray_dirs[object_mask]

                if self.minimum_mem:
                    _, neus_normals[object_mask], _ = self.get_neus_surface(
                        pnts, dirs, torch.zeros_like(dirs))
                else:
                    _, neus_normals[object_mask] = self.get_neus_with_grad(
                        pnts)

            sg_rgb = out['sg_rgb'].detach() + out['indir_rgb'].detach()
            sg_rgb = torch.clamp(sg_rgb, 0, 1)

            res.append({
                'normals': normals.detach(),
                'normal_neus': neus_normals.detach(),
                # 'sample_normals': sample_normals[mask].detach(),
            })

        batch_size = ground_truth['rgb'].shape[0]
        model_outputs = utils.merge_output(res, self.total_pixels, batch_size)

        plt.plot_norm(
            model_outputs,
            ground_truth['rgb'],
            self.plots_dir,
            self.cur_iter,
            self.img_res,
        )

        # log environment map
        lgtSGs = self.model.envmap_material_network.get_light()
        envmap = compute_envmap(lgtSGs=lgtSGs,
                                H=256,
                                W=512,
                                upper_hemi=self.model.envmap_material_network.upper_hemi)
        envmap = envmap.cpu().numpy()
        imageio.imwrite(os.path.join(self.plots_dir, 'envmap1_{}.png'.format(self.cur_iter)), envmap)

        self.model.train()
        self.is_training = True
        self.train_dataset.sampling_idx = sampling_idx

    def get_neus_surface(self,
                         points,
                         view_dirs,
                         pred_normals,
                         n_samp=32,
                         dist=0.05):
        t = torch.linspace(0, dist, n_samp, device=points.device)[:, None]
        xs = points[..., None, :] - t * view_dirs[..., None, :]
        xs = xs.view(-1, 3)

        sdfs = self.model.implicit_network(xs)[..., :1].view(-1, n_samp, 1)
        normals = self.model.implicit_network.gradient(xs).view(-1, n_samp, 3)

        estimated_next_sdf = torch.cat([sdfs[:, 1:], sdfs[:, -1:]],
                                       1).view(-1, 1)
        estimated_prev_sdf = torch.cat([sdfs[:, :-1], sdfs[:, -1:]],
                                       1).view(-1, 1)

        s = self.model.implicit_network.neus_model.dev(xs)

        prev_cdf = torch.sigmoid(estimated_prev_sdf * s)
        next_cdf = torch.sigmoid(estimated_next_sdf * s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(-1, n_samp).clip(0.01, 0.99)
        T = torch.cumprod(
            torch.cat([
                torch.ones(alpha.shape[0], 1).to(alpha.device),
                1. - alpha + 1e-10
            ], -1), -1)
        weight = (alpha * T[:, :-1])[..., None]

        res = 1 - weight.sum(-2)

        final_x = (xs.view(-1, n_samp, 3) * weight).sum(-2) + res * points
        final_normal = (normals * weight).sum(-2) + res * pred_normals

        pts_norm = torch.linalg.norm(xs, ord=2, dim=-1,
                                     keepdim=True).reshape(-1, n_samp)
        inside_sphere = (pts_norm < 1).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()
        # Eikonal loss
        gradient_error = (torch.linalg.norm(normals.reshape(-1, n_samp, 3), ord=2, dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        return final_x, final_normal, gradient_error

    def get_neus_with_grad(self, points):
        sdfs = self.model.implicit_network(points)[..., :1].view(-1, 1)
        normals = self.model.implicit_network.gradient(points).view(-1, 3)
        return sdfs, normals

    def sample_eikonal(self, points):
        x = points[:, None, :]
        noise = torch.rand(128, 3, device=points.device) * 2 - 1
        x = x + noise * 0.1
        x = x.view(-1, 3)
        normals = self.model.implicit_network.gradient(x).view(-1, 3)
        gradient_error = (torch.linalg.norm(normals, ord=2, dim=-1) - 1.0) ** 2
        return gradient_error.mean()

    def pbr_step(self, model_input):
        loss = 0.
        points = model_input['points']
        normals = model_input['normals']  # get from the mesh in the neus stage(can be processed by ourselves)
        object_mask = model_input['object_mask']

        sg_envmap_material = self.model.envmap_material_network(points[object_mask], train_spec=False, train_norm=True)
        pred_normal = sg_envmap_material['sg_normal_map']

        pbar_loss_dict = {}

        normal_loss = torch.nn.MSELoss()(pred_normal, normals[object_mask])

        if self.cur_iter > 500:  # after 500 epoch, we will add smooth loss
            d_normal = sg_envmap_material['sg_normal_map']
            d_xi_normal = sg_envmap_material['random_xi_normal']

            smooth_loss = torch.nn.L1Loss()(d_normal, d_xi_normal)
            loss = loss + smooth_loss
            pbar_loss_dict["smooth_loss"] = smooth_loss

        loss = loss + normal_loss
        pbar_loss_dict["normal_loss"] = normal_loss

        if self.minimum_mem:
            return loss, pbar_loss_dict

        pnts = points[object_mask]
        neus_sdfs, neus_normals = self.get_neus_with_grad(pnts)

        neus_normal_loss = torch.nn.MSELoss()(neus_normals,
                                              normals[object_mask])
        loss = loss + neus_normal_loss
        pbar_loss_dict["neus_normal_loss"] = neus_normal_loss

        point_loss = torch.nn.MSELoss()(neus_sdfs, torch.zeros_like(neus_sdfs))
        loss = loss + point_loss
        pbar_loss_dict["point_loss"] = point_loss

        eikonal = self.sample_eikonal(pnts)
        loss = loss + eikonal
        pbar_loss_dict["eikonal"] = eikonal

        return loss, pbar_loss_dict

    def get_sg_render(self,
                      points,
                      view_dirs,
                      indir_lgtSGs,
                      albedo_ratio=None,
                      fun_spec=False,
                      lin_diff=False,
                      train_spec=False,
                      indir_integral=None,
                      **kwargs):
        from model.sg_render import render_with_all_sg
        view_dirs = view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-6)
        normals = self.model.get_idr_render(points, view_dirs, normal_only=True)
        ret = {
            'normals': normals,
        }

        assert train_spec == self.train_spec

        # sg renderer
        sg_envmap_material = self.model.envmap_material_network(
            points, train_spec=train_spec)

        if albedo_ratio is not None:
            sg_envmap_material['sg_diffuse_albedo'] = sg_envmap_material['sg_diffuse_albedo'] * albedo_ratio

        diffuse_albedo = sg_envmap_material['sg_diffuse_albedo']
        roughness = sg_envmap_material['sg_roughness']
        metallic = sg_envmap_material['sg_metallic']
        normal_map = sg_envmap_material['sg_normal_map']

        # do not render
        ret["sg_rgb"] = torch.ones_like(diffuse_albedo)
        ret["indir_rgb"] = torch.zeros_like(diffuse_albedo)
        ret['sg_diffuse_rgb'] = torch.zeros_like(diffuse_albedo)
        ret['sg_specular_rgb'] = torch.zeros_like(diffuse_albedo)
        ret['indir_diffuse_rgb'] = torch.zeros_like(diffuse_albedo)
        ret['indir_specular_rgb'] = torch.zeros_like(diffuse_albedo)
        ret['vis_shadow'] = torch.zeros_like(diffuse_albedo)

        # use diffuse albedo to hold normal (and random xi normal)
        ret.update({
            'diffuse_albedo': normal_map,
            'roughness': roughness,
            'metallic': metallic,
            'normal_map': normal_map,
            'random_xi_roughness': sg_envmap_material['random_xi_roughness'],
            'random_xi_metallic': sg_envmap_material['random_xi_metallic'],
            'random_xi_diffuse_albedo': sg_envmap_material['random_xi_normal']
        })

        return ret

    def batching(self, model_input):
        # model_input['uv'] = model_input['uv'].reshape(-1, 1, 2)
        # model_input['intrinsics'] = model_input['intrinsics'].expand(512, -1, -1)
        # model_input['pose'] = model_input['pose'].expand(512, -1, -1)
        # model_input['object_mask'] = model_input['object_mask'].reshape(-1, 1)

        model_input = self.tex_space_sampler.simple_data_batch(model_input)

        return model_input

    def run(self):
        print("training...")
        self.cur_iter = self.start_epoch * len(self.train_dataloader)
        if hasattr(self.model.ray_tracer, "generate"):  # pass the sdf function into ray tracer
            self.model.ray_tracer.generate(lambda x: self.model.implicit_network(x)[:, 0])
        self.model.get_sg_render = self.get_sg_render  # decorator function for BRDF model
        self.train_spec = True
        self.is_training = True

        for epoch in range(self.start_epoch, self.nepochs + 1):
            self.train_dataset.change_sampling_idx(self.num_pixels)

            if self.cur_iter > self.max_niters:  # final plot
                self.save_checkpoints(epoch)
                self.plot_to_disk()
                print('Training has reached max number of iterations: {}; exiting...'.format(self.cur_iter))
                exit(0)

            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                if self.cur_iter % self.ckpt_freq == 0:
                    self.save_checkpoints(epoch)

                if self.cur_iter % self.plot_freq == 0 and self.cur_iter > 0:
                    if self.minimum_mem:
                        with torch.no_grad():
                            self.plot_to_disk()
                    else:
                        self.plot_to_disk()

                for key in model_input.keys():
                    model_input[key] = model_input[key].cuda()

                model_input = self.batching(model_input)

                loss, pbar_loss_dict = self.pbr_step(model_input)

                self.c_optimizer.zero_grad()
                loss.backward()
                self.c_optimizer.step()

                if self.cur_iter % 50 == 0:  # print yo the console every 50 iters
                    post_fix = ", ".join(
                        map(lambda it: f"{it[0]}={it[1].item() if isinstance(it[1], torch.Tensor) else it[1]}",
                            pbar_loss_dict.items()))
                    print('{0} [{1}] ({2}/{3}): loss = {4}, sg_lr = {5}, '.format(self.expname, epoch, indices,
                                                                                  self.n_batches, loss.item(),
                                                                                  self.c_scheduler.get_last_lr()[
                                                                                      0]) + post_fix)

                self.cur_iter += 1
                self.c_scheduler.step()  # update the learning rate
