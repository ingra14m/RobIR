import os
import sys
from datetime import datetime

import numpy as np
import torch
from pyhocon import ConfigFactory
from tensorboardX import SummaryWriter

import utils.general as utils
import utils.plots as plt
from model.loss import query_indir_illum
import itertools


class VisTrainRunner:
    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.exps_folder_name = kwargs['exps_folder_name']
        self.batch_size = kwargs['batch_size']
        self.nepochs = self.conf.get_int('train.illum_epoch')
        self.max_niters = kwargs['max_niters']
        self.index = kwargs['index']

        self.expname = 'Vis-' + kwargs['expname']

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
        self.illum_optimizer_params_subdir = "IllumOptimizerParameters"
        self.illum_scheduler_params_subdir = "IllumSchedulerParameters"
        self.vis_optimizer_params_subdir = "VisOptimizerParameters"
        self.vis_scheduler_params_subdir = "VisSchedulerParameters"
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.illum_optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.illum_scheduler_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.vis_optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.vis_scheduler_params_subdir))

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

        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=self.conf.get_config('model'))
        if torch.cuda.is_available():
            self.model.cuda()

        self.illum_loss = utils.get_class(self.conf.get_string('train.illum_loss_class'))(
            **self.conf.get_config('illum_loss'))
        self.illum_optimizer = torch.optim.Adam(self.model.indirect_illum_network.parameters(),
                                                lr=self.conf.get_float('train.illum_learning_rate'))
        self.illum_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.illum_optimizer,
                                                                    self.conf.get_list('train.illum_sched_milestones',
                                                                                       default=[]),
                                                                    gamma=self.conf.get_float(
                                                                        'train.illum_sched_factor', default=0.0))
        self.vis_optimizer = torch.optim.Adam(self.model.visibility_network.parameters(),
                                              lr=self.conf.get_float('train.illum_learning_rate'))
        self.vis_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.vis_optimizer,
                                                                  self.conf.get_list('train.illum_sched_milestones',
                                                                                     default=[]),
                                                                  gamma=self.conf.get_float('train.illum_sched_factor',
                                                                                            default=0.0))

        self.start_epoch = 0
        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

            print('Loading pretrained model: ', os.path.join(
                old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth"))

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"], strict=False)
            self.start_epoch = saved_model_state['epoch']

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.illum_optimizer_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.illum_optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.illum_scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.illum_scheduler.load_state_dict(data["scheduler_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.vis_optimizer_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.vis_optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.vis_scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.vis_scheduler.load_state_dict(data["scheduler_state_dict"])

        self.no_normal = False
        mat_dir = os.path.join('./', kwargs['exps_folder_name'], 'Norm-' + kwargs['expname'])
        if os.path.exists(mat_dir):
            timestamps = os.listdir(mat_dir)
            if len(timestamps) < 1:
                self.no_normal = True
            else:
                timestamp = sorted(timestamps)[-1]  # using the newest training result
        else:
            print('No Mat_model pretrain, please train it first!')
            self.no_normal = True

        if not self.no_normal:
            old_checkpnts_dir = os.path.join(mat_dir, timestamp, 'checkpoints')

            pth_path = os.path.join(old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth")
            print('Loading pretrained model: ', pth_path)

            saved_model_state = torch.load(pth_path)
            model_state = saved_model_state["model_state_dict"]
            for key in list(filter(lambda x: "normal_decoder_layer" not in x, model_state)):
                del model_state[key]
            self.model.load_state_dict(model_state, strict=False)

        self.num_pixels = self.conf.get_int('train.illum_num_pixels')  # use 2048 if oom
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.ckpt_freq = self.conf.get_int('train.ckpt_freq')

    def save_checkpoints(self, epoch):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.illum_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.illum_optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.illum_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.illum_optimizer_params_subdir, "latest.pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.illum_scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.illum_scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.illum_scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.illum_scheduler_params_subdir, "latest.pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.vis_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.vis_optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.vis_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.vis_optimizer_params_subdir, "latest.pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.vis_scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.vis_scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.vis_scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.vis_scheduler_params_subdir, "latest.pth"))

    def plot_to_disk(self):
        self.model.eval()
        sampling_idx = self.train_dataset.sampling_idx
        self.train_dataset.change_sampling_idx(-1)
        indices, model_input, ground_truth = next(
            itertools.islice(self.plot_dataloader, self.index, None))

        for key in model_input.keys():
            model_input[key] = model_input[key].cuda()

        rand_dir = torch.rand(3).cuda()
        rand_dir[:2] -= 0.5
        # rand_dir = torch.tensor([0.2620, -0.2081,  0.0800]).cuda()
        # rand_dir = rand_dir / (torch.norm(rand_dir, dim=-1, keepdim=True) + 1e-4)
        rand_shift = torch.ones(1, 1).cuda()
        rand_shift = torch.clip(rand_shift * (self.cur_iter % 3) * 0.5, 0.1, 1.0)

        print("[Random] hdr shift:", rand_shift.item())

        split = utils.split_input(model_input, self.total_pixels)
        res = []
        for s in split:
            s['hdr_shift'] = rand_shift.expand(s['uv'].shape[1], 1)
            out = self.model(s, trainstage="Illum")
            trace_outputs = self.model.trace_radiance(out, nsamp=8)

            _, pred_vis = torch.max(trace_outputs["pred_vis"].detach(), dim=-1)

            def hdr2ldr(x, t=0.01):
                return self.model.gamma.hdr_shift.hdr2ldr(x, t)

            pred_integral = out['indir_integral'].detach()
            pred_integral_ldr = hdr2ldr(pred_integral)

            pred_vis = torch.mean(pred_vis.float(), axis=1)
            gt_vis = torch.mean((~trace_outputs["gt_vis"]).float(), axis=1)[:, 0]

            gt_integral = trace_outputs['gt_integral'].detach()
            gt_integral_ldr = hdr2ldr(gt_integral)

            res.append({
                'gt_integral': gt_integral,
                'gt_integral_ldr': gt_integral_ldr,
                'pred_integral': pred_integral,
                'pred_integral_ldr': pred_integral_ldr,
                'pred_vis': pred_vis,
                'gt_vis': gt_vis,
            })

        batch_size = ground_truth['rgb'].shape[0]
        model_outputs = utils.merge_output(res, self.total_pixels, batch_size)

        plt.plot_illum(
            model_outputs,
            ground_truth['rgb'],
            self.plots_dir,
            self.cur_iter,
            self.img_res,
        )
        self.model.train()
        self.train_dataset.sampling_idx = sampling_idx

    def run(self):
        print("training...")
        self.cur_iter = self.start_epoch * len(self.train_dataloader)
        if hasattr(self.model.ray_tracer, "generate"):
            self.model.ray_tracer.generate(lambda x: self.model.implicit_network(x)[:, 0])
        if hasattr(self.model, "octree_ray_tracer"):
            self.model.octree_ray_tracer.generate(lambda x: self.model.implicit_network(x)[:, 0])
        self.model.gamma.hdr_shift.fit_data(self.train_dataset)
        self.anneal_t = 0.0

        for epoch in range(self.start_epoch, self.nepochs + 1):
            self.train_dataset.change_sampling_idx(self.num_pixels)

            if self.cur_iter > self.max_niters:
                self.save_checkpoints(epoch)
                self.plot_to_disk()
                print('Training has reached max number of iterations: {}; exiting...'.format(self.cur_iter))
                exit(0)

            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                if self.cur_iter % self.ckpt_freq == 0:
                    self.save_checkpoints(epoch)

                # add the second one not to plot in the first iter
                if self.cur_iter % self.plot_freq == 0 and self.cur_iter > 0:
                    self.plot_to_disk()

                for key in model_input.keys():
                    model_input[key] = model_input[key].cuda()

                model_input['hdr_shift'] = torch.rand_like(model_input["uv"][0, ..., :1])
                model_outputs = self.model(model_input, trainstage='Illum')
                trace_outputs = self.model.trace_radiance(model_outputs, nsamp=512)

                if model_outputs["network_object_mask"].any():
                    # self.anneal_t = 1 - np.clip(self.cur_iter / 2000, 0.0, 1.0)
                    radiance_loss, visibility_loss = self.illum_loss(model_outputs, trace_outputs, self.anneal_t)

                    # update vis
                    self.vis_optimizer.zero_grad()
                    visibility_loss.backward()
                    self.vis_optimizer.step()

                    # update illum
                    self.illum_optimizer.zero_grad()
                    radiance_loss.backward()
                    self.illum_optimizer.step()

                    if self.cur_iter % 50 == 0:
                        print('{0} [{1}] ({2}/{3}): radiance_loss = {4}, visibility_loss = {5}, anneal_t = {6}'
                              .format(self.expname, epoch, data_index, self.n_batches,
                                      radiance_loss.item(), visibility_loss.item(), self.anneal_t))
                        self.writer.add_scalar('radiance_loss', radiance_loss.item(), self.cur_iter)
                        self.writer.add_scalar('visibility_loss', visibility_loss.item(), self.cur_iter)

                self.cur_iter += 1
                self.illum_scheduler.step()
                self.vis_scheduler.step()
