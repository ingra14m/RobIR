import dataclasses
import gin
import tqdm
import os
import torch
from typing import List
import numpy as np
import gc
import time
import imageio
from torch.utils.tensorboard import SummaryWriter
from trimesh import Trimesh


to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


@gin.configurable("Log")
@dataclasses.dataclass
class Logger:

    log_dir: str = None
    exp_name: str = "experiment"
    ckpt_path: str = None
    no_reload: bool = False
    reload_ignore: List[str] = None
    save_every: int = 10000  # The number of steps to save a checkpoint.
    print_every: int = 100  # The number of steps between reports to tensorboard.
    render_every: int = 10000 # The number of steps between test rendering.
    gc_every: int = 10000  # The number of steps between garbage collections.

    def __post_init__(self):
        assert self.log_dir is not None
        os.makedirs(os.path.join(self.log_dir, self.exp_name), exist_ok=True)
        self._init_step = 0
        self._global_step = 0
        self._resume_time = 0
        self._start_time = time.time()
        self._modules = {}
        self._stats_trace = []
        self._writer = SummaryWriter(str(self.path_with_new_idx("exp")))
        print(f'[Tensorboard] tensorboard --logdir={self.path_of("")}')

    def load_state(self, **modules):
        # Load checkpoints
        if self.ckpt_path is not None:
            ckpts = [self.ckpt_path]
        else:
            ckpts = [os.path.join(self.log_dir, self.exp_name, f)
                     for f in sorted(os.listdir(os.path.join(self.log_dir, self.exp_name))) if 'tar' in f]

        print('Found ckpts', ckpts)
        if len(ckpts) > 0 and not self.no_reload:
            ckpt_path = ckpts[-1]
            print('Reloading from', ckpt_path)
            ckpt = torch.load(ckpt_path)
            start = ckpt['global_step']
            if 'resume_time' in ckpt:
                self._resume_time = ckpt['resume_time']

            for key in modules:
                if self.reload_ignore is not None and key in self.reload_ignore:
                    continue
                assert key in ckpt
                modules[key].load_state_dict(ckpt[key])
        else:
            start = 0

        self._init_step = start + 1
        self._global_step = start + 1

        self._modules.update(modules)
        return self._init_step

    def save_state(self, **modules):
        modules.update(self._modules)
        path = os.path.join(self.log_dir, self.exp_name, '{:06d}.tar'.format(self._global_step))

        state = {
            'global_step': self._global_step,
            'resume_time': self.time_cost()
        }

        for key in modules:
            state[key] = modules[key].state_dict()

        torch.save(state, path)
        print('Saved checkpoints at', path, f'(Time {self.time_cost()} s)')

    def path_with_new_idx(self, *paths):
        for i in range(10000):
            tmp_paths = list(paths)
            assert "." not in tmp_paths[-1]
            tmp_paths[-1] = tmp_paths[-1] + f"-{i}"
            tmp_path = self.path_of(*tmp_paths)
            if os.path.exists(tmp_path):
                continue
            return tmp_path
        return self.path_of(*paths)

    def path_of(self, *paths):
        return os.path.join(self.log_dir, self.exp_name, *paths)

    def time_cost(self, total=True):
        return time.time() - self._start_time + (self._resume_time if total else 0)

    def trange(self, max_step):
        if self.gc_every > 0:
            gc.disable()  # Disable automatic garbage collection for efficiency.
        for _ in tqdm.trange(self._init_step, max_step + 1):
            i = self._global_step
            if i == 1:      # 在进行了第一次训练后，存储所有用到地超参数
                self._writer.add_text('gin/train', gin.config.markdown(gin.operative_config_str()), 0)
            yield i
            if self.save_every > 0 and i % self.save_every == 0:
                self.save_state()
            if self.gc_every > 0 and i % self.gc_every == 0:
                gc.collect()
            self._global_step = i + 1

    def test_trange(self, frame_num):
        gc.enable()
        start_time = time.time()
        for i in tqdm.trange(0, frame_num):
            yield i
        end_time = time.time()
        print(f"Render {frame_num} frames in {end_time - start_time} s")
        self.__test_render_time = end_time - start_time

    def skip_test_cases(self, test_dataset):
        for i in range(1, self._global_step):
            if self.render_every > 0 and i % self.render_every == 0:
                next(test_dataset)

    def should_test(self):
        self.__render_start_time = time.time()
        return self.render_every > 0 and self._global_step % self.render_every == 0

    def should_save(self):
        return self.save_every > 0 and self._global_step % self.save_every == 0

    def log_images(self, **images):
        render_cost = (time.time() - self.__render_start_time)
        print("Render finished in", render_cost, "s")
        # vis_suite = vis.visualize_suite(pred_distance, pred_acc)

        for k in images:
            rgb8 = to8b(images[k].detach().cpu().numpy())
            filename = os.path.join(self.log_dir, self.exp_name, 'images', f'{self._global_step:03d}-{k}.png')
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            imageio.imwrite(filename, rgb8)

        # psnr = math.mse_to_psnr(((pred_color - test_case['pixels']) ** 2).mean())
        # ssim = ssim_fn(pred_color, test_case['pixels'])
        # eval_time = time.time() - t_eval_start
        # num_rays = jnp.prod(jnp.array(test_case['rays'].directions.shape[:-1]))
        # rays_per_sec = num_rays / eval_time
        # summary_writer.scalar('test_rays_per_sec', rays_per_sec, step)
        # print(f'Eval {step}: {eval_time:0.3f}s., {rays_per_sec:0.0f} rays/sec')
        # summary_writer.scalar('test_psnr', psnr, step)
        # summary_writer.scalar('test_ssim', ssim, step)
        # self._writer.add_image('test_pred_color', pred_color, self._global_step)
        # for k, v in vis_suite.items():
        #     summary_writer.image('test_pred_' + k, v, step)
        # summary_writer.image('test_pred_acc', pred_acc, step)
        # summary_writer.image('test_target', test_case['pixels'], step)

    def log_metrics(self, **metrics):
        if self.print_every > 0 and self._global_step % self.print_every == 0:
            log_str = f"[TRAIN] Iter: {self._global_step}"
            max_item = 3
            for k in metrics:
                t = metrics[k]
                log_str += f" {k}: {t.item() if isinstance(t, torch.Tensor) else t}"
                max_item -= 1
                if max_item<= 0:
                    break
            tqdm.tqdm.write(log_str)

        self._stats_trace.append(metrics)
        self.log_scalars("metrics", **metrics)

    def log_scalars(self, tag, **scalars):
        for k in scalars:
            t = scalars[k]
            self._writer.add_scalar(f'{tag}/{k}', t.item() if isinstance(t, torch.Tensor) else t, self._global_step)

    def log_video(self, **rgbs):
        desc = {}
        folder = f"test_{self._global_step - 1:06d}"
        os.makedirs(self.path_of(folder), exist_ok=True)
        for k in rgbs:
            if not isinstance(rgbs[k], (list, tuple)) or len(rgbs[k]) == 0 or \
                    not isinstance(rgbs[k][0], (np.ndarray, torch.Tensor)):
                desc[k] = rgbs[k]
                continue
            frames = np.stack(list(map(lambda x: x.cpu().detach().numpy(), rgbs[k])), 0)
            moviebase = self.path_of(folder, f'{k}.mp4')
            if 'rgb' not in k:
                frames = frames / np.max(frames)
            imageio.mimwrite(moviebase, to8b(frames), fps=30, quality=8)
        desc['render_time'] = self.__test_render_time
        self.log_json(f"{folder}\\description", **desc)

    def log_mesh(self, mesh: Trimesh):
        export_path = self.path_of("meshes", f"mesh_{self._global_step:06d}.ply")
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        mesh.export(export_path)

    def log_json(self, tag, **kwargs):
        import json
        with open(self.path_of(tag + ".json"), 'w') as fp:
            json.dump(kwargs, fp, indent=4)
