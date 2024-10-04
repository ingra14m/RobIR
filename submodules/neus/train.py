

# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Training script for Nerf."""
import functools
import shutil

import torch.random
from absl import app
from dev.misc_old import utils
from misc.math import learning_rate_decay, mse_to_psnr
from dev.mip.volume_rendering import *
from dataset import load_dataset
from model import *
from optimization.log import Logger
from optimization.regular import *
import gin


@gin.configurable
def render_image(rays, render_fn, chunk=8192):
    """Render all the pixels of an image (in test mode).

    Args:
      render_fn: function, jit-ed render function.
      rays: a `Rays` namedtuple, the rays to be rendered.
      rng: jnp.ndarray, random number generator (used in training mode only).
      chunk: int, the size of chunks to render sequentially.

    Returns:
      rgb: jnp.ndarray, rendered color image.
      disp: jnp.ndarray, rendered disparity image.
      acc: jnp.ndarray, rendered accumulated weights per pixel.
    """
    height, width = rays[0].shape[:2]
    num_rays = height * width
    rays = torch_tree_map(lambda r: r.reshape(num_rays, -1), rays)

    results = []
    for i in range(0, num_rays, chunk):
        chunk_rays = torch_tree_map(lambda r: r[i:i + chunk], rays)
        chunk_results = render_fn(chunk_rays)[-1]
        ret = torch_tree_map(lambda x: x, chunk_results)
        results += [ret]

    def combine_dicts(dicts):
        res = {}
        for key in ["rgb", "dist", "acc"]:
            res[key] = []
        for obj in dicts:
            for key in res:
                res[key] += [obj[key]]
        return res

    rgb, distance, acc = map(lambda x: torch.cat(x, 0).reshape(height, width, -1), combine_dicts(results).values())

    return {
        "rgb": rgb,
        "dist": distance,
        "acc": acc
    }


@gin.configurable("Trainer")
@dataclasses.dataclass
class TrainConfig:
    lr_init: float = 5e-4  # The initial learning rate.
    lr_final: float = 5e-6  # The final learning rate.
    lr_delay_steps: int = 2500  # The number of "warmup" learning steps.
    lr_delay_mult: float = 0.01  # How much sever the "warmup" should be.
    max_steps: int = 200000  # The number of optimization steps.
    device: str = "cuda"  # The default running device.
    weight_decay_mult: float = 0.0001  # The multiplier on weight decay.

    disable_multiscale_loss: bool = False # If True, disable multiscale loss.
    coarse_loss_mult: float = 0.1  # How much to downweight the coarse loss(es).
    grad_max_norm: float = 0.  # Gradient clipping magnitude, disabled if == 0.

    model_class: type = VNeRF


FLAGS = flags.FLAGS
utils.define_common_flags()


def train_step(model, batch, state: utils.TrainState, config: TrainConfig):

    def tree_sum_fn(fn):
        return utils.torch_tree_reduce(lambda x, y: x + fn(y), list(model.parameters()), init=0)

    weight_l2 = config.weight_decay_mult * (
            tree_sum_fn(lambda z: torch.sum(z ** 2)) /
            tree_sum_fn(lambda z: torch.prod(torch.tensor(z.shape))))

    rays = utils.torch_tree_map(lambda x: x.to(config.device), batch['rays'])
    pixels = utils.torch_tree_map(lambda x: x.to(config.device), batch['pixels'])
    ret = volume_rendering(rays, model)

    mask = rays.lossmult
    if config.disable_multiscale_loss:
        mask = torch.ones_like(mask)

    losses = []
    for level in ret:
        losses.append((mask * (level['rgb'] - pixels[..., :3]) ** 2).sum() / mask.sum())
    # losses = torch.cat(losses, 0)
    # loss = (config.coarse_loss_mult * torch.sum(losses[:-1]) + losses[-1] + weight_l2)
    loss = losses[-1] + weight_l2
    for l0 in losses[:-1]:
        loss = loss + config.coarse_loss_mult * l0

    # # TODO: L1 Loss in NeuS
    # mask_sum = mask.sum() + 1e-5
    # color_error = (ret[-1]['rgb'] - pixels[..., :3]) * mask
    # color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
    # loss = color_fine_loss

    # regularization
    loss = loss + accumulate_reg(ret[-1]['acc'])
    loss = loss + similarity_reg(ret[-1]['sim_or_grad'])
    loss = loss + sparsity_reg(ret[-1]['weights'])
    loss = loss + eikonal_reg(ret[-1]['means'], ret[-1]['sim_or_grad'])

    state.optimizer.zero_grad()
    loss.backward()

    if config.grad_max_norm >= 1e-10:
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_max_norm)
    state.optimizer.step()

    # def tree_norm(tree):
    #     return jnp.sqrt(
    #         jax.tree_util.tree_reduce(
    #             lambda x, y: x + jnp.sum(y ** 2), tree, initializer=0))
    #
    # if config.grad_max_val > 0:
    #     clip_fn = lambda z: jnp.clip(z, -config.grad_max_val, config.grad_max_val)
    #     grad = jax.tree_util.tree_map(clip_fn, grad)
    #
    # grad_abs_max = jax.tree_util.tree_reduce(
    #     lambda x, y: jnp.maximum(x, jnp.max(jnp.abs(y))), grad, initializer=0)
    #
    # grad_norm = tree_norm(grad)
    # if config.grad_max_norm > 0:
    #     mult = jnp.minimum(1, config.grad_max_norm / (1e-7 + grad_norm))
    #     grad = jax.tree_util.tree_map(lambda z: mult * z, grad)
    # grad_norm_clipped = tree_norm(grad)
    #
    # new_optimizer = state.optimizer.apply_gradient(grad, learning_rate=lr)
    # new_state = state.replace(optimizer=new_optimizer)

    losses = list(map(lambda x: x.item(), losses))
    psnrs = mse_to_psnr(np.array(losses))
    stats = utils.Stats(
        loss=loss.item(),
        losses=losses,
        weight_l2=weight_l2,
        psnr=psnrs[-1],
        psnrs=psnrs,
        grad_norm=0,
        grad_abs_max=0,
        grad_norm_clipped=0,
    )

    return stats


def main(unused_argv):
    gin_files = utils.parse_gin_file()

    torch.random.manual_seed(20200823)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    np.random.seed(20201473)

    dataset = load_dataset('train')
    test_dataset = load_dataset('test')

    config = TrainConfig()
    logger = Logger()

    for gin_file in gin_files:
        shutil.copy(gin_file, logger.path_of(os.path.basename(gin_file)))

    model = config.model_class()
    model.to(config.device)

    # TODO: initialize make NeuS broken
    # def init_weight(m):
    #     if type(m) == nn.Linear:
    #         nn.init.orthogonal_(m.weight, 1)
    #
    # model.apply(init_weight)

    optimizer = torch.optim.Adam(model.parameters(), config.lr_init)    # TODO: , betas=(0.9, 0.99))

    logger.load_state(model=model, optimizer=optimizer)
    logger.skip_test_cases(test_dataset)

    learning_rate_fn = functools.partial(
        learning_rate_decay,
        lr_init=config.lr_init,
        lr_final=config.lr_final,
        max_steps=config.max_steps,
        lr_delay_steps=config.lr_delay_steps,
        lr_delay_mult=config.lr_delay_mult)

    state = utils.TrainState(optimizer)

    if not flags.FLAGS.test:

        for step, batch in zip(logger.trange(config.max_steps), dataset):
            lr = learning_rate_fn(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            stats = train_step(model, batch, state, config)
            logger.log_metrics(Loss=stats.loss, PSNR=stats.psnr, LR=lr)

            if logger.should_test():
                torch.cuda.empty_cache()
                test_case = next(test_dataset)
                test_case = utils.torch_tree_map(lambda x: x.to(config.device), test_case)
                render_fn = functools.partial(
                    volume_rendering,
                    nerf=model,
                    perturb=False,
                    raw_noise_std=0.,
                )
                with torch.no_grad():
                    ret = render_image(test_case['rays'], render_fn)

                disp_map = 1. / torch.max(1e-10 * torch.ones_like(ret['dist'], device=ret['dist'].device),
                                          ret['dist'] / (ret['acc'] + 1e-8))
                ret['dist'] = disp_map
                logger.log_images(**ret)

        # reload test data for rendering
        test_dataset = load_dataset('test')
    else:
        print("Test and rendering only")

    frames = {}

    def add_frame(**kv_pairs):
        for k in kv_pairs:
            if k not in frames:
                frames[k] = []
            frames[k].append(kv_pairs[k])

    for i, test_case in zip(logger.test_trange(100), test_dataset):
        test_case = utils.torch_tree_map(lambda x: x.to(config.device), test_case)
        render_fn = functools.partial(
            volume_rendering,
            nerf=model,
            perturb=False,
            raw_noise_std=0.,
        )
        with torch.no_grad():
            ret = render_image(test_case['rays'], render_fn)
        mse_loss = ((ret['rgb'] - test_case['pixels'][..., :3]) ** 2).mean().cpu()
        psnr = mse_to_psnr(mse_loss)
        add_frame(rgb=ret['rgb'].cpu(), psnr=psnr.item(), mse=mse_loss.item())

    def calc_avg(tag):
        mean = np.array(frames[tag]).mean().item()
        frames[f'mean_{tag}'] = mean

    [calc_avg(t) for t in ['psnr', 'mse']]
    logger.log_video(**frames)


from model.neus_fields import *
from dev.neus.neus_renderer import *
from dataset.neus_dataset import *


def neus_train(argv):
    batch_size = 512
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    from pyhocon import ConfigFactory
    conf = ConfigFactory.parse_string("data_dir = G:\\Repository\\NeuS\\public_data\\bmvs_dog \n" +
                                      "render_cameras_name = cameras_sphere.npz \n" +
                                      "object_cameras_name = cameras_sphere.npz \n")
    dataset = Dataset(conf)
    image_perm = torch.randperm(dataset.n_images)

    from tqdm import tqdm
    device = "cuda"
    params_to_train = []
    nerf_outside = NeRF(d_in=4).to(device)
    sdf_network = SDFNetwork(d_in=3, d_out=257, d_hidden=256, n_layers=8).to(device)
    deviation_network = SingleVarianceNetwork(init_val=0.3).to(device)
    color_network = RenderingNetwork(d_feature=256, mode='idr', d_in=9, d_out=3, d_hidden=256, n_layers=4).to(device)
    params_to_train += list(nerf_outside.parameters())
    params_to_train += list(sdf_network.parameters())
    params_to_train += list(deviation_network.parameters())
    params_to_train += list(color_network.parameters())

    optimizer = torch.optim.Adam(params_to_train, lr=5e-4)
    renderer = NeuSRenderer(nerf_outside,
                            sdf_network,
                            deviation_network,
                            color_network)

    for iter_i in tqdm(range(20000)):
        data = dataset.gen_random_rays_at(image_perm[iter_i % len(image_perm)], batch_size)

        rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
        near, far = dataset.near_far_from_sphere(rays_o, rays_d)

        background_rgb = None
        if 'use_white_bkgd':
            background_rgb = torch.ones([1, 3])

        mask_weight = 0.0
        igr_weight = 0.1

        if mask_weight > 0.0:
            mask = (mask > 0.5).float()
        else:
            mask = torch.ones_like(mask)

        mask_sum = mask.sum() + 1e-5
        render_out = renderer.render(rays_o, rays_d, near, far,
                                          background_rgb=background_rgb,
                                          cos_anneal_ratio=1.)

        color_fine = render_out['color_fine']
        s_val = render_out['s_val']
        cdf_fine = render_out['cdf_fine']
        gradient_error = render_out['gradient_error']
        weight_max = render_out['weight_max']
        weight_sum = render_out['weight_sum']

        # Loss
        color_error = (color_fine - true_rgb) * mask
        color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
        psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb) ** 2 * mask).sum() / (mask_sum * 3.0)).sqrt())

        eikonal_loss = gradient_error

        mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

        loss = color_fine_loss + \
               eikonal_loss * igr_weight + \
               mask_loss * mask_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_i % 100 == 0:
            log_str = f"[TRAIN] Iter: {iter_i} Loss: {loss.item()} PSNR: {psnr.item()}"
            tqdm.write(log_str)


if __name__ == '__main__':
  app.run(main)
