import dataclasses
import gin
from dataset.mip_dateset import get_dataset
from dataset.load_blender import load_blender_data
import numpy as np
import torch
from misc.utils import torch_tree_map
from misc.defs import Rays
from dataset.neus_dataset import Dataset as NeuSDataset

""" mip-nerf dataset """


@gin.configurable("Data")
@dataclasses.dataclass
class DataConfig:
    dataset_dir: str = None
    dataset_loader: str = 'blender'  # The type of dataset loader to use.
    near: float = 1.
    far: float = 6.
    white_bkgd: bool = True
    batching: str = 'all_images'  # Batch composition, [single_image, all_images].
    batch_size: int = 4096  # The number of rays/pixels in each batch.
    factor: int = 0  # The downsample factor of images, 0 for no downsampling.
    render_path: bool = False
    spherify: bool = False  # Set to True for spherical 360 scenes.
    half_res: bool = False # mip nerf not implemented
    test_skip: int = 8
    version: str = "mip"
    llffhold: int = 8  # Use every Nth image for the test set. Used only by LLFF.
    alpha_as_mask: bool = False


def load_dataset(split_tag):
    config = DataConfig()
    assert config.dataset_dir is not None
    if config.version == "mip":
        return get_dataset(split_tag, config.dataset_dir, config)
    elif config.version == "naive":
        return VanillaDataset(split_tag, config)
    elif config.version == "neus":
        return NeuSDatasetWrapper(split_tag, config)
    elif config.version == "mvs":
        return NeuSDatasetWrapper(split_tag, config, ext="jpg")

""" vanilla nerf dataset """


# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


class VanillaDataset:

    def __init__(self, split_tag, config: DataConfig):
        assert config.dataset_loader == "blender"
        K = None

        """ load blender """
        images, poses, render_poses, hwf, i_split = load_blender_data(config.dataset_dir, config.half_res, config.test_skip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, config.dataset_dir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        masks = images[..., -1:]

        if config.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]

        if K is None:
            K = np.array([
                [focal, 0, 0.5 * W],
                [0, focal, 0.5 * H],
                [0, 0, 1]
            ])

        if split_tag == 'test':
            render_poses = np.array(poses[i_test])
        render_poses = torch.Tensor(render_poses)

        N_rand = config.batch_size
        poses = torch.Tensor(poses)

        i_split = i_train
        if split_tag == "val":
            i_split = i_val
        elif split_tag == "test":
            i_split = i_test

        def iterate():
            precrop_iters = 500
            while True:
                img_i = np.random.choice(i_split)
                target = images[img_i]
                target = torch.Tensor(target)
                pose = poses[img_i, :3, :4]
                mask = torch.Tensor(masks[img_i])

                if N_rand is not None:
                    rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                    if split_tag == "train":
                        if precrop_iters > 0:
                            precrop_iters -= 1
                            dH = int(H // 2)
                            dW = int(W // 2)
                            coords = torch.stack(
                                torch.meshgrid(
                                    torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                                    torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW)
                                ), -1)
                        else:
                            coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)), -1)  # (H, W, 2)

                        coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
                        select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                        select_coords = coords[select_inds].long()  # (N_rand, 2)
                        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                        target = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                        mask = mask[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 1)

                    dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
                    ones = torch.ones_like(rays_d[..., :1])

                    rays = Rays(
                        origins=rays_o,
                        directions=rays_d,
                        viewdirs=dirs,
                        radii=torch.zeros_like(ones),
                        lossmult=mask,
                        near=ones * near,
                        far=ones * far)

                    if split_tag == "test":
                        rays = torch_tree_map(lambda x: x.reshape(H, W, -1), rays)

                    yield {
                        "pixels": target,
                        "rays": rays,
                    }

        self.iterate = iterate

    def __iter__(self):
        return self.iterate()

    def __next__(self):
        if not hasattr(self, "__iterated"):
            self.__iterated = self.iterate()

        return next(self.__iterated)


class NeuSDatasetWrapper:

    def __init__(self, split_tag, config, **kwargs):
        self.config = config
        from pyhocon import ConfigFactory
        conf = ConfigFactory.parse_string(f"data_dir = {config.dataset_dir}\n" +
            "render_cameras_name = cameras_sphere.npz \n" +
            "object_cameras_name = cameras_sphere.npz \n")
        self.dataset = NeuSDataset(conf, **kwargs)

        def iterate():
            image_perm = torch.randperm(self.dataset.n_images)
            iter_step = 0
            while True:
                if split_tag == "test":
                    rays = self.image(2 if config.factor == 0 else 2 * config.factor)
                    shape = rays.origins.shape
                    # rays = torch_tree_map(lambda x: x.reshape(self.dataset.H, self.dataset.W, -1), rays)
                    yield {
                        "pixels": torch.zeros(shape[0], shape[1], 3),
                        "rays": rays,
                    }

                else:
                    data = self.dataset.gen_random_rays_at(image_perm[iter_step % len(image_perm)],
                                                           config.batch_size)
                    iter_step += 1
                    rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
                    near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

                    dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
                    ones = torch.ones_like(rays_d[..., :1])
                    rays = Rays(rays_o, rays_d, dirs, torch.zeros_like(ones), mask, ones * near, ones * far)


                    yield {
                        "pixels": true_rgb,
                        "rays": rays,
                    }

        self.iterate = iterate

    def image(self, resolution_level=4):
        idx = np.random.randint(self.dataset.n_images)

        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape

        near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)
        dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        ones = torch.ones_like(rays_d[..., :1])

        return Rays(rays_o, rays_d, dirs, torch.zeros_like(ones), ones, ones * near, ones * far)

    def __iter__(self):
        return self.iterate()

    def __next__(self):
        if not hasattr(self, "__iterated"):
            self.__iterated = self.iterate()

        return next(self.__iterated)
