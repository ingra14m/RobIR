import os
import torch
import numpy as np

from os import path

from utils import rend_util
import json
import imageio

import glob
import dataclasses
from datasets import mip_utils as utils
from PIL import Image


@dataclasses.dataclass
class DataConfig:
    dataset_dir: str = None
    dataset_loader: str = 'llff'  # The type of dataset loader to use.
    near: float = 0.01
    far: float = 6.
    white_bkgd: bool = True
    batching: str = 'all_images'  # Batch composition, [single_image, all_images].
    batch_size: int = 1024  # The number of rays/pixels in each batch.
    factor: int = 8  # The downsample factor of images, 0 for no downsampling.
    render_path: bool = False
    spherify: bool = True  # Set to True for spherical 360 scenes.
    half_res: bool = False  # mip nerf not implemented
    test_skip: int = 8
    llffhold: int = 8  # Use every Nth image for the test set. Used only by LLFF.
    alpha_as_mask: bool = True


class LLFFDataset(torch.utils.data.Dataset):
    def __init__(self,
                 instance_dir,
                 frame_skip,
                 split='train',
                 blender=False
                 ):
        self.instance_dir = instance_dir
        print('Creating dataset from: ', self.instance_dir)
        assert os.path.exists(self.instance_dir), "Data directory is empty"

        config = DataConfig()

        self.daemon = True
        self.split = split
        self.data_dir = instance_dir
        self.near = config.near
        self.far = config.far
        self.batch_size = config.batch_size
        self.batching = config.batching
        self.render_path = config.render_path

        self.split = split

        """Initialize training."""
        self._load_renderings(config)
        self._generate_rays()

        if config.batching == 'all_images':
            # flatten the ray and image dimension together.
            self.images = self.images.reshape([-1, 3])
            self.rays = utils.namedtuple_map(lambda r: r.reshape([-1, r.shape[-1]]),
                                             self.rays)
        elif config.batching == 'single_image':
            self.images = self.images.reshape([-1, self.resolution, 3])
            self.rays = utils.namedtuple_map(
                lambda r: r.reshape([-1, self.resolution, r.shape[-1]]), self.rays)
        else:
            raise NotImplementedError(
                f'{config.batching} batching strategy is not implemented.')

        self.single_imgname = None
        self.single_imgname_idx = None
        self.sampling_idx = None

    def _load_renderings(self, config):
        """Load images from disk."""
        # Load images.
        imgdir_suffix = ''
        if config.factor > 0:
            imgdir_suffix = '_{}'.format(config.factor)
            factor = config.factor
        else:
            factor = 1
        imgdir = path.join(self.data_dir, 'images' + imgdir_suffix)
        if not utils.file_exists(imgdir):
            raise ValueError('Image folder {} does not exist.'.format(imgdir))
        imgfiles = [
            path.join(imgdir, f)
            for f in sorted(utils.listdir(imgdir))
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')
        ]
        images = []
        for imgfile in imgfiles:
            with utils.open_file(imgfile, 'rb') as imgin:
                image = np.array(Image.open(imgin), dtype=np.float32) / 255.
                images.append(image)
        images = np.stack(images, axis=-1)

        # Load poses and bds.
        with utils.open_file(path.join(self.data_dir, 'poses_bounds.npy'),
                             'rb') as fp:
            poses_arr = np.load(fp)
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
        bds = poses_arr[:, -2:].transpose([1, 0])
        if poses.shape[-1] != images.shape[-1]:
            raise RuntimeError('Mismatch between imgs {} and poses {}'.format(
                images.shape[-1], poses.shape[-1]))

        # Update poses according to downsampling.
        poses[:2, 4, :] = np.array(images.shape[:2]).reshape([2, 1])
        poses[2, 4, :] = poses[2, 4, :] * 1. / factor

        # Correct rotation matrix ordering and move variable dim to axis 0.
        poses = np.concatenate(
            [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        images = np.moveaxis(images, -1, 0)
        bds = np.moveaxis(bds, -1, 0).astype(np.float32)

        # Rescale according to a default bd factor.
        scale = 1. / (bds.min() * .75)
        poses[:, :3, 3] *= scale
        bds *= scale

        # Recenter poses.
        poses = self._recenter_poses(poses)

        # Generate a spiral/spherical ray path for rendering videos.
        if config.spherify:
            poses = self._generate_spherical_poses(poses, bds)
            self.spherify = True
        else:
            self.spherify = False
        if not config.spherify and self.split == 'test':
            self._generate_spiral_poses(poses, bds)

        # Select the split.
        i_test = np.arange(images.shape[0])[::config.llffhold]
        i_train = np.array(
            [i for i in np.arange(int(images.shape[0])) if i not in i_test])
        if self.split == 'train':
            indices = i_train
        else:
            indices = i_test
        images = images[indices]
        poses = poses[indices]

        self.images = images
        self.camtoworlds = poses[:, :3, :4]
        self.focal = poses[0, -1, -1]
        self.h, self.w = images.shape[1:3]
        self.resolution = self.h * self.w
        if config.render_path:
            self.n_examples = self.render_poses.shape[0]
        else:
            self.n_examples = images.shape[0]

        self.old_init()

    def old_init(self):
        self.n_cameras = self.n_examples
        # self.image_paths = image_paths

        self.single_imgname = None
        self.single_imgname_idx = None
        self.sampling_idx = None

        self.intrinsics_all = []
        intrinsics = [[self.focal, 0, self.w / 2], [0, self.focal, self.h / 2], [0, 0, 1]]
        intrinsics = np.array(intrinsics).astype(np.float32)
        for i in range(self.n_cameras):
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())

        scale = 2.0
        print("Scale {}".format(scale))
        self.camtoworlds[..., 3] /= scale

        self.pose_all = torch.from_numpy(self.camtoworlds).float()
        self.pose_all = torch.cat(
            [self.pose_all, torch.tensor([0, 0, 0, 1. / scale])[None, None, :].expand(self.pose_all.shape[0], -1, -1)],
            -2)

        self.rgb_images = torch.tensor(self.images.reshape(-1, self.w * self.h, 3))
        self.object_masks = torch.ones_like(self.rgb_images[..., -1:], dtype=torch.bool)

        self.img_res = self.images.shape[1:3]
        self.total_pixels = self.img_res[0] * self.img_res[1]

    def _generate_rays(self):
        """Generate normalized device coordinate rays for llff."""
        if self.split == 'test':
            n_render_poses = self.render_poses.shape[0]
            self.camtoworlds = np.concatenate([self.render_poses, self.camtoworlds],
                                              axis=0)

        """Generating rays for all images."""
        x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
            np.arange(self.w, dtype=np.float32),  # X-Axis (columns)
            np.arange(self.h, dtype=np.float32),  # Y-Axis (rows)
            indexing='xy')
        camera_dirs = np.stack(
            [(x - self.w * 0.5 + 0.5) / self.focal,
             -(y - self.h * 0.5 + 0.5) / self.focal, -np.ones_like(x)],
            axis=-1)
        directions = ((camera_dirs[None, ..., None, :] *
                       self.camtoworlds[:, None, None, :3, :3]).sum(axis=-1))
        origins = np.broadcast_to(self.camtoworlds[:, None, None, :3, -1],
                                  directions.shape)
        viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

        # Distance from each unit-norm direction vector to its x-axis neighbor.
        dx = np.sqrt(
            np.sum((directions[:, :-1, :, :] - directions[:, 1:, :, :]) ** 2, -1))
        dx = np.concatenate([dx, dx[:, -2:-1, :]], 1)
        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.

        radii = dx[..., None] * 2 / np.sqrt(12)

        ones = np.ones_like(origins[..., :1])
        self.rays = utils.Rays(
            origins=origins,
            directions=directions,
            viewdirs=viewdirs,
            radii=radii,
            lossmult=ones,
            near=ones * self.near,
            far=ones * self.far)

        if not self.spherify:
            ndc_origins, ndc_directions = convert_to_ndc(self.rays.origins,
                                                         self.rays.directions,
                                                         self.focal, self.w, self.h)

            mat = ndc_origins
            # Distance from each unit-norm direction vector to its x-axis neighbor.
            dx = np.sqrt(np.sum((mat[:, :-1, :, :] - mat[:, 1:, :, :]) ** 2, -1))
            dx = np.concatenate([dx, dx[:, -2:-1, :]], 1)

            dy = np.sqrt(np.sum((mat[:, :, :-1, :] - mat[:, :, 1:, :]) ** 2, -1))
            dy = np.concatenate([dy, dy[:, :, -2:-1]], 2)
            # Cut the distance in half, and then round it out so that it's
            # halfway between inscribed by / circumscribed about the pixel.
            radii = (0.5 * (dx + dy))[..., None] * 2 / np.sqrt(12)

            ones = np.ones_like(ndc_origins[..., :1])
            self.rays = utils.Rays(
                origins=ndc_origins,
                directions=ndc_directions,
                viewdirs=self.rays.directions,
                radii=radii,
                lossmult=ones,
                near=ones * self.near,
                far=ones * self.far)

        # Split poses from the dataset and generated poses
        if self.split == 'test':
            self.camtoworlds = self.camtoworlds[n_render_poses:]
            split = [np.split(r, [n_render_poses], 0) for r in self.rays]
            split0, split1 = zip(*split)
            self.render_rays = utils.Rays(*split0)
            self.rays = utils.Rays(*split1)

    def _generate_spherical_poses(self, poses, bds):
        """Generate a 360 degree spherical path for rendering."""
        # pylint: disable=g-long-lambda
        p34_to_44 = lambda p: np.concatenate([
            p,
            np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])
        ], 1)
        rays_d = poses[:, :3, 2:3]
        rays_o = poses[:, :3, 3:4]

        def min_line_dist(rays_o, rays_d):
            a_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
            b_i = -a_i @ rays_o
            pt_mindist = np.squeeze(-np.linalg.inv(
                (np.transpose(a_i, [0, 2, 1]) @ a_i).mean(0)) @ (b_i).mean(0))
            return pt_mindist

        pt_mindist = min_line_dist(rays_o, rays_d)
        center = pt_mindist
        up = (poses[:, :3, 3] - center).mean(0)
        vec0 = self._normalize(up)
        vec1 = self._normalize(np.cross([.1, .2, .3], vec0))
        vec2 = self._normalize(np.cross(vec0, vec1))
        pos = center
        c2w = np.stack([vec1, vec2, vec0, pos], 1)
        poses_reset = (
                np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4]))
        rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))
        sc = 1. / rad
        poses_reset[:, :3, 3] *= sc
        bds *= sc
        rad *= sc
        centroid = np.mean(poses_reset[:, :3, 3], 0)
        zh = centroid[2]
        radcircle = np.sqrt(rad ** 2 - zh ** 2)
        new_poses = []

        for th in np.linspace(0., 2. * np.pi, 120):
            camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
            up = np.array([0, 0, -1.])
            vec2 = self._normalize(camorigin)
            vec0 = self._normalize(np.cross(vec2, up))
            vec1 = self._normalize(np.cross(vec2, vec0))
            pos = camorigin
            p = np.stack([vec0, vec1, vec2, pos], 1)
            new_poses.append(p)

        new_poses = np.stack(new_poses, 0)
        new_poses = np.concatenate([
            new_poses,
            np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)
        ], -1)
        poses_reset = np.concatenate([
            poses_reset[:, :3, :4],
            np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)
        ], -1)
        if self.split == 'test':
            self.render_poses = new_poses[:, :3, :4]
        return poses_reset

    def _viewmatrix(self, z, up, pos):
        """Construct lookat view matrix."""
        vec2 = self._normalize(z)
        vec1_avg = up
        vec0 = self._normalize(np.cross(vec1_avg, vec2))
        vec1 = self._normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, pos], 1)
        return m

    def _normalize(self, x):
        """Normalization helper function."""
        return x / np.linalg.norm(x)

    def _recenter_poses(self, poses):
        """Recenter poses according to the original NeRF code."""
        poses_ = poses.copy()
        bottom = np.reshape([0, 0, 0, 1.], [1, 4])
        c2w = self._poses_avg(poses)
        c2w = np.concatenate([c2w[:3, :4], bottom], -2)
        bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
        poses = np.concatenate([poses[:, :3, :4], bottom], -2)
        poses = np.linalg.inv(c2w) @ poses
        poses_[:, :3, :4] = poses[:, :3, :4]
        poses = poses_
        return poses

    def _poses_avg(self, poses):
        """Average poses according to the original NeRF code."""
        hwf = poses[0, :3, -1:]
        center = poses[:, :3, 3].mean(0)
        vec2 = self._normalize(poses[:, :3, 2].sum(0))
        up = poses[:, :3, 1].sum(0)
        c2w = np.concatenate([self._viewmatrix(vec2, up, center), hwf], 1)
        return c2w

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        if self.single_imgname_idx is not None:
            idx = self.single_imgname_idx

        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx],
            "object_mask": self.object_masks[idx],
        }
        ground_truth = {
            "rgb": self.rgb_images[idx]
        }

        if self.split == 'test':
            ground_truth["envmap6_rgb"] = self.envmap6_images[idx]
            ground_truth["envmap12_rgb"] = self.envmap12_images[idx]

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            sample["object_mask"] = self.object_masks[idx][self.sampling_idx]
            sample["uv"] = uv[self.sampling_idx, :]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input,
        # ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]
