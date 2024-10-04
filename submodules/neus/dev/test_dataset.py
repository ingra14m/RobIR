from dataset import *
from dev.misc_old.utils import *
from absl import app


define_common_flags()


@gin.configurable()
@dataclasses.dataclass
class Config:
  """Configuration flags for everything."""
  dataset_loader: str = 'multicam'  # The type of dataset loader to use.
  batching: str = 'all_images'  # Batch composition, [single_image, all_images].
  batch_size: int = 4096  # The number of rays/pixels in each batch.
  factor: int = 0  # The downsample factor of images, 0 for no downsampling.
  spherify: bool = False  # Set to True for spherical 360 scenes.
  render_path: bool = False  # If True, render a path. Used only by LLFF.
  near: float = 2.  # Near plane distance.
  far: float = 6.  # Far plane distance.
  white_bkgd: bool = True  # If True, use white as the background (black o.w.).

  llffhold: int = 8  # Use every Nth image for the test set. Used only by LLFF.
  lr_init: float = 5e-4  # The initial learning rate.
  lr_final: float = 5e-6  # The final learning rate.
  lr_delay_steps: int = 2500  # The number of "warmup" learning steps.
  lr_delay_mult: float = 0.01  # How much sever the "warmup" should be.
  grad_max_norm: float = 0.  # Gradient clipping magnitude, disabled if == 0.
  grad_max_val: float = 0.  # Gradient clipping value, disabled if == 0.
  max_steps: int = 1000000  # The number of optimization steps.

  save_every: int = 100000  # The number of steps to save a checkpoint.
  print_every: int = 100  # The number of steps between reports to tensorboard.
  gc_every: int = 10000  # The number of steps between garbage collections.
  test_render_interval: int = 1  # The interval between images saved to disk.

  disable_multiscale_loss: bool = False  # If True, disable multiscale loss.

  randomized: bool = True  # Use randomized stratified sampling.

  coarse_loss_mult: float = 0.1  # How much to downweight the coarse loss(es).

  weight_decay_mult: float = 0.  # The multiplier on weight decay.


def load_config():
  gin.parse_config_files_and_bindings(flags.FLAGS.gin_file,
                                      flags.FLAGS.gin_param)
  return Config()


def test_loader(arg_v):
    config = load_config()

    # class DatasetConfig:
    #     white_bkgd = True
    #     render_path = False
    #     dataset_loader: str = 'blender'  # The type of dataset loader to use.
    #     batching: str = 'all_images'  # Batch composition, [single_image, all_images].
    #     batch_size: int = 4096  # The number of rays/pixels in each batch.
    #     near = 1.
    #     far = 6.
    #     factor: int = 0  # The downsample factor of images, 0 for no downsampling.
    #     spherify: bool = False  # Set to True for spherical 360 scenes.
    #
    # config = DatasetConfig()

    # dataset = get_dataset('train', r'G:\Repository\nerf-pytorch\data\nerf_synthetic\lego', config)

    dataset = load_dataset('train')
    for b in dataset:
        sp = torch_tree_map(lambda x: list(x.shape), b)
        sp = torch_tree_reduce(lambda x, y: x + y.sum(), b, 0)
        print(sp)


def test_utils(arg_v):
    config = load_config()
    a = {
        "a": torch.tensor([1, 2, 3]),
        "b": [
            (torch.tensor([1.]), torch.tensor([2.])),
            torch.tensor([3])
        ]
    }

    sp = torch_tree_reduce(lambda x, y: x + y.sum(), a, 0)
    print(sp)


if __name__ == '__main__':
    app.run(test_utils)
