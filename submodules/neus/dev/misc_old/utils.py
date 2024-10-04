"""Utility functions."""
import collections
import os
from os import path
from absl import flags
import dataclasses
import gin
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

# gin.add_config_file_search_path('../')
# gin.add_config_file_search_path('config')
# gin.add_config_file_search_path('../config')
#
#
# gin.config.external_configurable(F.relu, module='F')
# gin.config.external_configurable(torch.sigmoid, module='F')
# gin.config.external_configurable(F.softmax, module='F')


@dataclasses.dataclass
class TrainState:
  optimizer: torch.optim.Optimizer


@dataclasses.dataclass
class Stats:
  loss: float
  losses: float
  weight_l2: float
  psnr: float
  psnrs: float
  grad_norm: float
  grad_abs_max: float
  grad_norm_clipped: float


Rays = collections.namedtuple(
    'Rays',
    ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far'))


def define_common_flags():
  # Define the flags used by both train.py and eval.py
  flags.DEFINE_multi_string('gin_file', None,
                            'List of paths to the config files.')
  flags.DEFINE_multi_string(
      'gin_param', None, 'Newline separated list of Gin parameter bindings.')
  flags.DEFINE_string('train_dir', None, 'where to store ckpts and logs')
  flags.DEFINE_string('data_dir', None, 'input data directory.')
  flags.DEFINE_integer(
      'chunk', 8192,
      'the size of chunks for evaluation inferences, set to the value that'
      'fits your GPU/TPU memory.')
  flags.DEFINE_boolean('test', None, 'just test the model and generate a video')


def parse_gin_file():
  gin.parse_config_files_and_bindings(flags.FLAGS.gin_file, flags.FLAGS.gin_param)
  return flags.FLAGS.gin_file


def open_file(pth, mode='r'):
  return open(pth, mode=mode)


def file_exists(pth):
  return path.exists(pth)


def listdir(pth):
  return os.listdir(pth)


def isdir(pth):
  return path.isdir(pth)


def makedirs(pth):
  os.makedirs(pth)


def namedtuple_map(fn, tup):
  """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
  return type(tup)(*map(fn, tup))


""" torch tree utils """

def torch_tree_map(fn, obj):
  if isinstance(obj, (list, tuple)):
    res = []
    for i, o in enumerate(obj):
      res.append(torch_tree_map(fn, o))
    try:
      return type(obj)(*res)
    except TypeError:
      return type(obj)(res)

  if isinstance(obj, dict):
    res = {}
    for k, o in obj.items():
      res[k] = torch_tree_map(fn, o)
    return res

  return fn(obj)

def torch_tree_reduce(fn, obj, init=None):
  if isinstance(obj, (list, tuple)):
    for i, o in enumerate(obj):
      init = torch_tree_reduce(fn, o, init=init)
    return init

  if isinstance(obj, dict):
    for k, o in obj.items():
      init = torch_tree_reduce(fn, o, init=init)
    return init
  if init is None:
    return obj
  return fn(init, obj)


def shard(xs):
  # """Split data into shards for multiple devices along the first dimension."""
  # return jax.tree_map(
  #     lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]), xs)
  return torch_tree_map(torch.tensor, xs)

def to_device(xs):
  # """Transfer data to devices (GPU/TPU)."""
  # return jax.tree_map(jnp.array, xs)
  return torch_tree_map(torch.tensor, xs)

def unshard(x, padding=0):
  """Collect the sharded tensor to the shape before sharding."""
  y = x.reshape([x.shape[0] * x.shape[1]] + list(x.shape[2:]))
  if padding > 0:
    y = y[:-padding]
  return y


def save_img_uint8(img, pth):
  """Save an image (probably RGB) in [0, 1] to disk as a uint8 PNG."""
  with open_file(pth, 'wb') as f:
    Image.fromarray(
        (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)).save(
            f, 'PNG')


def save_img_float32(depthmap, pth):
  """Save an image (probably a depthmap) to disk as a float32 TIFF."""
  with open_file(pth, 'wb') as f:
    Image.fromarray(np.nan_to_num(depthmap).astype(np.float32)).save(f, 'TIFF')


def nan(x, grad_of=None):
  x_nan = (x.isnan().any() or x.isinf().any()).item()
  if x_nan:
    return True
  if grad_of is not None:
    g = torch.autograd.grad(grad_of.sum(), x, retain_graph=True)[0]
    return (g.isnan().any() or g.isinf().any()).item()
  return False


def prox_gradients(func, x, dx):
 y0 = func(x)
 grads = []
 for i in range(x.shape[-1]):
   ofs = torch.zeros_like(x)
   ofs[..., i] = dx
   y1 = func(x + ofs)
   grads.append((y1 - y0) / dx)
 return torch.cat(grads, -1)

