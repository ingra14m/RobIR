import scipy as sp
import numpy as np
from scipy import signal
import matplotlib.cm as cm


def sinebow(h):
  """A cyclic and uniform colormap, see http://basecase.org/env/on-rainbows."""
  f = lambda x: np.sin(np.pi * x)**2
  return np.stack([f(3 / 6 - h), f(5 / 6 - h), f(7 / 6 - h)], -1)


def convolve2d(z, f):
  return signal.convolve2d(z, f, mode='same')


def depth_to_normals(depth):
  """Assuming `depth` is orthographic, linearize it to a set of normals."""
  f_blur = np.array([1, 2, 1]) / 4
  f_edge = np.array([-1, 0, 1]) / 2
  dy = convolve2d(depth, f_blur[None, :] * f_edge[:, None])
  dx = convolve2d(depth, f_blur[:, None] * f_edge[None, :])
  inv_denom = 1 / np.sqrt(1 + dx**2 + dy**2)
  normals = np.stack([dx * inv_denom, dy * inv_denom, inv_denom], -1)
  return normals


def visualize_depth(depth,
                    acc=None,
                    near=None,
                    far=None,
                    ignore_frac=0,
                    curve_fn=lambda x: -np.log(x + np.finfo(np.float32).eps),
                    modulus=0.,
                    colormap=None):
  """Visualize a depth map.

  Args:
    depth: A depth map.
    acc: An accumulation map, in [0, 1].
    near: The depth of the near plane, if None then just use the min().
    far: The depth of the far plane, if None then just use the max().
    ignore_frac: What fraction of the depth map to ignore when automatically
      generating `near` and `far`. Depends on `acc` as well as `depth'.
    curve_fn: A curve function that gets applied to `depth`, `near`, and `far`
      before the rest of visualization. Good choices: x, 1/(x+eps), log(x+eps).
        Note that the default choice will flip the sign of depths, so that the
        default colormap (turbo) renders "near" as red and "far" as blue.
    modulus: If > 0, mod the normalized depth by `modulus`. Use (0, 1].
    colormap: A colormap function. If None (default), will be set to
      matplotlib's turbo if modulus==0, sinebow otherwise.

  Returns:
    An RGB visualization of `depth`.
  """
  if acc is None:
    acc = np.ones_like(depth)
  acc = np.where(np.isnan(depth), np.zeros_like(acc), acc)

  # Sort `depth` and `acc` according to `depth`, then identify the depth values
  # that span the middle of `acc`, ignoring `ignore_frac` fraction of `acc`.
  sortidx = np.argsort(depth.reshape([-1]))
  depth_sorted = depth.reshape([-1])[sortidx]
  acc_sorted = acc.reshape([-1])[sortidx]
  cum_acc_sorted = np.cumsum(acc_sorted)
  mask = ((cum_acc_sorted >= cum_acc_sorted[-1] * ignore_frac) &
          (cum_acc_sorted <= cum_acc_sorted[-1] * (1 - ignore_frac)))
  depth_keep = depth_sorted[mask]

  # If `near` or `far` are None, use the highest and lowest non-NaN values in
  # `depth_keep` as automatic near/far planes.
  eps = np.finfo(np.float32).eps
  near = near or depth_keep[0] - eps
  far = far or depth_keep[-1] + eps

  # Curve all values.
  depth, near, far = [curve_fn(x) for x in [depth, near, far]]

  # Wrap the values around if requested.
  if modulus > 0:
    value = np.mod(depth, modulus) / modulus
    colormap = colormap or sinebow
  else:
    # Scale to [0, 1].
    value = np.nan_to_num(
        np.clip((depth - np.minimum(near, far)) / np.abs(far - near), 0, 1))
    colormap = colormap or cm.get_cmap('turbo')

  vis = colormap(value)[:, :, :3]

  # Set non-accumulated pixels to white.
  vis = vis * acc[:, :, None] + (1 - acc)[:, :, None]

  return vis


def visualize_normals(depth, acc, scaling=None):
  """Visualize fake normals of `depth` (optionally scaled to be isotropic)."""
  if scaling is None:
    mask = ~np.isnan(depth)
    x, y = np.meshgrid(
        np.arange(depth.shape[1]), np.arange(depth.shape[0]), indexing='xy')
    xy_var = (np.var(x[mask]) + np.var(y[mask])) / 2
    z_var = np.var(depth[mask])
    scaling = np.sqrt(xy_var / z_var)

  scaled_depth = scaling * depth
  normals = depth_to_normals(scaled_depth)
  vis = np.isnan(normals) + np.nan_to_num((normals + 1) / 2, 0)

  # Set non-accumulated pixels to white.
  if acc is not None:
    vis = vis * acc[:, :, None] + (1 - acc)[:, :, None]

  return vis


def visualize_suite(depth, acc):
  """A wrapper around other visualizations for easy integration."""
  vis = {
      'depth': visualize_depth(depth, acc),
      'depth_mod': visualize_depth(depth, acc, modulus=0.1),
      'depth_normals': visualize_normals(depth, acc)
  }
  return vis

