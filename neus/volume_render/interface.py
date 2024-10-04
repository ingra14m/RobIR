from misc.utils import *
from misc.defs import *
from volume_render.mip_render import render_mip
from volume_render.sdf_render import render_neus
import functools


def mip_render_fn(model, rays: Rays, is_eval=False, **kwargs):
    if is_eval:
        with torch.no_grad():
            render_fn = functools.partial(
                render_mip,
                nerf=model,
                is_eval=is_eval
            )
            return render_image(rays, lambda x: render_fn(x)[-1])
    return render_mip(rays, model, is_eval=is_eval)[-1]


def neus_render_fn(model: ISDF, rays: Rays, is_eval=False, global_step=-1, anneal_end=50000, **kwargs):
    cos_anneal_ratio = np.min([1.0, global_step / anneal_end]) if global_step >= 0 else 1.0
    if is_eval:
        with torch.no_grad():
            render_fn = functools.partial(
                render_neus,
                model=model,
                is_eval=is_eval,
                cos_anneal_ratio=cos_anneal_ratio
            )
            return render_image(rays, render_fn)
    return render_neus(rays,
                       model=model,
                       is_eval=is_eval,
                       cos_anneal_ratio=cos_anneal_ratio)


render_fns = {
    "mip": mip_render_fn,
    "neus": neus_render_fn,
}


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
        chunk_results = render_fn(chunk_rays)
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
