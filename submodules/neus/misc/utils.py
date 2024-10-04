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


def prox_gradients(func, x, dx):
    y0 = func(x)
    assert y0.shape[-1] == 1
    grads = []
    for i in range(x.shape[-1]):
        ofs = torch.zeros_like(x)
        ofs[..., i] = dx
        y1 = func(x + ofs)
        grads.append((y1 - y0) / dx)
    return torch.cat(grads, -1)


def prox_tangent_gradients(func, x, dx, axis):
    y0 = func(x)
    assert y0.shape[-1] == 1
    grads = []
    for u in axis:
        ofs = dx * u
        y1 = func(x + ofs)
        grads.append((y1 - y0) / dx)
    return torch.cat(grads, -1)


@gin.configurable
class BBox(torch.nn.Module):

    """
    Define a bounding box in the scene, which encodes the inside positions into [0, 1]

    @call: xyz in R3 -> normalized uvw in [0, 1]
    @method inv: normalized uvw in [0, 1] -> xyz in R3
    @method inside: xyz whether inside the box
    """

    @staticmethod
    def from_dict(config):

        """
        single value: the (half) size of the box (centered by origin)
        3-values array: the (half) size of the box (centered by origin)
        2-element array: the min and max of the box, respectively
        dictionary: the kwargs of the init method, box_size > box_min > box_max > box_center (priority)
        """
        if config is None:
            return BBox(box_min=0, box_max=1)

        if isinstance(config, (int, float)):
            return BBox(box_size=config)

        if isinstance(config, (tuple, list)):
            if len(config) == 3:
                return BBox(box_size=config)
            if len(config) == 2:
                return BBox(box_min=config[0], box_max=config[1])
            if len(config) == 6:
                return BBox(box_min=config[0:3], box_max=config[3:6])
            raise NotImplementedError

        wrap_config = {}
        for k in config:
            if "box" not in k and k not in ["dims", "input_half_size", "learnable"]:
                k = "box_" + k
            wrap_config[k] = config[k]
        return BBox(**config)

    def __init__(self, box_size=None, box_min=None, box_max=None, box_center=None, dims=3, input_half_size=True):
        super(BBox, self).__init__()
        self.dims = dims
        def brd_cst(x): return torch.tensor(x).expand(dims)

        if input_half_size:
            if box_min is None and box_size is not None:
                box_size = box_size * 2

        if box_size is not None:
            self.box_size = brd_cst(box_size)
            if box_min is not None:
                self.box_center = box_min + self.box_size / 2
            else:
                self.box_center = brd_cst(0) if box_center is None else box_center
        else:
            assert box_min is not None and box_max is not None
            self.box_size = brd_cst(box_max) - brd_cst(box_min)
            self.box_center = (brd_cst(box_max) + brd_cst(box_min)) / 2

        # box center is not actually used
        self.box_min = self.box_center - self.box_size / 2
        assert (self.box_size > 0).all()

    def inv(self, local_x):
        assert local_x.shape[-1] == self.dims
        return local_x * self.box_size + self.box_min

    def inside(self, x):
        assert x.shape[-1] == self.dims
        lower = x > self.box_min
        upper = x < self.box_min + self.box_size
        lower = torch.prod(lower, dim=-1, keepdim=True)
        upper = torch.prod(upper, dim=-1, keepdim=True)
        return torch.logical_and(lower, upper)

    def forward(self, x):
        assert x.shape[-1] == self.dims
        return (x - self.box_min) / self.box_size


def make_bbox(bbox) -> BBox:
    return BBox.from_dict(bbox)
