import gin
import dataclasses
import torch
import torch.nn.functional as F
import collections


Rays = collections.namedtuple(
    'Rays',
    ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far'))


class IComp:

    def radius(self) -> float:
        raise NotImplementedError

    def background(self, x, dirs):
        raise NotImplementedError


class ISDF(IComp):

    def sdf(self, x):
        raise NotImplementedError

    def sdf_and_feat(self, x):
        raise NotImplementedError

    def color(self, x, gradients, dirs, feature_vector):
        raise NotImplementedError

    def grad(self, x):
        raise NotImplementedError

    def dev(self, x):
        raise NotImplementedError


class IMip:

    def color_and_density_of_gaussian(self, means, covs, dirs):
        raise NotImplementedError


gin.add_config_file_search_path('../')
gin.add_config_file_search_path('config')
gin.add_config_file_search_path('../config')


gin.config.external_configurable(F.relu, module='F')
gin.config.external_configurable(F.sigmoid, module='F')
gin.config.external_configurable(F.softmax, module='F')


def identity(x): return x


gin.config.external_configurable(identity, module='F')
