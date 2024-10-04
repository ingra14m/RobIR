import torch
import gin
from torch import nn
from torch.nn import functional as F
#import tinycudann as tcnn
from misc.schedule import Curve
from misc.utils import make_bbox


class Embed:

    def feature_dim(self) -> int:
        raise NotImplementedError


@gin.register
@gin.configurable
class PE(nn.Module, Embed):
    def __init__(self, input_dims=3, num_freq=10, include_input=True, log_sampling=True, schedule=None):
        super(PE, self).__init__()
        self.kwargs = {
            'input_dims': input_dims,
            'include_input': include_input,
            'max_freq_log2': num_freq - 1,
            'num_freqs': num_freq,
            'log_sampling': log_sampling,
            'periodic_fns': [torch.sin, torch.cos],
        }
        self.create_embedding_fn()

        self.window_curve = None if schedule is None else Curve(schedule)

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

    def forward(self, inputs):
        return self.embed(inputs)

    def feature_dim(self) -> int:
        return self.out_dim

    def windowed_embed(self, x):
        code = self.embed(x)
        if self.window_curve is None:
            return code
        start = 0
        if self.kwargs["include_input"]:
            start = 3
        init_shape = list(code.shape[:-1])
        w_code = code[..., start:].view(init_shape + [-1, 2, 3])
        window = self.get_cosine_easing_window()
        w_code = (window.view(-1, 1, 1) * w_code).view(init_shape + [-1])
        return torch.cat([code[..., :start], w_code], -1)

    def get_cosine_easing_window(self):
        alpha = self.window_curve()
        window = self.cosine_easing_window(0, self.kwargs['max_freq_log2'], self.kwargs['num_freqs'], alpha)
        return window

    @classmethod
    def cosine_easing_window(cls, min_freq_log2, max_freq_log2, num_bands, alpha):
        """Eases in each frequency one by one with a cosine.

        This is equivalent to taking a Tukey window and sliding it to the right
        along the frequency spectrum.

        Args:
          min_freq_log2: the lower frequency band.
          max_freq_log2: the upper frequency band.
          num_bands: the number of frequencies.
          alpha: will ease in each frequency as alpha goes from 0.0 to num_freqs.

        Returns:
          A 1-d numpy array with num_sample elements containing the window.
        """
        if max_freq_log2 is None:
            max_freq_log2 = num_bands - 1.0
        bands = torch.linspace(min_freq_log2, max_freq_log2, num_bands)
        x = torch.clip(alpha - bands, 0.0, 1.0)
        return 0.5 * (1 + torch.cos(torch.pi * x + torch.pi))


@gin.register
@gin.configurable
class Grid(nn.Module, Embed):

    def __init__(self, n_cells=128, out_dim=3):
        super().__init__()
        self.out_dim = out_dim
        N = n_cells
        self.grid = nn.Parameter(torch.randn(1, self.out_dim, N, N, N))

    def forward(self, inputs):
        sh = F.grid_sample(self.grid, inputs.view(1, -1, 1, 1, 3))
        sh = sh.view(self.out_dim, -1).permute(1, 0)
        return sh

    def feature_dim(self) -> int:
        return self.out_dim


def tcnn_encoding(max_level, n_feature, in_dim, hashmap_size=20):

    encoding_config = {
        "otype": "HashGrid",
        "n_levels": max_level,
        "n_features_per_level": n_feature,
        "log2_hashmap_size": hashmap_size,
        "base_resolution": 16,
        "per_level_scale": 1.5,
        "interpolation": "Smoothstep",
    }

    enc = tcnn.Encoding(in_dim, encoding_config)
    return enc


@gin.register
@gin.configurable
class Hash(nn.Module, Embed):

    def __init__(self, n_levels=16, n_features=2, in_dim=3, schedule=None, bbox=None):
        super(Hash, self).__init__()
        assert n_features in [2 ** i for i in range(6)]
        assert n_levels <= 16

        if n_features <= 8:
            encodings = [tcnn_encoding(n_levels, n_features, in_dim)]
        else:
            encodings = []
            n_feat = 0
            while n_feat < n_features:
                encodings.append(tcnn_encoding(n_levels, 8, in_dim))
                n_feat += 8

        self.encodings = nn.ModuleList(encodings)
        self.n_levels = n_levels
        self.n_features = min(n_features, 8)
        self.n_output_dims = n_levels * n_features

        self.window_curve = None if schedule is None else Curve(schedule)
        self.bbox = make_bbox(bbox)

    def forward(self, x):
        x = self.bbox(x)

        init_shape = x.shape
        x = x.view(-1, init_shape[-1])
        codes = torch.cat([enc(x).view(-1, self.n_levels, self.n_features) for enc in self.encodings], -1)
        shape = list(init_shape[:-1]) + [-1]
        return codes.view(*shape).float()

    def feature_dim(self) -> int:
        return self.n_output_dims

    def windowed_embed(self, x):
        code = self(x)
        if self.window_curve is None:
            return code
        init_shape = list(code.shape[:-1])
        w_code = code.view(init_shape + [self.n_levels, -1])
        window = self.get_cosine_easing_window()
        w_code = (window.view(-1, 1) * w_code).view(init_shape + [-1])
        return w_code

    def get_cosine_easing_window(self):
        alpha = self.window_curve()
        window = self.cosine_easing_window(0, self.n_levels, self.n_levels, alpha)
        return window

    @classmethod
    def cosine_easing_window(cls, min_freq_log2, max_freq_log2, num_bands, alpha):
        """Eases in each frequency one by one with a cosine.

        This is equivalent to taking a Tukey window and sliding it to the right
        along the frequency spectrum.

        Args:
          min_freq_log2: the lower frequency band.
          max_freq_log2: the upper frequency band.
          num_bands: the number of frequencies.
          alpha: will ease in each frequency as alpha goes from 0.0 to num_freqs.

        Returns:
          A 1-d numpy array with num_sample elements containing the window.
        """
        if max_freq_log2 is None:
            max_freq_log2 = num_bands - 1.0
        bands = torch.linspace(min_freq_log2, max_freq_log2, num_bands)
        x = torch.clip(alpha - bands, 0.0, 1.0)
        return 0.5 * (1 + torch.cos(torch.pi * x + torch.pi))


def expected_sin(x, x_var):
    def safe_trig_helper(x, fn, t=100 * torch.pi):
        return fn(torch.where(torch.abs(x) < t, x, x % t))

    """Estimates mean and variance of sin(z), z ~ N(x, var)."""
    # When the variance is wide, shrink sin towards zero.
    y = torch.exp(-0.5 * x_var) * safe_trig_helper(x, torch.sin)
    y_var = F.relu(0.5 * (1 - torch.exp(-2 * x_var) * safe_trig_helper(2 * x, torch.cos)) - y ** 2)
    return y, y_var


def integrated_pos_enc(x_coord, min_deg, max_deg, diag=False):
    """Encode `x` with sinusoids scaled by 2^[min_deg:max_deg-1].

    Args:
      x_coord: a tuple containing: x, jnp.ndarray, variables to be encoded. Should
        be in [-pi, pi]. x_cov, jnp.ndarray, covariance matrices for `x`.
      min_deg: int, the min degree of the encoding.
      max_deg: int, the max degree of the encoding.
      diag: bool, if true, expects input covariances to be diagonal (full
        otherwise).

    Returns:
      encoded: jnp.ndarray, encoded variables.
    """
    if diag:
        x, x_cov_diag = x_coord
        scales = torch.tensor([2 ** i for i in range(min_deg, max_deg)], device=x.device)
        shape = list(x.shape[:-1]) + [-1]
        y = torch.reshape(x[..., None, :] * scales[:, None], shape)
        y_var = torch.reshape(x_cov_diag[..., None, :] * scales[:, None] ** 2, shape)
    else:
        x, x_cov = x_coord
        num_dims = x.shape[-1]
        basis = torch.cat(
            [2 ** i * torch.eye(num_dims, device=x.device) for i in range(min_deg, max_deg)], 1)
        y = x @ basis
        # Get the diagonal of a covariance matrix (ie, variance). This is equivalent
        # to jax.vmap(jnp.diag)((basis.T @ covs) @ basis).
        y_var = torch.sum((x_cov @ basis) * basis, -2)

    return expected_sin(
        torch.cat([y, y + 0.5 * torch.pi], dim=-1),
        torch.cat([y_var] * 2, dim=-1))[0]


@gin.register
@gin.configurable
class IPE(nn.Module, Embed):

    def __init__(self, min_deg=0, max_deg=16, in_dim=3, diag=True):
        super(IPE, self).__init__()
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.diag = diag
        self.in_dim = in_dim

    def forward(self, mean, cov):
        if not self.diag:
            cov = torch.diagonal(cov, 0, 1, 2)
        enc = integrated_pos_enc(
            (mean, cov),
            self.min_deg,
            self.max_deg,
        )
        return enc

    def feature_dim(self) -> int:
        return (self.max_deg - self.min_deg) * 2 * self.in_dim


@gin.register
class Id(nn.Module, Embed):

    def forward(self, inputs):
        return inputs

    def feature_dim(self) -> int:
        return 3

