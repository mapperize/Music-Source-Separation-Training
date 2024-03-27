from functools import partial

import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from beartype.typing import Tuple, Optional, List, Callable
from beartype import beartype

from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.triton.layernorm import RMSNorm as fusedRMSNorm

# helper functions

def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


# norm

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class MoELayer(Module):
    """
    Theory:
    https://arxiv.org/abs/2402.01771
    https://arxiv.org/abs/2401.04081
    https://arxiv.org/pdf/2210.05144

    Mamba as the expert
    TODO: Block should be replaced with custom kernel for parallel computation
    TODO: https://github.com/microsoft/tutel
    """
    def __init__(
            self, d_model, d_state = 16, d_conv = 4, expand = 2, eps = 1e-5,
            num_experts = 4, top_k = 2
        ):
        super().__init__()

        self.top_k = top_k
        self.num_experts = num_experts
        self.router = Linear(d_model, num_experts)
        self.norm = fusedRMSNorm(d_model, eps=eps)

        self.experts = ModuleList(
            [
                Mamba(d_model, d_state, d_conv, expand, eps)
                for i in range(num_experts)
            ]   
        )
    
    def forward(self, x, residual = None, params = None):
        
        x_shape = x.shape
        x, residual = self.norm(x, residual = residual, prenorm = True)

        route = self.router(x)
        route = route.view(-1, self.num_experts)
        route = torch.softmax(route, dim=1)

        k_probs, k_indices = torch.topk(route, k=self.top_k, dim=1)
        
        x = x.view(-1, x_shape[-1])

        for idx, expert in enumerate(self.experts):
            for k in self.top_k:
                indices = (k_indices[:, k] == idx).nonzero()
                if indices.numel() > 0:
                    x[indices] = expert(x[indices], inference_params = params)
                    x[indices] *= k_probs[:, k][indices].unsqueeze(1)

        x = x.view(*x_shape)
        return x, residual


class MambaLayer(Module):
    def __init__(self, d_model, d_state = 16, d_conv = 4, expand = 2, eps = 1e-5, **kwargs):
        super().__init__()
        self.norm = fusedRMSNorm(d_model, eps=eps)
        self.mamba = Mamba(
                d_model=d_model,        # Model dimension d_model
                d_state=d_state,        # SSM state expansion factor
                d_conv=d_conv,          # Local convolution width
                expand=expand,          # Block expansion factor
        )
    
    def forward(self, x, residual = None, params = None):
        
        x, residual = self.norm(x, residual = residual, prenorm = True)
        x = self.mamba(x, inference_params = params)

        return x, residual


class MambaModule(Module):
    def __init__(
            self, d_model, depth = 1, eps = 1e-5

            attn_state = 16, attn_conv = 4, attn_expand = 2,    # attn-sized mamba
            ff_state = 16, ff_conv = 4, ff_expand = 2,          # ff-sized mamba

            use_moe = False, num_experts = None, top_k = None   # moe params
        ):
        super().__init__()

        layer = MoELayer if use_moe else MambaLayer
        kwargs = {
            d_state: ff_state, d_conv: ff_conv, d_expand: ff_expand,
            num_experts: num_experts, top_k: top_k
        }
        
        self.layers = ModuleList(
            [
                MambaLayer(d_model=d_model, d_state=attn_state, d_conv=attn_conv, expand=attn_expand, eps=eps),
                layer(d_model=d_model, eps=eps **kwargs)
            ]
            for i in range(depth)
        )

        self.norm = fusedRMSNorm(d_model, eps = eps)

    def forward(self, x, params = None):

        residual = None
        for layer in self.layers:
            x, residual = layer(x, residual, params)
        
        return self.norm(x, residual = residual)


# bandsplit module

class BandSplit(Module):
    @beartype
    def __init__(
            self,
            dim,
            dim_inputs: Tuple[int, ...]
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(
                RMSNorm(dim_in),
                nn.Linear(dim_in, dim)
            )

            self.to_features.append(net)

    def forward(self, x):
        x = x.split(self.dim_inputs, dim=-1)

        outs = []
        for split_input, to_feature in zip(x, self.to_features):
            split_output = to_feature(split_input)
            outs.append(split_output)

        return torch.stack(outs, dim=-2)


def MLP(
        dim_in,
        dim_out,
        dim_hidden=None,
        depth=1,
        activation=nn.Tanh
):
    dim_hidden = default(dim_hidden, dim_in)

    net = []
    dims = (dim_in, *((dim_hidden,) * (depth - 1)), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)

        net.append(nn.Linear(layer_dim_in, layer_dim_out))

        if is_last:
            continue

        net.append(activation())

    return nn.Sequential(*net)


class MaskEstimator(Module):
    @beartype
    def __init__(
            self,
            dim,
            dim_inputs: Tuple[int, ...],
            depth,
            mlp_expansion_factor=4
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for dim_in in dim_inputs:
            net = []

            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth),
                nn.GLU(dim=-1)
            )

            self.to_freqs.append(mlp)

    def forward(self, x):
        x = x.unbind(dim=-2)

        outs = []

        for band_features, mlp in zip(x, self.to_freqs):
            freq_out = mlp(band_features)
            outs.append(freq_out)

        return torch.cat(outs, dim=-1)


# main class

DEFAULT_FREQS_PER_BANDS = (
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    12, 12, 12, 12, 12, 12, 12, 12,
    24, 24, 24, 24, 24, 24, 24, 24,
    48, 48, 48, 48, 48, 48, 48, 48,
    128, 129,
)


class BSMamba(Module):

    @beartype
    def __init__(
            self,
            *,

            stereo=False,
            num_stems=1,
            freqs_per_bands: Tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,

            depth,
            time_mamba_depth=2,
            freq_mamba_depth=2,
            d_model = 192,
            attn_state = 16,
            attn_conv = 4,
            attn_expand = 2,
            ff_state = 16,
            ff_conv = 4,
            ff_expand = 2,
            use_moe = True,
            num_experts = 4,
            top_k = 2,

            stft_n_fft=2048,
            stft_hop_length=512,
            # 10ms at 44100Hz, from sections 4.1, 4.4 in the paper - @faroit recommends // 2 or // 4 for better reconstruction
            stft_win_length=2048,
            stft_normalized=False,
            stft_window_fn: Optional[Callable] = None,
            mask_estimator_depth=2,
            multi_stft_resolution_loss_weight=1.,
            multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
            multi_stft_hop_size=147,
            multi_stft_normalized=False,
            multi_stft_window_fn: Callable = torch.hann_window
    ):
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems

        self.layers = ModuleList([])

        """
            Roughly 3 * expand * d_model^2 parameters
            Parameters	Layers	Model dim.
            130M	24	768
            370M	48	1024
            790M	48	1536
            1.4B	48	2048
            2.8B	64	2560
        """

        mamba_kwargs = dict(
            d_model = d_model,
            attn_state = attn_state,
            attn_conv = attn_conv,
            attn_expand = attn_expand,
            ff_state = ff_state,
            ff_conv = ff_conv,
            ff_expand = ff_expand,
            use_moe = use_moe,
            num_experts = num_experts,
            top_k = top_k
        )
        
        for _ in range(depth):
            mamba_modules = [
                MambaModule(depth = time_mamba_depth, **mamba_kwargs),
                MambaModule(depth = freq_mamba_depth, **mamba_kwargs)
            ]
            self.layers.append(nn.ModuleList(mamba_modules))

        #self.final_norm = RMSNorm(dim) # Not required

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized
        )

        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)

        freqs = torch.stft(torch.randn(1, 4096), **self.stft_kwargs, return_complex=True).shape[1]

        assert len(freqs_per_bands) > 1
        assert sum(
            freqs_per_bands) == freqs, f'the number of freqs in the bands must equal {freqs} based on the STFT settings, but got {sum(freqs_per_bands)}'

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)

        self.band_split = BandSplit(
            dim=dim,
            dim_inputs=freqs_per_bands_with_complex
        )

        self.mask_estimators = nn.ModuleList([])

        for _ in range(num_stems):
            mask_estimator = MaskEstimator(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth
            )

            self.mask_estimators.append(mask_estimator)

        # for the multi-resolution stft loss

        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn

        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size,
            normalized=multi_stft_normalized
        )

    def forward(
            self,
            raw_audio,
            target=None,
            return_loss_breakdown=False
    ):
        """
        einops

        b - batch
        f - freq
        t - time
        s - audio channel (1 for mono, 2 for stereo)
        n - number of 'stems'
        c - complex (2)
        d - feature dimension
        """

        device = raw_audio.device

        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, 'b t -> b 1 t')

        channels = raw_audio.shape[1]
        assert (not self.stereo and channels == 1) or (
                    self.stereo and channels == 2), 'stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)'

        # to stft

        raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, '* t')

        stft_window = self.stft_window_fn(device=device)

        stft_repr = torch.stft(raw_audio, **self.stft_kwargs, window=stft_window, return_complex=True)
        stft_repr = torch.view_as_real(stft_repr)

        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')
        stft_repr = rearrange(stft_repr,
                              'b s f t c -> b (f s) t c')  # merge stereo / mono into the frequency, with frequency leading dimension, for band splitting

        x = rearrange(stft_repr, 'b f t c -> b t (f c)')

        x = self.band_split(x)

        # axial / hierarchical mamba

        for time_mamba, freq_mamba in self.layers:

            x = rearrange(x, 'b t f d -> b f t d')
            x, ps = pack([x], '* t d')

            x = time_mamba(x)

            x, = unpack(x, ps, '* t d')
            x = rearrange(x, 'b f t d -> b t f d')
            x, ps = pack([x], '* f d')

            x = freq_mamba(x)

            x, = unpack(x, ps, '* f d')

        #x = self.final_norm(x)

        num_stems = len(self.mask_estimators)

        mask = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        mask = rearrange(mask, 'b n t (f c) -> b n f t c', c=2)

        # modulate frequency representation

        stft_repr = rearrange(stft_repr, 'b f t c -> b 1 f t c')

        # complex number multiplication

        stft_repr = torch.view_as_complex(stft_repr)
        mask = torch.view_as_complex(mask)

        stft_repr = stft_repr * mask

        # istft

        stft_repr = rearrange(stft_repr, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)

        recon_audio = torch.istft(stft_repr, **self.stft_kwargs, window=stft_window, return_complex=False)

        recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', s=self.audio_channels, n=num_stems)

        if num_stems == 1:
            recon_audio = rearrange(recon_audio, 'b 1 s t -> b s t')

        # if a target is passed in, calculate loss for learning

        if not exists(target):
            return recon_audio

        if self.num_stems > 1:
            assert target.ndim == 4 and target.shape[1] == self.num_stems

        if target.ndim == 2:
            target = rearrange(target, '... t -> ... 1 t')

        target = target[..., :recon_audio.shape[-1]]  # protect against lost length on istft

        loss = F.l1_loss(recon_audio, target)

        multi_stft_resolution_loss = 0.

        for window_size in self.multi_stft_resolutions_window_sizes:
            res_stft_kwargs = dict(
                n_fft=max(window_size, self.multi_stft_n_fft),  # not sure what n_fft is across multi resolution stft
                win_length=window_size,
                return_complex=True,
                window=self.multi_stft_window_fn(window_size, device=device),
                **self.multi_stft_kwargs,
            )

            recon_Y = torch.stft(rearrange(recon_audio, '... s t -> (... s) t'), **res_stft_kwargs)
            target_Y = torch.stft(rearrange(target, '... s t -> (... s) t'), **res_stft_kwargs)

            multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(recon_Y, target_Y)

        weighted_multi_resolution_loss = multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight

        total_loss = loss + weighted_multi_resolution_loss

        if not return_loss_breakdown:
            return total_loss

        return total_loss, (loss, multi_stft_resolution_loss)