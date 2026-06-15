"""WAN VAE for video encode/decode."""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as safetensors_load_file

from utils import DEVICE, get_available_gpu_memory

# Latent normalization constants (WAN 2.2)
LATENTS_MEAN = (
  -0.7571,
  -0.7089,
  -0.9113,
  0.1075,
  -0.1745,
  0.9653,
  -0.1517,
  1.5508,
  0.4134,
  -0.0715,
  0.5517,
  -0.3632,
  -0.1922,
  -0.9497,
  0.2503,
  -0.2921,
)
LATENTS_STD = (
  2.8184,
  1.4541,
  2.3275,
  2.6558,
  1.2196,
  1.7708,
  2.6052,
  2.0743,
  3.2687,
  2.1526,
  2.8652,
  1.5579,
  1.6382,
  1.1253,
  2.8251,
  1.9160,
)

CACHE_T = 2


class DiagonalGaussianDistribution:
  def __init__(self, parameters: torch.Tensor):
    self.mean, _ = torch.chunk(parameters, 2, dim=1)

  def mode(self) -> torch.Tensor:
    return self.mean


class CausalConv3d(nn.Conv3d):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
    super().__init__(
      in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
    )
    self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
    self.padding = (0, 0, 0)

  def forward(self, x, cache_x=None):
    padding = list(self._padding)
    if cache_x is not None and self._padding[4] > 0:
      cache_x = cache_x.to(x.device)
      x = torch.cat([cache_x, x], dim=2)
      padding[4] -= cache_x.shape[2]
    return super().forward(F.pad(x, padding))


class RMS_norm(nn.Module):
  def __init__(self, dim, channel_first=True, images=True, bias=False):
    super().__init__()
    broadcastable_dims = (1, 1, 1) if not images else (1, 1)
    shape = (dim, *broadcastable_dims) if channel_first else (dim,)
    self.channel_first = channel_first
    self.scale = dim**0.5
    self.gamma = nn.Parameter(torch.ones(shape))
    self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

  def forward(self, x):
    return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):
  def forward(self, x):
    return super().forward(x.float()).type_as(x)


class Resample(nn.Module):
  def __init__(self, dim, mode):
    assert mode in ("none", "upsample2d", "upsample3d", "downsample2d", "downsample3d")
    super().__init__()
    self.dim = dim
    self.mode = mode
    if mode == "upsample2d":
      self.resample = nn.Sequential(
        Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), nn.Conv2d(dim, dim // 2, 3, padding=1)
      )
    elif mode == "upsample3d":
      self.resample = nn.Sequential(
        Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), nn.Conv2d(dim, dim // 2, 3, padding=1)
      )
      self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
    elif mode == "downsample2d":
      self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
    elif mode == "downsample3d":
      self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
      self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
    else:
      self.resample = nn.Identity()

  def forward(self, x, feat_cache=None, feat_idx=[0], final=False):  # noqa: B006
    b, c, t, h, w = x.size()
    if self.mode == "upsample3d":
      if feat_cache is not None:
        idx = feat_idx[0]
        if feat_cache[idx] is None:
          feat_cache[idx] = "Rep"
          feat_idx[0] += 1
        else:
          cache_x = x[:, :, -CACHE_T:, :, :]
          if feat_cache[idx] == "Rep":
            x = self.time_conv(x)
          else:
            x = self.time_conv(x, feat_cache[idx])
          feat_cache[idx] = cache_x
          feat_idx[0] += 1
          x = x.reshape(b, 2, c, t, h, w)
          x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
          x = x.reshape(b, c, t * 2, h, w)
    t = x.shape[2]
    x = x.permute(0, 2, 1, 3, 4).flatten(0, 1).contiguous(memory_format=torch.channels_last)
    x = self.resample(x)
    x = x.unflatten(0, (-1, t)).permute(0, 2, 1, 3, 4).contiguous(memory_format=torch.channels_last_3d)
    if self.mode == "downsample3d":
      if feat_cache is not None:
        idx = feat_idx[0]
        if feat_cache[idx] is None:
          feat_cache[idx] = x
        else:
          cache_x = x[:, :, -1:, :, :]
          x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
          feat_cache[idx] = cache_x
          deferred_x = feat_cache[idx + 1]
          if deferred_x is not None:
            x = torch.cat([deferred_x, x], 2)
            feat_cache[idx + 1] = None
          if x.shape[2] == 1 and not final:
            feat_cache[idx + 1] = x
            x = None
        feat_idx[0] += 2
    return x


class ResidualBlock(nn.Module):
  def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
    super().__init__()
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.residual = nn.Sequential(
      RMS_norm(in_dim, images=False),
      nn.SiLU(),
      CausalConv3d(in_dim, out_dim, 3, padding=1),
      RMS_norm(out_dim, images=False),
      nn.SiLU(),
      nn.Dropout(dropout),
      CausalConv3d(out_dim, out_dim, 3, padding=1),
    )
    self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

  def forward(self, x, feat_cache=None, feat_idx=[0], final=False):  # noqa: B006
    h = self.shortcut(x)
    x = self.residual[0](x)
    x = self.residual[1](x)
    idx = feat_idx[0]
    cache_x = x[:, :, -CACHE_T:, :, :].clone()
    if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
      cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
    x = self.residual[2](x, feat_cache[idx])
    feat_cache[idx] = cache_x
    feat_idx[0] += 1
    x = self.residual[3](x)
    x = self.residual[4](x)
    x = self.residual[5](x)
    idx = feat_idx[0]
    cache_x = x[:, :, -CACHE_T:, :, :].clone()
    if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
      cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
    x = self.residual[6](x, feat_cache[idx])
    feat_cache[idx] = cache_x
    feat_idx[0] += 1
    return x + h


class AttentionBlock(nn.Module):
  def __init__(self, dim: int):
    super().__init__()
    self.dim = dim
    self.norm = RMS_norm(dim)
    self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
    self.proj = nn.Conv2d(dim, dim, 1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    identity = x
    b, c, t, h, w = x.size()
    x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    x = self.norm(x)
    q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)
    x = F.scaled_dot_product_attention(q, k, v).squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
    x = self.proj(x).view(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    return x + identity


class WanEncoder3d(nn.Module):
  def __init__(
    self,
    dim=128,
    z_dim=4,
    dim_mult=(1, 2, 4, 4),
    num_res_blocks=2,
    attn_scales=(),
    temperal_downsample=(True, True, False),
    dropout=0.0,
  ):
    super().__init__()
    self.dim = dim
    self.z_dim = z_dim
    self.temperal_downsample = temperal_downsample
    dims = [dim * u for u in [1] + list(dim_mult)]
    scale = 1.0
    self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)
    self.downsamples = nn.ModuleList()
    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
      for _ in range(num_res_blocks):
        self.downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
        if scale in attn_scales:
          self.downsamples.append(AttentionBlock(out_dim))
        in_dim = out_dim
      if i != len(dim_mult) - 1:
        mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
        self.downsamples.append(Resample(out_dim, mode=mode))
        scale /= 2.0
    self.middle = nn.Sequential(
      ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim), ResidualBlock(out_dim, out_dim, dropout)
    )
    self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(), CausalConv3d(out_dim, z_dim, 3, padding=1))

  def forward(self, x, feat_cache=None, feat_idx=[0], final=False):  # noqa: B006
    if feat_cache is not None:
      idx = feat_idx[0]
      cache_x = x[:, :, -CACHE_T:, :, :]
      x = self.conv1(x, feat_cache[idx])
      feat_cache[idx] = cache_x
      feat_idx[0] += 1
    else:
      x = self.conv1(x)
    for layer in self.downsamples:
      if feat_cache is not None:
        x = layer(x, feat_cache, feat_idx, final=final)
        if x is None:
          return None
      else:
        x = layer(x)
    for layer in self.middle:
      if isinstance(layer, ResidualBlock) and feat_cache is not None:
        x = layer(x, feat_cache, feat_idx, final=final)
      else:
        x = layer(x)
    for layer in self.head:
      if isinstance(layer, CausalConv3d) and feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :]
        x = layer(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
      else:
        x = layer(x)
    return x


def _count_conv3d(model: nn.Module) -> int:
  return sum(1 for m in model.modules() if isinstance(m, CausalConv3d))


def _count_cache_layers(model: nn.Module) -> int:
  return sum(
    1 for m in model.modules() if isinstance(m, CausalConv3d) or (isinstance(m, Resample) and m.mode == "downsample3d")
  )


class Decoder3d(nn.Module):
  def __init__(
    self,
    dim=128,
    z_dim=4,
    dim_mult=(1, 2, 4, 4),
    num_res_blocks=2,
    attn_scales=(),
    temperal_upsample=(False, True, True),
    dropout=0.0,
  ):
    super().__init__()
    self.dim = dim
    self.z_dim = z_dim
    dim_mult = list(dim_mult)
    dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
    scale = 1.0 / 2 ** (len(dim_mult) - 2)
    self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)
    self.middle = nn.Sequential(
      ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]), ResidualBlock(dims[0], dims[0], dropout)
    )
    upsamples = []
    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
      if i in (1, 2, 3):
        in_dim = in_dim // 2
      for _ in range(num_res_blocks + 1):
        upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
        if scale in attn_scales:
          upsamples.append(AttentionBlock(out_dim))
        in_dim = out_dim
      if i != len(dim_mult) - 1:
        mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
        upsamples.append(Resample(out_dim, mode=mode))
        scale *= 2.0
    self.upsamples = nn.Sequential(*upsamples)
    self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(), CausalConv3d(out_dim, 3, 3, padding=1))

  def forward(self, x, feat_cache=None, feat_idx=[0]):  # noqa: B006
    if feat_cache is not None:
      idx = feat_idx[0]
      cache_x = x[:, :, -CACHE_T:, :, :].clone()
      if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
        cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
      x = self.conv1(x, feat_cache[idx])
      feat_cache[idx] = cache_x
      feat_idx[0] += 1
    else:
      x = self.conv1(x)
    for layer in self.middle:
      if isinstance(layer, ResidualBlock) and feat_cache is not None:
        x = layer(x, feat_cache, feat_idx)
      else:
        x = layer(x)
    for layer in self.upsamples:
      if feat_cache is not None:
        x = layer(x, feat_cache, feat_idx)
      else:
        x = layer(x)
    for layer in self.head:
      if isinstance(layer, CausalConv3d) and feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
          cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
        x = layer(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
      else:
        x = layer(x)
    return x


def _to_channels_last(model: nn.Module) -> nn.Module:
  for module in model.modules():
    if isinstance(module, nn.Conv2d):
      module.weight.data = module.weight.data.contiguous(memory_format=torch.channels_last)
    elif isinstance(module, nn.Conv3d):
      module.weight.data = module.weight.data.contiguous(memory_format=torch.channels_last_3d)
  return model


class Wan2_1_VAE(nn.Module):
  # Architecture constants
  Z_DIM = 16
  BASE_DIM = 96
  DIM_MULT = (1, 2, 4, 4)
  NUM_RES_BLOCKS = 2
  ATTN_SCALES = ()
  TEMPERAL_DOWNSAMPLE = (False, True, True)

  def __init__(self):
    super().__init__()
    self.z_dim = self.Z_DIM
    self.encoder = WanEncoder3d(
      self.BASE_DIM,
      self.Z_DIM * 2,
      self.DIM_MULT,
      self.NUM_RES_BLOCKS,
      self.ATTN_SCALES,
      self.TEMPERAL_DOWNSAMPLE,
    )
    self.conv1 = CausalConv3d(self.Z_DIM * 2, self.Z_DIM * 2, 1)
    self.conv2 = CausalConv3d(self.Z_DIM, self.Z_DIM, 1)
    self.decoder = Decoder3d(
      self.BASE_DIM,
      self.Z_DIM,
      self.DIM_MULT,
      self.NUM_RES_BLOCKS,
      self.ATTN_SCALES,
      self.TEMPERAL_DOWNSAMPLE[::-1],
    )

  def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
    dtype = next(self.parameters()).dtype
    with torch.amp.autocast("cuda", dtype=dtype):
      self.clear_cache()
      t = x.shape[2]
      t = 1 + ((t - 1) // 4) * 4
      iter_ = 1 + (t - 1) // 2
      feat_map = [None] * _count_cache_layers(self.encoder) if iter_ > 1 else None
      for i in range(iter_):
        conv_idx = [0]
        if i == 0:
          out = self.encoder(x[:, :, :1, :, :], feat_cache=feat_map, feat_idx=conv_idx)
        else:
          out_ = self.encoder(
            x[:, :, 1 + 2 * (i - 1) : 1 + 2 * i, :, :],
            feat_cache=feat_map,
            feat_idx=conv_idx,
            final=(i == (iter_ - 1)),
          )
          if out_ is None:
            continue
          out = torch.cat([out, out_], 2)
      enc = self.conv1(out)
      mu, logvar = enc[:, : self.z_dim], enc[:, self.z_dim :]
      return DiagonalGaussianDistribution(torch.cat([mu, logvar], dim=1))

  def decode(self, z: torch.Tensor) -> torch.Tensor:
    dtype = next(self.parameters()).dtype
    with torch.amp.autocast("cuda", dtype=dtype):
      self.clear_cache()
      x = self.conv2(z)
      for i in range(z.shape[2]):
        self._conv_idx = [0]
        if i == 0:
          out = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
        else:
          out_ = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
          out = torch.cat([out, out_], 2)
      self.clear_cache()
      return out

  def clear_cache(self):
    self._conv_num = _count_conv3d(self.decoder)
    self._conv_idx = [0]
    self._feat_map = [None] * self._conv_num

  def load(self, model_path: str) -> None:
    import os
    from flashpack import assign_from_file

    fp_path = model_path.replace(".safetensors", ".flashpack")
    if os.path.exists(fp_path):
      print(f"Loading VAE from {fp_path} (flashpack). avail mem: {get_available_gpu_memory():.2f} GB")
      t0 = time.perf_counter()
      assign_from_file(self, fp_path, device=str(DEVICE), strict_params=True, strict_buffers=True)
      self.eval().requires_grad_(False)
      _to_channels_last(self)
      print(f"  VAE load: flashpack={time.perf_counter()-t0:.2f}s")
      return

    print(f"Loading VAE from {model_path}. avail mem: {get_available_gpu_memory():.2f} GB")
    t0 = time.perf_counter()
    state_dict = safetensors_load_file(model_path, device=str(DEVICE))
    t_read = time.perf_counter() - t0
    t1 = time.perf_counter()
    self.load_state_dict(state_dict, strict=True, assign=True)
    t_copy = time.perf_counter() - t1
    self.eval().requires_grad_(False)
    _to_channels_last(self)
    print(f"  VAE load: read={t_read:.2f}s  load_state_dict={t_copy:.2f}s")
