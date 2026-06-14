"""WanModel: diffusion transformer for image-to-video generation."""

import math
import time

import torch
import torch.nn as nn
from safetensors.torch import load_file as safetensors_load_file

from layers import (
  MLP,
  LayerNormScaleShift,
  ModulateProjection,
  MulAdd,
  NDRotaryEmbedding,
  PatchEmbed,
  RMSNorm,
  ScaleResidualLayerNormScaleShift,
  TimestepEmbedder,
  WanAttention,
  apply_flashinfer_rope_qk_inplace,
)
from lora import get_param_names_mapping
from utils import get_available_gpu_memory

__all__ = ["WanModel", "FP8Linear", "RowWiseFP8Linear", "replace_ffn_linears_with_fp8", "replace_attn_linears_with_fp8", "replace_last_n_ffn_with_fp8", "replace_last_n_attn_with_rowwise_fp8"]

_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0


class FP8Linear(nn.Module):
  """FP8 linear via _scaled_mm with per-tensor dynamic activation scale."""

  def __init__(self, weight_fp16: torch.Tensor, bias: torch.Tensor | None = None):
    super().__init__()
    out_features, in_features = weight_fp16.shape
    self.in_features = in_features
    self.out_features = out_features
    amax = weight_fp16.detach().float().abs().amax()
    scale = (amax / _FP8_MAX).clamp_min(1e-12)
    w_fp8 = (weight_fp16.detach().float() / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    self.register_buffer("weight", w_fp8)
    self.register_buffer("weight_scale", scale.float().reshape(1))
    self.bias = nn.Parameter(bias.clone()) if bias is not None else None

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    x_2d = x.reshape(-1, self.in_features)
    amax = x_2d.abs().amax().float()  # fp32 scalar for scale math
    scale_a = (amax / _FP8_MAX).clamp_min(1e-12).reshape(1)
    # skip bf16→fp32 upcasting: fp8 has only 3 mantissa bits so bf16 precision is sufficient
    x_fp8 = (x_2d / scale_a.to(x_2d.dtype)).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    out = torch._scaled_mm(x_fp8, self.weight.T, scale_a=scale_a, scale_b=self.weight_scale, out_dtype=x.dtype)
    if self.bias is not None:
      out = out + self.bias.to(out.dtype)
    return out.reshape(*orig_shape[:-1], self.out_features)


class RowWiseFP8Linear(nn.Module):
  """FP8 linear with per-token (row-wise) activation scale — more accurate for noisy activations."""

  def __init__(self, weight_fp16: torch.Tensor, bias: torch.Tensor | None = None):
    super().__init__()
    out_features, in_features = weight_fp16.shape
    self.in_features = in_features
    self.out_features = out_features
    amax = weight_fp16.detach().float().abs().amax()
    scale = (amax / _FP8_MAX).clamp_min(1e-12)
    w_fp8 = (weight_fp16.detach().float() / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    self.register_buffer("weight", w_fp8)
    # rowwise requires scale_b shape (1, out_features)
    self.register_buffer("weight_scale", scale.float().reshape(1, 1).expand(1, out_features).contiguous())
    self.bias = nn.Parameter(bias.clone()) if bias is not None else None

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    x_2d = x.reshape(-1, self.in_features)
    amax = x_2d.abs().amax(dim=1, keepdim=True).float()  # (M, 1) fp32 for scale math
    scale_a = (amax / _FP8_MAX).clamp_min(1e-12)
    x_fp8 = (x_2d / scale_a.to(x_2d.dtype)).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    out = torch._scaled_mm(x_fp8, self.weight.T, scale_a=scale_a, scale_b=self.weight_scale, out_dtype=x.dtype)
    if self.bias is not None:
      out = out + self.bias.to(out.dtype)
    return out.reshape(*orig_shape[:-1], self.out_features)

  def forward_prequantized(self, x_fp8: torch.Tensor, scale_a: torch.Tensor, orig: torch.Tensor) -> torch.Tensor:
    """Use pre-quantized x_fp8/scale_a (skip per-token amax+quantize for sharing across Q,K,V)."""
    out = torch._scaled_mm(x_fp8, self.weight.T, scale_a=scale_a, scale_b=self.weight_scale, out_dtype=orig.dtype)
    if self.bias is not None:
      out = out + self.bias.to(out.dtype)
    return out.reshape(*orig.shape[:-1], self.out_features)


def replace_ffn_linears_with_fp8(model: "WanModel") -> None:
  for block in model.blocks:
    ffn = block.ffn
    ffn.fc_in = FP8Linear(ffn.fc_in.weight, ffn.fc_in.bias)
    ffn.fc_out = FP8Linear(ffn.fc_out.weight, ffn.fc_out.bias)


def replace_attn_linears_with_fp8(model: "WanModel") -> None:
  """Replace attention projection linears with FP8 — inputs are RMSNorm outputs (Gaussian)."""
  for block in model.blocks:
    block.to_q = FP8Linear(block.to_q.weight, block.to_q.bias)
    block.to_k = FP8Linear(block.to_k.weight, block.to_k.bias)
    block.to_v = FP8Linear(block.to_v.weight, block.to_v.bias)
    block.to_out = FP8Linear(block.to_out.weight, block.to_out.bias)
    block.attn2.to_q = FP8Linear(block.attn2.to_q.weight, block.attn2.to_q.bias)
    block.attn2.to_k = FP8Linear(block.attn2.to_k.weight, block.attn2.to_k.bias)
    block.attn2.to_v = FP8Linear(block.attn2.to_v.weight, block.attn2.to_v.bias)
    block.attn2.to_out = FP8Linear(block.attn2.to_out.weight, block.attn2.to_out.bias)


def replace_last_n_ffn_with_fp8(model: "WanModel", n: int) -> None:
  """Replace FFN linears in the last n transformer blocks with FP8."""
  for block in model.blocks[-n:]:
    ffn = block.ffn
    ffn.fc_in = FP8Linear(ffn.fc_in.weight, ffn.fc_in.bias)
    ffn.fc_out = FP8Linear(ffn.fc_out.weight, ffn.fc_out.bias)


def replace_last_n_attn_with_rowwise_fp8(model: "WanModel", n: int) -> None:
  """Replace attention projection linears in the last n blocks with rowwise FP8."""
  for block in model.blocks[-n:]:
    block.to_q = RowWiseFP8Linear(block.to_q.weight, block.to_q.bias)
    block.to_k = RowWiseFP8Linear(block.to_k.weight, block.to_k.bias)
    block.to_v = RowWiseFP8Linear(block.to_v.weight, block.to_v.bias)
    block.to_out = RowWiseFP8Linear(block.to_out.weight, block.to_out.bias)
    block.attn2.to_q = RowWiseFP8Linear(block.attn2.to_q.weight, block.attn2.to_q.bias)
    block.attn2.to_k = RowWiseFP8Linear(block.attn2.to_k.weight, block.attn2.to_k.bias)
    block.attn2.to_v = RowWiseFP8Linear(block.attn2.to_v.weight, block.attn2.to_v.bias)
    block.attn2.to_out = RowWiseFP8Linear(block.attn2.to_out.weight, block.attn2.to_out.bias)


# Checkpoint key → model key remapping
PARAM_NAMES_MAPPING = {
  r"^patch_embedding\.(.*)$": r"patch_embedding.proj.\1",
  r"^text_embedding\.0\.(.*)$": r"condition_embedder.text_embedder.fc_in.\1",
  r"^text_embedding\.2\.(.*)$": r"condition_embedder.text_embedder.fc_out.\1",
  r"^time_embedding\.0\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_in.\1",
  r"^time_embedding\.2\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_out.\1",
  r"^time_projection\.1\.(.*)$": r"condition_embedder.time_modulation.linear.\1",
  r"^head\.head\.(.*)$": r"proj_out.\1",
  r"^head\.modulation$": r"scale_shift_table",
  r"^blocks\.(\d+)\.self_attn\.q\.(.*)$": r"blocks.\1.to_q.\2",
  r"^blocks\.(\d+)\.self_attn\.k\.(.*)$": r"blocks.\1.to_k.\2",
  r"^blocks\.(\d+)\.self_attn\.v\.(.*)$": r"blocks.\1.to_v.\2",
  r"^blocks\.(\d+)\.self_attn\.o\.(.*)$": r"blocks.\1.to_out.\2",
  r"^blocks\.(\d+)\.self_attn\.norm_q\.(.*)$": r"blocks.\1.norm_q.\2",
  r"^blocks\.(\d+)\.self_attn\.norm_k\.(.*)$": r"blocks.\1.norm_k.\2",
  r"^blocks\.(\d+)\.cross_attn\.q\.(.*)$": r"blocks.\1.attn2.to_q.\2",
  r"^blocks\.(\d+)\.cross_attn\.k\.(.*)$": r"blocks.\1.attn2.to_k.\2",
  r"^blocks\.(\d+)\.cross_attn\.v\.(.*)$": r"blocks.\1.attn2.to_v.\2",
  r"^blocks\.(\d+)\.cross_attn\.o\.(.*)$": r"blocks.\1.attn2.to_out.\2",
  r"^blocks\.(\d+)\.cross_attn\.norm_q\.(.*)$": r"blocks.\1.attn2.norm_q.\2",
  r"^blocks\.(\d+)\.cross_attn\.norm_k\.(.*)$": r"blocks.\1.attn2.norm_k.\2",
  r"^blocks\.(\d+)\.ffn\.0\.(.*)$": r"blocks.\1.ffn.fc_in.\2",
  r"^blocks\.(\d+)\.ffn\.2\.(.*)$": r"blocks.\1.ffn.fc_out.\2",
  r"^blocks\.(\d+)\.norm3\.(.*)$": r"blocks.\1.self_attn_residual_norm.norm.\2",
  r"^blocks\.(\d+)\.modulation$": r"blocks.\1.scale_shift_table",
}

# LoRA key remapping (handles both diffusers and kohya naming)
LORA_PARAM_NAMES_MAPPING = {
  r"^blocks[._](\d+)[._]self_attn[._]q\.(.*)$": r"blocks.\1.to_q.\2",
  r"^blocks[._](\d+)[._]self_attn[._]k\.(.*)$": r"blocks.\1.to_k.\2",
  r"^blocks[._](\d+)[._]self_attn[._]v\.(.*)$": r"blocks.\1.to_v.\2",
  r"^blocks[._](\d+)[._]self_attn[._]o\.(.*)$": r"blocks.\1.to_out.\2",
  r"^blocks[._](\d+)[._]cross_attn[._]q\.(.*)$": r"blocks.\1.attn2.to_q.\2",
  r"^blocks[._](\d+)[._]cross_attn[._]k\.(.*)$": r"blocks.\1.attn2.to_k.\2",
  r"^blocks[._](\d+)[._]cross_attn[._]v\.(.*)$": r"blocks.\1.attn2.to_v.\2",
  r"^blocks[._](\d+)[._]cross_attn[._]o\.(.*)$": r"blocks.\1.attn2.to_out.\2",
  r"^blocks[._](\d+)[._]ffn[._]0\.(.*)$": r"blocks.\1.ffn.fc_in.\2",
  r"^blocks[._](\d+)[._]ffn[._]2\.(.*)$": r"blocks.\1.ffn.fc_out.\2",
}


class WanCrossAttention(nn.Module):
  def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, eps: float = 1e-6):
    assert dim % num_heads == 0
    super().__init__()
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.to_q = nn.Linear(dim, dim)
    self.to_k = nn.Linear(dim, dim)
    self.to_v = nn.Linear(dim, dim)
    self.to_out = nn.Linear(dim, dim)
    self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
    self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
    self.attn = WanAttention(num_heads=num_heads, head_size=self.head_dim, causal=False)

  def forward(self, x, context):
    q = self.norm_q(self.to_q(x)).unflatten(2, (self.num_heads, self.head_dim))
    k = self.norm_k(self.to_k(context)).unflatten(2, (self.num_heads, self.head_dim))
    v = self.to_v(context).unflatten(2, (self.num_heads, self.head_dim))
    return self.to_out(self.attn(q, k, v).flatten(2))


class WanTransformerBlock(nn.Module):
  def __init__(self, dim: int, ffn_dim: int, num_heads: int, eps: float = 1e-6):
    super().__init__()
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.norm1 = LayerNormScaleShift(dim, eps=eps, elementwise_affine=False, dtype=torch.float32)
    self.to_q = nn.Linear(dim, dim, bias=True)
    self.to_k = nn.Linear(dim, dim, bias=True)
    self.to_v = nn.Linear(dim, dim, bias=True)
    self.to_out = nn.Linear(dim, dim, bias=True)
    self.attn1 = WanAttention(num_heads=num_heads, head_size=self.head_dim, causal=False)
    self.norm_q = RMSNorm(dim, eps=eps)
    self.norm_k = RMSNorm(dim, eps=eps)
    self.self_attn_residual_norm = ScaleResidualLayerNormScaleShift(
      dim, eps=eps, elementwise_affine=True, dtype=torch.float32
    )
    self.attn2 = WanCrossAttention(dim, num_heads, qk_norm=True, eps=eps)
    self.cross_attn_residual_norm = ScaleResidualLayerNormScaleShift(
      dim, eps=eps, elementwise_affine=False, dtype=torch.float32
    )
    self.ffn = MLP(dim, ffn_dim, act_type="gelu_pytorch_tanh")
    self.mlp_residual = MulAdd()
    self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

  def forward(self, hidden_states, encoder_hidden_states, temb, freqs_cis):
    orig_dtype = hidden_states.dtype
    e = self.scale_shift_table + temb.float()
    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = e.chunk(6, dim=1)

    norm_hidden_states = self.norm1(hidden_states, shift_msa, scale_msa)
    if isinstance(self.to_q, RowWiseFP8Linear):
      x_2d = norm_hidden_states.reshape(-1, self.to_q.in_features)
      scale_a = (x_2d.abs().amax(dim=1, keepdim=True).float() / _FP8_MAX).clamp_min(1e-12)
      x_fp8 = (x_2d / scale_a.to(x_2d.dtype)).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
      q_proj = self.to_q.forward_prequantized(x_fp8, scale_a, norm_hidden_states)
      k_proj = self.to_k.forward_prequantized(x_fp8, scale_a, norm_hidden_states)
      v_proj = self.to_v.forward_prequantized(x_fp8, scale_a, norm_hidden_states)
    else:
      q_proj = self.to_q(norm_hidden_states)
      k_proj = self.to_k(norm_hidden_states)
      v_proj = self.to_v(norm_hidden_states)
    query = self.norm_q(q_proj).unflatten(2, (self.num_heads, self.head_dim))
    key = self.norm_k(k_proj).unflatten(2, (self.num_heads, self.head_dim))
    value = v_proj.unflatten(2, (self.num_heads, self.head_dim))

    cos, sin = freqs_cis
    cos_sin_cache = torch.cat([cos.contiguous(), sin.contiguous()], dim=-1)
    query, key = apply_flashinfer_rope_qk_inplace(query, key, cos_sin_cache, is_neox=False)

    attn_output = self.to_out(self.attn1(query, key, value).flatten(2))
    # skip fuse_scale_shift_kernel(shift=0, scale=0) which is a no-op identity
    hidden_states = hidden_states + attn_output * gate_msa
    norm_hidden_states = self.self_attn_residual_norm.norm(hidden_states).to(orig_dtype)
    hidden_states = hidden_states.to(orig_dtype)

    attn_output = self.attn2(norm_hidden_states, encoder_hidden_states)
    norm_hidden_states, hidden_states = self.cross_attn_residual_norm(
      hidden_states, attn_output, 1, c_shift_msa, c_scale_msa
    )
    norm_hidden_states = norm_hidden_states.to(orig_dtype)
    hidden_states = hidden_states.to(orig_dtype)

    ff_output = self.ffn(norm_hidden_states)
    return self.mlp_residual(ff_output, c_gate_msa, hidden_states).to(orig_dtype)


class WanTimeTextImageEmbedding(nn.Module):
  def __init__(self, dim: int, time_freq_dim: int, text_embed_dim: int):
    super().__init__()
    self.time_embedder = TimestepEmbedder(dim, frequency_embedding_size=time_freq_dim, act_layer="silu")
    self.time_modulation = ModulateProjection(dim, factor=6, act_layer="silu")
    self.text_embedder = MLP(text_embed_dim, dim, dim, act_type="gelu_pytorch_tanh")

  def forward(self, timestep, encoder_hidden_states_text):
    temb = self.time_embedder(timestep)
    timestep_proj = self.time_modulation(temb).unflatten(-1, (6, -1))
    encoder_hidden_states_text = self.text_embedder(encoder_hidden_states_text)
    return temb, timestep_proj, encoder_hidden_states_text


class WanModel(nn.Module):
  # Architecture constants (WAN 2.2 14B)
  PATCH_SIZE = (1, 2, 2)
  NUM_ATTENTION_HEADS = 40
  ATTENTION_HEAD_DIM = 128
  IN_CHANNELS = 36
  OUT_CHANNELS = 16
  TEXT_DIM = 4096
  FREQ_DIM = 256
  FFN_DIM = 13824
  NUM_LAYERS = 40
  EPS = 1e-6

  PARAM_NAMES_MAPPING = PARAM_NAMES_MAPPING
  LORA_PARAM_NAMES_MAPPING = LORA_PARAM_NAMES_MAPPING

  def __init__(self):
    super().__init__()
    inner_dim = self.NUM_ATTENTION_HEADS * self.ATTENTION_HEAD_DIM
    self.hidden_size = inner_dim
    self.num_attention_heads = self.NUM_ATTENTION_HEADS
    self.in_channels = self.IN_CHANNELS
    self.out_channels = self.OUT_CHANNELS
    self.num_channels_latents = self.OUT_CHANNELS
    self.patch_size = self.PATCH_SIZE

    self.patch_embedding = PatchEmbed(
      in_chans=self.IN_CHANNELS, embed_dim=inner_dim, patch_size=self.PATCH_SIZE, flatten=False
    )
    self.condition_embedder = WanTimeTextImageEmbedding(
      dim=inner_dim, time_freq_dim=self.FREQ_DIM, text_embed_dim=self.TEXT_DIM
    )
    self.blocks = nn.ModuleList(
      [
        WanTransformerBlock(dim=inner_dim, ffn_dim=self.FFN_DIM, num_heads=self.NUM_ATTENTION_HEADS, eps=self.EPS)
        for _ in range(self.NUM_LAYERS)
      ]
    )
    self.norm_out = LayerNormScaleShift(inner_dim, eps=self.EPS, elementwise_affine=False, dtype=torch.float32)
    self.proj_out = nn.Linear(inner_dim, self.OUT_CHANNELS * math.prod(self.PATCH_SIZE), bias=True)
    self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

    d = self.hidden_size // self.NUM_ATTENTION_HEADS
    rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
    self.rotary_emb = NDRotaryEmbedding(rope_dim_list=rope_dim_list, rope_theta=10000, dtype=torch.float64)

  def forward(
    self, hidden_states: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: list[torch.Tensor]
  ) -> torch.Tensor:
    batch_size, _, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    freqs_cis = self.rotary_emb.forward_from_grid(
      (post_patch_num_frames, post_patch_height, post_patch_width),
      start_frame=0,
      device=str(hidden_states.device),
    )

    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    temb, timestep_proj, encoder_hidden_states = self.condition_embedder(timestep, encoder_hidden_states)

    for block in self.blocks:
      hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, freqs_cis)

    shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

    hidden_states = self.norm_out(hidden_states, shift, scale)
    hidden_states = self.proj_out(hidden_states)
    hidden_states = hidden_states.reshape(
      batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

  def load(self, model_path: str) -> None:
    print(f"Loading Transformer from {model_path}. avail mem: {get_available_gpu_memory():.2f} GB")
    t0 = time.perf_counter()
    state_dict = safetensors_load_file(model_path)
    t_read = time.perf_counter() - t0

    t1 = time.perf_counter()
    mapping_fn = get_param_names_mapping(self.PARAM_NAMES_MAPPING)
    state_dict = {mapping_fn(k): v for k, v in state_dict.items()}
    t_rename = time.perf_counter() - t1

    t2 = time.perf_counter()
    self.load_state_dict(state_dict, strict=True)
    torch.cuda.synchronize()
    t_copy = time.perf_counter() - t2

    self.eval().requires_grad_(False)
    print(f"  Transformer load: read={t_read:.2f}s  rename={t_rename:.2f}s  load_state_dict={t_copy:.2f}s")
