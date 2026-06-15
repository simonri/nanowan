"""All NN layers, kernels, and custom-op infrastructure in one file."""

import functools
import inspect
import math
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.library import Library

# --------------------------------------------------------------------------- #
# Custom-op registration infrastructure (from wan/utils/common.py and
# wan/utils/custom_op.py)
# --------------------------------------------------------------------------- #

sglang_lib = Library("sglang", "FRAGMENT")


def direct_register_custom_op(
  op_name: str,
  op_func: Callable,
  mutates_args: list[str],
  fake_impl: Callable | None = None,
  target_lib: Library | None = None,
) -> None:
  lib = target_lib if target_lib is not None else sglang_lib
  schema_str = torch.library.infer_schema(op_func, mutates_args=mutates_args)
  lib.define(f"{op_name}{schema_str}")
  lib.impl(op_name, op_func, "CUDA")
  if fake_impl is not None:
    lib._register_fake(op_name, fake_impl)


def register_custom_op(
  fn: Callable | None = None,
  *,
  op_name: str | None = None,
  mutates_args: list[str] | None = None,
  **extra_kwargs,
) -> Callable:
  if not ("out_shape" in extra_kwargs or "fake_impl" in extra_kwargs):
    extra_kwargs["out_shape"] = None

  def decorator(op_func):
    name = op_name or op_func.__name__
    if "fake_impl" in extra_kwargs:
      fake = extra_kwargs["fake_impl"]
    else:
      out_shape = extra_kwargs.get("out_shape")
      signature = inspect.signature(op_func)

      def _fake(*args, **kwargs):
        if out_shape is None:
          return None
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        try:
          ref = bound.args[out_shape] if isinstance(out_shape, int) else bound.arguments[out_shape]
        except (IndexError, KeyError) as err:
          raise RuntimeError(f"Cannot find output at {out_shape!r} for op {name!r}") from err
        return torch.empty_like(ref)

      fake = _fake
    if not hasattr(torch.ops.sglang, name):
      direct_register_custom_op(op_name=name, op_func=op_func, mutates_args=mutates_args or [], fake_impl=fake)
    return op_func

  if fn is not None:
    return decorator(fn)
  return decorator


def register_custom_op_from_extern(
  fn: Callable,
  *,
  op_name: str | None = None,
  mutates_args: list[str] | None = None,
  out_shape: int | str | None = None,
  out_dtype: torch.dtype | None = None,
  fake_impl: Callable | None = None,
  computed_args: dict | None = None,
) -> Callable:
  name = op_name or fn.__name__
  computed_args = computed_args or {}

  if computed_args:
    original_fn = fn
    original_sig = inspect.signature(fn)
    new_params = [p for pn, p in original_sig.parameters.items() if pn not in computed_args]
    new_sig = original_sig.replace(parameters=new_params)

    def wrapper(*args, **kwargs):
      bound = new_sig.bind(*args, **kwargs)
      bound.apply_defaults()
      for arg_name, compute_fn in computed_args.items():
        bound.arguments[arg_name] = compute_fn(**bound.arguments)
      return original_fn(**bound.arguments)

    wrapper.__name__ = fn.__name__
    wrapper.__qualname__ = fn.__qualname__
    wrapper.__module__ = fn.__module__
    wrapper.__signature__ = new_sig  # type: ignore
    wrapper.__annotations__ = {k: v for k, v in getattr(fn, "__annotations__", {}).items() if k not in computed_args}
    fn = wrapper

  fake_sig = inspect.signature(fn)
  if fake_impl is None and out_shape is not None:

    def _fake_impl(*args, **kwargs):
      bound = fake_sig.bind(*args, **kwargs)
      bound.apply_defaults()
      try:
        ref = bound.args[out_shape] if isinstance(out_shape, int) else bound.arguments[out_shape]
      except (IndexError, KeyError) as err:
        raise RuntimeError(f"Cannot find {out_shape!r} for extern op {name!r}") from err
      if out_dtype is not None:
        return torch.empty(ref.shape, dtype=out_dtype, device=ref.device)
      return torch.empty_like(ref)

    fake_impl = _fake_impl
  elif fake_impl is None:

    def fake_impl(*args, **kwargs):
      return None

  direct_register_custom_op(op_name=name, op_func=fn, mutates_args=mutates_args or [], fake_impl=fake_impl)
  return fn


class CustomOp(nn.Module):
  op_registry: dict[str, type["CustomOp"]] = {}

  def __init__(self) -> None:
    super().__init__()
    self._forward_method = self.dispatch_forward()

  def forward(self, *args, **kwargs) -> Any:
    return self._forward_method(*args, **kwargs)

  def forward_native(self, *args, **kwargs) -> Any:
    raise NotImplementedError

  def forward_cuda(self, *args, **kwargs) -> Any:
    raise NotImplementedError

  def dispatch_forward(self) -> Callable:
    return self.forward_cuda

  @classmethod
  def register(cls, name: str) -> Callable:
    def decorator(op_cls):
      assert name not in cls.op_registry
      op_cls.name = name
      cls.op_registry[name] = op_cls
      return op_cls

    return decorator


# --------------------------------------------------------------------------- #
# Triton scale-shift kernel (from wan/kernels/scale_shift.py)
# --------------------------------------------------------------------------- #

import triton
import triton.language as tl


@triton.jit
def _fuse_scale_shift_kernel_blc(
  x_ptr,
  shift_ptr,
  scale_ptr,
  scale_constant: tl.constexpr,
  y_ptr,
  B,
  L,
  C,
  stride_x_b,
  stride_x_l,
  stride_x_c,
  stride_s_b,
  stride_s_l,
  stride_s_c,
  stride_sc_b,
  stride_sc_l,
  stride_sc_c,
  SCALE_IS_SCALAR: tl.constexpr,
  SHIFT_IS_SCALAR: tl.constexpr,
  BLOCK_L: tl.constexpr,
  BLOCK_C: tl.constexpr,
):
  pid_l = tl.program_id(0)
  pid_c = tl.program_id(1)
  pid_b = tl.program_id(2)
  l_offsets = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
  c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
  mask_l = l_offsets < L
  mask_c = c_offsets < C
  mask = mask_l[:, None] & mask_c[None, :]
  x_off = pid_b * stride_x_b + l_offsets[:, None] * stride_x_l + c_offsets[None, :] * stride_x_c
  x = tl.load(x_ptr + x_off, mask=mask, other=0)
  if SHIFT_IS_SCALAR:
    shift_val = tl.load(shift_ptr)
    shift = tl.full((BLOCK_L, BLOCK_C), shift_val, dtype=shift_val.dtype)
  else:
    s_off = pid_b * stride_s_b + l_offsets[:, None] * stride_s_l + c_offsets[None, :] * stride_s_c
    shift = tl.load(shift_ptr + s_off, mask=mask, other=0)
  if SCALE_IS_SCALAR:
    scale_val = tl.load(scale_ptr)
    scale = tl.full((BLOCK_L, BLOCK_C), scale_val, dtype=scale_val.dtype)
  else:
    sc_off = pid_b * stride_sc_b + l_offsets[:, None] * stride_sc_l + c_offsets[None, :] * stride_sc_c
    scale = tl.load(scale_ptr + sc_off, mask=mask, other=0)
  y = x * (scale_constant + scale) + shift
  tl.store(y_ptr + x_off, y, mask=mask)


def fuse_scale_shift_kernel(
  x: torch.Tensor,
  scale: torch.Tensor,
  shift: torch.Tensor,
  scale_constant: float = 1.0,
  block_l: int = 128,
  block_c: int = 128,
) -> torch.Tensor:
  assert x.is_cuda and scale.is_cuda
  assert x.is_contiguous()
  B, L, C = x.shape
  output = torch.empty_like(x)

  if scale.dim() == 0 or (scale.dim() == 1 and scale.numel() == 1):
    scale_blc = scale.reshape(1)
  elif scale.dim() == 2:
    scale_blc = scale[:, None, :]
  else:
    scale_blc = scale

  if shift.dim() == 0 or (shift.dim() == 1 and shift.numel() == 1):
    shift_blc = shift.reshape(1)
  elif shift.dim() == 2:
    shift_blc = shift[:, None, :]
  else:
    shift_blc = shift

  need_scale_scalar = scale_blc.dim() == 1 and scale_blc.numel() == 1
  need_shift_scalar = shift_blc.dim() == 1 and shift_blc.numel() == 1

  if not need_scale_scalar:
    scale_exp = scale_blc.expand(B, L, C)
    s_sb, s_sl, s_sc = scale_exp.stride()
  else:
    s_sb = s_sl = s_sc = 0

  if not need_shift_scalar:
    shift_exp = shift_blc.expand(B, L, C)
    sh_sb, sh_sl, sh_sc = shift_exp.stride()
  else:
    sh_sb = sh_sl = sh_sc = 0

  grid = (triton.cdiv(L, block_l), triton.cdiv(C, block_c), B)
  _fuse_scale_shift_kernel_blc[grid](
    x,
    shift_blc if need_shift_scalar else shift_exp,
    scale_blc if need_scale_scalar else scale_exp,
    scale_constant,
    output,
    B,
    L,
    C,
    x.stride(0),
    x.stride(1),
    x.stride(2),
    sh_sb,
    sh_sl,
    sh_sc,
    s_sb,
    s_sl,
    s_sc,
    SCALE_IS_SCALAR=need_scale_scalar,
    SHIFT_IS_SCALAR=need_shift_scalar,
    BLOCK_L=block_l,
    BLOCK_C=block_c,
    num_warps=4,
    num_stages=2,
  )
  return output


# --------------------------------------------------------------------------- #
# Triton RMS-norm one-pass kernel (from wan/kernels/rmsnorm_onepass.py)
# --------------------------------------------------------------------------- #


@triton.jit
def _rms_norm_tiled_onepass(
  y_ptr,
  x_ptr,
  w_ptr,
  SEQ: tl.constexpr,
  DIM: tl.constexpr,
  EPS: tl.constexpr,
  BLOCK_SIZE_SEQ: tl.constexpr,
  BLOCK_SIZE_DIM: tl.constexpr,
):
  seq_blk_id = tl.program_id(0)
  seq_id = seq_blk_id * BLOCK_SIZE_SEQ
  seq_offset = seq_id + tl.arange(0, BLOCK_SIZE_SEQ)[:, None]
  s_mask = seq_offset < SEQ
  d_offset = tl.arange(0, BLOCK_SIZE_DIM)[None, :]
  d_mask = d_offset < DIM
  y_blk = y_ptr + seq_offset * DIM + d_offset
  x_blk = x_ptr + seq_offset * DIM + d_offset
  mask = s_mask & d_mask
  x = tl.load(x_blk, mask=mask, other=0.0).to(tl.float32)
  mean_square = tl.sum(x * x, axis=1, keep_dims=True) / DIM
  rstd = tl.math.rsqrt(mean_square + EPS)
  w = tl.load(w_ptr + d_offset, mask=d_mask)
  tl.store(y_blk, x * rstd * w, mask=mask)


@register_custom_op(op_name="triton_one_pass_rms_norm_cuda", out_shape="x")
def _triton_one_pass_rms_norm_cuda(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
  shape = x.shape
  x = x.contiguous()
  y = torch.empty_like(x)
  x_view = x.reshape(-1, shape[-1])
  y_view = y.reshape(-1, shape[-1])
  S, D = x_view.shape
  block_size_seq = min(16, triton.next_power_of_2(max(1, S // 512)))
  grid = (triton.cdiv(S, block_size_seq),)
  with torch.get_device_module().device(x.device):
    _rms_norm_tiled_onepass[grid](
      y_view,
      x_view,
      w,
      S,
      D,
      eps,
      BLOCK_SIZE_DIM=triton.next_power_of_2(D),
      BLOCK_SIZE_SEQ=block_size_seq,
    )
  return y


def triton_one_pass_rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
  return _triton_one_pass_rms_norm_cuda(x, w, eps)


# --------------------------------------------------------------------------- #
# Flash-attention v4 wrapper (from wan/kernels/flash_attention_v4.py)
# --------------------------------------------------------------------------- #

try:
  from flash_attn.cute import flash_attn_varlen_func as _flash_attn_varlen_func
except Exception:
  _flash_attn_varlen_func = None


def flash_attn_varlen_func(
  q,
  k,
  v,
  cu_seqlens_q=None,
  cu_seqlens_k=None,
  max_seqlen_q=None,
  max_seqlen_k=None,
  seqused_q=None,
  seqused_k=None,
  page_table=None,
  softmax_scale=None,
  causal=False,
  qv=None,
  window_size=(-1, -1),
  softcap=0.0,
  num_splits=1,
  pack_gqa=None,
  return_lse=False,
  score_mod=None,
  aux_tensors=None,
):
  if _flash_attn_varlen_func is None:
    raise RuntimeError("flash_attn not available")
  ws = tuple(None if v == -1 else v for v in window_size)
  out, lse = _flash_attn_varlen_func(
    q,
    k,
    v,
    qv=qv,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    seqused_q=seqused_q,
    seqused_k=seqused_k,
    page_table=page_table,
    softmax_scale=softmax_scale,
    causal=causal,
    softcap=softcap,
    window_size=ws,
    num_splits=num_splits,
    pack_gqa=pack_gqa,
    score_mod=score_mod,
    aux_tensors=aux_tensors,
    return_lse=return_lse,
  )
  return (out, lse) if return_lse else out


def _flash_attn_varlen_func_fake(
  q,
  k,
  v,
  cu_seqlens_q=None,
  cu_seqlens_k=None,
  max_seqlen_q=None,
  max_seqlen_k=None,
  **kwargs,
) -> torch.Tensor:
  def _maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x

  q, k, v = [_maybe_contiguous(t) for t in (q, k, v)]
  num_head, head_dim = q.shape[-2:]
  if cu_seqlens_q is None:
    batch_size, seqlen_q = q.shape[:2]
  else:
    batch_size = cu_seqlens_q.shape[0] - 1
    seqlen_q = None
  head_dim_v = v.shape[-1]
  q_batch_seqlen_shape = (batch_size, seqlen_q) if cu_seqlens_q is None else (q.shape[0],)
  return q.new_empty(*q_batch_seqlen_shape, num_head, head_dim_v)


@register_custom_op(fake_impl=_flash_attn_varlen_func_fake)
def flash_attn_varlen_func_op(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  cu_seqlens_q: torch.Tensor | None = None,
  cu_seqlens_k: torch.Tensor | None = None,
  max_seqlen_q: int | None = None,
  max_seqlen_k: int | None = None,
  seqused_q: torch.Tensor | None = None,
  seqused_k: torch.Tensor | None = None,
  page_table: torch.Tensor | None = None,
  softmax_scale: float | None = None,
  causal: bool = False,
  qv: torch.Tensor | None = None,
  window_size: list[int] | None = None,
  softcap: float = 0.0,
  num_splits: int = 1,
  pack_gqa: bool | None = None,
  return_lse: bool = False,
) -> torch.Tensor:
  ws = tuple(window_size) if window_size is not None else (-1, -1)
  return flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    seqused_q=seqused_q,
    seqused_k=seqused_k,
    page_table=page_table,
    softmax_scale=softmax_scale,
    causal=causal,
    qv=qv,
    window_size=ws,
    softcap=softcap,
    num_splits=num_splits,
    pack_gqa=pack_gqa,
    return_lse=return_lse,
  )


# --------------------------------------------------------------------------- #
# Activation functions (from wan/layers/activation.py)
# --------------------------------------------------------------------------- #

_ACTIVATIONS = {
  "gelu": nn.GELU,
  "gelu_pytorch_tanh": lambda: nn.GELU(approximate="tanh"),
  "silu": nn.SiLU,
}


def get_act_fn(name: str) -> nn.Module:
  name = name.lower()
  if name not in _ACTIVATIONS:
    raise ValueError(f"Unknown activation: {name!r}")
  return _ACTIVATIONS[name]()


# --------------------------------------------------------------------------- #
# FP8 linear layer using sgl_kernel fp8 GEMM
# --------------------------------------------------------------------------- #

import sgl_kernel


class FP8Linear(nn.Module):
  def __init__(self, in_features: int, out_features: int, bias: bool = True):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weight = nn.Parameter(
      torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn),
      requires_grad=False,
    )
    self.register_buffer("weight_scale", torch.ones(1, dtype=torch.float32))
    self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    x_flat = x.reshape(-1, x.shape[-1]).contiguous()
    n_tokens = x_flat.shape[0]
    x_q = torch.empty_like(x_flat, dtype=torch.float8_e4m3fn)
    scales_a = torch.empty(n_tokens, dtype=torch.float32, device=x.device)
    sgl_kernel.sgl_per_token_quant_fp8(x_flat, x_q, scales_a)
    scales_b = self.weight_scale.expand(self.out_features).contiguous()
    bias = self.bias.to(x.dtype) if self.bias is not None else None
    out = sgl_kernel.fp8_scaled_mm(
      x_q,
      self.weight.T,  # column-major view (no copy) — required by fp8_scaled_mm
      scales_a,
      scales_b,
      out_dtype=x.dtype,
      bias=bias,
    )
    return out.reshape(*orig_shape[:-1], self.out_features)


# --------------------------------------------------------------------------- #
# MLP (from wan/layers/mlp.py)
# --------------------------------------------------------------------------- #


class MLP(nn.Module):
  def __init__(
    self,
    input_dim: int,
    mlp_hidden_dim: int,
    output_dim: int | None = None,
    act_type: str = "gelu_pytorch_tanh",
    fp8: bool = False,
  ):
    super().__init__()
    Linear = FP8Linear if fp8 else nn.Linear
    self.fc_in = Linear(input_dim, mlp_hidden_dim, bias=True)
    self.act = get_act_fn(act_type)
    self.fc_out = Linear(mlp_hidden_dim, output_dim if output_dim is not None else input_dim, bias=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.fc_out(self.act(self.fc_in(x)))


# --------------------------------------------------------------------------- #
# Layer norms (from wan/layers/layernorm.py)
# --------------------------------------------------------------------------- #

from sgl_kernel.elementwise import fused_add_rmsnorm, rmsnorm


@CustomOp.register("rms_norm")
class RMSNorm(CustomOp):
  def __init__(
    self, hidden_size: int, eps: float = 1e-6, dtype: torch.dtype = torch.float32, var_hidden_size: int | None = None
  ):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps
    self.hidden_size = hidden_size
    self.variance_size_override = None if var_hidden_size == hidden_size else var_hidden_size

  def forward_cuda(self, x: torch.Tensor, residual: torch.Tensor | None = None):
    shape = x.shape
    x = x.reshape(-1, shape[-1])
    if residual is not None:
      residual_shape = residual.shape
      residual = residual.view(-1, shape[-1])
    if x.dtype == torch.float:
      if residual is None and self.variance_size_override is None:
        return self.forward_native(x).view(shape)
      out = self.forward_native(x, residual)
      if residual is not None:
        return out[0].view(shape), out[1].view(residual_shape)
      return out.view(shape)
    elif self.variance_size_override is not None:
      return self.forward_native(x, residual)
    elif residual is not None:
      fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
      return x.view(shape), residual.view(residual_shape)
    else:
      if x.shape[-1] <= 128:
        out = triton_one_pass_rms_norm(x, self.weight.data, self.variance_epsilon)
      else:
        out = rmsnorm(x, self.weight.data, self.variance_epsilon)
    return out.view(shape)

  def forward_native(self, x: torch.Tensor, residual: torch.Tensor | None = None):
    if residual is not None:
      x = x + residual
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * x

  def extra_repr(self) -> str:
    return f"hidden_size={self.hidden_size}, eps={self.variance_epsilon}"


class FP32LayerNorm(nn.LayerNorm):
  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    origin_dtype = inputs.dtype
    return F.layer_norm(
      inputs.float(),
      self.normalized_shape,
      self.weight.float() if self.weight is not None else None,
      self.bias.float() if self.bias is not None else None,
      self.eps,
    ).to(origin_dtype)


class _ScaleResidualNormScaleShift(CustomOp):
  norm_type: str

  def __init__(
    self,
    hidden_size: int,
    eps: float = 1e-6,
    elementwise_affine: bool = False,
    dtype: torch.dtype = torch.float32,
    prefix: str = "",
  ):
    super().__init__()
    self.eps = eps
    self.dtype = dtype
    if self.norm_type == "rms":
      self.norm = RMSNorm(hidden_size, eps=eps, dtype=dtype)
    elif self.norm_type == "layer":
      self.norm = FP32LayerNorm(hidden_size, elementwise_affine=elementwise_affine, eps=eps, dtype=dtype)

  def forward_cuda(self, residual, x, gate, shift, scale):
    return self.forward_native(residual, x, gate, shift, scale)

  def forward_native(self, residual, x, gate, shift, scale):
    if isinstance(gate, int):
      assert gate == 1
      residual_output = residual + x
    elif isinstance(gate, torch.Tensor):
      residual_output = residual + x * gate
    normalized = self.norm(residual_output)
    modulated = fuse_scale_shift_kernel(normalized, scale, shift)
    return modulated, residual_output


class _NormScaleShift(CustomOp):
  norm_type: str

  def __init__(
    self,
    hidden_size: int,
    eps: float = 1e-6,
    elementwise_affine: bool = False,
    dtype: torch.dtype = torch.float32,
    prefix: str = "",
  ):
    super().__init__()
    self.eps = eps
    if self.norm_type == "rms":
      self.norm = RMSNorm(hidden_size, eps=eps, dtype=dtype)
    elif self.norm_type == "layer":
      self.norm = FP32LayerNorm(hidden_size, elementwise_affine=elementwise_affine, eps=eps, dtype=dtype)

  def forward_cuda(self, x, shift, scale):
    return self.forward_native(x, shift, scale)

  def forward_native(self, x, shift, scale):
    normalized = self.norm(x)
    return fuse_scale_shift_kernel(normalized, scale, shift).to(x.dtype)


class LayerNormScaleShift(_NormScaleShift):
  norm_type = "layer"


class ScaleResidualLayerNormScaleShift(_ScaleResidualNormScaleShift):
  norm_type = "layer"


# --------------------------------------------------------------------------- #
# Fused elementwise mul-add (from wan/layers/elementwise.py)
# --------------------------------------------------------------------------- #


class MulAdd(CustomOp):
  def __init__(self, prefix: str = ""):
    super().__init__()

  def forward_native(self, a, b, c, k: int = 0):
    return c + a * (k + b)

  def forward_cuda(self, a, b, c, k: int = 0):
    return fuse_scale_shift_kernel(a, b, c, scale_constant=k)


# --------------------------------------------------------------------------- #
# Timestep / patch / modulate embeddings (from wan/layers/visual_embedding.py)
# --------------------------------------------------------------------------- #


def timestep_embedding(
  t: torch.Tensor, dim: int, max_period: int = 10000, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
  half = dim // 2
  freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=dtype, device=t.device) / half)
  args = t[:, None].float() * freqs[None]
  embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
  if dim % 2:
    embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
  return embedding


class TimestepEmbedder(nn.Module):
  def __init__(
    self,
    hidden_size,
    act_layer="silu",
    frequency_embedding_size=256,
    max_period=10000,
    dtype=None,
    freq_dtype=torch.float32,
  ):
    super().__init__()
    self.frequency_embedding_size = frequency_embedding_size
    self.max_period = max_period
    self.mlp = MLP(frequency_embedding_size, hidden_size, hidden_size, act_type=act_layer)
    self.freq_dtype = freq_dtype

  def forward(self, t: torch.Tensor) -> torch.Tensor:
    t_freq = timestep_embedding(t, self.frequency_embedding_size, self.max_period, dtype=self.freq_dtype).to(
      self.mlp.fc_in.weight.dtype
    )
    return self.mlp(t_freq)


class PatchEmbed(nn.Module):
  def __init__(
    self,
    patch_size=16,
    in_chans=3,
    embed_dim=768,
    norm_layer=None,
    flatten=True,
    bias=True,
    dtype=None,
    prefix: str = "",
  ):
    super().__init__()
    if isinstance(patch_size, (list, tuple)):
      if len(patch_size) == 1:
        patch_size = (1, patch_size[0], patch_size[0])
      elif len(patch_size) == 2:
        patch_size = (1, patch_size[0], patch_size[1])
    else:
      patch_size = (1, patch_size, patch_size)
    self.patch_size = patch_size
    self.flatten = flatten
    self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias, dtype=dtype)
    self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

  def forward(self, x):
    if x.dim() == 5:
      B, C, T, H, W = x.shape
      pt, ph, pw = self.patch_size
      if T % pt == 0 and H % ph == 0 and W % pw == 0:
        T_ = T // pt
        H_ = H // ph
        W_ = W // pw
        x = x.reshape(B, C, T_, pt, H_, ph, W_, pw)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        x = x.reshape(B, T_ * H_ * W_, C * pt * ph * pw)
        w = self.proj.weight.reshape(self.proj.weight.shape[0], -1)
        x = F.linear(x, w, self.proj.bias)
        if not self.flatten:
          x = x.reshape(B, T_, H_, W_, -1).permute(0, 4, 1, 2, 3).contiguous()
        x = self.norm(x)
        return x
    x = self.proj(x)
    if self.flatten:
      x = x.flatten(2).transpose(1, 2)
    return self.norm(x)


class ModulateProjection(nn.Module):
  def __init__(self, hidden_size: int, factor: int = 2, act_layer: str = "silu", dtype: torch.dtype | None = None):
    super().__init__()
    self.factor = factor
    self.hidden_size = hidden_size
    self.linear = nn.Linear(hidden_size, hidden_size * factor, bias=True)
    self.act = get_act_fn(act_layer)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.linear(self.act(x))


# --------------------------------------------------------------------------- #
# N-D rotary positional embedding (from wan/layers/mrope.py)
# --------------------------------------------------------------------------- #


def _to_tuple(x, dim=2):
  if isinstance(x, int):
    return (x,) * dim
  elif len(x) == dim:
    return x
  raise ValueError(f"Expected length {dim} or int, got {x}")


class OneDRotaryEmbedding(nn.Module):
  def __init__(
    self,
    dim: int,
    theta: float = 10000.0,
    theta_rescale_factor: float = 1.0,
    interpolation_factor: float = 1.0,
    dtype: torch.dtype = torch.float32,
    use_real: bool = False,
    repeat_interleave_real: bool = False,
  ):
    super().__init__()
    assert dim % 2 == 0
    self.dim = dim
    self.theta = theta
    self.theta_rescale_factor = theta_rescale_factor
    self.interpolation_factor = interpolation_factor
    self.dtype = dtype
    self.use_real = use_real
    self.repeat_interleave_real = repeat_interleave_real

  def build_freqs(self, device):
    return 1.0 / (
      self.theta ** (torch.arange(0, self.dim, 2, dtype=self.dtype, device=device)[: (self.dim // 2)] / self.dim)
    )

  def build_freqs_outer(self, pos, device):
    theta = self.theta
    if self.theta_rescale_factor != 1.0:
      theta *= self.theta_rescale_factor ** (self.dim / (self.dim - 2))
    freqs = self.build_freqs(device)
    freqs = torch.outer(pos * self.interpolation_factor, freqs)
    freqs_cos = freqs.cos()
    freqs_sin = freqs.sin()
    if self.use_real and self.repeat_interleave_real:
      freqs_cos = freqs_cos.repeat_interleave(2, dim=1)
      freqs_sin = freqs_sin.repeat_interleave(2, dim=1)
    return freqs_cos.float(), freqs_sin.float()

  @functools.lru_cache(maxsize=16)  # noqa: B019
  def forward_from_grid(self, seq_len: int, start_pos: int, device_str: str):
    device = torch.device(device_str)
    pos = torch.arange(start_pos, start_pos + seq_len, dtype=self.dtype, device=device)
    return self.build_freqs_outer(pos, device)


class NDRotaryEmbedding(nn.Module):
  def __init__(
    self,
    rope_dim_list,
    rope_theta,
    theta_rescale_factor=1.0,
    interpolation_factor=1.0,
    use_real=False,
    repeat_interleave_real=False,
    dtype=torch.float32,
  ):
    super().__init__()
    self.rope_dim_list = rope_dim_list
    self.ndim = len(rope_dim_list)
    self.rope_theta = rope_theta
    self.dtype = dtype
    if isinstance(theta_rescale_factor, (int, float)):
      self.theta_rescale_factor = [theta_rescale_factor] * self.ndim
    else:
      self.theta_rescale_factor = theta_rescale_factor
    if isinstance(interpolation_factor, (int, float)):
      self.interpolation_factor = [interpolation_factor] * self.ndim
    else:
      self.interpolation_factor = interpolation_factor

    self.rope_generators = nn.ModuleList()
    _config_to_gen_idx: dict = {}
    self.dim_idx_to_gen_idx: list = []
    for i in range(self.ndim):
      dim = rope_dim_list[i]
      rescale = self.theta_rescale_factor[i]
      interp = self.interpolation_factor[i]
      config_key = (dim, rescale, interp, use_real, repeat_interleave_real)
      if config_key not in _config_to_gen_idx:
        gen = OneDRotaryEmbedding(
          dim=dim,
          theta=rope_theta,
          theta_rescale_factor=rescale,
          interpolation_factor=interp,
          dtype=dtype,
          use_real=use_real,
          repeat_interleave_real=repeat_interleave_real,
        )
        _config_to_gen_idx[config_key] = len(self.rope_generators)
        self.rope_generators.append(gen)
      self.dim_idx_to_gen_idx.append(_config_to_gen_idx[config_key])

  def forward_from_grid(self, grid_size, start_frame=0, device=None):
    return self._forward_cached_from_grid(grid_size, start_frame, device)

  @functools.lru_cache(maxsize=16)  # noqa: B019
  def _forward_cached_from_grid(self, grid_size, start_frame, device_str):
    device = torch.device(device_str)
    sizes = _to_tuple(grid_size, dim=self.ndim)
    num_tokens = 1
    for s in sizes:
      num_tokens *= int(s)
    head_dim_half = sum(self.rope_dim_list) // 2
    cos = torch.empty((num_tokens, head_dim_half), device=device, dtype=self.dtype)
    sin = torch.empty((num_tokens, head_dim_half), device=device, dtype=self.dtype)
    col_offset = 0
    for i in range(self.ndim):
      dim_i = self.rope_dim_list[i]
      dim_i_half = dim_i // 2
      size_i = int(sizes[i])
      base_offset = start_frame if (i == 0 and start_frame > 0) else 0
      gen = self.rope_generators[self.dim_idx_to_gen_idx[i]]
      cos_1d, sin_1d = gen.forward_from_grid(size_i, base_offset, str(device))
      repeats_per_entry = 1
      for j in range(i + 1, self.ndim):
        repeats_per_entry *= int(sizes[j])
      tile_count = 1
      for j in range(i):
        tile_count *= int(sizes[j])
      cos_expanded = cos_1d.repeat_interleave(repeats_per_entry, dim=0)
      sin_expanded = sin_1d.repeat_interleave(repeats_per_entry, dim=0)
      if tile_count > 1:
        cos_expanded = cos_expanded.repeat(tile_count, 1)
        sin_expanded = sin_expanded.repeat(tile_count, 1)
      cos[:, col_offset : col_offset + dim_i_half] = cos_expanded
      sin[:, col_offset : col_offset + dim_i_half] = sin_expanded
      col_offset += dim_i_half
    return cos.float(), sin.float()


# --------------------------------------------------------------------------- #
# FlashInfer rotary embedding (from wan/layers/rotary_embedding/utils.py)
# --------------------------------------------------------------------------- #

try:
  from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace as _flashinfer_apply_rope_inplace
except Exception:
  _flashinfer_apply_rope_inplace = None

if _flashinfer_apply_rope_inplace is not None:
  flashinfer_apply_rope_inplace = register_custom_op_from_extern(
    _flashinfer_apply_rope_inplace,
    op_name="flashinfer_apply_rope_with_cos_sin_cache_inplace",
    mutates_args=["query", "key"],
  )
else:
  flashinfer_apply_rope_inplace = None


def apply_flashinfer_rope_qk_inplace(q, k, cos_sin_cache, *, head_size=None, is_neox=False, positions=None):
  if q.dim() != 4 or k.dim() != 4 or q.shape != k.shape:
    raise ValueError()
  bsz, seqlen, nheads, d = q.shape
  if head_size is None:
    head_size = d

  if flashinfer_apply_rope_inplace is None:
    half_size = cos_sin_cache.shape[-1] // 2
    if positions is None:
      cos = cos_sin_cache[:seqlen, :half_size].to(q.dtype)
      sin = cos_sin_cache[:seqlen, half_size:].to(q.dtype)
      cos = cos.unsqueeze(0).expand(bsz, -1, -1).reshape(bsz * seqlen, -1)
      sin = sin.unsqueeze(0).expand(bsz, -1, -1).reshape(bsz * seqlen, -1)
    else:
      positions = positions.to(cos_sin_cache.device).view(-1)
      cos = cos_sin_cache[positions, :half_size].to(q.dtype)
      sin = cos_sin_cache[positions, half_size:].to(q.dtype)
    q_flat = q.reshape(bsz * seqlen, nheads, d)
    k_flat = k.reshape(bsz * seqlen, nheads, d)

    def _apply_rotary(x, cos, sin, interleaved):
      x1, x2 = x[..., ::2], x[..., 1::2]
      c, s = cos[:, None, :], sin[:, None, :]
      if interleaved:
        return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)
      half = x.shape[-1] // 2
      return torch.cat([x[..., :half] * c - x[..., half:] * s, x[..., :half] * s + x[..., half:] * c], dim=-1)

    q_rot = _apply_rotary(q_flat, cos, sin, interleaved=not is_neox)
    k_rot = _apply_rotary(k_flat, cos, sin, interleaved=not is_neox)
    return q_rot.view(bsz, seqlen, nheads, d), k_rot.view(bsz, seqlen, nheads, d)

  if positions is None:
    pos_1d = torch.arange(seqlen, device=q.device, dtype=torch.long)
    positions = pos_1d if bsz == 1 else pos_1d.repeat(bsz)

  q_flat = q.reshape(bsz * seqlen, nheads * d).contiguous()
  k_flat = k.reshape(bsz * seqlen, nheads * d).contiguous()
  flashinfer_apply_rope_inplace(
    positions=positions,
    query=q_flat,
    key=k_flat,
    head_size=d,
    cos_sin_cache=cos_sin_cache,
    is_neox=is_neox,
  )
  return q_flat.view(bsz, seqlen, nheads, d), k_flat.view(bsz, seqlen, nheads, d)


# --------------------------------------------------------------------------- #
# WanAttention (from wan/layers/attention/layer.py)
# --------------------------------------------------------------------------- #


class WanAttention(nn.Module):
  def __init__(self, num_heads, head_size, softmax_scale=None, causal=False):
    super().__init__()
    self.num_heads = num_heads
    self.head_size = head_size
    self.softmax_scale = softmax_scale
    self.causal = causal

  def forward(self, q, k, v):
    return flash_attn_varlen_func_op(
      q=q,
      k=k,
      v=v,
      max_seqlen_q=q.shape[1],
      max_seqlen_k=k.shape[1],
      softmax_scale=self.softmax_scale,
      causal=self.causal,
    )
