"""T5 text encoder (UMT5-XXL)."""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as safetensors_load_file

from utils import DEVICE, get_available_gpu_memory

# Architecture constants (UMT5-XXL used by WAN 2.2)
VOCAB_SIZE = 256384
DIM = 4096
DIM_ATTN = 4096
DIM_FFN = 10240
NUM_HEADS = 64
NUM_LAYERS = 24
NUM_BUCKETS = 32
SHARED_POS = False
DROPOUT = 0.1


def fp16_clamp(x: torch.Tensor) -> torch.Tensor:
  if x.dtype == torch.float16 and torch.isinf(x).any():
    clamp = torch.finfo(x.dtype).max - 1000
    x = torch.clamp(x, min=-clamp, max=clamp)
  return x


class GELU(nn.Module):
  def forward(self, x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class T5LayerNorm(nn.Module):
  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.dim = dim
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
    if self.weight.dtype in (torch.float16, torch.bfloat16):
      x = x.type_as(self.weight)
    return self.weight * x


class T5Attention(nn.Module):
  def __init__(self, dim: int, dim_attn: int, num_heads: int, dropout: float = 0.1):
    super().__init__()
    self.dim = dim
    self.dim_attn = dim_attn
    self.num_heads = num_heads
    self.head_dim = dim_attn // num_heads
    self.q = nn.Linear(dim, dim_attn, bias=False)
    self.k = nn.Linear(dim, dim_attn, bias=False)
    self.v = nn.Linear(dim, dim_attn, bias=False)
    self.o = nn.Linear(dim_attn, dim, bias=False)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor, context=None, mask=None, pos_bias=None) -> torch.Tensor:
    context = x if context is None else context
    b, n, c = x.size(0), self.num_heads, self.head_dim
    q = self.q(x).view(b, -1, n, c)
    k = self.k(context).view(b, -1, n, c)
    v = self.v(context).view(b, -1, n, c)
    attn_bias = x.new_zeros(b, n, q.size(1), k.size(1))
    if pos_bias is not None:
      attn_bias += pos_bias
    if mask is not None:
      mask = mask.view(b, 1, 1, -1) if mask.ndim == 2 else mask.unsqueeze(1)
      attn_bias.masked_fill_(mask == 0, torch.finfo(x.dtype).min)
    attn = torch.einsum("binc,bjnc->bnij", q, k) + attn_bias
    attn = F.softmax(attn.float(), dim=-1).type_as(attn)
    x = torch.einsum("bnij,bjnc->binc", attn, v)
    x = x.reshape(b, -1, n * c)
    x = self.o(x)
    return self.dropout(x)


class T5FeedForward(nn.Module):
  def __init__(self, dim: int, dim_ffn: int, dropout: float = 0.1):
    super().__init__()
    self.gate = nn.Sequential(nn.Linear(dim, dim_ffn, bias=False), GELU())
    self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
    self.fc2 = nn.Linear(dim_ffn, dim, bias=False)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.fc1(x) * self.gate(x)
    x = self.dropout(x)
    x = self.fc2(x)
    return self.dropout(x)


class T5RelativeEmbedding(nn.Module):
  def __init__(self, num_buckets: int, num_heads: int, bidirectional: bool, max_dist: int = 128):
    super().__init__()
    self.num_buckets = num_buckets
    self.num_heads = num_heads
    self.bidirectional = bidirectional
    self.max_dist = max_dist
    self.embedding = nn.Embedding(num_buckets, num_heads)

  def forward(self, lq: int, lk: int) -> torch.Tensor:
    device = self.embedding.weight.device
    rel_pos = torch.arange(lk, device=device).unsqueeze(0) - torch.arange(lq, device=device).unsqueeze(1)
    rel_pos = self._relative_position_bucket(rel_pos)
    rel_pos_embeds = self.embedding(rel_pos)
    return rel_pos_embeds.permute(2, 0, 1).unsqueeze(0).contiguous()

  def _relative_position_bucket(self, rel_pos: torch.Tensor) -> torch.Tensor:
    if self.bidirectional:
      num_buckets = self.num_buckets // 2
      rel_buckets = (rel_pos > 0).long() * num_buckets
      rel_pos = torch.abs(rel_pos)
    else:
      num_buckets = self.num_buckets
      rel_buckets = 0
      rel_pos = -torch.min(rel_pos, torch.zeros_like(rel_pos))
    max_exact = num_buckets // 2
    rel_pos_large = (
      max_exact
      + (
        torch.log(rel_pos.float() / max_exact) / math.log(self.max_dist / max_exact) * (num_buckets - max_exact)
      ).long()
    )
    rel_pos_large = torch.min(rel_pos_large, torch.full_like(rel_pos_large, num_buckets - 1))
    rel_buckets += torch.where(rel_pos < max_exact, rel_pos, rel_pos_large)
    return rel_buckets


class T5SelfAttention(nn.Module):
  def __init__(
    self,
    dim: int,
    dim_attn: int,
    dim_ffn: int,
    num_heads: int,
    num_buckets: int,
    shared_pos: bool = True,
    dropout: float = 0.1,
  ):
    super().__init__()
    self.shared_pos = shared_pos
    self.norm1 = T5LayerNorm(dim)
    self.attn = T5Attention(dim, dim_attn, num_heads, dropout)
    self.norm2 = T5LayerNorm(dim)
    self.ffn = T5FeedForward(dim, dim_ffn, dropout)
    self.pos_embedding = None if shared_pos else T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True)

  def forward(self, x: torch.Tensor, mask=None, pos_bias=None) -> torch.Tensor:
    e = pos_bias if self.shared_pos else self.pos_embedding(x.size(1), x.size(1))
    x = fp16_clamp(x + self.attn(self.norm1(x), mask=mask, pos_bias=e))
    return fp16_clamp(x + self.ffn(self.norm2(x)))


class T5Encoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.shared_pos = SHARED_POS
    self.token_embedding = nn.Embedding(VOCAB_SIZE, DIM)
    self.pos_embedding = T5RelativeEmbedding(NUM_BUCKETS, NUM_HEADS, bidirectional=True) if SHARED_POS else None
    self.dropout = nn.Dropout(DROPOUT)
    self.blocks = nn.ModuleList(
      [T5SelfAttention(DIM, DIM_ATTN, DIM_FFN, NUM_HEADS, NUM_BUCKETS, SHARED_POS, DROPOUT) for _ in range(NUM_LAYERS)]
    )
    self.norm = T5LayerNorm(DIM)

  def forward(self, ids: torch.Tensor, mask=None) -> torch.Tensor:
    x = self.token_embedding(ids)
    x = self.dropout(x)
    e = self.pos_embedding(x.size(1), x.size(1)) if self.shared_pos else None
    for block in self.blocks:
      x = block(x, mask, pos_bias=e)
    x = self.norm(x)
    return self.dropout(x)

  def load(self, model_path: str) -> None:
    import os
    from flashpack import assign_from_file

    fp_path = model_path.replace(".safetensors", ".flashpack")
    if os.path.exists(fp_path):
      print(f"Loading T5 encoder from {fp_path} (flashpack). avail mem: {get_available_gpu_memory():.2f} GB")
      t0 = time.perf_counter()
      assign_from_file(self, fp_path, device=str(DEVICE), strict_params=True, strict_buffers=True)
      self.eval().requires_grad_(False)
      print(f"  T5 load: flashpack={time.perf_counter()-t0:.2f}s")
      return

    print(f"Loading T5 encoder from {model_path}. avail mem: {get_available_gpu_memory():.2f} GB")
    t0 = time.perf_counter()
    state_dict = safetensors_load_file(model_path, device=str(DEVICE))
    t_read = time.perf_counter() - t0
    t1 = time.perf_counter()
    self.load_state_dict(state_dict, strict=True, assign=True)
    t_copy = time.perf_counter() - t1
    self.eval().requires_grad_(False)
    print(f"  T5 load: read={t_read:.2f}s  load_state_dict={t_copy:.2f}s")
