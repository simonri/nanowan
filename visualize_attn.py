"""Attention visualization for WAN 2.2 I2V generation.

Patches the T5 encoder and WAN cross-attention to capture weights during inference,
then produces a three-panel figure:

  ① T5 self-attention heatmap (token × token, avg over 24 layers & 64 heads)
  ② Cross-attention spatial maps per top token (which video regions attend to which word)
  ③ Cross-attention evolution across denoising steps

Usage:
    uv run python visualize_attn.py
Output:
    attention_viz.png
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import PIL.Image
from transformers import AutoTokenizer

from model import WanCrossAttention, WanModel
from prepare import (
  DIT_DTYPE,
  FLOW_SHIFT,
  HEIGHT,
  HIGH_NOISE_LORAS,
  HIGH_NOISE_PATH,
  HIGH_NOISE_STRENGTHS,
  IMAGE_PATH,
  LOW_NOISE_LORAS,
  LOW_NOISE_PATH,
  LOW_NOISE_STRENGTHS,
  NUM_FRAMES,
  NUM_STEPS,
  PROMPT,
  T5_PATH,
  TEXT_LEN,
  TOKENIZER_ID,
  VAE_PATH,
  WIDTH,
)
from run import encode_image, load_transformer
from scheduler import FlowMatchEulerDiscreteScheduler
from t5 import T5Attention, T5Encoder
from utils import DEVICE, set_default_torch_dtype, skip_init_modules
from vae import Wan2_1_VAE

# ---------------------------------------------------------------------------
# Video/patch geometry (derived from model constants)
# ---------------------------------------------------------------------------
LAT_H = HEIGHT // 8       # 80
LAT_W = WIDTH // 8        # 44
LAT_F = 1 + (NUM_FRAMES - 1) // 4  # 21
POST_F = LAT_F            # patch_t = 1, so no temporal reduction
POST_H = LAT_H // 2       # patch_h = 2 → 40
POST_W = LAT_W // 2       # patch_w = 2 → 22
NUM_PATCHES = POST_F * POST_H * POST_W  # 18480

# ---------------------------------------------------------------------------
# Shared state populated by the hooks
# ---------------------------------------------------------------------------
_t5_layers: list[np.ndarray] = []         # per-layer [sl, sl] float32
_cross_step_layers: list[np.ndarray] = [] # within one step: per-layer [P, sl]
_cross_steps: list[np.ndarray] = []       # per-step avg [P, sl]
_seq_len: int = 0                         # real (non-padding) token count
_token_strings: list[str] = []            # decoded token strings
_is_special: list[bool] = []              # True for special/sentinel tokens

# ---------------------------------------------------------------------------
# ① T5 self-attention patch
#    Re-implements T5Attention.forward identically, but saves the softmax'd
#    attention matrix (averaged over heads) to _t5_layers.
# ---------------------------------------------------------------------------
_orig_t5_forward = T5Attention.forward


def _t5_attn_hook(self, x, context=None, mask=None, pos_bias=None):
  context = x if context is None else context
  b, n, c = x.size(0), self.num_heads, self.head_dim
  q = self.q(x).view(b, -1, n, c)
  k = self.k(context).view(b, -1, n, c)
  v = self.v(context).view(b, -1, n, c)
  attn_bias = x.new_zeros(b, n, q.size(1), k.size(1))
  if pos_bias is not None:
    attn_bias += pos_bias
  if mask is not None:
    mv = mask.view(b, 1, 1, -1) if mask.ndim == 2 else mask.unsqueeze(1)
    attn_bias.masked_fill_(mv == 0, torch.finfo(x.dtype).min)
  attn = torch.einsum("binc,bjnc->bnij", q, k) + attn_bias
  attn = F.softmax(attn.float(), dim=-1).type_as(attn)
  sl = _seq_len
  # avg over batch & heads, keep only real tokens
  _t5_layers.append(attn[0].float().mean(0)[:sl, :sl].cpu().numpy())
  x = torch.einsum("bnij,bjnc->binc", attn, v)
  x = x.reshape(b, -1, n * c)
  x = self.o(x)
  return self.dropout(x)


# ---------------------------------------------------------------------------
# ② WAN cross-attention patch
#    Computes Q×K in chunked float32 for visualization (avoids materialising
#    the full 18480×512 matrix at once), then delegates the actual output to
#    the original FlashAttention path — no change to model numerics.
# ---------------------------------------------------------------------------
_orig_cross_forward = WanCrossAttention.forward

_VIS_CHUNK = 512  # patches per chunk; keep GPU peak memory low


def _cross_attn_hook(self, x, context):
  q = self.norm_q(self.to_q(x)).unflatten(2, (self.num_heads, self.head_dim))  # [b, P, H, d]
  k = self.norm_k(self.to_k(context)).unflatten(2, (self.num_heads, self.head_dim))  # [b, T, H, d]
  v = self.to_v(context).unflatten(2, (self.num_heads, self.head_dim))

  # Visualisation: Q×K in float32, chunked over patches, averaged over heads
  sl = _seq_len
  scale = self.head_dim**-0.5
  q_f = q.permute(0, 2, 1, 3).float()  # [b, H, P, d]  — new tensor, q unchanged
  k_f = k.permute(0, 2, 1, 3).float()  # [b, H, T, d]
  chunks = []
  for i in range(0, q_f.shape[2], _VIS_CHUNK):
    s = torch.einsum("bhpd,bhtd->bhpt", q_f[:, :, i : i + _VIS_CHUNK], k_f) * scale
    w = s.softmax(dim=-1)              # [b, H, chunk, T]
    chunks.append(w[0].mean(0)[:, :sl].cpu().float())  # [chunk, sl]
    del s, w
  _cross_step_layers.append(torch.cat(chunks, dim=0).numpy())  # [P, sl]
  del q_f, k_f, chunks

  # Forward pass: use the original FlashAttention (unchanged numerics)
  return self.to_out(self.attn(q, k, v).flatten(2))


def _finalize_step():
  """Average per-layer cross-attention maps for the completed step."""
  if _cross_step_layers:
    _cross_steps.append(np.stack(_cross_step_layers, 0).mean(0))  # [P, sl]
    _cross_step_layers.clear()


# ---------------------------------------------------------------------------
# Text encoding with T5 attention capture
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_text_with_attn(prompt: str) -> torch.Tensor:
  global _seq_len, _token_strings, _is_special
  T5Attention.forward = _t5_attn_hook
  _t5_layers.clear()

  print("Loading T5 encoder...")
  t5 = T5Encoder()
  t5.load(T5_PATH)
  t5.to(DEVICE, dtype=torch.bfloat16)

  print("Loading tokenizer...")
  tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
  enc = tokenizer(
    [prompt],
    padding="max_length",
    truncation=True,
    max_length=TEXT_LEN,
    return_attention_mask=True,
    return_tensors="pt",
  ).to(DEVICE)
  input_ids = enc["input_ids"]
  attention_mask = enc["attention_mask"]

  _seq_len = int(attention_mask.gt(0).sum(dim=1)[0])
  raw = tokenizer.convert_ids_to_tokens(input_ids[0, :_seq_len].tolist())
  _token_strings = [t.replace("▁", " ").strip() or "▁" for t in raw]
  special_ids = set(tokenizer.all_special_ids)
  _is_special = [input_ids[0, i].item() in special_ids for i in range(_seq_len)]

  print(f"  Prompt tokenised to {_seq_len} real tokens (+ {TEXT_LEN - _seq_len} padding)")

  with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    hidden = t5(input_ids, attention_mask)

  emb = hidden[0, :_seq_len]
  pad = emb.new_zeros(TEXT_LEN - _seq_len, emb.size(1))
  prompt_embeds = torch.cat([emb, pad], dim=0).unsqueeze(0)

  T5Attention.forward = _orig_t5_forward
  del t5
  torch.cuda.empty_cache()
  return prompt_embeds


# ---------------------------------------------------------------------------
# Denoising loop with cross-attention capture
# ---------------------------------------------------------------------------
@torch.no_grad()
def denoise_with_attn(
  prompt_embeds: torch.Tensor,
  image_latent: torch.Tensor,
  high_model: WanModel,
  low_model: WanModel,
  seed: int = 42,
) -> torch.Tensor:
  WanCrossAttention.forward = _cross_attn_hook
  _cross_step_layers.clear()
  _cross_steps.clear()

  scheduler = FlowMatchEulerDiscreteScheduler(shift=FLOW_SHIFT)
  scheduler.set_timesteps(NUM_STEPS, device=DEVICE)

  g = torch.Generator(device=DEVICE).manual_seed(seed)
  latents = torch.randn(1, 16, LAT_F, LAT_H, LAT_W, generator=g, device=DEVICE, dtype=DIT_DTYPE)
  enc = prompt_embeds.to(DEVICE, dtype=DIT_DTYPE)
  img = image_latent.to(DEVICE, dtype=DIT_DTYPE)

  with torch.amp.autocast("cuda", dtype=DIT_DTYPE):
    for i, t in enumerate(scheduler.timesteps):
      model = high_model if i < NUM_STEPS // 2 else low_model
      noise_pred = model(
        hidden_states=torch.cat([latents, img], dim=1),
        timestep=t.repeat(1),
        encoder_hidden_states=enc,
      )
      _finalize_step()
      latents = scheduler.step(noise_pred, t, latents)
      print(f"  Step {i + 1}/{NUM_STEPS} done")

  WanCrossAttention.forward = _orig_cross_forward
  return latents


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_all(out_path: str = "attention_viz.png"):
  toks = _token_strings
  sl = _seq_len

  # ① T5: average over layers → [sl, sl]
  t5_avg = np.stack(_t5_layers, 0).mean(0)

  # ② Cross-attention: average over steps → [P, sl], then reshape to spatial
  cross_avg = np.stack(_cross_steps, 0).mean(0)        # [P, sl]
  cross_spatial = cross_avg.reshape(POST_F, POST_H, POST_W, sl)  # [F, H, W, sl]

  # Select top tokens by average attention, excluding special/sentinel tokens
  TOP_K = 8
  importance = cross_avg.mean(0)
  content_indices = [i for i in range(sl) if not _is_special[i]]
  top_ids = sorted(content_indices, key=lambda i: importance[i], reverse=True)[:TOP_K]

  # ③ Evolution: mean attention per token per step → [steps, sl]
  evolution = np.stack([s.mean(0) for s in _cross_steps], 0)

  # ---- Build figure --------------------------------------------------------
  fig = plt.figure(figsize=(24, 22))
  gs = gridspec.GridSpec(3, 1, height_ratios=[1.0, 1.35, 0.75], hspace=0.55)

  # --- Panel ①: T5 self-attention ---
  ax1 = fig.add_subplot(gs[0])
  vmax = float(np.percentile(t5_avg, 99))
  im1 = ax1.imshow(t5_avg, aspect="auto", cmap="Blues", vmin=0.0, vmax=vmax, interpolation="nearest")
  ax1.set_title(
    "① T5 Self-Attention (avg over 24 layers × 64 heads)\n"
    "Row = query token, Column = key token (being attended to)",
    fontsize=11,
  )
  stride = max(1, sl // 50)
  ticks = list(range(0, sl, stride))
  ax1.set_xticks(ticks)
  ax1.set_xticklabels([toks[i] for i in ticks], rotation=90, fontsize=6)
  ax1.set_yticks(ticks)
  ax1.set_yticklabels([toks[i] for i in ticks], fontsize=6)
  fig.colorbar(im1, ax=ax1, fraction=0.015, pad=0.01, label="Attention weight")

  # --- Panel ②: Spatial cross-attention maps ---
  show_frames = [0, POST_F // 2]
  frame_labels = ["frame 0", f"frame {POST_F // 2}"]
  gs2 = gridspec.GridSpecFromSubplotSpec(2, TOP_K, subplot_spec=gs[1], hspace=0.08, wspace=0.08)
  first_ax = None
  for col, tid in enumerate(top_ids):
    for row, fi in enumerate(show_frames):
      ax = fig.add_subplot(gs2[row, col])
      if first_ax is None:
        first_ax = ax
      m = cross_spatial[fi, :, :, tid]
      m_norm = (m - m.min()) / (m.max() - m.min() + 1e-9)
      ax.imshow(m_norm, cmap="hot", aspect="auto", interpolation="bilinear")
      if row == 0:
        label = toks[tid] if tid < len(toks) else f"#{tid}"
        ax.set_title(f'"{label}"\nrank {col + 1}', fontsize=7, pad=2)
      if col == 0:
        ax.set_ylabel(frame_labels[row], fontsize=7)
      ax.set_xticks([])
      ax.set_yticks([])

  # Title for panel ②: position relative to the first subplot after layout
  if first_ax is not None:
    first_ax.annotate(
      f"② Cross-Attention Spatial Maps — top {TOP_K} tokens by avg patch attention "
      f"(avg over all steps & layers, brightness = strength)",
      xy=(0, 1), xycoords="axes fraction",
      xytext=(0, 28), textcoords="offset points",
      fontsize=9, ha="left", va="bottom",
    )

  # --- Panel ③: Evolution ---
  ax3 = fig.add_subplot(gs[2])
  im3 = ax3.imshow(evolution.T, aspect="auto", cmap="plasma", interpolation="nearest")
  ax3.set_title(
    "③ Cross-Attention Evolution per Denoising Step\n"
    "(y = text token, brightness = avg attention over all patches, layers & heads)",
    fontsize=11,
  )
  ax3.set_xlabel("Denoising step  (0 = high noise → last = low noise)")
  ax3.set_ylabel("Token index")
  ax3.set_xticks(range(len(_cross_steps)))
  ax3.set_xticklabels([f"step {i}" for i in range(len(_cross_steps))], fontsize=8)
  stride2 = max(1, sl // 40)
  yticks = list(range(0, sl, stride2))
  ax3.set_yticks(yticks)
  ax3.set_yticklabels([toks[i] if i < len(toks) else "" for i in yticks], fontsize=7)
  # Mark the high→low model swap boundary (first half = high_model, second = low_model)
  swap_x = NUM_STEPS // 2 - 0.5
  ax3.axvline(swap_x, color="white", linewidth=1.5, linestyle="--", alpha=0.85)
  ax3.text(swap_x + 0.1, sl * 0.01, "model swap\n(high→low)", color="white", fontsize=7, va="top")
  fig.colorbar(im3, ax=ax3, fraction=0.015, pad=0.01, label="Avg attn weight")

  plt.savefig(out_path, dpi=150, bbox_inches="tight")
  print(f"Saved → {out_path}")
  plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
  # 1. Encode text + capture T5 attention
  prompt_embeds = encode_text_with_attn(PROMPT)

  # 2. Load VAE, encode image, then unload VAE to free memory
  print("Loading VAE...")
  with skip_init_modules(), set_default_torch_dtype(torch.float32):
    vae = Wan2_1_VAE()
  vae.load(VAE_PATH)
  vae.to(DEVICE)
  image = PIL.Image.open(IMAGE_PATH).convert("RGB").resize((WIDTH, HEIGHT), PIL.Image.LANCZOS)
  image_latent = encode_image(image, vae)
  del vae
  torch.cuda.empty_cache()

  # 3. Load transformers with LoRAs
  print("Loading high-noise transformer...")
  high_model = load_transformer(HIGH_NOISE_PATH, HIGH_NOISE_LORAS, HIGH_NOISE_STRENGTHS)
  print("Loading low-noise transformer...")
  low_model = load_transformer(LOW_NOISE_PATH, LOW_NOISE_LORAS, LOW_NOISE_STRENGTHS)

  # 4. Denoise with cross-attention capture
  print("Denoising with attention capture (cross-attn uses chunked matmul instead of FlashAttn)...")
  denoise_with_attn(prompt_embeds, image_latent, high_model, low_model)

  # 5. Plot and save
  print("Plotting...")
  plot_all("attention_viz.png")


if __name__ == "__main__":
  main()
