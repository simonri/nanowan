"""One-time script: convert safetensors model files to flashpack format.

For T5 and VAE: direct repack (no remapping, no fp8).
For transformers: apply key remapping + absorb KJ fp8 scales → re-quantize to fp8
with new max-abs scale → store weight + weight_scale buffer in flashpack.

Run once; subsequent inference loads from .flashpack files automatically.
"""

import time

import torch
from flashpack import pack_to_file
from safetensors.torch import load_file

from lora import get_param_names_mapping
from model import PARAM_NAMES_MAPPING
from prepare import HIGH_NOISE_PATH, LOW_NOISE_PATH, T5_PATH, VAE_PATH


def convert_plain(src: str) -> None:
  dst = src.replace(".safetensors", ".flashpack")
  print(f"Converting {src} → {dst}")
  t0 = time.perf_counter()
  state_dict = load_file(src)
  t_read = time.perf_counter() - t0
  t1 = time.perf_counter()
  pack_to_file(state_dict, dst, target_dtype=None, silent=False)
  print(f"  read={t_read:.2f}s  pack={time.perf_counter()-t1:.2f}s")


def convert_transformer_fp8(src: str) -> None:
  dst = src.replace(".safetensors", ".flashpack")
  print(f"Converting transformer {src} → {dst}")
  t0 = time.perf_counter()
  state_dict = load_file(src)
  t_read = time.perf_counter() - t0

  # Apply checkpoint key → model key remapping
  mapping_fn = get_param_names_mapping(PARAM_NAMES_MAPPING)
  state_dict = {mapping_fn(k): v for k, v in state_dict.items()}

  # Absorb KJ-format per-tensor fp8 scales into weights, store new scale as buffer
  if "scaled_fp8" in state_dict:
    state_dict.pop("scaled_fp8")
    scale_keys = [k for k in list(state_dict.keys()) if k.endswith(".scale_weight")]
    n_absorbed = 0
    for sk in scale_keys:
      wk = sk.removesuffix(".scale_weight") + ".weight"
      if wk in state_dict:
        scale = state_dict.pop(sk).float()  # scalar
        w_true = state_dict[wk].float() * scale  # true fp32 weight
        new_scale = (w_true.abs().max() / 448.0).clamp(min=1e-12)
        state_dict[wk] = (w_true / new_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        # weight_scale buffer matches FP8Linear.register_buffer("weight_scale", ...) shape [1]
        state_dict[sk.removesuffix(".scale_weight") + ".weight_scale"] = new_scale.unsqueeze(0).to(torch.float32)
        n_absorbed += 1
      else:
        state_dict.pop(sk)
    print(f"  absorbed {n_absorbed} fp8 scales")

  t1 = time.perf_counter()
  pack_to_file(state_dict, dst, target_dtype=None, silent=False)
  print(f"  read={t_read:.2f}s  pack={time.perf_counter()-t1:.2f}s")


if __name__ == "__main__":
  convert_plain(T5_PATH)
  convert_plain(VAE_PATH)
  convert_transformer_fp8(HIGH_NOISE_PATH)
  convert_transformer_fp8(LOW_NOISE_PATH)
  print("Done.")
