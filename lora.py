"""LoRA: parameter-name mapping, format normalization, application."""

import re
from collections.abc import Callable, Mapping
from enum import Enum

import torch
import torch.nn as nn
from safetensors.torch import load_file as safetensors_load_file

# --------------------------------------------------------------------------- #
# Parameter name mapping
# --------------------------------------------------------------------------- #


def get_param_names_mapping(mapping_dict: dict[str, str]) -> Callable[[str], str]:
  def mapping_fn(name: str) -> str:
    for pattern, replacement in mapping_dict.items():
      if re.match(pattern, name) is not None:
        return re.sub(pattern, replacement, name)
    return name

  return mapping_fn


# --------------------------------------------------------------------------- #
# LoRA format detection and normalization
# --------------------------------------------------------------------------- #


class LoRAFormat(Enum):
  STANDARD = "standard"
  WAN = "wan"
  KOHYA = "kohya"


KOHYA_PREFIXES = ("lora_unet_", "lora_te_", "lora_te1_", "lora_te2_")


def detect_lora_format(state_dict: Mapping[str, torch.Tensor]) -> LoRAFormat:
  keys = list(state_dict.keys())
  if not keys:
    return LoRAFormat.STANDARD
  if any(".lora_A." in k or ".lora_B." in k for k in keys):
    return LoRAFormat.STANDARD
  if sum(k.startswith(KOHYA_PREFIXES) for k in keys) > len(keys) // 2:
    return LoRAFormat.KOHYA
  if any(k.startswith("diffusion_model.") for k in keys):
    return LoRAFormat.WAN
  return LoRAFormat.STANDARD


def _swap_down_up_to_A_B(name: str) -> str:
  return name.replace("lora_down.weight", "lora_A.weight").replace("lora_up.weight", "lora_B.weight")


def normalize_lora_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
  fmt = detect_lora_format(state_dict)
  print(f"Detected LoRA format: {fmt}")
  if fmt == LoRAFormat.KOHYA:
    out: dict[str, torch.Tensor] = {}
    for name, weight in state_dict.items():
      if name.startswith(("lora_te_", "lora_te1_", "lora_te2_")):
        continue
      if name.startswith("lora_unet_"):
        name = name[len("lora_unet_") :]
      out[_swap_down_up_to_A_B(name)] = weight
    return out
  if fmt == LoRAFormat.WAN:
    out = {}
    for name, weight in state_dict.items():
      name = name.removeprefix("diffusion_model.")
      out[_swap_down_up_to_A_B(name)] = weight
    return out
  return dict(state_dict)


# --------------------------------------------------------------------------- #
# High-level apply function
# --------------------------------------------------------------------------- #


def apply_loras(
  model: nn.Module,
  lora_paths: list[str],
  strengths: list[float],
  lora_param_names_mapping: dict[str, str],
) -> None:
  """Bake all LoRAs into model weights in-place."""
  mapping_fn = get_param_names_mapping(lora_param_names_mapping)

  for lora_path, strength in zip(lora_paths, strengths, strict=True):
    print(f"Applying LoRA {lora_path} strength={strength}")
    raw = safetensors_load_file(lora_path)
    state_dict = normalize_lora_state_dict(raw)

    # Collect alpha/rank metadata
    alphas: dict[str, float] = {}
    ranks: dict[str, int] = {}
    for key, tensor in state_dict.items():
      if key.endswith(".alpha"):
        alphas[key[: -len(".alpha")]] = tensor.item()
      elif ".lora_A." in key:
        ranks[key[: key.index(".lora_A.")]] = tensor.shape[0]

    # Group A/B pairs by base key
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in state_dict.items():
      if key.endswith(".alpha"):
        continue
      if ".lora_A." in key:
        base = key[: key.index(".lora_A.")]
        pairs.setdefault(base, {})["A"] = tensor
      elif ".lora_B." in key:
        base = key[: key.index(".lora_B.")]
        pairs.setdefault(base, {})["B"] = tensor

    n_merged = 0
    for base_key, ab in pairs.items():
      if "A" not in ab or "B" not in ab:
        continue
      mapped_full = mapping_fn(base_key + ".lora_A.weight")
      if ".lora_A." in mapped_full:
        mapped_module = mapped_full[: mapped_full.index(".lora_A.")]
      else:
        mapped_module = mapped_full.removesuffix(".weight")
      try:
        target = model.get_submodule(mapped_module)
      except AttributeError:
        print(f"  Warning: no submodule {mapped_module!r} for LoRA key {base_key!r}")
        continue
      if not isinstance(target, nn.Linear):
        continue

      rank = ranks.get(base_key)
      alpha = alphas.get(base_key)
      scale = strength * (alpha / rank if alpha is not None and rank is not None and alpha != rank else 1.0)

      lora_A = ab["A"].to(target.weight)
      lora_B = ab["B"].to(target.weight)
      target.weight.data.add_(lora_B @ lora_A, alpha=scale)
      n_merged += 1

    print(f"  Merged {n_merged} LoRA layers")
    assert n_merged > 0, f"No LoRA layers merged from {lora_path} — check key mapping"
