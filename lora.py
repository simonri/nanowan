"""LoRA: parameter-name mapping, format normalization, application."""

import re
from collections.abc import Callable, Mapping
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file as safetensors_load_file

LORA_MERGE_CHUNK_BYTES = 32 * 1024 * 1024


# --------------------------------------------------------------------------- #
# Parameter name mapping (from wan/loader/utils.py)
# --------------------------------------------------------------------------- #


def get_param_names_mapping(
  mapping_dict: dict[str, str | tuple[str, int, int]],
) -> Callable[[str], tuple[str, Any, Any]]:
  def mapping_fn(name: str) -> tuple[str, Any, Any]:
    merge_index = None
    total_split_params = None
    max_steps = max(8, len(mapping_dict) * 2)
    applied_patterns: set[str] = set()
    visited_names: set[str] = {name}

    for _ in range(max_steps):
      transformed = False
      for pattern, replacement in mapping_dict.items():
        if pattern in applied_patterns:
          continue
        if re.match(pattern, name) is None:
          continue

        curr_merge_index = None
        curr_total_split_params = None
        if isinstance(replacement, tuple):
          curr_merge_index = replacement[1]
          curr_total_split_params = replacement[2]
          replacement = replacement[0]

        new_name = re.sub(pattern, replacement, name)
        if new_name != name:
          if curr_merge_index is not None:
            merge_index = curr_merge_index
            total_split_params = curr_total_split_params
          name = new_name
          applied_patterns.add(pattern)
          if name in visited_names:
            transformed = False
            break
          visited_names.add(name)
          transformed = True
          break

      if not transformed:
        break

    return name, merge_index, total_split_params

  return mapping_fn


# --------------------------------------------------------------------------- #
# LoRA format detection and normalization (from wan/pipeline/lora_format_adapter.py)
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
# LoRA layers (from wan/layers/lora/linear.py)
# --------------------------------------------------------------------------- #

LoRAWeightEntry = tuple[
  nn.Parameter,
  nn.Parameter,
  str | None,
  float,
  int | None,
  int | None,
]


class BaseLayerWithLoRA(nn.Module):
  def __init__(self, base_layer: nn.Module, lora_rank: int | None = None, lora_alpha: int | None = None):
    super().__init__()
    self.base_layer = base_layer
    self.merged: bool = False
    self.cpu_weight: torch.Tensor | None = None
    self.disable_lora: bool = True
    self.lora_rank = lora_rank
    self.lora_alpha = lora_alpha
    self.lora_weights_list: list[LoRAWeightEntry] = []

  def _ensure_cpu_weight_snapshot(self) -> None:
    if self.cpu_weight is None:
      self.cpu_weight = self.base_layer.weight.detach().to("cpu").clone()

  @property
  def weight(self) -> torch.Tensor:
    return self.base_layer.weight

  @property
  def bias(self) -> torch.Tensor:
    return self.base_layer.bias

  @staticmethod
  def _as_mutable_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_inference():
      with torch.inference_mode(False):
        return tensor.detach().clone()
    return tensor

  @torch.no_grad()
  def _merge_lora_into_data(self, data: torch.Tensor, lora_list: list[LoRAWeightEntry]) -> None:
    for lora_A, lora_B, _, lora_strength, lora_rank, lora_alpha in lora_list:
      lora_A_sliced = lora_A.to(data)
      lora_B_sliced = lora_B.to(data)
      scale = lora_strength
      if lora_alpha is not None and lora_rank is not None and lora_alpha != lora_rank:
        scale *= lora_alpha / lora_rank
      data_2d = data.reshape(-1, data.shape[-1]) if data.dim() > 2 else data
      lora_B_2d = lora_B_sliced.reshape(-1, lora_B_sliced.shape[-1]) if lora_B_sliced.dim() > 2 else lora_B_sliced
      chunk_rows = max(1, LORA_MERGE_CHUNK_BYTES // (data_2d.shape[-1] * max(1, data_2d.element_size())))
      for start in range(0, lora_B_2d.shape[0], chunk_rows):
        end = min(start + chunk_rows, lora_B_2d.shape[0])
        chunk_delta = lora_B_2d[start:end] @ lora_A_sliced
        data_2d[start:end].add_(chunk_delta, alpha=scale)

  @torch.no_grad()
  def merge_lora_weights(self) -> None:
    if self.disable_lora or not self.lora_weights_list:
      return
    if self.merged:
      self.unmerge_lora_weights()
    self._ensure_cpu_weight_snapshot()
    current_device = self.base_layer.weight.data.device
    data = self.base_layer.weight.data.clone()
    data = self._as_mutable_tensor(data)
    self._merge_lora_into_data(data, self.lora_weights_list)
    self.base_layer.weight.data = self._as_mutable_tensor(data.to(current_device, non_blocking=True))
    self.merged = True

  @torch.no_grad()
  def unmerge_lora_weights(self) -> None:
    if self.disable_lora or not self.merged:
      return
    current_device = self.base_layer.weight.data.device
    cpu_weight_on_device = self.cpu_weight.to(current_device, non_blocking=True)
    if self.base_layer.weight.data.is_inference():
      self.base_layer.weight.data = self._as_mutable_tensor(cpu_weight_on_device)
    else:
      self.base_layer.weight.data.copy_(cpu_weight_on_device)
    self.merged = False

  def set_lora_weights(
    self, A: torch.Tensor, B: torch.Tensor, lora_path: str | None = None, strength: float = 1.0
  ) -> None:
    self.lora_weights_list.append(
      (
        nn.Parameter(A),
        nn.Parameter(B),
        lora_path,
        strength,
        self.lora_rank,
        self.lora_alpha,
      )
    )
    self.disable_lora = False
    self.merge_lora_weights()


class LinearWithLoRA(BaseLayerWithLoRA):
  def __init__(self, base_layer: nn.Linear, lora_rank: int | None = None, lora_alpha: int | None = None):
    super().__init__(base_layer, lora_rank, lora_alpha)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.base_layer(x)


def wrap_with_lora_layer(
  layer: nn.Module, lora_rank: int | None = None, lora_alpha: int | None = None
) -> nn.Module | None:
  if isinstance(layer, nn.Linear):
    return LinearWithLoRA(layer, lora_rank=lora_rank, lora_alpha=lora_alpha)
  return None


def replace_submodule(model: nn.Module, module_name: str, new_module: nn.Module) -> nn.Module:
  parent = model.get_submodule(".".join(module_name.split(".")[:-1]))
  setattr(parent, module_name.split(".")[-1], new_module)
  return new_module


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

    # Normalize alpha/rank metadata
    alphas: dict[str, float] = {}
    ranks: dict[str, int] = {}
    for key, tensor in state_dict.items():
      if key.endswith(".alpha"):
        base = key[: -len(".alpha")]
        alphas[base] = tensor.item()
      elif ".lora_A." in key:
        base = key[: key.index(".lora_A.")]
        ranks[base] = tensor.shape[0]

    # Group A/B pairs
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
      # Map the FULL key (including the lora_A suffix) so patterns like
      # r"^blocks[._](\d+)[._]self_attn[._]q\.(.*)$" can match the trailing part.
      mapped_full, _, _ = mapping_fn(base_key + ".lora_A.weight")
      if ".lora_A." in mapped_full:
        mapped_module = mapped_full[: mapped_full.index(".lora_A.")]
      else:
        mapped_module = mapped_full.removesuffix(".weight")

      # Find the target submodule
      try:
        target = model.get_submodule(mapped_module)
      except AttributeError:
        print(f"  Warning: no submodule {mapped_module!r} for LoRA key {base_key!r}")
        continue

      # Ensure it's wrapped with LoRA
      if not isinstance(target, BaseLayerWithLoRA):
        wrapped = wrap_with_lora_layer(target)
        if wrapped is None:
          continue
        replace_submodule(model, mapped_module, wrapped)
        target = wrapped

      lora_rank = ranks.get(base_key)
      lora_alpha = alphas.get(base_key)
      target.lora_rank = lora_rank
      target.lora_alpha = int(lora_alpha) if lora_alpha is not None else lora_rank

      A = ab["A"]
      B = ab["B"]
      target.set_lora_weights(A, B, lora_path=lora_path, strength=strength)
      n_merged += 1

    print(f"  Merged {n_merged} LoRA layers")
    assert n_merged > 0, f"No LoRA layers merged from {lora_path} — check key mapping"
