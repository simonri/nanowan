"""Fixed constants, evaluation utilities, and output helpers. Do not modify."""

import os

import imageio
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Input / output
# ---------------------------------------------------------------------------

IMAGE_PATH = "./i2v_input.JPG"
PROMPT = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."  # noqa: E501
OUTPUT_PATH = "output.mp4"

# ---------------------------------------------------------------------------
# Output geometry (must not be reduced)
# ---------------------------------------------------------------------------

HEIGHT = 352
WIDTH = 640
NUM_FRAMES = 81  # pixel frames; latent frames = 1 + (81-1)//4 = 21
FPS = 16

# ---------------------------------------------------------------------------
# Text encoding
# ---------------------------------------------------------------------------

TEXT_LEN = 512
TOKENIZER_ID = "google/umt5-xxl"

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------

MODEL_DIR = "models"
T5_PATH = f"{MODEL_DIR}/text_encoders/umt5-xxl-enc-bf16.safetensors"
VAE_PATH = f"{MODEL_DIR}/vae/Wan2_1_VAE_bf16.safetensors"
HIGH_NOISE_PATH = f"{MODEL_DIR}/diffusion_models/Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors"
LOW_NOISE_PATH = f"{MODEL_DIR}/diffusion_models/Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors"
HIGH_NOISE_LORAS = [f"{MODEL_DIR}/loras/lightning_high_noise_model.safetensors"]
HIGH_NOISE_STRENGTHS = [1.0]
LOW_NOISE_LORAS = [f"{MODEL_DIR}/loras/lightning_low_noise_model.safetensors"]
LOW_NOISE_STRENGTHS = [1.0]

# ---------------------------------------------------------------------------
# Reference latents for correctness checking
# ---------------------------------------------------------------------------

LATENTS_REF_PATH = "latents_ref.pt"

NUM_STEPS = 8  # total denoising steps; first half = high noise, second = low noise
FLOW_SHIFT = 5.0
DIT_DTYPE = torch.float16
VAE_DTYPE = torch.float32


def save_latents_ref(latents: torch.Tensor) -> None:
  torch.save(latents.cpu(), LATENTS_REF_PATH)
  print(f"Saved reference latents → {LATENTS_REF_PATH}")


def compare_latents_ref(latents: torch.Tensor) -> float | None:
  if not os.path.exists(LATENTS_REF_PATH):
    return None
  ref = torch.load(LATENTS_REF_PATH, map_location=latents.device, weights_only=True)
  return (latents.float() - ref.float()).pow(2).mean().sqrt().item()


# ---------------------------------------------------------------------------
# Video saving
# ---------------------------------------------------------------------------


def save_mp4(video: torch.Tensor, path: str, fps: int = FPS) -> None:
  """video: [1, 3, F, H, W] float32 in [0, 1]."""
  frames = video[0].permute(1, 2, 3, 0).cpu().float().numpy()  # [F, H, W, 3]
  frames = (frames * 255).clip(0, 255).astype(np.uint8)
  writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
  for frame in frames:
    writer.append_data(frame)
  writer.close()
  print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Summary output (ground-truth metric — do not modify)
# ---------------------------------------------------------------------------


def print_summary(
  latents: torch.Tensor,
  denoising_seconds: float,
  total_seconds: float,
  load_seconds: float,
  decode_seconds: float,
) -> None:
  peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
  latent_rmse = compare_latents_ref(latents)
  print(f"Total: {total_seconds:.2f}s")
  print("---")
  print(f"denoising_seconds: {denoising_seconds:.2f}")
  print(f"total_seconds:     {total_seconds:.2f}")
  print(f"load_seconds:      {load_seconds:.2f}")
  print(f"decode_seconds:    {decode_seconds:.2f}")
  print(f"peak_vram_mb:      {peak_vram_mb:.1f}")
  print(f"latent_rmse:       {latent_rmse:.6f}" if latent_rmse is not None else "latent_rmse:       N/A")
