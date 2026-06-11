"""Minimal WAN 2.2 I2V inference script.

Two-stage denoising: 4 steps with the high-noise transformer, 4 steps with
the low-noise transformer.  Both transformers receive lightning LoRAs before
running.
"""

import time

import imageio
import numpy as np
import PIL.Image
import torch
from transformers import AutoTokenizer

from lora import apply_loras
from model import WanModel
from scheduler import FlowMatchEulerDiscreteScheduler
from t5 import T5Encoder
from utils import DEVICE, normalize, numpy_to_pt, pil_to_numpy, set_default_torch_dtype, skip_init_modules
from vae import LATENTS_MEAN, LATENTS_STD, Wan2_1_VAE

# ---------------------------------------------------------------------------
# User-configurable constants
# ---------------------------------------------------------------------------

IMAGE_PATH = "./i2v_input.JPG"
PROMPT = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
OUTPUT_PATH = "output.mp4"

# Inference geometry
HEIGHT = 480
WIDTH = 832
NUM_FRAMES = 81  # pixel frames; latent frames = 1 + (81-1)//4 = 21
NUM_STEPS = 8  # total steps; first half = high noise, second = low noise
FLOW_SHIFT = 5.0
FPS = 16

# Dtype: fp16 checkpoints → use torch.float16
DIT_DTYPE = torch.float16
VAE_DTYPE = torch.float32

# Model paths
MODEL_DIR = "models"
TOKENIZER_ID = "google/umt5-xxl"
T5_PATH = f"{MODEL_DIR}/text_encoders/umt5-xxl-enc-bf16.safetensors"
VAE_PATH = f"{MODEL_DIR}/vae/Wan2_1_VAE_bf16.safetensors"
HIGH_NOISE_PATH = f"{MODEL_DIR}/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"
LOW_NOISE_PATH = f"{MODEL_DIR}/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"

# LoRAs applied to both transformers (lightning 4-step distillation)
HIGH_NOISE_LORAS = [f"{MODEL_DIR}/loras/lightning_high_noise_model.safetensors"]
HIGH_NOISE_STRENGTHS = [1.0]
LOW_NOISE_LORAS = [f"{MODEL_DIR}/loras/lightning_low_noise_model.safetensors"]
LOW_NOISE_STRENGTHS = [1.0]

# T5 text length
TEXT_LEN = 512

# ---------------------------------------------------------------------------
# Text encoding
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_text(prompt: str) -> torch.Tensor:
  print("Loading T5 encoder...")
  t5 = T5Encoder()
  t5.load(T5_PATH)
  t5.to(DEVICE, dtype=torch.bfloat16)

  print("Loading tokenizer...")
  tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

  tokens = tokenizer(
    [prompt],
    padding="max_length",
    truncation=True,
    max_length=TEXT_LEN,
    return_attention_mask=True,
    return_tensors="pt",
  ).to(DEVICE)

  input_ids = tokens["input_ids"]
  attention_mask = tokens["attention_mask"]

  with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    hidden_states = t5(input_ids, attention_mask)  # [1, TEXT_LEN, 4096]

  # Trim to actual sequence length then pad back to TEXT_LEN
  seq_len = int(attention_mask.gt(0).sum(dim=1)[0])
  embedding = hidden_states[0, :seq_len]  # [seq_len, 4096]
  pad = embedding.new_zeros(TEXT_LEN - seq_len, embedding.size(1))
  embedding = torch.cat([embedding, pad], dim=0)  # [TEXT_LEN, 4096]
  prompt_embeds = embedding.unsqueeze(0)  # [1, TEXT_LEN, 4096]

  del t5
  torch.cuda.empty_cache()
  return prompt_embeds


# ---------------------------------------------------------------------------
# Image encoding (VAE encode)
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_image(image: PIL.Image.Image, vae: Wan2_1_VAE) -> torch.Tensor:
  """Returns the normalized image latent [1, 20, LAT_F, LAT_H, LAT_W]."""
  # Preprocess: convert to [-1, 1] tensor
  img = pil_to_numpy(image)  # [1, H, W, 3]
  img = numpy_to_pt(img)  # [1, 3, H, W]
  img = normalize(img)  # [1, 3, H, W] in [-1, 1]
  img = img.unsqueeze(2)  # [1, 3, 1, H, W]

  # Pad with zeros for remaining frames so VAE can infer latent temporal dim
  img = torch.cat(
    [
      img,
      img.new_zeros(1, 3, NUM_FRAMES - 1, HEIGHT, WIDTH),
    ],
    dim=2,
  ).to(DEVICE, dtype=VAE_DTYPE)  # [1, 3, 81, H, W]

  # VAE encode
  latent_dist = vae.encode(img)
  latent_raw = latent_dist.mode()  # [1, 16, 21, H//8, W//8]

  # Normalize: (raw - mean) / std
  mean = torch.tensor(LATENTS_MEAN, device=DEVICE, dtype=latent_raw.dtype).view(1, 16, 1, 1, 1)
  std = torch.tensor(LATENTS_STD, device=DEVICE, dtype=latent_raw.dtype).view(1, 16, 1, 1, 1)
  latent_norm = (latent_raw - mean) / std  # [1, 16, 21, H//8, W//8]

  # Build mask [1, 4, 21, H//8, W//8]: 1 for frames belonging to the first pixel frame
  lat_h = HEIGHT // 8
  lat_w = WIDTH // 8

  mask = torch.ones(1, 1, NUM_FRAMES, lat_h, lat_w)
  mask[:, :, 1:] = 0
  first_mask = torch.repeat_interleave(mask[:, :, :1], repeats=4, dim=2)  # [1,1,4,H,W]
  mask = torch.cat([first_mask, mask[:, :, 1:]], dim=2)  # [1,1,84,H,W]
  mask = mask.view(1, -1, 4, lat_h, lat_w).transpose(1, 2)  # [1,4,21,H,W]
  mask = mask.to(DEVICE, dtype=latent_norm.dtype)

  image_latent = torch.cat([mask, latent_norm], dim=1)  # [1, 20, 21, H//8, W//8]
  return image_latent


# ---------------------------------------------------------------------------
# VAE decode
# ---------------------------------------------------------------------------


@torch.no_grad()
def decode_latents(latents: torch.Tensor, vae: Wan2_1_VAE) -> torch.Tensor:
  """latents: [1, 16, 21, H//8, W//8] → video [1, 3, 81, H, W] in [0, 1]."""
  mean = torch.tensor(LATENTS_MEAN, device=DEVICE, dtype=torch.float32).view(1, 16, 1, 1, 1)
  std = torch.tensor(LATENTS_STD, device=DEVICE, dtype=torch.float32).view(1, 16, 1, 1, 1)
  latents_raw = latents.float() * std + mean
  video = vae.decode(latents_raw)  # [1, 3, F, H, W] in [-1, 1]
  video = (video / 2 + 0.5).clamp(0, 1)  # [0, 1]
  return video


# ---------------------------------------------------------------------------
# Load transformer with LoRAs baked in
# ---------------------------------------------------------------------------


def load_transformer(ckpt_path: str, lora_paths: list[str], lora_strengths: list[float]) -> WanModel:
  with skip_init_modules(), set_default_torch_dtype(DIT_DTYPE):
    model = WanModel()
  model.load(ckpt_path)
  model.to(DEVICE)
  if lora_paths:
    apply_loras(model, lora_paths, lora_strengths, WanModel.LORA_PARAM_NAMES_MAPPING)
  return model


# ---------------------------------------------------------------------------
# Denoising loop
# ---------------------------------------------------------------------------


@torch.no_grad()
def denoise(
  prompt_embeds: torch.Tensor,  # [1, TEXT_LEN, 4096]
  image_latent: torch.Tensor,  # [1, 20, 21, H//8, W//8]
  high_model: WanModel,
  low_model: WanModel,
  seed: int = 42,
) -> torch.Tensor:
  scheduler = FlowMatchEulerDiscreteScheduler(shift=FLOW_SHIFT)
  scheduler.set_timesteps(NUM_STEPS, device=DEVICE)

  lat_h = HEIGHT // 8
  lat_w = WIDTH // 8
  lat_f = 1 + (NUM_FRAMES - 1) // 4  # = 21

  g = torch.Generator(device=DEVICE).manual_seed(seed)
  latents = torch.randn(1, 16, lat_f, lat_h, lat_w, generator=g, device=DEVICE, dtype=DIT_DTYPE)

  encoder_hidden_states = prompt_embeds.to(DEVICE, dtype=DIT_DTYPE)
  image_latent = image_latent.to(DEVICE, dtype=DIT_DTYPE)

  with torch.amp.autocast("cuda", dtype=DIT_DTYPE):
    for step_index, t in enumerate(scheduler.timesteps):
      model = high_model if step_index < NUM_STEPS // 2 else low_model
      model_input = torch.cat([latents, image_latent], dim=1)  # [1, 36, 21, H//8, W//8]
      timestep = t.repeat(1)  # [1]
      noise_pred = model(
        hidden_states=model_input,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
      )
      latents = scheduler.step(noise_pred, t, latents)

  return latents


# ---------------------------------------------------------------------------
# Save MP4
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
# Main
# ---------------------------------------------------------------------------


def main():
  t_total = time.perf_counter()

  # 1. Encode text (unloads T5 after)
  prompt_embeds = encode_text(PROMPT)

  # 2. Load VAE and encode image
  print("Loading VAE...")
  with skip_init_modules(), set_default_torch_dtype(torch.float32):
    vae = Wan2_1_VAE()
  vae.load(VAE_PATH)
  vae.to(DEVICE)

  print(f"Loading input image from {IMAGE_PATH} ...")
  image = PIL.Image.open(IMAGE_PATH).convert("RGB").resize((WIDTH, HEIGHT), PIL.Image.LANCZOS)
  image_latent = encode_image(image, vae)

  # 3. Load transformers with LoRAs
  print("Loading high-noise transformer...")
  high_model = load_transformer(HIGH_NOISE_PATH, HIGH_NOISE_LORAS, HIGH_NOISE_STRENGTHS)

  print("Loading low-noise transformer...")
  low_model = load_transformer(LOW_NOISE_PATH, LOW_NOISE_LORAS, LOW_NOISE_STRENGTHS)

  # 4. Denoise
  print(f"Denoising ({NUM_STEPS} steps: {NUM_STEPS // 2} high-noise + {NUM_STEPS // 2} low-noise)...")
  t_denoise = time.perf_counter()
  latents = denoise(prompt_embeds, image_latent, high_model, low_model)
  torch.cuda.synchronize()
  print(f"  Denoising: {time.perf_counter() - t_denoise:.2f}s")

  # 5. Decode with VAE
  print("Decoding...")
  t_decode = time.perf_counter()
  video = decode_latents(latents, vae)
  torch.cuda.synchronize()
  print(f"  Decoding: {time.perf_counter() - t_decode:.2f}s")

  # 6. Save
  save_mp4(video, OUTPUT_PATH)
  print(f"Total: {time.perf_counter() - t_total:.2f}s")


if __name__ == "__main__":
  main()
