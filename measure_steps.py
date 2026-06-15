"""Measure per-step GPU and wall time to find any CPU-GPU sync overhead."""
import os
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

import time
import torch
from prepare import DIT_DTYPE, TEXT_LEN, HEIGHT, WIDTH, NUM_FRAMES, NUM_STEPS, FLOW_SHIFT
from model import WanModel, replace_last_n_ffn_with_fp8, replace_last_n_attn_with_rowwise_fp8, replace_ffn_linears_with_fp8, replace_attn_linears_with_fp8
from scheduler import FlowMatchEulerDiscreteScheduler
from utils import DEVICE, set_default_torch_dtype, skip_init_modules

# Load models with random weights (no disk I/O needed)
with skip_init_modules(), set_default_torch_dtype(DIT_DTYPE):
    high_model = WanModel()
    low_model = WanModel()

high_model.to(DEVICE)
low_model.to(DEVICE)

replace_last_n_ffn_with_fp8(high_model, n=25)
replace_last_n_attn_with_rowwise_fp8(high_model, n=15)
replace_ffn_linears_with_fp8(low_model)
replace_attn_linears_with_fp8(low_model)

high_model = torch.compile(high_model, mode='default')
low_model = torch.compile(low_model, mode='default')

lat_h = HEIGHT // 8
lat_w = WIDTH // 8
lat_f = 1 + (NUM_FRAMES - 1) // 4
_dummy = torch.zeros(1, 36, lat_f, lat_h, lat_w, device=DEVICE, dtype=DIT_DTYPE)
_ts = torch.tensor([500.0], device=DEVICE, dtype=torch.float32)
_enc = torch.zeros(1, TEXT_LEN, 4096, device=DEVICE, dtype=DIT_DTYPE)

# Warmup (compile + allocator warm)
with torch.no_grad(), torch.amp.autocast('cuda', dtype=DIT_DTYPE):
    _ = high_model(hidden_states=_dummy, timestep=_ts, encoder_hidden_states=_enc)
    _ = low_model(hidden_states=_dummy, timestep=_ts, encoder_hidden_states=_enc)
    _ = high_model(hidden_states=_dummy, timestep=_ts, encoder_hidden_states=_enc)
    _ = low_model(hidden_states=_dummy, timestep=_ts, encoder_hidden_states=_enc)
torch.cuda.synchronize()

# Setup scheduler
scheduler = FlowMatchEulerDiscreteScheduler(shift=FLOW_SHIFT)
scheduler.set_timesteps(NUM_STEPS, device=DEVICE)

# Setup latents
latents = torch.randn(1, 16, lat_f, lat_h, lat_w, device=DEVICE, dtype=DIT_DTYPE)
image_latent = torch.zeros(1, 20, lat_f, lat_h, lat_w, device=DEVICE, dtype=DIT_DTYPE)
encoder_hidden_states = _enc

# Run the actual denoising loop with per-step timing
print(f"\nPer-step GPU vs wall timing ({NUM_STEPS} steps):")
print(f"{'step':>5}  {'model':>8}  {'GPU(s)':>8}  {'wall(s)':>8}  {'overhead(ms)':>12}")

total_wall = 0
total_gpu = 0

with torch.no_grad(), torch.amp.autocast('cuda', dtype=DIT_DTYPE):
    for step_index, t in enumerate(scheduler.timesteps):
        model = high_model if step_index < NUM_STEPS // 2 else low_model
        model_name = 'high' if step_index < NUM_STEPS // 2 else 'low'

        model_input = torch.cat([latents, image_latent], dim=1)
        timestep = t.repeat(1)

        # GPU timing
        e_start = torch.cuda.Event(enable_timing=True)
        e_end = torch.cuda.Event(enable_timing=True)

        t0 = time.perf_counter()
        e_start.record()
        noise_pred = model(hidden_states=model_input, timestep=timestep, encoder_hidden_states=encoder_hidden_states)
        e_end.record()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        gpu = e_start.elapsed_time(e_end) / 1000.0

        latents = scheduler.step(noise_pred, t, latents)

        total_wall += wall
        total_gpu += gpu
        overhead_ms = (wall - gpu) * 1000
        print(f"{step_index:>5}  {model_name:>8}  {gpu:>8.3f}  {wall:>8.3f}  {overhead_ms:>12.1f}")

print(f"\nTotal GPU: {total_gpu:.3f}s  Total wall (no sync): {total_wall:.3f}s")
print(f"Note: Each step syncs GPU (synchronize call), so wall includes sync overhead")
