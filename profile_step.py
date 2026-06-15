"""Profile one high_model forward pass with current config."""
import os
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

import torch
from prepare import DIT_DTYPE, TEXT_LEN, HEIGHT, WIDTH, NUM_FRAMES
from model import WanModel, replace_last_n_ffn_with_fp8, replace_last_n_attn_with_rowwise_fp8
from utils import DEVICE, set_default_torch_dtype, skip_init_modules

with skip_init_modules(), set_default_torch_dtype(DIT_DTYPE):
    model = WanModel()
model.to(DEVICE)

replace_last_n_ffn_with_fp8(model, n=25)
replace_last_n_attn_with_rowwise_fp8(model, n=15)
model = torch.compile(model, mode='default')

lat_h = HEIGHT // 8
lat_w = WIDTH // 8
lat_f = 1 + (NUM_FRAMES - 1) // 4
_dummy = torch.zeros(1, 36, lat_f, lat_h, lat_w, device=DEVICE, dtype=DIT_DTYPE)
_ts = torch.tensor([500.0], device=DEVICE, dtype=torch.float32)
_enc = torch.zeros(1, TEXT_LEN, 4096, device=DEVICE, dtype=DIT_DTYPE)

with torch.no_grad(), torch.amp.autocast('cuda', dtype=DIT_DTYPE):
    _ = model(hidden_states=_dummy, timestep=_ts, encoder_hidden_states=_enc)
    _ = model(hidden_states=_dummy, timestep=_ts, encoder_hidden_states=_enc)
torch.cuda.synchronize()

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_flops=False,
) as prof:
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=DIT_DTYPE):
        _ = model(hidden_states=_dummy, timestep=_ts, encoder_hidden_states=_enc)

torch.cuda.synchronize()

# Print top-30 CUDA operations by self time
print("\n=== TOP 30 CUDA KERNELS BY SELF TIME ===")
table = prof.key_averages(group_by_input_shape=False).table(
    sort_by="self_cuda_time_total", row_limit=30)
print(table)

# Print total CUDA time
events = prof.key_averages()
total_cuda = sum(e.self_cuda_time_total for e in events) / 1e6
print(f"\nTotal self CUDA time: {total_cuda:.3f}s")

# Group by operation name prefix
from collections import defaultdict
groups = defaultdict(float)
for e in events:
    name = e.key
    if 'flash_attn' in name.lower() or 'flash_fwd' in name.lower():
        groups['FA3 attention'] += e.self_cuda_time_total
    elif 'scaled_mm' in name.lower() or 'fp8' in name.lower():
        groups['FP8 GEMMs'] += e.self_cuda_time_total
    elif 'mm' in name.lower() or 'gemm' in name.lower() or 'sgemm' in name.lower():
        groups['BF16 GEMMs'] += e.self_cuda_time_total
    elif 'fuse_scale_shift' in name.lower() or 'scale_shift' in name.lower():
        groups['scale_shift'] += e.self_cuda_time_total
    elif 'rmsnorm' in name.lower() or 'rms_norm' in name.lower():
        groups['RMSNorm'] += e.self_cuda_time_total
    elif 'rope' in name.lower() or 'rotary' in name.lower():
        groups['RoPE'] += e.self_cuda_time_total
    else:
        groups['other'] += e.self_cuda_time_total

print("\n=== GROUPED SUMMARY ===")
total = sum(groups.values())
for k, v in sorted(groups.items(), key=lambda x: -x[1]):
    print(f"  {k:30s}: {v/1e6:7.1f}ms  ({100*v/total:5.1f}%)")
