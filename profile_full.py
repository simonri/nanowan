"""Profile one high_model forward: full breakdown of all operation groups."""
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
model = torch.compile(model, mode='default', dynamic=False)

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

events = prof.key_averages()
T = 'self_device_time_total'  # new name in this pytorch version
total_cuda = sum(getattr(e, T) for e in events) / 1e6

# Comprehensive grouping
from collections import defaultdict
groups = defaultdict(float)
kernel_list = []
for e in events:
    name = e.key.lower()
    t = getattr(e, T)
    kernel_list.append((t, e.key, getattr(e, 'count', 1)))
    if 'flash_attn' in name or 'flash_fwd' in name:
        groups['FA3 attention'] += t
    elif 'scaled_mm' in name or ('fp8' in name and 'mm' in name):
        groups['FP8 GEMMs'] += t
    elif 'sgemm' in name or 'hgemm' in name or 'bgemm' in name or ('gemm' in name and 'flash' not in name):
        groups['BF16/FP16 GEMMs'] += t
    elif 'ampere_bf16' in name or 'ampere_fp16' in name or 'ampere_h16816' in name or 'sm80_xmma' in name:
        groups['BF16/FP16 GEMMs'] += t
    elif '_mm' in name or 'cublaslt' in name or 'cutlass_gemm' in name or 'matmul' in name:
        groups['BF16/FP16 GEMMs'] += t
    elif 'fuse_scale_shift' in name:
        groups['scale_shift Triton'] += t
    elif 'rms_norm' in name or 'rmsnorm' in name or 'one_pass_rms' in name:
        groups['RMSNorm Triton'] += t
    elif 'rope' in name or 'rotary' in name:
        groups['RoPE'] += t
    elif 'layer_norm' in name or 'layernorm' in name or 'welford' in name or 'vectorized_layer_norm' in name:
        groups['LayerNorm'] += t
    elif 'gelu' in name or 'activation' in name:
        groups['GELU'] += t
    elif 'elementwise' in name or 'vectorized_elementwise' in name:
        groups['elementwise compiled'] += t
    elif 'cat' in name or 'fill' in name or 'zeros_' in name or 'copy_' in name:
        groups['memory ops'] += t
    else:
        groups['other'] += t

print(f"\nTotal self CUDA time: {total_cuda:.3f}s")
print("\n=== FULL GROUPED SUMMARY ===")
total = sum(groups.values())
for k, v in sorted(groups.items(), key=lambda x: -x[1]):
    print(f"  {k:40s}: {v/1e6:7.1f}ms  ({100*v/total:5.1f}%)")

print("\n=== TOP 50 CUDA KERNELS BY SELF TIME ===")
kernel_list.sort(reverse=True)
for t, name, cnt in kernel_list[:50]:
    print(f"  {t/1e3:8.2f}ms (n={cnt:3d})  {name[:100]}")
