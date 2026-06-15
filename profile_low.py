"""Profile real low_model forward pass."""
import os
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
import torch
from lora import apply_loras
from prepare import DIT_DTYPE, TEXT_LEN, HEIGHT, WIDTH, NUM_FRAMES, LOW_NOISE_PATH, LOW_NOISE_LORAS, LOW_NOISE_STRENGTHS
from model import WanModel, replace_ffn_linears_with_fp8, replace_attn_linears_with_fp8
from utils import DEVICE, set_default_torch_dtype, skip_init_modules

print("Loading real low_model weights...")
with skip_init_modules(), set_default_torch_dtype(DIT_DTYPE):
    model = WanModel()
model.load(LOW_NOISE_PATH)
model.to(DEVICE)
apply_loras(model, LOW_NOISE_LORAS, LOW_NOISE_STRENGTHS, WanModel.LORA_PARAM_NAMES_MAPPING)
replace_ffn_linears_with_fp8(model)
replace_attn_linears_with_fp8(model)
model = torch.compile(model, mode='default')

lat_h, lat_w, lat_f = HEIGHT//8, WIDTH//8, 1+(NUM_FRAMES-1)//4
dummy = torch.zeros(1, 36, lat_f, lat_h, lat_w, device=DEVICE, dtype=DIT_DTYPE)
ts = torch.tensor([500.0], device=DEVICE, dtype=torch.float32)
enc = torch.zeros(1, TEXT_LEN, 4096, device=DEVICE, dtype=DIT_DTYPE)

print("Warmup...")
with torch.no_grad(), torch.amp.autocast('cuda', dtype=DIT_DTYPE):
    _ = model(hidden_states=dummy, timestep=ts, encoder_hidden_states=enc)
    _ = model(hidden_states=dummy, timestep=ts, encoder_hidden_states=enc)
torch.cuda.synchronize()

print("Profiling...")
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=DIT_DTYPE):
        _ = model(hidden_states=dummy, timestep=ts, encoder_hidden_states=enc)
torch.cuda.synchronize()

for e in sorted(prof.key_averages(), key=lambda x: -x.self_device_time_total)[:8]:
    print(f'  {e.key[:60]:60s}: avg={e.self_device_time_total/max(e.count,1):.0f}us x {e.count} = {e.self_device_time_total/1e6:.3f}s')
total = sum(e.self_device_time_total for e in prof.key_averages())
print(f'Total CUDA: {total/1e6:.3f}s')

e_start = torch.cuda.Event(enable_timing=True)
e_end = torch.cuda.Event(enable_timing=True)
e_start.record()
with torch.no_grad(), torch.amp.autocast('cuda', dtype=DIT_DTYPE):
    _ = model(hidden_states=dummy, timestep=ts, encoder_hidden_states=enc)
e_end.record()
torch.cuda.synchronize()
print(f'CUDA event timing: {e_start.elapsed_time(e_end)/1000:.3f}s')
