"""Benchmark fuse_scale_shift_kernel with different block/warp configs."""
import os
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
import torch
from layers import _fuse_scale_shift_kernel_blc, fuse_scale_shift_kernel
import triton
from utils import DEVICE

L, C = 32760, 5120
x = torch.randn(1, L, C, device=DEVICE, dtype=torch.bfloat16)
scale = torch.randn(1, 1, C, device=DEVICE, dtype=torch.float32).expand(1, L, C).contiguous()
shift = torch.randn(1, 1, C, device=DEVICE, dtype=torch.float32).expand(1, L, C).contiguous()
out = torch.empty_like(x)

def run(block_l, block_c, num_warps, n=200):
    # warmup
    for _ in range(5):
        grid = (triton.cdiv(L, block_l), triton.cdiv(C, block_c), 1)
        _fuse_scale_shift_kernel_blc[grid](
            x, shift, scale, 1.0, out,
            1, L, C,
            x.stride(0), x.stride(1), x.stride(2),
            shift.stride(0), shift.stride(1), shift.stride(2),
            scale.stride(0), scale.stride(1), scale.stride(2),
            SCALE_IS_SCALAR=False, SHIFT_IS_SCALAR=False,
            BLOCK_L=block_l, BLOCK_C=block_c,
            num_warps=num_warps, num_stages=2,
        )
    torch.cuda.synchronize()
    e_start = torch.cuda.Event(enable_timing=True)
    e_end = torch.cuda.Event(enable_timing=True)
    e_start.record()
    for _ in range(n):
        grid = (triton.cdiv(L, block_l), triton.cdiv(C, block_c), 1)
        _fuse_scale_shift_kernel_blc[grid](
            x, shift, scale, 1.0, out,
            1, L, C,
            x.stride(0), x.stride(1), x.stride(2),
            shift.stride(0), shift.stride(1), shift.stride(2),
            scale.stride(0), scale.stride(1), scale.stride(2),
            SCALE_IS_SCALAR=False, SHIFT_IS_SCALAR=False,
            BLOCK_L=block_l, BLOCK_C=block_c,
            num_warps=num_warps, num_stages=2,
        )
    e_end.record()
    torch.cuda.synchronize()
    ms = e_start.elapsed_time(e_end) / n
    blocks = triton.cdiv(L, block_l) * triton.cdiv(C, block_c)
    print(f"  BL={block_l:3d} BC={block_c:3d} warps={num_warps:2d}  blocks={blocks:5d}  {ms:.3f} ms")
    return ms

print(f"[L={L}, C={C}] scale_shift kernel microbenchmark")
print("config                             time")
configs = [
    (128, 128, 4),   # current default
    (128, 128, 8),
    (128, 128, 16),
    (128, 256, 4),
    (128, 256, 8),
    (128, 256, 16),
    (256, 128, 4),
    (256, 128, 8),
    (256, 256, 4),
    (256, 256, 8),
    (64,  128, 4),
    (64,  128, 8),
    (128, 512, 4),
    (128, 512, 8),
]
results = []
for bl, bc, nw in configs:
    ms = run(bl, bc, nw)
    results.append((ms, bl, bc, nw))

results.sort()
print("\nTop 5 fastest:")
for ms, bl, bc, nw in results[:5]:
    print(f"  BL={bl} BC={bc} warps={nw}: {ms:.3f} ms  ({results[0][0]/ms*100:.0f}% of baseline)")
