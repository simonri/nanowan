"""Benchmark scale_shift Triton kernel with different params."""
import os
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import triton
import triton.language as tl


@triton.jit
def _fuse_scale_shift_bench(
    x_ptr, shift_ptr, scale_ptr, scale_constant: tl.constexpr, y_ptr,
    B, L, C, stride_x_b, stride_x_l, stride_x_c,
    stride_s_b, stride_s_l, stride_s_c, stride_sc_b, stride_sc_l, stride_sc_c,
    BLOCK_L: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid_l = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_b = tl.program_id(2)
    l_offsets = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_l = l_offsets < L
    mask_c = c_offsets < C
    mask = mask_l[:, None] & mask_c[None, :]
    x_off = pid_b * stride_x_b + l_offsets[:, None] * stride_x_l + c_offsets[None, :] * stride_x_c
    x = tl.load(x_ptr + x_off, mask=mask, other=0)
    sc_off = pid_b * stride_sc_b + l_offsets[:, None] * stride_sc_l + c_offsets[None, :] * stride_sc_c
    scale = tl.load(scale_ptr + sc_off, mask=mask, other=0)
    sh_off = pid_b * stride_s_b + l_offsets[:, None] * stride_s_l + c_offsets[None, :] * stride_s_c
    shift = tl.load(shift_ptr + sh_off, mask=mask, other=0)
    y = x * (scale_constant + scale) + shift
    tl.store(y_ptr + x_off, y, mask=mask)


def bench(x, scale, shift, bl, bc, nw, ns, iters=300):
    B, L, C = x.shape
    output = torch.empty_like(x)
    grid = (triton.cdiv(L, bl), triton.cdiv(C, bc), B)
    kwargs = dict(
        num_warps=nw, num_stages=ns,
    )
    for _ in range(10):
        _fuse_scale_shift_bench[grid](
            x, shift, scale, 1.0, output, B, L, C,
            x.stride(0), x.stride(1), x.stride(2),
            shift.stride(0), shift.stride(1), shift.stride(2),
            scale.stride(0), scale.stride(1), scale.stride(2),
            bl, bc, **kwargs)
    torch.cuda.synchronize()
    e_start = torch.cuda.Event(enable_timing=True)
    e_end = torch.cuda.Event(enable_timing=True)
    e_start.record()
    for _ in range(iters):
        _fuse_scale_shift_bench[grid](
            x, shift, scale, 1.0, output, B, L, C,
            x.stride(0), x.stride(1), x.stride(2),
            shift.stride(0), shift.stride(1), shift.stride(2),
            scale.stride(0), scale.stride(1), scale.stride(2),
            bl, bc, **kwargs)
    e_end.record()
    torch.cuda.synchronize()
    blocks = triton.cdiv(L, bl) * triton.cdiv(C, bc) * B
    return e_start.elapsed_time(e_end) / iters, blocks


if __name__ == '__main__':
    B, L, C = 1, 8190, 5120
    x = torch.randn(B, L, C, device='cuda', dtype=torch.bfloat16)
    scale = torch.randn(B, L, C, device='cuda', dtype=torch.bfloat16)
    shift = torch.randn(B, L, C, device='cuda', dtype=torch.bfloat16)

    configs = [
        (128, 128, 4, 2),  # baseline
        (128, 128, 4, 4),
        (128, 128, 8, 2),
        (128, 128, 8, 4),
        (128, 128, 16, 4),
        (64, 256, 4, 2),
        (64, 256, 8, 4),
        (32, 512, 8, 4),
        (16, 1024, 8, 4),
        (16, 2048, 8, 4),
        (8, 4096, 8, 4),
    ]

    print(f'{"BL":>6} {"BC":>6} {"nw":>4} {"ns":>4} {"us/call":>10} {"blocks":>8} {"speedup":>8}')
    baseline_t = None
    for bl, bc, nw, ns in configs:
        t, blocks = bench(x, scale, shift, bl, bc, nw, ns)
        if baseline_t is None:
            baseline_t = t
            speedup = "1.00x (baseline)"
        else:
            speedup = f'{baseline_t / t:.2f}x'
        print(f'{bl:>6} {bc:>6} {nw:>4} {ns:>4} {t*1000:>10.1f} {blocks:>8} {speedup}')
