"""Test script to compare attention backends: sdpa, flex, and tilelang.

This script compares:
1. Numerical correctness (forward and backward)
2. Performance (latency and throughput)
"""

import torch
import torch.nn.functional as F
import time
import argparse
from typing import Dict, List, Tuple, Optional
import math

# Check available backends
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_AVAILABLE = True
except ImportError:
    FLEX_AVAILABLE = False
    print("FlexAttention not available")

try:
    from tilelang_attention import (
        tilelang_attention,
        tilelang_block_diff_attention,
        create_block_diff_mask,
        TILELANG_AVAILABLE
    )
except ImportError:
    TILELANG_AVAILABLE = False
    print("TileLang not available")
except Exception as e:
    TILELANG_AVAILABLE = False
    print(f"TileLang import error: {e}")


def create_block_diff_mask_torch(seq_len: int, block_size: int, device: torch.device) -> torch.Tensor:
    """Create BD3LM block diffusion mask using PyTorch."""
    n = seq_len
    total_len = 2 * n

    q_idx = torch.arange(total_len, device=device)[:, None]
    kv_idx = torch.arange(total_len, device=device)[None, :]

    x0_flag_q = (q_idx >= n)
    x0_flag_kv = (kv_idx >= n)

    block_q = torch.where(x0_flag_q, (q_idx - n) // block_size, q_idx // block_size)
    block_kv = torch.where(x0_flag_kv, (kv_idx - n) // block_size, kv_idx // block_size)

    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
    offset_block_causal = (block_q > block_kv) & x0_flag_kv & (~x0_flag_q)
    block_causal = (block_q >= block_kv) & x0_flag_kv & x0_flag_q

    mask = block_diagonal | offset_block_causal | block_causal
    return mask


def sdpa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    """SDPA attention with mask. Input shape: [B, S, H, D]"""
    batch, seq_len, heads, dim = q.shape

    # Transpose to [B, H, S, D]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Convert bool mask to float mask for SDPA (match query dtype)
    attn_mask = torch.zeros_like(mask, dtype=q.dtype)
    attn_mask = attn_mask.masked_fill(~mask, float('-inf'))

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return out.transpose(1, 2)  # Back to [B, S, H, D]


def flex_attention_fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      seq_len: int, block_size: int) -> torch.Tensor:
    """FlexAttention with block diff mask. Input shape: [B, S, H, D]"""
    from functools import partial

    def block_diff_mask_fn(b, h, q_idx, kv_idx, block_size=None, n=None):
        x0_flag_q = (q_idx >= n)
        x0_flag_kv = (kv_idx >= n)
        block_q = torch.where(x0_flag_q, (q_idx - n) // block_size, q_idx // block_size)
        block_kv = torch.where(x0_flag_kv, (kv_idx - n) // block_size, kv_idx // block_size)
        block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
        offset_block_causal = (block_q > block_kv) & x0_flag_kv & (~x0_flag_q)
        block_causal = (block_q >= block_kv) & x0_flag_kv & x0_flag_q
        return block_diagonal | offset_block_causal | block_causal

    n = seq_len // 2
    block_mask = create_block_mask(
        partial(block_diff_mask_fn, block_size=block_size, n=n),
        B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len
    )

    # Transpose to [B, H, S, D]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    out = flex_attention(q, k, v, block_mask=block_mask)
    return out.transpose(1, 2)


# Compiled version of flex_attention for better performance
_compiled_flex_attention = None

def flex_attention_compiled_fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                               block_mask) -> torch.Tensor:
    """Compiled FlexAttention. Input shape: [B, S, H, D]"""
    global _compiled_flex_attention

    if _compiled_flex_attention is None:
        _compiled_flex_attention = torch.compile(flex_attention, fullgraph=True, mode="max-autotune-no-cudagraphs")

    # Transpose to [B, H, S, D]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    out = _compiled_flex_attention(q, k, v, block_mask=block_mask)
    return out.transpose(1, 2)


def benchmark_fn(fn, *args, warmup: int = 10, rep: int = 100, **kwargs) -> Tuple[float, float]:
    """Benchmark a function and return mean and std latency in ms."""
    # Warmup
    for _ in range(warmup):
        out = fn(*args, **kwargs)
        if out.requires_grad:
            out.sum().backward(retain_graph=True)

    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(rep):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = fn(*args, **kwargs)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return sum(times) / len(times), (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5


def benchmark_backward(fn, *args, warmup: int = 10, rep: int = 100, **kwargs) -> Tuple[float, float]:
    """Benchmark backward pass."""
    # Warmup
    for _ in range(warmup):
        out = fn(*args, **kwargs)
        grad = torch.randn_like(out)
        out.backward(grad, retain_graph=True)

    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(rep):
        out = fn(*args, **kwargs)
        grad = torch.randn_like(out)
        torch.cuda.synchronize()
        start = time.perf_counter()
        out.backward(grad, retain_graph=True)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return sum(times) / len(times), (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5


def test_numerical_correctness(
    batch: int = 2,
    seq_len: int = 512,  # This is n, total will be 2n
    heads: int = 8,
    dim: int = 64,
    block_size: int = 64,
    device: str = 'cuda'
):
    """Test numerical correctness between backends."""
    print("\n" + "="*60)
    print("NUMERICAL CORRECTNESS TEST")
    print("="*60)
    print(f"Config: batch={batch}, seq_len={seq_len*2}, heads={heads}, dim={dim}, block_size={block_size}")

    torch.manual_seed(42)

    total_seq = seq_len * 2
    q = torch.randn(batch, total_seq, heads, dim, device=device, dtype=torch.float16, requires_grad=True)
    k = torch.randn(batch, total_seq, heads, dim, device=device, dtype=torch.float16, requires_grad=True)
    v = torch.randn(batch, total_seq, heads, dim, device=device, dtype=torch.float16, requires_grad=True)

    # Create mask
    mask = create_block_diff_mask_torch(seq_len, block_size, device)

    results = {}
    grads = {}

    # SDPA (reference)
    print("\n[SDPA] Running...")
    q_sdpa = q.detach().clone().requires_grad_(True)
    k_sdpa = k.detach().clone().requires_grad_(True)
    v_sdpa = v.detach().clone().requires_grad_(True)
    out_sdpa = sdpa_attention(q_sdpa, k_sdpa, v_sdpa, mask)
    grad_out = torch.randn_like(out_sdpa)
    out_sdpa.backward(grad_out)
    results['sdpa'] = out_sdpa.detach()
    grads['sdpa'] = (q_sdpa.grad.detach(), k_sdpa.grad.detach(), v_sdpa.grad.detach())
    print(f"  Output shape: {out_sdpa.shape}")

    # FlexAttention
    if FLEX_AVAILABLE:
        print("\n[FlexAttention] Running...")
        try:
            q_flex = q.detach().clone().requires_grad_(True)
            k_flex = k.detach().clone().requires_grad_(True)
            v_flex = v.detach().clone().requires_grad_(True)
            out_flex = flex_attention_fn(q_flex, k_flex, v_flex, total_seq, block_size)
            out_flex.backward(grad_out)
            results['flex'] = out_flex.detach()
            grads['flex'] = (q_flex.grad.detach(), k_flex.grad.detach(), v_flex.grad.detach())
            print(f"  Output shape: {out_flex.shape}")
        except Exception as e:
            print(f"  Error: {e}")
            results['flex'] = None
    else:
        print("\n[FlexAttention] Not available")
        results['flex'] = None

    # TileLang
    if TILELANG_AVAILABLE:
        print("\n[TileLang] Running...")
        try:
            q_tile = q.detach().clone().requires_grad_(True)
            k_tile = k.detach().clone().requires_grad_(True)
            v_tile = v.detach().clone().requires_grad_(True)
            # TileLang uses block_size directly, computes mask inline
            out_tile = tilelang_block_diff_attention(q_tile, k_tile, v_tile, block_size)
            out_tile.backward(grad_out)
            results['tilelang'] = out_tile.detach()
            grads['tilelang'] = (q_tile.grad.detach(), k_tile.grad.detach(), v_tile.grad.detach())
            print(f"  Output shape: {out_tile.shape}")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results['tilelang'] = None
            grads['tilelang'] = None
    else:
        print("\n[TileLang] Not available")
        results['tilelang'] = None
        grads['tilelang'] = None

    # Compare results
    print("\n" + "-"*60)
    print("FORWARD PASS COMPARISON (vs SDPA reference)")
    print("-"*60)

    ref = results['sdpa']
    for name, out in results.items():
        if name == 'sdpa' or out is None:
            continue

        # Compute metrics
        max_diff = (out - ref).abs().max().item()
        mean_diff = (out - ref).abs().mean().item()
        rel_diff = ((out - ref).abs() / (ref.abs() + 1e-6)).mean().item()

        is_close = torch.allclose(out, ref, rtol=1e-2, atol=1e-2)

        print(f"\n{name.upper()} vs SDPA:")
        print(f"  Max absolute diff:  {max_diff:.6e}")
        print(f"  Mean absolute diff: {mean_diff:.6e}")
        print(f"  Mean relative diff: {rel_diff:.6e}")
        print(f"  torch.allclose (rtol=1e-2, atol=1e-2): {is_close}")

    # Compare gradients
    print("\n" + "-"*60)
    print("BACKWARD PASS COMPARISON (vs SDPA reference)")
    print("-"*60)

    ref_grads = grads['sdpa']
    for name, g in grads.items():
        if name == 'sdpa' or g is None:
            continue

        print(f"\n{name.upper()} vs SDPA:")
        for i, gname in enumerate(['dQ', 'dK', 'dV']):
            max_diff = (g[i] - ref_grads[i]).abs().max().item()
            mean_diff = (g[i] - ref_grads[i]).abs().mean().item()
            is_close = torch.allclose(g[i], ref_grads[i], rtol=1e-2, atol=1e-2)
            print(f"  {gname}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, allclose={is_close}")


def test_performance(
    batch: int = 4,
    seq_len: int = 1024,  # This is n, total will be 2n
    heads: int = 12,
    dim: int = 64,
    block_size: int = 64,
    device: str = 'cuda',
    warmup: int = 20,
    rep: int = 100
):
    """Benchmark performance of different backends."""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)
    print(f"Config: batch={batch}, seq_len={seq_len*2}, heads={heads}, dim={dim}, block_size={block_size}")
    print(f"Warmup={warmup}, Repetitions={rep}")

    torch.manual_seed(42)

    total_seq = seq_len * 2

    # Calculate FLOPs
    # Attention FLOPs: 4 * batch * heads * seq^2 * dim (2 for QK^T, 2 for softmax@V)
    flops = 4 * batch * heads * total_seq * total_seq * dim

    results = {}

    # SDPA
    print("\n[SDPA] Benchmarking...")
    q = torch.randn(batch, total_seq, heads, dim, device=device, dtype=torch.float16, requires_grad=True)
    k = torch.randn(batch, total_seq, heads, dim, device=device, dtype=torch.float16, requires_grad=True)
    v = torch.randn(batch, total_seq, heads, dim, device=device, dtype=torch.float16, requires_grad=True)
    mask = create_block_diff_mask_torch(seq_len, block_size, device)

    fwd_mean, fwd_std = benchmark_fn(sdpa_attention, q, k, v, mask, warmup=warmup, rep=rep)

    q = torch.randn(batch, total_seq, heads, dim, device=device, dtype=torch.float16, requires_grad=True)
    k = torch.randn(batch, total_seq, heads, dim, device=device, dtype=torch.float16, requires_grad=True)
    v = torch.randn(batch, total_seq, heads, dim, device=device, dtype=torch.float16, requires_grad=True)
    bwd_mean, bwd_std = benchmark_backward(sdpa_attention, q, k, v, mask, warmup=warmup, rep=rep)

    results['sdpa'] = {
        'fwd_mean': fwd_mean, 'fwd_std': fwd_std,
        'bwd_mean': bwd_mean, 'bwd_std': bwd_std
    }
    print(f"  Forward:  {fwd_mean:.3f} ± {fwd_std:.3f} ms")
    print(f"  Backward: {bwd_mean:.3f} ± {bwd_std:.3f} ms")

    # FlexAttention (compiled)
    if FLEX_AVAILABLE:
        print("\n[FlexAttention (compiled)] Benchmarking...")
        try:
            from functools import partial

            def block_diff_mask_fn(b, h, q_idx, kv_idx, block_size=None, n=None):
                x0_flag_q = (q_idx >= n)
                x0_flag_kv = (kv_idx >= n)
                block_q = torch.where(x0_flag_q, (q_idx - n) // block_size, q_idx // block_size)
                block_kv = torch.where(x0_flag_kv, (kv_idx - n) // block_size, kv_idx // block_size)
                block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
                offset_block_causal = (block_q > block_kv) & x0_flag_kv & (~x0_flag_q)
                block_causal = (block_q >= block_kv) & x0_flag_kv & x0_flag_q
                return block_diagonal | offset_block_causal | block_causal

            n = total_seq // 2
            block_mask = create_block_mask(
                partial(block_diff_mask_fn, block_size=block_size, n=n),
                B=None, H=None, Q_LEN=total_seq, KV_LEN=total_seq
            )

            q = torch.randn(batch, total_seq, heads, dim, device=device, dtype=torch.float16, requires_grad=True)
            k = torch.randn(batch, total_seq, heads, dim, device=device, dtype=torch.float16, requires_grad=True)
            v = torch.randn(batch, total_seq, heads, dim, device=device, dtype=torch.float16, requires_grad=True)

            # Pre-compile with extra warmup
            print("  Compiling (this may take a moment)...")
            for _ in range(3):
                _ = flex_attention_compiled_fn(q, k, v, block_mask)
            torch.cuda.synchronize()

            fwd_mean, fwd_std = benchmark_fn(flex_attention_compiled_fn, q, k, v, block_mask, warmup=warmup, rep=rep)

            q = torch.randn(batch, total_seq, heads, dim, device=device, dtype=torch.float16, requires_grad=True)
            k = torch.randn(batch, total_seq, heads, dim, device=device, dtype=torch.float16, requires_grad=True)
            v = torch.randn(batch, total_seq, heads, dim, device=device, dtype=torch.float16, requires_grad=True)
            bwd_mean, bwd_std = benchmark_backward(flex_attention_compiled_fn, q, k, v, block_mask, warmup=warmup, rep=rep)

            results['flex_compiled'] = {
                'fwd_mean': fwd_mean, 'fwd_std': fwd_std,
                'bwd_mean': bwd_mean, 'bwd_std': bwd_std
            }
            print(f"  Forward:  {fwd_mean:.3f} ± {fwd_std:.3f} ms")
            print(f"  Backward: {bwd_mean:.3f} ± {bwd_std:.3f} ms")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[FlexAttention] Not available")

    # TileLang
    if TILELANG_AVAILABLE:
        print("\n[TileLang] Benchmarking...")
        try:
            q = torch.randn(batch, total_seq, heads, dim, device=device, dtype=torch.float16, requires_grad=True)
            k = torch.randn(batch, total_seq, heads, dim, device=device, dtype=torch.float16, requires_grad=True)
            v = torch.randn(batch, total_seq, heads, dim, device=device, dtype=torch.float16, requires_grad=True)

            # Warmup for compilation
            print("  Compiling (this may take a moment)...")
            for _ in range(3):
                _ = tilelang_block_diff_attention(q, k, v, block_size)
            torch.cuda.synchronize()

            fwd_mean, fwd_std = benchmark_fn(
                tilelang_block_diff_attention, q, k, v, block_size, warmup=warmup, rep=rep
            )

            q = torch.randn(batch, total_seq, heads, dim, device=device, dtype=torch.float16, requires_grad=True)
            k = torch.randn(batch, total_seq, heads, dim, device=device, dtype=torch.float16, requires_grad=True)
            v = torch.randn(batch, total_seq, heads, dim, device=device, dtype=torch.float16, requires_grad=True)
            bwd_mean, bwd_std = benchmark_backward(
                tilelang_block_diff_attention, q, k, v, block_size, warmup=warmup, rep=rep
            )

            results['tilelang'] = {
                'fwd_mean': fwd_mean, 'fwd_std': fwd_std,
                'bwd_mean': bwd_mean, 'bwd_std': bwd_std
            }
            print(f"  Forward:  {fwd_mean:.3f} ± {fwd_std:.3f} ms")
            print(f"  Backward: {bwd_mean:.3f} ± {bwd_std:.3f} ms")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[TileLang] Not available")

    # Summary
    print("\n" + "-"*60)
    print("PERFORMANCE SUMMARY")
    print("-"*60)
    print(f"\n{'Backend':<15} {'Forward (ms)':<18} {'Backward (ms)':<18} {'Total (ms)':<15} {'TFLOPs':<10}")
    print("-"*76)

    for name, r in results.items():
        total = r['fwd_mean'] + r['bwd_mean']
        # For backward, roughly 2.5x forward FLOPs
        total_flops = flops + 2.5 * flops
        tflops = total_flops / (total * 1e-3) / 1e12
        print(f"{name:<15} {r['fwd_mean']:.3f} ± {r['fwd_std']:.3f}      {r['bwd_mean']:.3f} ± {r['bwd_std']:.3f}      {total:.3f}          {tflops:.2f}")

    # Speedup comparison
    if 'sdpa' in results:
        print("\n" + "-"*60)
        print("SPEEDUP vs SDPA")
        print("-"*60)
        sdpa_total = results['sdpa']['fwd_mean'] + results['sdpa']['bwd_mean']
        for name, r in results.items():
            if name == 'sdpa':
                continue
            total = r['fwd_mean'] + r['bwd_mean']
            speedup = sdpa_total / total
            print(f"{name}: {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Test attention backends")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length (n, total will be 2n)")
    parser.add_argument("--heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--block-size", type=int, default=64, help="Block size for block diffusion")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--rep", type=int, default=100, help="Benchmark repetitions")
    parser.add_argument("--skip-numerical", action="store_true", help="Skip numerical test")
    parser.add_argument("--skip-perf", action="store_true", help="Skip performance test")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")

    print(f"\nAvailable backends:")
    print(f"  SDPA: True (built-in)")
    print(f"  FlexAttention: {FLEX_AVAILABLE}")
    print(f"  TileLang: {TILELANG_AVAILABLE}")

    if not args.skip_numerical:
        test_numerical_correctness(
            batch=args.batch,
            seq_len=args.seq_len,
            heads=args.heads,
            dim=args.dim,
            block_size=args.block_size,
            device=device
        )

    if not args.skip_perf and device == 'cuda':
        test_performance(
            batch=args.batch,
            seq_len=args.seq_len,
            heads=args.heads,
            dim=args.dim,
            block_size=args.block_size,
            device=device,
            warmup=args.warmup,
            rep=args.rep
        )


if __name__ == "__main__":
    main()
