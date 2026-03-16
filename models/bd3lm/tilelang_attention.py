"""TileLang Flash Attention backend for BD3LM.

This module provides a TileLang-based flash attention implementation
with support for BD3LM's block diffusion attention mask.
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional, Tuple

try:
    import tilelang
    import tilelang.language as T
    from tilelang.utils.target import Target
    # Set target explicitly to avoid auto-detection issues
    TILELANG_TARGET = Target('cuda')
    TILELANG_AVAILABLE = True
except ImportError:
    TILELANG_AVAILABLE = False
    TILELANG_TARGET = None
    tilelang = None
    T = None
except Exception as e:
    print(f"TileLang init warning: {e}")
    TILELANG_AVAILABLE = False
    TILELANG_TARGET = None
    tilelang = None
    T = None


def check_tilelang_available():
    if not TILELANG_AVAILABLE:
        raise RuntimeError(
            "TileLang is not installed. Please install it with: "
            "pip install tilelang"
        )


# ============================================================================
# Forward Kernel with Block Diffusion Mask (computed inline)
# ============================================================================

if TILELANG_AVAILABLE:
    @tilelang.jit(
        out_idx=[3, 4],
        target=TILELANG_TARGET,
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
    )
    def _flashattn_block_diff_fwd_kernel(
        batch: int,
        heads: int,
        seq_len: int,  # This is the FULL sequence length (2n)
        dim: int,
        block_size: int,  # Block size for block diffusion
        block_M: int = 64,
        block_N: int = 64
    ):
        """Flash attention forward kernel with BD3LM block diffusion mask computed inline."""
        scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
        shape = [batch, seq_len, heads, dim]
        dtype = T.float16
        accum_dtype = T.float32
        n = seq_len // 2  # Half sequence length

        @T.prim_func
        def flash_fwd_block_diff(
            Q: T.Tensor(shape, dtype),
            K: T.Tensor(shape, dtype),
            V: T.Tensor(shape, dtype),
            Output: T.Tensor(shape, dtype),
            lse: T.Tensor([batch, heads, seq_len], accum_dtype),
        ):
            with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=128) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)

                T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                # Use large negative finite value instead of -inf to avoid NaN from -inf - (-inf)
                T.fill(scores_max, -1e30)

                loop_range = T.ceildiv(seq_len, block_N)

                for k in T.Pipelined(loop_range, num_stages=1):
                    T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)

                    # First compute QK^T
                    T.clear(acc_s)
                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    # Apply block diffusion mask using a large negative value instead of -inf
                    # to avoid numerical issues with reduce operations
                    mask_value = -1e9  # Large negative but finite
                    for i, j in T.Parallel(block_M, block_N):
                        q_idx = bx * block_M + i
                        kv_idx = k * block_N + j

                        # Check if indices are in x0 region (>= n)
                        x0_flag_q = q_idx >= n
                        x0_flag_kv = kv_idx >= n

                        # Compute block indices
                        block_q = T.if_then_else(
                            x0_flag_q,
                            (q_idx - n) // block_size,
                            q_idx // block_size
                        )
                        block_kv = T.if_then_else(
                            x0_flag_kv,
                            (kv_idx - n) // block_size,
                            kv_idx // block_size
                        )

                        # Block Diagonal: same block, same region
                        block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

                        # Offset Block Causal: q in xt, kv in x0, q_block > kv_block
                        offset_block_causal = (block_q > block_kv) & x0_flag_kv & (~x0_flag_q)

                        # Block Causal: both in x0, q_block >= kv_block
                        block_causal = (block_q >= block_kv) & x0_flag_kv & x0_flag_q

                        # Combined mask - set invalid positions to large negative value
                        valid = block_diagonal | offset_block_causal | block_causal
                        acc_s[i, j] = T.if_then_else(valid, acc_s[i, j], mask_value)
                    T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)

                    T.copy(scores_max, scores_max_prev)
                    # Compute block max with clear=True, then explicitly take max with running max
                    T.reduce_max(acc_s, scores_sum, dim=1, clear=True)  # Use scores_sum as temp
                    for i in T.Parallel(block_M):
                        scores_max[i] = T.max(scores_max_prev[i], scores_sum[i])
                    for i in T.Parallel(block_M):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] *= scores_scale[i]
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.copy(acc_s, acc_s_cast)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                    T.reduce_sum(acc_s, scores_sum, dim=1, clear=True)
                    for i in T.Parallel(block_M):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])
                for i in T.Parallel(block_M):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                T.copy(logsum, lse[bz, by, bx * block_M : (bx + 1) * block_M])

        return flash_fwd_block_diff


    @tilelang.jit(
        out_idx=[3, 4],
        target=TILELANG_TARGET,
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
    )
    def _flashattn_causal_fwd_kernel(
        batch: int,
        heads: int,
        seq_len: int,
        dim: int,
        block_M: int = 64,
        block_N: int = 64
    ):
        """Flash attention forward kernel for causal attention."""
        scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
        shape = [batch, seq_len, heads, dim]
        dtype = T.float16
        accum_dtype = T.float32

        @T.prim_func
        def flash_fwd_causal(
            Q: T.Tensor(shape, dtype),
            K: T.Tensor(shape, dtype),
            V: T.Tensor(shape, dtype),
            Output: T.Tensor(shape, dtype),
            lse: T.Tensor([batch, heads, seq_len], accum_dtype),
        ):
            with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=128) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)

                T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                # Use large negative finite value instead of -inf
                T.fill(scores_max, -1e30)

                loop_range = T.ceildiv((bx + 1) * block_M, block_N)

                for k in T.Pipelined(loop_range, num_stages=1):
                    T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)

                    # First compute QK^T
                    T.clear(acc_s)
                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    # Then apply causal mask using large negative but finite value
                    mask_value = -1e9
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            bx * block_M + i >= k * block_N + j,
                            acc_s[i, j],
                            mask_value
                        )

                    T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)

                    T.copy(scores_max, scores_max_prev)
                    T.reduce_max(acc_s, scores_sum, dim=1, clear=True)  # Use scores_sum as temp
                    for i in T.Parallel(block_M):
                        scores_max[i] = T.max(scores_max_prev[i], scores_sum[i])
                    for i in T.Parallel(block_M):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] *= scores_scale[i]
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.copy(acc_s, acc_s_cast)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                    T.reduce_sum(acc_s, scores_sum, dim=1, clear=True)
                    for i in T.Parallel(block_M):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])
                for i in T.Parallel(block_M):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                T.copy(logsum, lse[bz, by, bx * block_M : (bx + 1) * block_M])

        return flash_fwd_causal


# ============================================================================
# Backward Kernels
# ============================================================================

if TILELANG_AVAILABLE:
    def make_dq_layout(dQ):
        return T.Layout(
            dQ.shape,
            lambda b, l, h, d: [b, l // 8, h, d // 8, (d % 2), 4 * (l % 8) + (d % 8) // 2]
        )

    @tilelang.jit(
        out_idx=[2],
        target=TILELANG_TARGET,
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
    )
    def _flashattn_bwd_preprocess(batch: int, heads: int, seq_len: int, dim: int):
        dtype = T.float16
        accum_dtype = T.float32
        shape = [batch, seq_len, heads, dim]
        blk = 32

        @T.prim_func
        def flash_bwd_prep(
            O: T.Tensor(shape, dtype),
            dO: T.Tensor(shape, dtype),
            Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
        ):
            with T.Kernel(heads, T.ceildiv(seq_len, blk), batch) as (bx, by, bz):
                o = T.alloc_fragment([blk, blk], dtype)
                do = T.alloc_fragment([blk, blk], dtype)
                acc = T.alloc_fragment([blk, blk], accum_dtype)
                delta = T.alloc_fragment([blk], accum_dtype)
                T.clear(acc)
                for k in range(T.ceildiv(dim, blk)):
                    T.copy(O[bz, by * blk : (by + 1) * blk, bx, k * blk : (k + 1) * blk], o)
                    T.copy(dO[bz, by * blk : (by + 1) * blk, bx, k * blk : (k + 1) * blk], do)
                    for i, j in T.Parallel(blk, blk):
                        acc[i, j] += o[i, j] * do[i, j]
                T.reduce_sum(acc, delta, 1)
                T.copy(delta, Delta[bz, bx, by * blk : (by + 1) * blk])

        return flash_bwd_prep


    @tilelang.jit(
        out_idx=[1],
        target=TILELANG_TARGET,
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
    )
    def _flashattn_bwd_postprocess(batch: int, heads: int, seq_len: int, dim: int):
        dtype = T.float16
        accum_dtype = T.float32
        shape = [batch, seq_len, heads, dim]
        blk = 64

        @T.prim_func
        def flash_bwd_post(
            dQ: T.Tensor(shape, accum_dtype),
            dQ_out: T.Tensor(shape, dtype),
        ):
            with T.Kernel(T.ceildiv(seq_len, blk), heads, batch, threads=128) as (bx, by, bz):
                T.annotate_layout({dQ: make_dq_layout(dQ)})
                T.copy(
                    dQ[bz, bx * blk : (bx + 1) * blk, by, :],
                    dQ_out[bz, bx * blk : (bx + 1) * blk, by, :],
                )

        return flash_bwd_post


    @tilelang.jit(
        target=TILELANG_TARGET,
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        }
    )
    def _flashattn_block_diff_bwd_kernel(
        batch: int,
        heads: int,
        seq_len: int,  # Full sequence length (2n)
        dim: int,
        block_size: int,  # Block size for block diffusion
        block_M: int = 64,
        block_N: int = 64
    ):
        """Flash attention backward kernel with block diffusion mask."""
        sm_scale = (1.0 / dim) ** 0.5
        scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
        shape = [batch, seq_len, heads, dim]
        dtype = T.float16
        accum_dtype = T.float32
        n = seq_len // 2

        @T.prim_func
        def flash_bwd_block_diff(
            Q: T.Tensor(shape, dtype),
            K: T.Tensor(shape, dtype),
            V: T.Tensor(shape, dtype),
            dO: T.Tensor(shape, dtype),
            lse: T.Tensor([batch, heads, seq_len], accum_dtype),
            Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
            dQ: T.Tensor(shape, accum_dtype),
            dK: T.Tensor(shape, dtype),
            dV: T.Tensor(shape, dtype),
        ):
            with T.Kernel(heads, T.ceildiv(seq_len, block_M), batch, threads=128) as (bx, by, bz):
                K_shared = T.alloc_shared([block_M, dim], dtype)
                dsT_shared = T.alloc_shared([block_M, block_N], dtype)
                q = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_M, dim], dtype)
                qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
                dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
                qkT_cast = T.alloc_fragment([block_M, block_N], dtype)
                dsT_cast = T.alloc_fragment([block_M, block_N], dtype)
                lse_shared = T.alloc_shared([block_N], accum_dtype)
                delta = T.alloc_shared([block_N], accum_dtype)
                do = T.alloc_shared([block_N, dim], dtype)
                dv = T.alloc_fragment([block_M, dim], accum_dtype)
                dk = T.alloc_fragment([block_M, dim], accum_dtype)
                dq = T.alloc_fragment([block_N, dim], accum_dtype)
                dv_shared = T.alloc_shared([block_M, dim], dtype)
                dk_shared = T.alloc_shared([block_M, dim], dtype)

                T.annotate_layout({dQ: make_dq_layout(dQ)})
                T.copy(K[bz, by * block_M : (by + 1) * block_M, bx, :], K_shared)
                T.copy(V[bz, by * block_M : (by + 1) * block_M, bx, :], V_shared)
                T.clear(dv)
                T.clear(dk)

                loop_ed = T.ceildiv(seq_len, block_N)

                for k in T.Pipelined(0, loop_ed, num_stages=2):
                    T.copy(Q[bz, k * block_N : (k + 1) * block_N, bx, :], q)
                    T.clear(qkT)
                    T.gemm(K_shared, q, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(lse[bz, bx, k * block_N : (k + 1) * block_N], lse_shared)

                    for i, j in T.Parallel(block_M, block_N):
                        qkT[i, j] = T.exp2(qkT[i, j] * scale - lse_shared[j])

                    # Apply block diffusion mask
                    for i, j in T.Parallel(block_M, block_N):
                        kv_idx = by * block_M + i
                        q_idx = k * block_N + j

                        x0_flag_q = q_idx >= n
                        x0_flag_kv = kv_idx >= n

                        block_q = T.if_then_else(
                            x0_flag_q,
                            (q_idx - n) // block_size,
                            q_idx // block_size
                        )
                        block_kv = T.if_then_else(
                            x0_flag_kv,
                            (kv_idx - n) // block_size,
                            kv_idx // block_size
                        )

                        block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
                        offset_block_causal = (block_q > block_kv) & x0_flag_kv & (~x0_flag_q)
                        block_causal = (block_q >= block_kv) & x0_flag_kv & x0_flag_q
                        valid = block_diagonal | offset_block_causal | block_causal

                        qkT[i, j] = T.if_then_else(valid, qkT[i, j], 0)

                    T.copy(dO[bz, k * block_N : (k + 1) * block_N, bx, :], do)
                    T.clear(dsT)
                    T.gemm(V_shared, do, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(qkT, qkT_cast)
                    T.gemm(qkT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow)

                    T.copy(Delta[bz, bx, k * block_N : (k + 1) * block_N], delta)

                    for i, j in T.Parallel(block_M, block_N):
                        dsT_cast[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale
                    T.gemm(dsT_cast, q, dk, policy=T.GemmWarpPolicy.FullRow)

                    T.copy(dsT_cast, dsT_shared)
                    T.clear(dq)
                    T.gemm(dsT_shared, K_shared, dq, transpose_A=True)
                    for i, j in T.Parallel(block_N, dim):
                        T.atomic_add(dQ[bz, k * block_N + i, bx, j], dq[i, j])

                T.copy(dv, dv_shared)
                T.copy(dk, dk_shared)
                T.copy(dv_shared, dV[bz, by * block_M : (by + 1) * block_M, bx, :])
                T.copy(dk_shared, dK[bz, by * block_M : (by + 1) * block_M, bx, :])

        return flash_bwd_block_diff


# ============================================================================
# PyTorch Autograd Function
# ============================================================================

class TileLangBlockDiffAttention(torch.autograd.Function):
    """TileLang Flash Attention with BD3LM block diffusion mask."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_size: int
    ) -> torch.Tensor:
        check_tilelang_available()

        batch, seq_len, heads, dim = q.shape
        block_M = 64
        block_N = 64 if dim <= 128 else 32

        # Ensure contiguous and fp16
        q = q.contiguous().half()
        k = k.contiguous().half()
        v = v.contiguous().half()

        kernel = _flashattn_block_diff_fwd_kernel(
            batch, heads, seq_len, dim, block_size, block_M, block_N
        )
        o, lse = kernel(q, k, v)

        ctx.save_for_backward(q, k, v, o, lse)
        ctx.block_size = block_size
        ctx.block_M = block_M
        ctx.block_N = block_N

        return o

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, o, lse = ctx.saved_tensors
        block_size = ctx.block_size

        batch, seq_len, heads, dim = q.shape
        block_M = ctx.block_M
        block_N = 64 if dim <= 64 else 32

        # Ensure contiguous
        grad_output = grad_output.contiguous().half()

        # Preprocess
        kernel_prep = _flashattn_bwd_preprocess(batch, heads, seq_len, dim)
        delta = kernel_prep(o, grad_output)

        # Backward kernel
        kernel = _flashattn_block_diff_bwd_kernel(
            batch, heads, seq_len, dim, block_size, block_M, block_N
        )

        dq = torch.zeros(batch, seq_len, heads, dim, dtype=torch.float32, device=q.device)
        dk = torch.empty(batch, seq_len, heads, dim, dtype=torch.float16, device=q.device)
        dv = torch.empty(batch, seq_len, heads, dim, dtype=torch.float16, device=q.device)

        kernel(q, k, v, grad_output, lse, delta, dq, dk, dv)

        # Postprocess
        kernel_post = _flashattn_bwd_postprocess(batch, heads, seq_len, dim)
        dq = kernel_post(dq)

        return dq, dk, dv, None


class TileLangCausalAttention(torch.autograd.Function):
    """TileLang Flash Attention with causal mask."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        check_tilelang_available()

        batch, seq_len, heads, dim = q.shape
        block_M = 64
        block_N = 64 if dim <= 128 else 32

        # Ensure contiguous and fp16
        q = q.contiguous().half()
        k = k.contiguous().half()
        v = v.contiguous().half()

        kernel = _flashattn_causal_fwd_kernel(batch, heads, seq_len, dim, block_M, block_N)
        o, lse = kernel(q, k, v)

        ctx.save_for_backward(q, k, v, o, lse)
        ctx.block_M = block_M
        ctx.block_N = block_N

        return o

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # For now, fall back to PyTorch for backward
        # A full implementation would include causal backward kernel
        raise NotImplementedError("Causal backward not yet implemented for TileLang")


# ============================================================================
# High-level API for BD3LM Integration
# ============================================================================

def tilelang_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    is_causal: bool = False
) -> torch.Tensor:
    """
    TileLang flash attention function.

    Args:
        q: Query tensor of shape [batch, seq_len, heads, dim]
        k: Key tensor of shape [batch, seq_len, heads, dim]
        v: Value tensor of shape [batch, seq_len, heads, dim]
        mask: Unused (for API compatibility)
        is_causal: If True, use causal masking

    Returns:
        Output tensor of shape [batch, seq_len, heads, dim]
    """
    if is_causal:
        return TileLangCausalAttention.apply(q, k, v)
    else:
        # Default to block_size=1 for non-block-diff attention
        return TileLangBlockDiffAttention.apply(q, k, v, 1)


def tilelang_block_diff_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int
) -> torch.Tensor:
    """
    TileLang flash attention with BD3LM block diffusion mask.

    This function is specifically designed for BD3LM's block diffusion
    attention pattern which combines:
    - Block Diagonal Mask (M_BD): Self-attention within noised blocks
    - Offset Block Causal Mask (M_OBC): Cross-attention for conditional context
    - Block Causal Mask (M_BC): Attention to update x0

    Args:
        q: Query tensor of shape [batch, seq_len, heads, dim]
           where seq_len = 2n (concatenation of xt and x0)
        k: Key tensor of shape [batch, seq_len, heads, dim]
        v: Value tensor of shape [batch, seq_len, heads, dim]
        block_size: Block size for the block diffusion pattern

    Returns:
        Output tensor of shape [batch, seq_len, heads, dim]
    """
    return TileLangBlockDiffAttention.apply(q, k, v, block_size)


# ============================================================================
# Utility Functions
# ============================================================================

def create_block_diff_mask(
    seq_len: int,
    block_size: int,
    device: torch.device = None
) -> torch.Tensor:
    """
    Create BD3LM block diffusion attention mask.

    The mask is composed of three parts:
    - Block Diagonal Mask (M_BD): Self-attention within noised blocks
    - Offset Block Causal Mask (M_OBC): Cross-attention for conditional context
    - Block Causal Mask (M_BC): Attention to update x0

    Args:
        seq_len: Half of the total sequence length (n in the original code)
        block_size: Block size for the diffusion blocks
        device: Device to create the mask on

    Returns:
        Boolean mask of shape [2*seq_len, 2*seq_len]
    """
    n = seq_len
    total_len = 2 * n

    q_idx = torch.arange(total_len, device=device)[:, None]
    kv_idx = torch.arange(total_len, device=device)[None, :]

    # Indicate whether token belongs to xt or x0
    x0_flag_q = (q_idx >= n)
    x0_flag_kv = (kv_idx >= n)

    # Compute block indices
    block_q = torch.where(
        x0_flag_q,
        (q_idx - n) // block_size,
        q_idx // block_size
    )
    block_kv = torch.where(
        x0_flag_kv,
        (kv_idx - n) // block_size,
        kv_idx // block_size
    )

    # 1. Block Diagonal Mask (M_BD)
    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

    # 2. Offset Block-Causal Mask (M_OBC)
    offset_block_causal = (
        (block_q > block_kv) & x0_flag_kv & (~x0_flag_q)
    )

    # 3. Block-Causal Mask (M_BC)
    block_causal = (block_q >= block_kv) & x0_flag_kv & x0_flag_q

    # Combine masks
    mask = block_diagonal | offset_block_causal | block_causal

    return mask
