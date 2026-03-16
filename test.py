"""Test script for BD3LM (block diffusion) implementation.

Tests:
1. Model instantiation and parameter count
2. Flex attention mask creation
3. Forward pass correctness
4. Training speed benchmark (compared with LLaDA/MDM)
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

# Import local models
sys.path.insert(0, os.path.dirname(__file__))
from models import *


def test_model_instantiation():
    """Test BD3LM model creation and parameter count."""
    print("=" * 60)
    print("TEST 1: Model Instantiation & Parameter Count")
    print("=" * 60)

    config = AutoConfig.from_pretrained(
        "./configs/bdm_700m.json", trust_remote_code=True)
    model = AutoModel.from_config(config, trust_remote_code=True)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model type: {config.model_type}")
    print(f"  Architecture: Pure Transformer (LLaMA-style, no timestep conditioning)")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Num layers: {config.n_blocks}")
    print(f"  Num heads: {config.n_heads}")
    print(f"  Head dim: {config.head_dim}")
    print(f"  Intermediate size: {config.intermediate_size}")
    print(f"  Block size: {config.block_size}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Attention backend: {config.attn_backend}")
    print(f"  Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")
    print()

    # Also check LLaDA for comparison
    config_mdm = AutoConfig.from_pretrained(
        "./configs/mdm_700m.json", trust_remote_code=True)
    model_mdm = AutoModel.from_config(config_mdm, trust_remote_code=True)
    total_params_mdm = sum(p.numel() for p in model_mdm.parameters())
    print(f"  LLaDA (MDM) parameters: {total_params_mdm:,} ({total_params_mdm / 1e6:.1f}M)")
    print(f"  Parameter ratio BD3LM/LLaDA: {total_params / total_params_mdm:.3f}")
    print()

    return model, config


def test_flex_attention():
    """Test flex attention mask creation."""
    print("=" * 60)
    print("TEST 2: Flex Attention Support")
    print("=" * 60)

    try:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask
        print("  FlexAttention: AVAILABLE")
    except ImportError:
        print("  FlexAttention: NOT AVAILABLE (requires PyTorch >= 2.5)")
        return False

    from models.bd3lm.modeling_bd3lm import block_diff_mask, FLEX_ATTN_AVAILABLE
    from functools import partial

    print(f"  FLEX_ATTN_AVAILABLE flag: {FLEX_ATTN_AVAILABLE}")

    # Test mask creation
    L, block_size = 1024, 128
    start = time.time()
    mask = create_block_mask(
        partial(block_diff_mask, block_size=block_size, n=L),
        B=None, H=None, Q_LEN=L * 2, KV_LEN=L * 2)
    elapsed = time.time() - start
    print(f"  Mask creation time (L={L}, block_size={block_size}): {elapsed:.3f}s")
    print(f"  Mask type: {type(mask).__name__}")

    # Test with full model length
    L, block_size = 2048, 128
    start = time.time()
    mask = create_block_mask(
        partial(block_diff_mask, block_size=block_size, n=L),
        B=None, H=None, Q_LEN=L * 2, KV_LEN=L * 2)
    elapsed = time.time() - start
    print(f"  Mask creation time (L={L}, block_size={block_size}): {elapsed:.3f}s")
    print()
    return True


def test_forward_pass(model, config):
    """Test forward pass on single GPU."""
    print("=" * 60)
    print("TEST 3: Forward Pass (Single GPU)")
    print("=" * 60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device).to(torch.bfloat16)
    model.eval()

    B, L = 2, config.model_length
    mask_token_id = 126336  # LLaDA mask token

    # Create dummy [xt | x0] input
    x0 = torch.randint(0, config.vocab_size, (B, L), device=device)
    # Mask 50% of tokens
    mask_indices = torch.rand(B, L, device=device) < 0.5
    xt = x0.clone()
    xt[mask_indices] = mask_token_id
    model_input = torch.cat([xt, x0], dim=1)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        start = time.time()
        output = model(input_ids=model_input)
        torch.cuda.synchronize()
        elapsed = time.time() - start

    logits = output.logits if hasattr(output, 'logits') else output
    print(f"  Input shape: {model_input.shape}")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Expected: ({B}, {L}, {config.vocab_size})")
    assert logits.shape == (B, L, config.vocab_size), \
        f"Shape mismatch: {logits.shape} vs ({B}, {L}, {config.vocab_size})"
    print(f"  Shape check: PASSED")
    print(f"  Forward time: {elapsed:.3f}s")
    print()

    model = model.cpu()
    torch.cuda.empty_cache()


def benchmark_training_speed(config_path, model_name, num_steps=10):
    """Benchmark training speed for a model."""
    device = torch.device("cuda:0")

    config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
    if not hasattr(config, 'use_cache'):
        config.use_cache = False
    model = AutoModel.from_config(config, trust_remote_code=True)
    model = model.to(device).to(torch.bfloat16)
    model.train()

    # Enable gradient checkpointing
    if hasattr(model, 'gradient_checkpointing_enable'):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    total_params = sum(p.numel() for p in model.parameters())

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Model-specific setup
    if "bd" in config_path.lower():
        seq_len = config.model_length
        B = 4
        mask_token_id = 126336

        def make_batch():
            x0 = torch.randint(0, config.vocab_size, (B, seq_len), device=device)
            mask_indices = torch.rand(B, seq_len, device=device) < 0.5
            xt = x0.clone()
            xt[mask_indices] = mask_token_id
            model_input = torch.cat([xt, x0], dim=1)
            return model_input, x0, mask_indices
    else:
        seq_len = min(getattr(config, 'max_sequence_length', 2048), 2048)
        B = 4
        mask_token_id = 126336

        def make_batch():
            x0 = torch.randint(0, config.vocab_size, (B, seq_len), device=device)
            mask_indices = torch.rand(B, seq_len, device=device) < 0.5
            xt = x0.clone()
            xt[mask_indices] = mask_token_id
            return xt, x0, mask_indices

    # Warmup
    for _ in range(3):
        model_input, x0, mask_indices = make_batch()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(input_ids=model_input)
            logits = output.logits if hasattr(output, 'logits') else output
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                x0.reshape(-1))
        loss.backward()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    # Benchmark
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    total_tokens = 0

    for step in range(num_steps):
        model_input, x0, mask_indices = make_batch()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(input_ids=model_input)
            logits = output.logits if hasattr(output, 'logits') else output
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                x0.reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_tokens += B * seq_len

    torch.cuda.synchronize()
    elapsed = time.time() - start
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    tokens_per_sec = total_tokens / elapsed
    steps_per_sec = num_steps / elapsed

    print(f"  {model_name}:")
    print(f"    Parameters: {total_params / 1e6:.1f}M")
    print(f"    Seq length: {seq_len}, Batch size: {B}")
    print(f"    Steps: {num_steps}, Time: {elapsed:.2f}s")
    print(f"    Speed: {steps_per_sec:.2f} steps/s, {tokens_per_sec:.0f} tokens/s")
    print(f"    Peak GPU memory: {peak_mem:.2f} GB")
    print()

    # Cleanup
    del model, optimizer
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "params_m": total_params / 1e6,
        "seq_len": seq_len,
        "batch_size": B,
        "steps": num_steps,
        "time_s": elapsed,
        "steps_per_sec": steps_per_sec,
        "tokens_per_sec": tokens_per_sec,
        "peak_mem_gb": peak_mem,
    }


def main():
    print("\n" + "=" * 60)
    print("  BD3LM (Block Diffusion) Implementation Test")
    print("  Ported from WindowDiffusion")
    print("=" * 60 + "\n")

    # Test 1: Model instantiation
    model, config = test_model_instantiation()

    # Test 2: Flex attention
    flex_ok = test_flex_attention()

    # Test 3: Forward pass
    if torch.cuda.is_available():
        test_forward_pass(model, config)
    else:
        print("SKIP: No GPU available for forward pass test\n")

    # Test 4: Training speed benchmark
    if torch.cuda.is_available():
        print("=" * 60)
        print("TEST 4: Training Speed Benchmark (Single GPU)")
        print("=" * 60)

        num_steps = 10
        results = []

        # BD3LM (Block Diffusion)
        r1 = benchmark_training_speed(
            "./configs/bdm_700m.json", "BD3LM (Block Diffusion)", num_steps)
        results.append(r1)

        # LLaDA (Masked Diffusion)
        r2 = benchmark_training_speed(
            "./configs/mdm_700m.json", "LLaDA (Masked Diffusion)", num_steps)
        results.append(r2)

        # Summary
        print("=" * 60)
        print("SPEED COMPARISON SUMMARY")
        print("=" * 60)
        print(f"  {'Model':<30} {'Params':>8} {'tok/s':>10} {'step/s':>8} {'Mem':>8}")
        print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")
        for r in results:
            print(f"  {r['model']:<30} {r['params_m']:>6.1f}M "
                  f"{r['tokens_per_sec']:>9.0f} "
                  f"{r['steps_per_sec']:>7.2f} "
                  f"{r['peak_mem_gb']:>6.2f}GB")

        if len(results) >= 2:
            ratio = results[0]['tokens_per_sec'] / results[1]['tokens_per_sec']
            print(f"\n  BD3LM / LLaDA speed ratio: {ratio:.3f}x")
        print()

    print("All tests completed!")


if __name__ == "__main__":
    main()
