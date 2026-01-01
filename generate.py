import os
# Suppress TensorFlow/OneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM

# Register all models
try:
    from models.llada.configuration_llada import LLaDAConfig
    from models.llada.modeling_llada import LLaDAModelLM
    AutoConfig.register("llada", LLaDAConfig)
    AutoModel.register(LLaDAConfig, LLaDAModelLM)
except ImportError:
    pass

try:
    from models.duo.configuration_duo import DUOConfig
    from models.duo.modeling_duo import DUO
    AutoConfig.register("duo", DUOConfig)
    AutoModel.register(DUOConfig, DUO)
except ImportError:
    pass

try:
    from models.bd3lm.configuration_bd3lm import BD3LMConfig
    from models.bd3lm.modeling_bd3lm import BD3LM
    AutoConfig.register("bd3lm", BD3LMConfig)
    AutoModel.register(BD3LMConfig, BD3LM)
except ImportError:
    pass


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    """
    Calculates how many tokens to unmask/fix at each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


# ==============================================================================
# UDM (DUO) - Uniform Diffusion Generation
# ==============================================================================
@torch.no_grad()
def generate_duo(model, prompt, steps=128, gen_length=128, temperature=0., cfg_scale=0.):
    """
    Generates text using Uniform Diffusion (Random Noise -> Text).
    This generates the *entire* sequence simultaneously (Non-Autoregressive).
    """
    b, prompt_len = prompt.shape
    total_len = prompt_len + gen_length
    device = model.device
    vocab_size = model.config.vocab_size

    # 1. Initialize with Random Uniform Noise (Integers)
    x = torch.randint(0, vocab_size, (b, total_len), device=device)
    x[:, :prompt_len] = prompt.clone() # Keep prompt fixed

    for i in range(steps):
        # Time schedule: 1.0 -> 0.0
        t = 1.0 - (i / steps)
        
        # DUO expects 1D timesteps (B,)
        timesteps = torch.full((b,), t, device=device, dtype=torch.float32)
        
        # Model Forward
        logits = model(input_ids=x, timesteps=timesteps, return_dict=True).logits
        
        if cfg_scale > 0:
            # Simple CFG implementation placeholder (requires null condition logic)
            pass

        if temperature > 0:
            logits = add_gumbel_noise(logits, temperature)
        
        # Predict x0
        x0_pred = torch.argmax(logits, dim=-1)
        
        # Only update the generation part
        pred_x0_part = x0_pred[:, prompt_len:]
        
        # Confidence-based update (Denoising)
        # We assume a linear schedule: we fix (i+1)/steps portion of tokens
        probs = F.softmax(logits[:, prompt_len:], dim=-1)
        token_probs = torch.gather(probs, -1, pred_x0_part.unsqueeze(-1)).squeeze(-1)
        
        target_clean_count = int((i + 1) * (gen_length / steps))
        
        # Identify top-k most confident tokens to fix to x0
        _, topk_indices = torch.topk(token_probs, target_clean_count, dim=1)
        update_mask = torch.zeros_like(token_probs, dtype=torch.bool)
        update_mask.scatter_(1, topk_indices, True)
        
        # Refresh noise for tokens not yet fixed
        random_noise = torch.randint(0, vocab_size, (b, gen_length), device=device)
        
        next_x_part = torch.where(update_mask, pred_x0_part, random_noise)
        x[:, prompt_len:] = next_x_part

    return x


# ==============================================================================
# BDM (BD3LM) - Block Autoregressive Generation
# ==============================================================================
@torch.no_grad()
def generate_bdm(model, prompt, steps=64, gen_length=128, block_length=32, temperature=0., mask_id=126336):
    """
    Generates text block-by-block.
    For each block:
      1. Append [MASK] tokens.
      2. Run diffusion process on JUST the new block, conditioning on past blocks.
    """
    b, prompt_len = prompt.shape
    device = model.device
    
    # Start with the prompt as the "History"
    curr_seq = prompt.clone()
    
    num_blocks = gen_length // block_length
    
    for block_idx in range(num_blocks):
        # 1. Append [MASK] tokens for the new block
        mask_block = torch.full((b, block_length), mask_id, dtype=torch.long, device=device)
        curr_seq = torch.cat([curr_seq, mask_block], dim=1)
        
        # Indices of the current block in the full sequence
        start_idx = curr_seq.shape[1] - block_length
        end_idx = curr_seq.shape[1]
        
        # 2. Diffusion Denoising Loop for this block
        # We simulate x_t -> x_0 transition
        
        # We track which tokens in the block are "fixed" (unmasked)
        # Initially, all are masked (mask_index = True)
        block_mask_status = torch.ones((b, block_length), dtype=torch.bool, device=device)
        
        # Calculate schedule for fixing tokens
        # We need to fix ALL tokens by the end of 'steps'
        num_transfer = get_num_transfer_tokens(block_mask_status, steps)
        
        for i in range(steps):
            t = 1.0 - (i / steps)
            
            # Construct Timesteps
            ts_seq = torch.zeros((b, curr_seq.shape[1]), device=device, dtype=model.dtype)
            ts_seq[:, start_idx:end_idx] = t
            
            # Forward Pass
            # BDM needs sample_mode=True to handle the causal masking correctly
            logits = model(input_ids=curr_seq, timesteps=ts_seq, sample_mode=True).logits
            
            # Extract logits for the current block only
            block_logits = logits[:, start_idx:end_idx]
            
            if temperature > 0:
                block_logits = add_gumbel_noise(block_logits, temperature)
            
            x0_pred = torch.argmax(block_logits, dim=-1)
            
            # Confidence Calculation for Remasking
            probs = F.softmax(block_logits, dim=-1)
            token_conf = torch.gather(probs, -1, x0_pred.unsqueeze(-1)).squeeze(-1)
                        
            # Get confidence of currently MASKED tokens
            masked_conf = torch.where(block_mask_status, token_conf, -torch.inf)
            
            # Select top-k confident tokens to fix
            k_transfer = num_transfer[:, i]
            
            # Create a mask of tokens to fix this step
            fix_decision = torch.zeros_like(block_mask_status, dtype=torch.bool)
            
            for batch_i in range(b):
                k = k_transfer[batch_i].item()
                if k > 0:
                    _, top_indices = torch.topk(masked_conf[batch_i], k)
                    fix_decision[batch_i, top_indices] = True
            
            # Update sequence
            # Where fix_decision is True, set to x0_pred
            # Where block_mask_status was False (already fixed), keep existing
            # Where still masked, keep MASK
            
            # Update the underlying sequence tensor
            current_block_ids = curr_seq[:, start_idx:end_idx]
            new_block_ids = torch.where(fix_decision, x0_pred, current_block_ids)
            curr_seq[:, start_idx:end_idx] = new_block_ids
            
            # Update status
            block_mask_status = block_mask_status & (~fix_decision)

        # Ensure block is fully filled
        if block_mask_status.any():
             # One final pass with t=0
             ts_seq = torch.zeros((b, curr_seq.shape[1]), device=device, dtype=model.dtype)
             logits = model(input_ids=curr_seq, timesteps=ts_seq, sample_mode=True).logits
             final_pred = torch.argmax(logits[:, start_idx:end_idx], dim=-1)
             curr_seq[:, start_idx:end_idx] = torch.where(block_mask_status, final_pred, curr_seq[:, start_idx:end_idx])

    return curr_seq


# ==============================================================================
# MDM (LLaDA) - Masked Diffusion Generation
# ==============================================================================
@torch.no_grad()
def generate_llada(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0., mask_id=126336, remasking='low_confidence', cfg_scale=0.):
    """
    Standard Masked Diffusion. Can be pure (block=gen_len) or semi-AR (block<gen_len).
    """
    b = prompt.shape[0]
    device = model.device
    
    # Initialize full sequence with masks
    x = torch.full((b, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt.shape[1]] = prompt.clone()
    
    prompt_len = prompt.shape[1]
    
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks
    
    for num_block in range(num_blocks):
        # Define current block range
        start = prompt_len + num_block * block_length
        end = prompt_len + (num_block + 1) * block_length
        
        # Create mask index for *just this block*
        # (Though x is already initialized with masks, we logically focus here)
        block_mask_index = (x[:, start:end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
        for i in range(steps_per_block):
            # Mask index for the whole sequence (needed for remasking logic)
            full_mask_index = (x == mask_id)
            
            # Forward
            logits = model(x).logits
            
            if cfg_scale > 0:
                # CFG Logic would go here
                pass

            if temperature > 0:
                logits = add_gumbel_noise(logits, temperature)
            
            x0 = torch.argmax(logits, dim=-1)
            
            # Remasking Strategy
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=x0.unsqueeze(-1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand(x0.shape, device=device)
            else:
                raise NotImplementedError(remasking)
            
            # Force future blocks to be ignored (set confidence to -inf)
            x0_p[:, end:] = -np.inf
            
            # Current known state vs Predicted state
            # We want to keep 'x' where it is already unmasked
            # We want to update 'x' where it is masked, using x0
            
            curr_x0 = torch.where(full_mask_index, x0, x)
            confidence = torch.where(full_mask_index, x0_p, -np.inf)
            
            # Select tokens to transfer (unmask)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            
            for j in range(b):
                k = num_transfer_tokens[j, i]
                if k > 0:
                    _, select_index = torch.topk(confidence[j], k)
                    transfer_index[j, select_index] = True
            
            # Update x
            x[transfer_index] = curr_x0[transfer_index]

    return x


# ==============================================================================
# AR (Autoregressive) - Standard HF Generation
# ==============================================================================
@torch.no_grad()
def generate_ar(model, prompt, gen_length=128, temperature=0., **kwargs):
    do_sample = (temperature > 0)
    outputs = model.generate(
        input_ids=prompt,
        max_new_tokens=gen_length,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        pad_token_id=model.config.eos_token_id
    )
    return outputs


# ==============================================================================
# Main Router
# ==============================================================================
@torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    
    model_type = getattr(model.config, 'model_type', '')

    # DUO (Uniform)
    if model_type == 'duo':
        return generate_duo(
            model, prompt, steps, gen_length, temperature, cfg_scale
        )
    
    # BDM (Block Diffusion)
    elif model_type == 'bd3lm':
        # Ensure block_length matches config if possible, else use arg
        blk = getattr(model.config, 'block_size', block_length)
        return generate_bdm(
            model, prompt, steps, gen_length, blk, temperature, mask_id
        )

    # LLaDA (Masked Diffusion)
    elif model_type == 'llada':
        return generate_llada(
            model, prompt, steps, gen_length, block_length, temperature, mask_id, remasking, cfg_scale
        )

    # AR
    else:
        return generate_ar(model, prompt, gen_length, temperature)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--gen_length', type=int, default=128)
    parser.add_argument('--block_length', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.0)
    args = parser.parse_args()

    device = 'cuda'

    # Load Config to determine type
    try:
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # Load Model
    if config.model_type in ['llada', 'duo', 'bd3lm']:
        if config.model_type == 'bd3lm':
            config.attn_backend = 'sdpa' # Force SDPA for sampling
        model = AutoModel.from_pretrained(args.model_path, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

    # Load Tokenizer
    # Force load the base tokenizer to avoid mismatch/gibberish output
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", trust_remote_code=True)
        
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Sample Prompts
    prompts = [ 
        "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
        "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?"
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = [[{"role": "user", "content": p}] for p in prompts]
            prompts = [tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in messages]
        except Exception:
            pass

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    
    print(f"Generating with model type: {config.model_type}...")
    
    out = generate(
        model, 
        inputs['input_ids'], 
        steps=args.steps, 
        gen_length=args.gen_length, 
        block_length=args.block_length,
        temperature=args.temperature,
        mask_id=tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None else 126336
    )

    # Decode
    generated_text = tokenizer.batch_decode(out[:, inputs['input_ids'].shape[1]:], skip_special_tokens=False)
    
    for i, text in enumerate(generated_text):
        print(f"\nPrompt {i+1}: {prompts[i]}")
        print(f"Output: {text}")
        print("-" * 50)

if __name__ == '__main__':
    main()