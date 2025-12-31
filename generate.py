import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM

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


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens

@torch.no_grad()
def generate_duo(model, prompt, steps=128, gen_length=128, temperature=0., cfg_scale=0., vocab_size=50258):
    b, prompt_len = prompt.shape
    total_len = prompt_len + gen_length
    device = model.device

    x = torch.randint(0, vocab_size, (b, total_len), device=device)
    x[:, :prompt_len] = prompt.clone()    
    for i in range(steps):
        t = 1.0 - (i / steps)
        
        timesteps = torch.full((b,), t, device=device, dtype=torch.float32)
        
        if cfg_scale > 0.:
            logits = model(input_ids=x, timesteps=timesteps, return_dict=True).logits
        else:
            logits = model(input_ids=x, timesteps=timesteps, return_dict=True).logits

        if temperature > 0:
            logits = add_gumbel_noise(logits, temperature)
        
        x0_pred = torch.argmax(logits, dim=-1)
        pred_x0_part = x0_pred[:, prompt_len:]
                
        probs = F.softmax(logits[:, prompt_len:], dim=-1)
        token_probs = torch.gather(probs, -1, pred_x0_part.unsqueeze(-1)).squeeze(-1)
        
        target_clean_count = int((i + 1) * (gen_length / steps))        
        _, topk_indices = torch.topk(token_probs, target_clean_count, dim=1)
        
        update_mask = torch.zeros_like(token_probs, dtype=torch.bool)
        update_mask.scatter_(1, topk_indices, True)
        
        random_noise = torch.randint(0, vocab_size, (b, gen_length), device=device)
        
        next_x_part = torch.where(update_mask, pred_x0_part, random_noise)
        x[:, prompt_len:] = next_x_part

    timesteps = torch.zeros((b,), device=device, dtype=torch.float32)
    logits = model(input_ids=x, timesteps=timesteps, return_dict=True).logits
    final_x0 = torch.argmax(logits, dim=-1)
    x[:, prompt_len:] = final_x0[:, prompt_len:]
    
    return x

@torch.no_grad()
def generate_ar(model, prompt, gen_length=128, temperature=0., **kwargs):
    if temperature == 0:
        do_sample = False
    else:
        do_sample = True
    
    outputs = model.generate(
        input_ids=prompt,
        max_new_tokens=gen_length,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        pad_token_id=model.config.eos_token_id
    )
    return outputs

@torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False):
    
    model_type = getattr(model.config, 'model_type', '')

    if model_type == 'duo':
        return generate_duo(
            model=model,
            prompt=prompt,
            steps=steps,
            gen_length=gen_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            vocab_size=model.config.vocab_size
        )
    
    if model_type != 'llada':
        return generate_ar(model, prompt, gen_length=gen_length, temperature=temperature)
    
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            if logits_eos_inf:
                if logits.shape[-1] > 126081:
                    logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) 
            
            if confidence_eos_eot_inf:
                if logits.shape[-1] > 126348:
                    logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) 
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='GSAI-ML/LLaDA-8B-Instruct')
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--gen_length', type=int, default=128)
    parser.add_argument('--block_length', type=int, default=128)
    args = parser.parse_args()

    device = 'cuda'

    try:
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        if config.model_type in ['llada', 'duo']:
            model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model.config.model_type != 'duo':
        if tokenizer.pad_token_id is not None:
             pass

    prompts = [ "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
             "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?"]

    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt} for prompt in prompts]
        try:
            prompts = [tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False) for message in messages]
        except Exception:
            pass 

    encoded_outputs = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt"
    )
    input_ids = encoded_outputs['input_ids'].to(device)
    attention_mask = encoded_outputs['attention_mask'].to(device)

    out = generate(model, input_ids, attention_mask, steps=args.steps, gen_length=args.gen_length, block_length=args.block_length)
    output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    for o in output:
        print(o)
        print('-' * 50)

if __name__ == '__main__':
    main()
