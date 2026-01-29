import accelerate
import torch
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig
from generate import generate
from noise import LogLinearNoise

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@register_model("bdm_dist")
class BD3LMEvalHarness(LM):
    def __init__(
        self,
        pretrained='',
        mask_id=126336,
        max_length=1024,
        batch_size=32,
        mc_num=128,
        device="cuda",
        dtype=torch.bfloat16,
        # Generation args
        steps=64,
        gen_length=128,
        block_length=32,
        **kwargs,
    ):
        super().__init__()
        self.accelerator = accelerate.Accelerator()
        
        model_kwargs = {}
        if self.accelerator.num_processes > 1:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})

        config = AutoConfig.from_pretrained(pretrained, trust_remote_code=True, torch_dtype=dtype, **model_kwargs)
        config.attn_backend = 'sdpa'
        
        self.model = AutoModel.from_pretrained(pretrained, config=config, trust_remote_code=True, torch_dtype=dtype, **model_kwargs)
        self.model.eval()

        if self.accelerator.num_processes > 1:
            self.model = self.accelerator.prepare(self.model)
        else:
            self.model = self.model.to(self.accelerator.device)

        self.device = self.accelerator.device
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        
        if self.tokenizer.mask_token_id is not None:
            self.mask_id = self.tokenizer.mask_token_id
            print(f"Using tokenizer mask_id: {self.mask_id}")
        else:
            self.mask_id = mask_id
            print(f"Using default mask_id: {self.mask_id}")
            
        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        self.dtype = dtype
        
        # Generation parameters
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = getattr(config, 'block_size', block_length)
        
        self.noise = LogLinearNoise()

    def _sigma_from_p(self, p):
        return torch.min(- torch.log(1 - p), self.noise.sigma_max)

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape
        target_len = (l - prompt_index.sum()).item()
        
        k = torch.randint(1, target_len + 1, (), device=batch.device)
        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        timesteps_1d = (x / target_len) 

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)
        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]
            
        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)
        
        xt = torch.where(is_mask, self.mask_id, batch)
        
        model_input = torch.cat([xt, batch], dim=1)
        
        p_mask = timesteps_1d.unsqueeze(1).repeat(1, l)
        
        # Convert t (probability p) to sigma
        sigma = self._sigma_from_p(timesteps_1d)
        
        return model_input, sigma.to(dtype=self.dtype), p_mask.to(dtype=self.dtype), is_mask

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            model_input, sigma, t_2d, is_mask = self._forward_process(seq, prompt_index)
            
            # Pass sigma as timesteps
            logits = self.model(input_ids=model_input, timesteps=sigma, sample_mode=False).logits
            
            loss = F.cross_entropy(logits[is_mask], seq[is_mask], reduction='none')
            
            # Weighting by 1/t (which is -loss_scaling for LogLinearNoise)
            loss = loss / t_2d[is_mask]
            
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix = self.tokenizer(e["prefix"])["input_ids"]
            target = self.tokenizer(e["target"])["input_ids"]
            return {"prefix": prefix, "target": target}

        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds).map(_tokenize)
        
        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = torch.tensor(elem["prefix"], device=self.device)
                target = torch.tensor(elem["target"], device=self.device)
                ll = self.get_loglikelihood(prefix, target)
                out.append((ll, 0.0))
        return out

    def loglikelihood_rolling(self, requests):
        out = []
        for (string,) in tqdm(requests, desc="Computing rolling likelihood..."):
            tokens = self.tokenizer(string, return_tensors="pt")["input_ids"].to(self.device)
            seq = tokens.view(-1)
            
            total_ll = 0.0
            for i in range(0, len(seq), self.block_length):
                target_block = seq[i : i + self.block_length]
                if i == 0:
                    history = torch.tensor([], dtype=torch.long, device=self.device)
                else:
                    history_start = max(0, i - (self.max_length - self.block_length))
                    history = seq[history_start:i]
                
                block_ll = self.get_loglikelihood(history, target_block)
                total_ll += block_ll
            out.append(total_ll)
            
        return out
    
    def generate_until(self, requests):
        def _tokenize(e):
            return {
                "question": self.tokenizer(e["question"])["input_ids"],
                "question_text": e["question"],
                "until": e["until"],
            }

        ds = [{"question": req.args[0], "until": req.args[1]['until']} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        out = []
        for elem in tqdm(ds, desc="Generating..."):
            prompt = elem["question"].unsqueeze(0).to(self.device)
            stop_tokens = elem["until"]
 
            generated_answer = generate(
                self.model, 
                prompt, 
                steps=self.steps, 
                gen_length=self.gen_length, 
                block_length=self.block_length,
                temperature=0, 
                mask_id=self.mask_id
            )
            
            # Decode
            # Generated answer contains [Prompt + Gen]
            gen_only = generated_answer[0][prompt.shape[1]:]
            decoded_text = self.tokenizer.decode(gen_only, skip_special_tokens=False)
            
            # Handle stop tokens
            for stop_seq in stop_tokens:
                if stop_seq in decoded_text:
                    decoded_text = decoded_text.split(stop_seq)[0]

            # Encode and decode to ensure clean text output
            generated_answer_ids = self.tokenizer(decoded_text, add_special_tokens=False)["input_ids"]
            final_text = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
            
            out.append(final_text)

            if self.accelerator.num_processes > 1:
                self.accelerator.wait_for_everyone()

        return out

if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()