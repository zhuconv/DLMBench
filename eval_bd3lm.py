import accelerate
import torch
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from models import *

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@register_model("bd3lm_dist")
class BD3LMEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        mask_id=126336,
        max_length=1024,
        batch_size=32,
        mc_num=128,
        device="cuda",
        **kwargs,
    ):
        super().__init__()
        accelerator = accelerate.Accelerator()
        self.accelerator = accelerator if accelerator.num_processes > 1 else None
        
        model_kwargs = {}
        if self.accelerator:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})

        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, **model_kwargs)
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
        else: 
            self.model = self.model.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.mask_id = mask_id
        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        self.max_length = max_length

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape
        target_len = (l - prompt_index.sum()).item()
        
        k = torch.randint(1, target_len + 1, (), device=batch.device)
        
        # Calculate schedule
        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        
        timesteps_1d = (x / target_len) # Shape (B,)
        timesteps_2d = timesteps_1d.unsqueeze(1).repeat(1, l) # Shape (B, L)

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)
        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]
            
        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)
        
        noisy_batch = torch.where(is_mask, self.mask_id, batch)
        
        return noisy_batch, timesteps_1d, timesteps_2d

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            noisy_seq, t_1d, t_2d = self._forward_process(seq, prompt_index)
            
            # Pass 1D timesteps to model
            logits = self.model(input_ids=noisy_seq, timesteps=t_1d).logits
            
            mask_indices = noisy_seq == self.mask_id
            
            # Use 2D timesteps for weighting the loss
            p_mask = t_2d[mask_indices]

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask
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
        raise NotImplementedError
    
    def generate_until(self, requests):
        raise NotImplementedError("Generation not implemented yet for BD3LM eval")

if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()