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

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("udm_dist")
class DuoEvalHarness(LM):
    def __init__(
        self,
        pretrained='',
        mask_id=126336,
        max_length=1024,
        batch_size=32,
        mc_num=128,
        is_check_greedy=False,
        cfg=0.,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        remasking='low_confidence',
        device="cuda",
        **kwargs,
    ):
        '''
        Implementation of the Uniform Diffusion Model (DUO) evaluation harness.
        '''
        super().__init__()

        self.accelerator = accelerate.Accelerator()
        
        model_kwargs = {}
        if self.accelerator.num_processes > 1:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})

        config = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
        self.vocab_size = config.vocab_size

        self.model = AutoModel.from_pretrained(pretrained, trust_remote_code=True, torch_dtype=torch.bfloat16, **model_kwargs)
        self.model.eval()

        if self.accelerator.num_processes > 1:
            self.model = self.accelerator.prepare(self.model)
        else:
            self.model = self.model.to(self.accelerator.device)

        self.device = self.accelerator.device
        self._rank = self.accelerator.local_process_index
        self._world_size = self.accelerator.num_processes

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        self.mask_id = mask_id 

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.cfg = cfg
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking    

    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    def _forward_process(self, batch, prompt_index):
        """
        Applies Uniform Corruption.
        Returns 1D timesteps (one per sequence) to match DUO architecture.
        """
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        random_noise = torch.randint(0, self.vocab_size, batch.shape, device=batch.device)
        
        noisy_batch = torch.where(is_mask, random_noise, batch)

        timesteps = (x / target_len)
        
        return noisy_batch, timesteps, is_mask

    @torch.no_grad()
    def get_logits(self, batch, timesteps):
        logits = self.model(input_ids=batch, timesteps=timesteps, return_dict=True).logits
        return logits

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            noisy_seq, timesteps, is_mask = self._forward_process(seq, prompt_index)

            logits = self.get_logits(noisy_seq, timesteps)

            p_mask = timesteps.unsqueeze(1).expand_as(is_mask)
            
            loss = F.cross_entropy(logits[is_mask], seq[is_mask], reduction='none')
            
            loss = loss / p_mask[is_mask]
            
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prefix, target = prefix.to(self.device), target.to(self.device)
        
        seq[0, :len(prefix)] = prefix
        
        random_target = torch.randint(0, self.vocab_size, (len(target),), device=self.device)
        seq[0, len(prefix):] = random_target
        
        timesteps = torch.ones((1,), dtype=torch.float, device=self.device)

        logits = self.get_logits(seq, timesteps)
        
        target_logits = logits[0, len(prefix):]
        x0 = torch.argmax(target_logits, dim=-1)

        correct = target == x0
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        
        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests: list[Instance]):
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
                cfg_scale=self.cfg, 
                remasking=self.remasking, 
                mask_id=self.mask_id
            )
            
            generated_answer = self.tokenizer.decode(generated_answer[0][prompt.shape[1]:], skip_special_tokens=False)
            for stop_seq in stop_tokens:
                    if stop_seq in generated_answer:
                        generated_answer = generated_answer.split(stop_seq)[0]

            generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            generated_answer = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
            out.append(generated_answer)

            self.accelerator.wait_for_everyone()

        return out


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()