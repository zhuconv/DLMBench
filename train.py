#    Modification Copyright 2024 Jiajun Zhu
#    Modification Copyright 2024 Zhenyu He
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch
import torch.nn.functional as F
import builtins
import logging
import os
import math
import glob
import random
from itertools import chain
from dataclasses import dataclass, field
from typing import Optional

# import wandb
import transformers
from transformers import Trainer, AutoTokenizer, AutoConfig, AutoModel, PreTrainedTokenizerBase
from datasets import load_dataset, load_from_disk, IterableDataset
from models import *
from typing import Dict, List, Union



CPU_COUNT = os.cpu_count()


@dataclass
class ModelArguments:
    config: Optional[str] = field(default=None)
    # model_name_or_path: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    dataset_cache_dir: str = field(default=None, metadata={"help": "Path to the data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    context_len: int = field(
        default=2048,
        metadata={"help": "Training Context Length."},
    )
    resume_from_checkpoint: Optional[bool] = field(default=None)
    finetune_from_pretrained: Optional[str] = field(default=None)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def get_processed_dataset(tokenizer, data_args, training_args, cached='tokenized'):

    # "../../hf_datasets/SlimPajama-627B"
    dpt = data_args.dataset_cache_dir

    assert cached in ['raw', 'tokenized', 'grouped'], "cached should be one of ['raw', 'tokenized', 'grouped']"
    if cached == 'grouped':
        print("Loading datasets: Grouped")
        lm_datasets = load_dataset(
            "arrow",
            data_files={
                "train": f"{data_args.dataset_cache_dir}/{training_args.context_len}/lm_datasets_train_*.arrow",
                "validation": f"{data_args.dataset_cache_dir}/{training_args.context_len}/lm_datasets_validation_*.arrow",
                "test": f"{data_args.dataset_cache_dir}/{training_args.context_len}/lm_datasets_test_*.arrow"},
            num_proc=CPU_COUNT,
            split=None
        )
        return lm_datasets

    elif cached == 'raw' or cached is None:
        if cached is None:
            raw_datasets = load_dataset("DKYoon/SlimPajama-6B", split=["train", 'validation'])
        else:
            raw_datasets = load_dataset("json",  # 本地路径
                data_files={
                    "train": f"{dpt}/train/*/*.jsonl.zst",
                    "validation": f"{dpt}/validation/*/*.jsonl.zst",
                    "test": f"{dpt}/test/*/*.jsonl.zst"
                },
                num_proc=CPU_COUNT,
                split=None,
            )

        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        def tokenize_function(examples):
            return tokenizer(examples[text_column_name])


        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            num_proc=CPU_COUNT,
            load_from_cache_file=True,
            cache_file_names={"train": f"{data_args.dataset_cache_dir}/tokenized_datasets_train.arrow",\
                "validation": f"{data_args.dataset_cache_dir}/tokenized_datasets_validation.arrow", \
                "test": f"{data_args.dataset_cache_dir}/tokenized_datasets_test.arrow"},
            desc="Running tokenizer on dataset",
        )
    elif cached == 'tokenized':
        tokenized_datasets = load_dataset(
            "arrow",
            data_files={
                "train": f"{data_args.dataset_cache_dir}/tokenized_datasets_train_*.arrow",
                "validation": f"{data_args.dataset_cache_dir}/tokenized_datasets_validation_*.arrow",
                "test": f"{data_args.dataset_cache_dir}/tokenized_datasets_test_*.arrow"
            },
            num_proc=CPU_COUNT,
            split=None
        )

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // training_args.context_len) * training_args.context_len
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + training_args.context_len] for i in range(0, total_length, training_args.context_len)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    os.makedirs(f"{data_args.dataset_cache_dir}/{training_args.context_len}", exist_ok=True)

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=CPU_COUNT,
        load_from_cache_file=True,
        cache_file_names={"train": f"{data_args.dataset_cache_dir}/{training_args.context_len}/lm_datasets_train.arrow",\
            "validation": f"{data_args.dataset_cache_dir}/{training_args.context_len}/lm_datasets_validation.arrow", \
            "test": f"{data_args.dataset_cache_dir}/{training_args.context_len}/lm_datasets_test.arrow"},
        desc=f"Grouping texts in chunks of {training_args.context_len}",
    )
    return lm_datasets

def get_streaming_dataset(tokenizer, data_args, training_args, cached='tokenized'):

    # "../../hf_datasets/SlimPajama-627B"
    dpt = data_args.dataset_cache_dir

    assert cached in ['raw', 'tokenized', 'grouped', None], "cached should be one of ['raw', 'tokenized', 'grouped']"

    if cached == 'grouped':
        print("Loading datasets: Grouped")
        lm_datasets = load_dataset(
            "arrow",
            data_files={
                "train": f"{data_args.dataset_cache_dir}/{training_args.context_len}/lm_datasets_train_*.arrow",
                "validation": f"{data_args.dataset_cache_dir}/{training_args.context_len}/lm_datasets_validation_*.arrow",
                "test": f"{data_args.dataset_cache_dir}/{training_args.context_len}/lm_datasets_test_*.arrow"},
            split=None,
            streaming=True
        )
        return lm_datasets

    elif cached == 'tokenized':
        tokenized_datasets = load_dataset(
            "arrow",
            data_files={
                "train": f"{data_args.dataset_cache_dir}/tokenized_datasets_train_*.arrow",
                "validation": f"{data_args.dataset_cache_dir}/tokenized_datasets_validation_*.arrow",
                "test": f"{data_args.dataset_cache_dir}/tokenized_datasets_test_*.arrow"
            },
            split=None,
            streaming=True
        )

    else: # raw
        if cached is None:
            from datasets import IterableDatasetDict
            train_raw_dataset = load_dataset("DKYoon/SlimPajama-6B", split="train", streaming=True)
            val_raw_dataset = load_dataset("DKYoon/SlimPajama-6B", split='validation', streaming=True)
            raw_datasets = IterableDatasetDict({"train": train_raw_dataset, "validation": val_raw_dataset})
        else:
            assert cached == 'raw', "cached should be one of ['raw', 'tokenized', 'grouped', None]"
            raw_datasets = load_dataset("json",  # 本地路径
                data_files={
                    "train": f"{dpt}/train/*/*.jsonl.zst",
                    "validation": f"{dpt}/validation/*/*.jsonl.zst",
                    "test": f"{dpt}/test/*/*.jsonl.zst"
                },
                # num_proc=CPU_COUNT,
                split=None,
                streaming=True
            )

        def infer_columns_of_dataset(raw_datasets):
            default_cols = raw_datasets.features
        
            if default_cols is not None:
                return list(default_cols)
        
            first_example = next(iter(raw_datasets))
            if isinstance(first_example, dict):
                return list(first_example.keys())
            else:
                raise ValueError(f'Unable to infer column names from the data type: {type(first_example)}')


        # column_names = raw_datasets["train"].column_names
        column_names = infer_columns_of_dataset(raw_datasets["train"])
        text_column_name = "text" if "text" in column_names else column_names[0]

        def tokenize_function(examples):
            return tokenizer(examples[text_column_name])
        

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
        )

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // training_args.context_len) * training_args.context_len
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + training_args.context_len] for i in range(0, total_length, training_args.context_len)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    os.makedirs(f"{data_args.dataset_cache_dir}/{training_args.context_len}", exist_ok=True)

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
    )

    return lm_datasets

@dataclass
class DataCollatorForRandomTimeMask:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int = None         # 可选：方便开启 flash-attn 等
    all_attend: bool = False               # 若 True，则忽略 pad 的 attention_mask，全 1
    avoid_special_masking: bool = True     # 不遮盖特殊 token

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 1) pad（让 tokenizer 生成 attention_mask；也可以 all_attend=True 全 1）
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of
        )
        input_ids = batch["input_ids"]               # (B, L)
        attention_mask = batch["attention_mask"]     # (B, L)

        B, L = input_ids.size()
        device = input_ids.device

        # 2) 采样 t（按样本一个 t），扩展到 (B, L)
        t = torch.rand(B, 1, dtype=torch.float32)              # CPU
        t = torch.clamp_min(t, 1e-4)
        # Create a broadcasted version for masking logic (B, L)
        t_broadcast = t.view(B, 1).repeat(1, L).to(device)

        # 3) 形成可遮盖位置（默认不遮盖特殊符号）
        can_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
        if self.avoid_special_masking:
            special_ids = set(self.tokenizer.all_special_ids)
            if len(special_ids) > 0:
                special_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
                for sid in special_ids:
                    special_mask |= (input_ids == sid)
                can_mask &= ~special_mask
        # 若不想在 padding 上遮盖，也可加：can_mask &= (attention_mask.bool())

        # 4) 以概率 t 进行逐 token 采样，但只允许在 can_mask==True 的位置
        # Use broadcasted t for Bernoulli sampling
        bern = torch.bernoulli(t_broadcast).bool()
        mask = bern & can_mask

        # 5) 生成 corrupted 与 labels

        mask_token_id = self.tokenizer.mask_token_id
        if mask_token_id is None:
            raise ValueError("tokenizer.mask_token_id is None. Please set a mask token for the tokenizer.")
        corrupted = input_ids.masked_fill(mask, mask_token_id)
        labels = input_ids.masked_fill(~mask, -100)  # 只在被遮盖处监督

        if self.all_attend:
            attention_mask = torch.ones_like(attention_mask)

        return {
            "input_ids": corrupted,
            "labels": labels,
            "attention_mask": attention_mask,
            "t": t,  # 传给 Trainer.compute_loss 做 1/t 加权
        }

class RandomTimeMaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 取出 labels 和 t，避免传进 model(**inputs)
        labels = inputs.pop("labels")
        t = inputs.pop("t")  # (B, L)

        if "attention_mask" in inputs:
            inputs.pop("attention_mask")
        
        # Pass 1D timesteps to model
        inputs["timesteps"] = t.to(inputs["input_ids"].device)

        logits = model(**inputs)

        B, L, V = logits.shape

        if labels.shape[1] > L:
            labels = labels[:, :L]
        
        per_tok = F.cross_entropy(
            logits.view(-1, V),
            labels.view(-1),
            reduction="none",
            ignore_index=-100
        ).view(B, L)

        # 1/t 加权；注意 t 与 per_tok 的形状一致
        t_broadcast = t.view(B, 1).to(per_tok.device)
        loss = (per_tok / t_broadcast).mean()

        # 把 t 放回去，以免后续 callback 需要
        inputs["t"] = t.to(torch.bfloat16 if torch.cuda.is_available() else t.dtype)
        if return_outputs:
            return loss, outputs
        return loss

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    def make_rank0_print(training_args):
        def _print(*args, **kwargs):
            if training_args.process_index == 0:
                builtins.print(*args, **kwargs)
        return _print

    #! affecting all processes
    print = make_rank0_print(training_args)

    #! Config and Model
    count_func = lambda model: sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    config = AutoConfig.from_pretrained(model_args.config)
    model = AutoModel.from_config(config)
    if training_args.local_rank == 0:
        print(f"Training new model from scratch - Total Size={count_func(model)/2**20:.2f}M parameters")

    # elif model_args.model_name_or_path:
    #     # config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    #     model = AutoModel.from_pretrained(model_args.model_name_or_path)
    #     # if training_args.local_rank == 0:
    #     print(f"Finetuning model from {model_args.model_name_or_path} - Model Size={count_func(model)/2**20:.2f}M parameters")
    # else:
    #     raise NotImplementedError

    # determine if load from pretrained
    # if training_args.finetune_from_pretrained:
    #     pretrained_model = LlamaForCausalLM.from_pretrained(training_args.finetune_from_pretrained)
    #     checkpoint = pretrained_model.state_dict()
    #     def filter(key):
    #         rotary = 'sin_cached' not in key and 'cos_cached' not in key
    #         post_linear = "post_attention_linears" not in key
    #         pe_proj = "pe.proj" not in key
    #         return all((rotary, post_linear, pe_proj))
    #     filtered_checkpoint = {k: v for k, v in checkpoint.items() if filter(k)}
    #     model.load_state_dict(filtered_checkpoint, strict=False)

    # tokenizer = AutoTokenizer.from_pretrained(
    #     "/cusp-data-efa/peihaow/hf_models/llama-tokenizer",
    #     use_fast=True,
    # )
 
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", use_fast=True)


    MASK_TOKEN = "<|mdm_mask|>"
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)
        tokenizer.mask_token_id = mask_token_id
    print(f"*** Using {tokenizer.convert_ids_to_tokens(mask_token_id)} as mask_token ***")

    # if training_args.local_rank > 0: 
    #     torch.distributed.barrier()

    lm_datasets = get_streaming_dataset(tokenizer, data_args, training_args, cached=None)

    print(f"*** Datasets Loaded ***")

    train_dataset = lm_datasets["train"]
    valid_dataset = lm_datasets["validation"]

    # if training_args.local_rank == 0:
    #     torch.distributed.barrier()
    
    # data_collator = default_data_collator # DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_collator = DataCollatorForRandomTimeMask(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,     # 可按需设置
        all_attend=False,         # 若想“全 1 mask”，设 True
        avoid_special_masking=True
    )

    data_module = dict(
        train_dataset=train_dataset, eval_dataset=valid_dataset, data_collator=data_collator
        )

    # Tell Trainer not to attempt DataParallel
    model.is_parallelizable = True
    model.model_parallel = True

    #! For Iteratable: do not skip streaming dataset but use a new shuffle for resume.
    n_lastest_iter = 0
    if training_args.resume_from_checkpoint == True:
        # search for the latest checkpoint
        from pathlib import Path
        all_checkpoints = list(Path(training_args.output_dir).glob("checkpoint-*"))
        all_checkpoints = [x for x in all_checkpoints if (x / "trainer_state.json").exists() and not x.name.endswith("final")]
        if len(all_checkpoints) == 0:
            training_args.resume_from_checkpoint = None
            print("No checkpoint found, starting from scratch")
        else:
            all_checkpoints = [str(x) for x in all_checkpoints]
            latest_checkpoint = max(all_checkpoints, key=os.path.getctime)
            training_args.resume_from_checkpoint = latest_checkpoint
            print("Resuming from checkpoint", latest_checkpoint)
            n_lastest_iter = int(latest_checkpoint.split('-')[-1])

    if isinstance(train_dataset, IterableDataset):
        shuffle_seed = training_args.data_seed + n_lastest_iter if training_args.data_seed is not None else training_args.seed + n_lastest_iter
        train_dataset = train_dataset.shuffle(seed=shuffle_seed)
        training_args.ignore_data_skip = True
        print("*** Set ignore_data_skip=True for streaming mode to save time ***")


    trainer = RandomTimeMaskTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    model.config.use_cache = False

    if training_args.do_train:
        logging.info("*** Start Training ***")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_state()
        # trainer.save_model(output_dir=training_args.output_dir)
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    if training_args.do_eval:
        logging.info("*** Evaluate on valid set***")
        metrics = trainer.evaluate(eval_dataset=valid_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        

if __name__ == "__main__":
    # wandb.init(
    #     project="IBSSM",
    #     entity="jiajun_vita",
    #     id=os.getenv("SLURM_JOB_NAME", "interact"),
    #     resume='allow',
    #     )
    transformers.logging.set_verbosity_warning()
    train()