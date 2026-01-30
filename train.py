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

import builtins
import csv
import glob
import inspect
import logging
import math
import os
import random
from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F

# import wandb
import transformers
from datasets import IterableDataset, load_dataset, load_from_disk
from deepspeed.runtime.fp16.loss_scaler import LossScaler
from torch.serialization import add_safe_globals
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
)

from models import *
from noise import LogLinearNoise

add_safe_globals([LossScaler])


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
    # resume_from_checkpoint: Optional[bool] = field(default=None)
    finetune_from_pretrained: Optional[str] = field(default=None)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def get_processed_dataset(tokenizer, data_args, training_args, cached="tokenized"):
    # "../../hf_datasets/SlimPajama-627B"
    dpt = data_args.dataset_cache_dir

    assert cached in ["raw", "tokenized", "grouped", None], (
        "cached should be one of ['raw', 'tokenized', 'grouped', None]"
    )
    if cached == "grouped":
        print("Loading datasets: Grouped")
        lm_datasets = load_dataset(
            "arrow",
            data_files={
                "train": f"{data_args.dataset_cache_dir}/{training_args.context_len}/lm_datasets_train_*.arrow",
                "validation": f"{data_args.dataset_cache_dir}/{training_args.context_len}/lm_datasets_validation_*.arrow",
                "test": f"{data_args.dataset_cache_dir}/{training_args.context_len}/lm_datasets_test_*.arrow",
            },
            num_proc=CPU_COUNT,
            split=None,
        )
        return lm_datasets

    elif cached == "raw" or cached is None:
        if cached is None:
            raw_datasets = load_dataset("DKYoon/SlimPajama-6B", split=None)
        else:
            raw_datasets = load_dataset(
                "json",  # 本地路径
                data_files={
                    "train": f"{dpt}/train/*/*.jsonl.zst",
                    "validation": f"{dpt}/validation/*/*.jsonl.zst",
                    "test": f"{dpt}/test/*/*.jsonl.zst",
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
            cache_file_names={
                "train": f"{data_args.dataset_cache_dir}/tokenized_datasets_train.arrow",
                "validation": f"{data_args.dataset_cache_dir}/tokenized_datasets_validation.arrow",
                "test": f"{data_args.dataset_cache_dir}/tokenized_datasets_test.arrow",
            },
            desc="Running tokenizer on dataset",
        )
    elif cached == "tokenized":
        tokenized_datasets = load_dataset(
            "arrow",
            data_files={
                "train": f"{data_args.dataset_cache_dir}/tokenized_datasets_train_*.arrow",
                "validation": f"{data_args.dataset_cache_dir}/tokenized_datasets_validation_*.arrow",
                "test": f"{data_args.dataset_cache_dir}/tokenized_datasets_test_*.arrow",
            },
            num_proc=CPU_COUNT,
            split=None,
        )

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (
            total_length // training_args.context_len
        ) * training_args.context_len
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + training_args.context_len]
                for i in range(0, total_length, training_args.context_len)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    os.makedirs(
        f"{data_args.dataset_cache_dir}/{training_args.context_len}", exist_ok=True
    )

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=CPU_COUNT,
        load_from_cache_file=True,
        cache_file_names={
            "train": f"{data_args.dataset_cache_dir}/{training_args.context_len}/lm_datasets_train.arrow",
            "validation": f"{data_args.dataset_cache_dir}/{training_args.context_len}/lm_datasets_validation.arrow",
            "test": f"{data_args.dataset_cache_dir}/{training_args.context_len}/lm_datasets_test.arrow",
        },
        desc=f"Grouping texts in chunks of {training_args.context_len}",
    )
    return lm_datasets


def get_streaming_dataset(tokenizer, data_args, training_args, cached="tokenized"):
    # "../../hf_datasets/SlimPajama-627B"
    dpt = data_args.dataset_cache_dir

    assert cached in ["raw", "tokenized", "grouped", None], (
        "cached should be one of ['raw', 'tokenized', 'grouped']"
    )

    if cached == "grouped":
        print("Loading datasets: Grouped")
        lm_datasets = load_dataset(
            "arrow",
            data_files={
                "train": f"{data_args.dataset_cache_dir}/{training_args.context_len}/lm_datasets_train_*.arrow",
                "validation": f"{data_args.dataset_cache_dir}/{training_args.context_len}/lm_datasets_validation_*.arrow",
                "test": f"{data_args.dataset_cache_dir}/{training_args.context_len}/lm_datasets_test_*.arrow",
            },
            split=None,
            streaming=True,
        )
        return lm_datasets

    elif cached == "tokenized":
        tokenized_datasets = load_dataset(
            "arrow",
            data_files={
                "train": f"{data_args.dataset_cache_dir}/tokenized_datasets_train_*.arrow",
                "validation": f"{data_args.dataset_cache_dir}/tokenized_datasets_validation_*.arrow",
                "test": f"{data_args.dataset_cache_dir}/tokenized_datasets_test_*.arrow",
            },
            split=None,
            streaming=True,
        )

    else:  # raw
        if cached is None:
            from datasets import IterableDatasetDict

            train_raw_dataset = load_dataset(
                "DKYoon/SlimPajama-6B", split="train", streaming=True
            )
            val_raw_dataset = load_dataset(
                "DKYoon/SlimPajama-6B", split="validation", streaming=True
            )
            raw_datasets = IterableDatasetDict(
                {"train": train_raw_dataset, "validation": val_raw_dataset}
            )
        else:
            assert cached == "raw", (
                "cached should be one of ['raw', 'tokenized', 'grouped', None]"
            )
            raw_datasets = load_dataset(
                "json",  # 本地路径
                data_files={
                    "train": f"{dpt}/train/*/*.jsonl.zst",
                    "validation": f"{dpt}/validation/*/*.jsonl.zst",
                    "test": f"{dpt}/test/*/*.jsonl.zst",
                },
                # num_proc=CPU_COUNT,
                split=None,
                streaming=True,
            )

        def infer_columns_of_dataset(raw_datasets):
            default_cols = raw_datasets.features

            if default_cols is not None:
                return list(default_cols)

            first_example = next(iter(raw_datasets))
            if isinstance(first_example, dict):
                return list(first_example.keys())
            else:
                raise ValueError(
                    f"Unable to infer column names from the data type: {type(first_example)}"
                )

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
        total_length = (
            total_length // training_args.context_len
        ) * training_args.context_len
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + training_args.context_len]
                for i in range(0, total_length, training_args.context_len)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    os.makedirs(
        f"{data_args.dataset_cache_dir}/{training_args.context_len}", exist_ok=True
    )

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
    )

    return lm_datasets


@dataclass
class DataCollatorForMaskedDiffusion:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        input_ids = batch["input_ids"]
        B, L = input_ids.shape

        t = torch.rand(B, 1, device=input_ids.device)
        t = torch.clamp(t, 1e-4, 1.0)

        mask_prob = t.expand(B, L)
        mask_indices = torch.bernoulli(mask_prob).bool()

        labels = input_ids.clone()
        input_ids = input_ids.clone()
        input_ids[mask_indices] = self.tokenizer.mask_token_id

        labels[~mask_indices] = -100

        # LLaDA/MDM loss weight is -1/t
        loss_scale = -1.0 / t
        # Expand loss_scale to match (B, L) for the trainer
        loss_scale = loss_scale.expand(B, L).clone()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": batch["attention_mask"],
            "loss_scale": loss_scale,
        }


@dataclass
class DataCollatorForUniformDiffusion:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        input_ids = batch["input_ids"]
        B, L = input_ids.shape

        t = torch.rand(B, 1, device=input_ids.device)
        t = torch.clamp(t, 1e-4, 1.0)

        mask_prob = t.expand(B, L)
        corrupt_indices = torch.bernoulli(mask_prob).bool()

        # Replace with RANDOM tokens from vocab
        random_noise = torch.randint(
            0, self.tokenizer.vocab_size, (B, L), device=input_ids.device
        )
        corrupted_ids = torch.where(corrupt_indices, random_noise, input_ids)

        labels = input_ids.clone()
        labels[~corrupt_indices] = -100

        # UDM loss weight is -1/t
        loss_scale = -1.0 / t
        loss_scale = loss_scale.expand(B, L).clone()

        return {
            "input_ids": corrupted_ids,
            "labels": labels,
            "attention_mask": batch["attention_mask"],
            "loss_scale": loss_scale,
            "timesteps": t.squeeze(-1),
        }


@dataclass
class DataCollatorForBlockDiffusion:
    tokenizer: PreTrainedTokenizerBase
    block_size: int = 32
    pad_to_multiple_of: int = None

    def __post_init__(self):
        self.noise = LogLinearNoise()

    def _sigma_from_p(self, p):
        return torch.min(-torch.log(1 - p), self.noise.sigma_max)

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        input_ids = batch["input_ids"]
        B, L = input_ids.shape

        num_blocks = (L + self.block_size - 1) // self.block_size

        # Sample timesteps
        sampling_eps_min = 1e-3
        sampling_eps_max = 1.0
        t_blocks = torch.rand(B, num_blocks, device=input_ids.device)
        t_blocks = t_blocks * (sampling_eps_max - sampling_eps_min) + sampling_eps_min

        # Expand to token level
        t_expanded = t_blocks.repeat_interleave(self.block_size, dim=1)[:, :L]

        # Get noise params
        loss_scale, p = self.noise(t_expanded)

        # Create noisy input (xt)
        mask_indices = torch.rand_like(p) < p
        xt = input_ids.clone()
        xt[mask_indices] = self.tokenizer.mask_token_id
        sigma = self._sigma_from_p(p)

        # 5. Dual-Stream concatenation [xt, x0]
        model_input = torch.cat([xt, input_ids], dim=1)  # Shape: (B, 2L)

        # The clean history (x0) has 0 noise, so sigma=0
        sigma_clean = torch.zeros_like(sigma)
        sigma_full = torch.cat([sigma, sigma_clean], dim=1)  # Shape: (B, 2L)

        loss_scale_clean = torch.zeros_like(loss_scale)
        loss_scale_full = torch.cat(
            [loss_scale, loss_scale_clean], dim=1
        )  # Shape: (B, 2L)

        # Create labels
        labels = torch.full(
            (B, 2 * L), -100, dtype=input_ids.dtype, device=input_ids.device
        )
        labels_xt = input_ids.clone()
        labels_xt[~mask_indices] = -100
        labels[:, :L] = labels_xt

        return {
            "input_ids": model_input,
            "labels": labels,
            "timesteps": sigma_full,
            "loss_scale": loss_scale_full,
        }


class DiffusionTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.pop("labels")
        loss_scale = inputs.pop("loss_scale")

        timesteps = inputs.pop("timesteps", None)

        if "attention_mask" in inputs:
            inputs.pop("attention_mask")

        # Check if model accepts timesteps
        model_module = model.module if hasattr(model, "module") else model
        forward_params = inspect.signature(model_module.forward).parameters

        if "timesteps" in forward_params and timesteps is not None:
            inputs["timesteps"] = timesteps

        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        if logits.shape[1] != labels.shape[1]:
            labels = labels[:, : logits.shape[1]]
            loss_scale = loss_scale[:, : logits.shape[1]]

        B, L, V = logits.shape

        # Get log probabilities of the target tokens
        # Flatten for gather
        logits_flat = logits.reshape(-1, V)
        labels_flat = labels.reshape(-1)

        # We only care about labels that are NOT -100
        valid_mask = labels_flat != -100

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Compute log_softmax
        log_probs = F.log_softmax(logits_flat, dim=-1)

        # Gather the log-prob of the correct token
        target_log_probs = log_probs[
            torch.arange(logits_flat.shape[0], device=logits.device), labels_flat
        ]

        # Reshape back to (B, L)
        target_log_probs = target_log_probs.view(B, L)
        valid_mask = valid_mask.view(B, L)

        # Scale by loss scale
        per_tok_loss = loss_scale * target_log_probs

        # Mask out ignored tokens
        per_tok_loss = per_tok_loss * valid_mask.float()

        loss = per_tok_loss.sum() / (valid_mask.sum() + 1e-6)

        if return_outputs:
            return loss, outputs
        return loss


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    def make_rank0_print(training_args):
        def _print(*args, **kwargs):
            if training_args.process_index == 0:
                builtins.print(*args, **kwargs)

        return _print

    #! affecting all processes
    print = make_rank0_print(training_args)

    #! Config and Model
    count_func = lambda model: sum(
        {p.data_ptr(): p.numel() for p in model.parameters()}.values()
    )
    config = AutoConfig.from_pretrained(model_args.config, trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<|mask|>"})

    MASK_TOKEN = "<|mdm_mask|>"
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)
        tokenizer.mask_token_id = mask_token_id
    print(
        f"*** Using {tokenizer.convert_ids_to_tokens(mask_token_id)} as mask_token ***"
    )

    # Select model & collator
    if "bd" in model_args.config.lower():
        print("*** Using BDM: Block Masking + Dual Stream ***")
        data_collator = DataCollatorForBlockDiffusion(
            tokenizer=tokenizer, block_size=32
        )
        TrainerClass = DiffusionTrainer
        Model_CLS = AutoModel
    elif "udm" in model_args.config.lower():
        print("*** Using UDM: Random Token Corruption ***")
        data_collator = DataCollatorForUniformDiffusion(tokenizer=tokenizer)
        TrainerClass = DiffusionTrainer
        Model_CLS = AutoModel
    elif "mdm" in model_args.config.lower():
        print("*** Using MDM: Standard Masking ***")
        data_collator = DataCollatorForMaskedDiffusion(tokenizer=tokenizer)
        TrainerClass = DiffusionTrainer
        Model_CLS = AutoModel
    elif "ar" in model_args.config.lower():
        print("*** Using AR: Standard Causal LM ***")
        data_collator = transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        )
        TrainerClass = transformers.Trainer
        Model_CLS = transformers.AutoModelForCausalLM
        config.max_position_embeddings = training_args.context_len
    else:
        raise ValueError(f"Unknown model type in config: {model_args.config}")

    model = Model_CLS.from_config(config, trust_remote_code=True)

    # if training_args.local_rank == 0:
    print(
        f"Training model {model_args.config} - Total Size={count_func(model) / 2**20:.2f}M parameters"
    )

    lm_datasets = get_processed_dataset(
        tokenizer, data_args, training_args, cached=None
    )

    print(f"*** Datasets Loaded ***")

    train_dataset = lm_datasets["train"]
    valid_dataset = lm_datasets["validation"]

    data_module = dict(
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
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
        all_checkpoints = [
            x
            for x in all_checkpoints
            if (x / "trainer_state.json").exists() and not x.name.endswith("final")
        ]
        if len(all_checkpoints) == 0:
            training_args.resume_from_checkpoint = None
            print("No checkpoint found, starting from scratch")
        else:
            all_checkpoints = [str(x) for x in all_checkpoints]
            latest_checkpoint = max(all_checkpoints, key=os.path.getctime)
            training_args.resume_from_checkpoint = latest_checkpoint
            print("Resuming from checkpoint", latest_checkpoint)
            n_lastest_iter = int(latest_checkpoint.split("-")[-1])

    if isinstance(train_dataset, IterableDataset):
        shuffle_seed = (
            training_args.data_seed + n_lastest_iter
            if training_args.data_seed is not None
            else training_args.seed + n_lastest_iter
        )
        train_dataset = train_dataset.shuffle(seed=shuffle_seed)
        training_args.ignore_data_skip = True
        print("*** Set ignore_data_skip=True for streaming mode to save time ***")

    class CSVLoggerCallback(transformers.TrainerCallback):
        def __init__(self, log_path="training_logs.csv"):
            self.log_path = log_path
            self.header_written = False

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            row = {
                "step": state.global_step,
                "epoch": state.epoch,
                "loss": logs.get("loss", ""),
                "learning_rate": logs.get("learning_rate", ""),
            }
            file_exists = os.path.isfile(self.log_path)
            with open(self.log_path, mode="a", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["step", "epoch", "loss", "learning_rate"]
                )
                if not file_exists and not self.header_written:
                    writer.writeheader()
                    self.header_written = True
                writer.writerow(row)

    # Determine log path based on config name
    config_name = os.path.splitext(os.path.basename(model_args.config))[0]
    os.makedirs("logs", exist_ok=True)
    job_id = os.getenv("SLURM_JOB_ID", "interact")
    log_path = os.path.join("logs", f"training_logs_{config_name}_{job_id}.csv")
    print(f"*** Saving training logs to {log_path} ***")

    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        callbacks=[CSVLoggerCallback(log_path=log_path)],
    )
    trainer.tokenizer = tokenizer
    model.config.use_cache = False

    if training_args.do_train:
        logging.info("*** Start Training ***")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_state()
        # trainer.save_model(output_dir=training_args.output_dir)
        safe_save_model_for_hf_trainer(
            trainer=trainer, output_dir=training_args.output_dir
        )

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
