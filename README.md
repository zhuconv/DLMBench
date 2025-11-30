# Setup
## Environment
```bash
pip install uv
uv pip install transformers==4.52.0 lm_eval==0.4.8 accelerate==0.34.2 datasets==3.6.0
uv pip install deepspeed hf_xet
```

## Data Preparation
Nothing to do.
This is because in `train.py:get_streaming_dataset` we implement a default setting of loading streaming dataset from `DKYoon/SlimPajama-6B`.

# Evaluation
In this file, we provide the code for the evaluation of [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base),
[LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) and [LLaDA 1.5](https://arxiv.org/abs/2505.19223).

## Eval Harness
```bash
bash eval.sh
```

For **the ppl tasks of LLaDA-8B-Base**, the evaluation results are as follows:

|                | ARC-C | Hellaswag | TruthfulQA | WinoGrande | GPQA | PIQA | MMLU | CMMLU | C-Eval |
|----------------|:------:|:----------:|:-----------:|:-----------:|:----:|:----:|:----:|:----:|:----:|
| **w/o CFG**    | 45.9  | 70.5       | 46.1        | **74.8**    | 25.2 | 73.6 | 65.9 | 69.9 | 70.5 |
| **w/ CFG**     | **47.9** | **72.5** | **46.4**    | **74.8**    | **26.1** | **74.4** |  –   | – | – |

In the Tab.1 of [LLaDA paper](https://arxiv.org/pdf/2502.09992), we only report results w/o CFG to ensure a fair comparison
with autoregressive models. 


For **the gen tasks of LLaDA-8B-Base**, the evaluation result are as follows:

| Settings | BBH | GSM8K | Math | HumanEval | MBPP |
|:------------------------------------|:----:|:----:|:----:|:----:|:----:|
| **gen_length = 1024, steps = 1024, block_length = 1024** | 49.7 | 70.3 | 31.4 | 35.4 | 40.0 |
| **gen_length = 512, steps = 512, block_length = 512**   | 50.4 | 70.8 | 30.9 | 32.9 | 39.2 |
| **gen_length = 256, steps = 256, block_length = 256**   | 45.0 | 70.0 | 30.3 | 32.9 | 40.2 |

In the Tab.1 of [LLaDA paper](https://arxiv.org/pdf/2502.09992), we report the results with `gen_length = 1024, steps = 1024, block_length = 1024` for simplicity. 
However, as shown above, the performance across all three settings is consistent.

## Reversal Curse
We downloaded a [text file](https://wenku.baidu.com/view/f13866185fbfc77da369b1b3?wkts=1760409102730) containing a large collection of classical Chinese poetic lines from Baidu Wenku.
Using regular expressions, we extracted pairs of consecutive poetic lines (i.e., couplets) and stored them in a file named `data/poem_data.json`.

We provide the evaluation command as follows:
```
# generate the subsequent line
python eval_reverse.py  --type ftb --eos_inf

# generate the preceding line
python eval_reverse.py  --type btf --eos_inf
```

## Acknowledgments
Thanks [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) and [LLaDA](https://github.com/ML-GSAI/LLaDA)
for their great work!