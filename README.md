# Setup
## Environment
```bash
pip install uv
uv sync
```

## Data Preparation
Nothing to do.
This is because in `train.py:get_streaming_dataset` we implement a default setting of loading streaming dataset from `DKYoon/SlimPajama-6B`.

# Training

```bash
bash train.sh arm_700m # bdm_700m, mdm_700m, udm_700m
```

# Evaluation
In this file, we provide the code for the evaluation of [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base),
[LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) and [LLaDA 1.5](https://arxiv.org/abs/2505.19223).

## Eval Harness
```bash
MODEL_PAHT=<your_checkpoint> bash eval.sh # "ar", "mdm", "udm", "bdm" shoud be contained in the path to distinguish
```


## Acknowledgments
Thanks [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) and [LLaDA](https://github.com/ML-GSAI/LLaDA)
for their great work!