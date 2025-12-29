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

# Evaluation Harness

To evaluate a model, set the `MODEL_PATH` environment variable to your checkpoint directory and run `eval.sh`.
<font color="red">IMPORTANT:</font> The script detects the model architecture from the folder name.

```bash
# Example for an AR model
MODEL_PATH=output/ar_700m/checkpoint-500 bash eval.sh

# Example for a Masked Diffusion model
MODEL_PATH=output/mdm_700m/checkpoint-1000 bash eval.sh
```

## Acknowledgments
Thanks [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) and [LLaDA](https://github.com/ML-GSAI/LLaDA)
for their great work!