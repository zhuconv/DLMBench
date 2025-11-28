

# Set the environment variables first before running the command.
export HF_HOME="/vcc-data/peihaow/huggingface"
export HF_HUB_CACHE="/vcc-data/peihaow/huggingface/hub"
export HF_DATASETS_CACHE="/vcc-data/peihaow/huggingface/datasets"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_TOKEN=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

MODEL_PATH='output/bd3lm_400m'  # path to the fine-tuned LLaDA-400M model
accelerate launch eval_bd3lm.py \
    --tasks arc_challenge,piqa,hellaswag \
    --model bd3lm_dist \
    --batch_size 32 \
    --model_args model_path=$MODEL_PATH
