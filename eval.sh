# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
source .venv/bin/activate

# MODEL_PATH is expected to be passed as an environment variable.
# MODEL_PATH='ar_700m/checkpoint-5500'  # path to the fine-tuned LLaDA-700M model
if [[ $MODEL_PATH == *mdm* ]]; then
    SCRIPT='eval_mdm.py'
    MODEL='mdm_dist'
elif [[ $MODEL_PATH == *bdm* ]]; then
    SCRIPT='eval_bdm.py'
    MODEL='bdm_dist'
elif [[ $MODEL_PATH == *udm* ]]; then 
    SCRIPT='eval_udm.py'
    MODEL='udm_dist'
elif [[ $MODEL_PATH == *ar* ]]; then
    SCRIPT='-m lm_eval'
    MODEL='hf'
else
    echo "Unknown model type in MODEL_PATH: $MODEL_PATH"
    exit 1
fi

# Set a limit for quick testing. Comment out or set to empty for full evaluation.
LIMIT=20
LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

accelerate launch $SCRIPT \
    --tasks arc_challenge,piqa,hellaswag \
    --model $MODEL\
    --batch_size 32 \
    $LIMIT_ARG \
    --model_args pretrained=output/$MODEL_PATH > logs/eval_${MODEL_PATH//\//_}.log 2>&1 &