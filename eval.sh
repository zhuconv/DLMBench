

# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true


MODEL_PATH='ar_700m/checkpoint-5500'  # path to the fine-tuned LLaDA-400M model
if [[ $MODEL_PATH == *mdm* ]]; then
    SCRIPT='eval_mdm.py'
    MODEL='mdm_dist'
elif [[ $MODEL_PATH == *bdm* ]]; then
    SCRIPT='eval_bdm.py'
    MODEL='bdm_dist'
elif [[ $MODEL_PATH == *ar* ]]; then
    SCRIPT='-m lm_eval'
    MODEL='hf'
else
    echo "Unknown model type in MODEL_PATH: $MODEL_PATH"
    exit 1
fi
accelerate launch $SCRIPT \
    --tasks arc_challenge,piqa,hellaswag \
    --model $MODEL\
    --batch_size 32 \
    --model_args pretrained=output/$MODEL_PATH > logs/eval_${MODEL_PATH//\//_}.log 2>&1 &
