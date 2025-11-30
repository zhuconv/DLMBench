

# Set the environment variables first before running the command.
export HF_HOME="/vcc-data/peihaow/huggingface"
export HF_HUB_CACHE="/vcc-data/peihaow/huggingface/hub"
export HF_DATASETS_CACHE="/vcc-data/peihaow/huggingface/datasets"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_TOKEN=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

MODEL_PATH='output/duo_400m/checkpoint-400'  # path to the fine-tuned LLaDA-400M model

# conditional likelihood estimation benchmarks
# cfg=0 indicates that classifier-free guidance is not used
accelerate launch eval_duo.py --tasks arc_challenge,piqa,hellaswag --num_fewshot 0 --model llada_dist --batch_size 32 --model_args model_path=$MODEL_PATH,cfg=0.5,is_check_greedy=False,mc_num=128

# multiple-choice benchmarks with 5-shot learning
#accelerate launch eval_llada.py --tasks winogrande,mmlu,cmmlu,ceval-valid --num_fewshot 5 --model llada_dist --batch_size 8 --model_args model_path=$MODEL_PATH,cfg=0.0,is_check_greedy=False,mc_num=128

