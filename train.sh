#!/bin/bash
#SBATCH --output=./logs/train_%x_%j.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --partition=p5-main
#SBATCH --time=1-12:00:00
#SBATCH --ntasks-per-node=1         # One task per GPU? per node
#SBATCH --cpus-per-task=96

# source ~/.bashrc
# conda activate ibm
# cd /cusp-data-efa/peihaow/jz/IBSSM

export SLURM_CPU_BIND=none
# export HF_HOME="/vcc-data/peihaow/huggingface"
# export HF_HUB_CACHE="/vcc-data/peihaow/huggingface/hub"
# export HF_DATASETS_CACHE="/vcc-data/peihaow/huggingface/datasets"
export LOGLEVEL=INFO
export TRITON_CACHE_DIR=/tmp/triton_cache
export WANDB_MODE=disabled 
export CUDA_LAUNCH_BLOCKING=1


CONFIG_NAME=$1  # "mdm_700m" "bdm_700m" "ar_700m" "duo_700m"
# --- Detect nodes ---
if [[ -z "${SLURM_JOB_NAME:-}" || "$SLURM_JOB_NAME" == "intern" ]]; then
    echo "Interactive mode detected."
    NNODES=1
    # NGPUS=1
    # CONFIG_NAME="ar_700m"
    command="torchrun"
    head_node_ip=$(hostname --ip-address)
else
    nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
    head_node=${nodes[0]}
    NNODES=${#nodes[@]}
    command="srun torchrun"
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
fi

# NAME=$(basename $CONFIG .json)
CONFIG_PATH="./configs/${CONFIG_NAME}.json"

echo "Head Node IP: $head_node_ip; NNODES: $NNODES"

NGPUS=${NGPUS:-4}

### Hyperparameters
setting=${setting:-"20B/4k"}
# ==== batch size setting ====
if [[ $setting == *"4k"* ]]; then
  global_batch_size=512
  local_batch_size=$(( 512 / NNODES ))
  micro_batch_size=4   # = 4
fi
# ==== max_tokens ====
if [[ $setting == *"20B"* ]]; then
  max_tokens=$(( 100000000000 / 5 ))   # 20B = 2e10
elif [[ $setting == *"100B"* ]]; then
  max_tokens=100000000000              # 1e11
fi
# ==== final args ====
batch_size=$(( local_batch_size / NGPUS ))
gradient_accumulation_steps=$(( batch_size / micro_batch_size ))
max_steps=$(( max_tokens / (global_batch_size * 4096) )) 

# ../../hf_datasets/SlimPajama-627B
# "/vcc-data/peihaow/SlimPajama-627B"

# python -m debugpy --wait-for-client --listen 0.0.0.0:5000 -m torch.distributed.launch
export CUDA_VISIBLE_DEVICES=0,1,2,3

${command} --nproc_per_node $NGPUS --nnodes $NNODES \
        --rdzv_endpoint $head_node_ip:29512 \
        --rdzv_id $RANDOM \
        --rdzv_backend c10d \
        train.py \
        --deepspeed "ds_config.json" \
        --dataset_cache_dir "../hf_datasets/SlimPajama-627B" \
        --output_dir "output/${CONFIG_NAME}" \
        --config ${CONFIG_PATH} \
        --resume_from_checkpoint true \
        --per_device_train_batch_size $micro_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --report_to none \
        --max_steps $max_steps \
        --context_len 2048 \
        --warmup_ratio 0.01 \
        --weight_decay 0.1 \
        --learning_rate 4e-4 \
        --adam_beta1 0.9 \
        --adam_beta2 0.98 \
        --lr_scheduler_type cosine_with_min_lr \
        --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' \
        --save_steps 100 \
        --logging_steps 10 \
        --do_train True \
        --do_predict True \
        --save_strategy "steps" \
        --gradient_checkpointing False \
        --bf16 True