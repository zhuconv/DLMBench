#!/bin/bash
#SBATCH --output=./logs/train_%x_%j.log
#SBATCH --nodes=4
#SBATCH --partition=gh
#SBATCH --time=36:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=jiajunzhu@utexas.edu

# source ~/.bashrc
# conda activate ibm
source .venv/bin/activate

export LOGLEVEL=INFO
export TRITON_CACHE_DIR=/tmp/triton_cache
export WANDB_MODE=disabled 

# Set CUDA_HOME for DeepSpeed
if [ -z "$CUDA_HOME" ]; then
    if command -v nvcc &> /dev/null; then
        export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
    else
        # Fallback to common CUDA installation path
        export CUDA_HOME=/home1/apps/nvidia/Linux_aarch64/24.7/compilers
    fi
fi

CONFIG_NAME=$1  # "mdm_700m" "bdm_700m" "ar_700m" "udm_700m"

# --- Detect nodes ---
if [[ -z "${SLURM_JOB_NAME:-}" || "$SLURM_JOB_NAME" == "intern" ]]; then
    echo "Interactive mode detected."
    NNODES=8
    CONFIG_NAME=${CONFIG_NAME:-"ar_700m"}
    command="torchrun"
    head_node_ip=c608-052 #$(hostname --ip-address)
    # Auto-detect number of GPUs in interactive mode
    if [ -z "$NGPUS" ]; then
        NGPUS=$(nvidia-smi --list-gpus | wc -l)
        echo "Auto-detected $NGPUS GPU(s) available"
    fi
else
    CONFIG_NAME=${SLURM_JOB_NAME}
    nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
    head_node=${nodes[0]}
    NNODES=${#nodes[@]}
    command="srun torchrun"
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
fi

# NAME=$(basename $CONFIG .json)
CONFIG_PATH="./configs/${CONFIG_NAME}.json"

echo "Head Node IP: $head_node_ip; NNODES: $NNODES"

# For 8 nodes with 1 GPU each, total GPUs = 8
# Each node runs 1 process, so nproc_per_node=1
NGPUS_PER_NODE=1
TOTAL_GPUS=$((NNODES * NGPUS_PER_NODE))

echo "Total GPUs: $TOTAL_GPUS (${NNODES} nodes × ${NGPUS_PER_NODE} GPU/node)"

### Hyperparameters - Optimized for GH200 (100GB GPU)
setting=${setting:-"20B/4k"}
# ==== batch size setting ====
# Optimized: Increased micro_batch_size from 4 to 20 for better GPU utilization
# For 700M model on 100GB GPU, we can use much larger batch sizes
if [[ $setting == *"4k"* ]]; then
  global_batch_size=512
  local_batch_size=$(( 512 / NNODES ))
  # Increased from 4 to 20: better GPU utilization on 100GB GPUs
  # Adjust based on actual memory usage (target: 70-85% GPU memory)
  micro_batch_size=${MICRO_BATCH_SIZE:-8}
fi
# ==== max_tokens ====
if [[ $setting == *"20B"* ]]; then
  max_tokens=$(( 100000000000 / 5 ))   # 20B = 2e10
elif [[ $setting == *"100B"* ]]; then
  max_tokens=100000000000              # 1e11
fi
# ==== final args ====
batch_size=$(( local_batch_size / NGPUS_PER_NODE ))
gradient_accumulation_steps=$(( batch_size / micro_batch_size ))
# Use context_len in calculation instead of hardcoded 4096
context_len=${CONTEXT_LEN:-4096}
max_steps=$(( max_tokens / (global_batch_size * context_len) ))

# ../../hf_datasets/SlimPajama-627B
# "/vcc-data/peihaow/SlimPajama-627B"

# Set CUDA_VISIBLE_DEVICES - each node only sees GPU 0
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

${command} --nproc_per_node $NGPUS_PER_NODE --nnodes $NNODES \
        --rdzv_endpoint $head_node_ip:29512 \
        --rdzv_id 12345 \
        --rdzv_backend c10d \
        train.py \
        --deepspeed "${DS_CONFIG:-ds_config.json}" \
        --dataset_cache_dir "../hf_datasets/SlimPajama-627B" \
        --output_dir "output/${CONFIG_NAME}" \
        --config ${CONFIG_PATH} \
        --resume_from_checkpoint true \
        --per_device_train_batch_size $micro_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --report_to none \
        --max_steps $max_steps \
        --context_len $context_len \
        --warmup_ratio 0.01 \
        --weight_decay 0.1 \
        --learning_rate 4e-4 \
        --adam_beta1 0.9 \
        --adam_beta2 0.98 \
        --save_steps 100 \
        --logging_steps 10 \
        --do_train True \
        --do_predict True \
        --save_total_limit 5 \
        --save_strategy "steps" \
        --dataloader_num_workers 8 \
        --dataloader_pin_memory True \
        --gradient_checkpointing True \
        --bf16 True

