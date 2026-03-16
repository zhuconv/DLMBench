#!/bin/bash
# Script to get nodes from the last gh-dev job and launch vista_train.sh on each node

# Get the last gh-dev running job's node list
NODELIST=$(squeue -u $USER -p gh-dev -t R -h -o "%N" | tail -1)

if [ -z "$NODELIST" ]; then
    echo "No running gh-dev jobs found!"
    exit 1
fi

echo "Found node list: $NODELIST"

# Expand the node list (e.g., c608-[091-092,101-102] -> individual nodes)
NODES=$(scontrol show hostnames "$NODELIST")

if [ -z "$NODES" ]; then
    echo "Failed to expand node list!"
    exit 1
fi

# Convert to array
NODE_ARRAY=($NODES)
NNODES=${#NODE_ARRAY[@]}

echo "Expanded to $NNODES nodes:"
echo "$NODES"
echo ""

# Get the config name (first argument or default)
CONFIG_NAME=${1:-"ar_700m"}

# Export necessary environment variables
export LOGLEVEL=INFO
export TRITON_CACHE_DIR=/tmp/triton_cache
export WANDB_MODE=disabled

# Set CUDA_HOME
if [ -z "$CUDA_HOME" ]; then
    if command -v nvcc &> /dev/null; then
        export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
    else
        export CUDA_HOME=/home1/apps/nvidia/Linux_aarch64/24.7/compilers
    fi
fi

# Head node is the first node
HEAD_NODE=${NODE_ARRAY[0]}
HEAD_NODE_IP=$(ssh $HEAD_NODE "hostname --ip-address" 2>/dev/null || echo $HEAD_NODE)

echo "Head node: $HEAD_NODE (IP: $HEAD_NODE_IP)"
echo "Config: $CONFIG_NAME"
echo ""

# Calculate hyperparameters using the same logic as vista_train.sh
NGPUS_PER_NODE=1
TOTAL_GPUS=$((NNODES * NGPUS_PER_NODE))

echo "Total GPUs: $TOTAL_GPUS (${NNODES} nodes × ${NGPUS_PER_NODE} GPU/node)"

### Hyperparameters - Same logic as vista_train.sh
setting=${setting:-"20B/4k"}
# ==== batch size setting ====
if [[ $setting == *"4k"* ]]; then
  global_batch_size=512
  local_batch_size=$(( 512 / NNODES ))
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
context_len=${CONTEXT_LEN:-4096}
max_steps=$(( max_tokens / (global_batch_size * context_len) ))

echo "Calculated parameters:"
echo "  micro_batch_size: $micro_batch_size"
echo "  gradient_accumulation_steps: $gradient_accumulation_steps"
echo "  max_steps: $max_steps"
echo "  context_len: $context_len"
echo ""

# Launch on each node
echo "Launching vista_train.sh on all nodes..."

for i in "${!NODE_ARRAY[@]}"; do
    NODE=${NODE_ARRAY[$i]}
    echo "Starting on node $NODE (rank $i)..."
    
    ssh $NODE "cd /work/11012/jiajunzhu2002/vista/DLMBench && \
        source .venv/bin/activate && \
        export LOGLEVEL=INFO && \
        export TRITON_CACHE_DIR=/tmp/triton_cache && \
        export WANDB_MODE=disabled && \
        export CUDA_VISIBLE_DEVICES=0 && \
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
        export CUDA_HOME=${CUDA_HOME} && \
        torchrun --nproc_per_node $NGPUS_PER_NODE --nnodes $NNODES \
            --node_rank $i \
            --rdzv_endpoint ${HEAD_NODE_IP}:29512 \
            --rdzv_id 12345 \
            --rdzv_backend c10d \
            train.py \
            --deepspeed \"\${DS_CONFIG:-ds_config.json}\" \
            --dataset_cache_dir '../hf_datasets/SlimPajama-627B' \
            --output_dir 'output/${CONFIG_NAME}' \
            --config './configs/${CONFIG_NAME}.json' \
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
            --save_strategy 'steps' \
            --dataloader_num_workers 8 \
            --dataloader_pin_memory True \
            --gradient_checkpointing True \
            --bf16 True" &
    
    # Small delay to avoid race conditions
    sleep 1
done

echo ""
echo "All nodes launched! Waiting for processes..."
wait
echo "Done!"

