#!/bin/bash

# Test LoRA Training Script for InternVL3-8B
# This script uses a custom trainer with minimal data for testing

set -e

# Configuration
MODEL_NAME="OpenGVLab/InternVL3-8B"
DATASET_DIR="/scratch/czr/Video-Guard/datasets"
OUTPUT_DIR="./output_full_lora_streaming"
NUM_GPUS=4  # Use 4 GPUs
MAX_SAMPLES="127813 15144"  # Full dataset: all Shot2Story + all SafeWatch
MAX_LENGTH=16384  # Increased to handle longest samples and avoid truncation warnings
PER_DEVICE_BATCH_SIZE=1  # Reduced to avoid OOM with max_length=16384
GRADIENT_ACCUMULATION_STEPS=8  # Effective batch size (1×4×8=32)

# LoRA configuration - optimized for memory
USE_LORA=True
LORA_R=16  # Reduce rank to save memory
LORA_ALPHA=32  # Adjust alpha proportionally
LORA_DROPOUT=0.05

# NCCL configuration for network issues
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

echo "🧪 Starting Test LoRA Training..."
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_DIR"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo "Max samples: $MAX_SAMPLES (TEST MODE)"
echo "Max length: $MAX_LENGTH"
echo "LoRA enabled: $USE_LORA (r=$LORA_R, alpha=$LORA_ALPHA)"

# Run training with torchrun
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=29505 \
    custom_lora_trainer.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_dir $DATASET_DIR \
    --max_samples $MAX_SAMPLES \
    --max_length $MAX_LENGTH \
    --input_size 448 \
    --max_num_patches 12 \
    --use_lora $USE_LORA \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --num_train_epochs 1 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --bf16 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --logging_steps 1 \
    --save_steps 10 \
    --save_total_limit 2 \
    --run_name "test-lora-streaming"

echo "✅ Test LoRA Training completed!"
