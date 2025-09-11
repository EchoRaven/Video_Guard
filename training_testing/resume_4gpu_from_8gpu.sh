#!/bin/bash
# 4卡resume训练脚本 - 从8卡checkpoint恢复
# 调整gradient accumulation以保持相同的有效batch size

echo "🔄 从8卡checkpoint恢复，使用4卡继续训练..."
echo "配置:"
echo "  - GPUs: 4 (原8卡)"
echo "  - Batch size per GPU: 2"
echo "  - Max sequence length: 24576"
echo "  - Gradient accumulation: 4 (保持effective batch size = 4*2*4 = 32)"
echo "  - Resume from: checkpoint-3500"
echo ""

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=16
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

# 设置路径
CHECKPOINT_PATH="./output_8gpu_full/checkpoint-3500"
OUTPUT_DIR="./output_4gpu_resume"
mkdir -p $OUTPUT_DIR

# 复制训练状态文件（如果需要）
echo "📋 准备resume..."
if [ -f "$CHECKPOINT_PATH/training_state.pt" ]; then
    echo "  ✓ 找到训练状态文件"
fi

# 启动分布式训练
torchrun \
    --nproc_per_node=4 \
    --master_port=29501 \
    custom_lora_trainer.py \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path "OpenGVLab/InternVL3-8B" \
    --dataset_dir "/scratch/czr/Video-Guard/datasets" \
    --lora_path $CHECKPOINT_PATH \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 3 \
    --dataloader_num_workers 4 \
    --bf16 True \
    --gradient_checkpointing True \
    --use_lora True \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --max_samples 10000 20000 \
    --max_length 24576 \
    --trust_remote_code True \
    2>&1 | tee $OUTPUT_DIR/resume_training.log

echo ""
echo "✅ Resume训练完成！"
echo "模型保存在: $OUTPUT_DIR"
echo "训练日志: $OUTPUT_DIR/resume_training.log"