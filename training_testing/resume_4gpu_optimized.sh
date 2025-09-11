#!/bin/bash
# 4卡继续训练脚本 - 内存优化版本
# 减少batch size和序列长度以避免OOM

echo "🔄 使用4卡继续训练（内存优化版）..."
echo "配置:"
echo "  - GPUs: 4"
echo "  - Batch size per GPU: 1 (降低以节省内存)"
echo "  - Max sequence length: 16384 (降低以节省内存)"
echo "  - Gradient accumulation: 8 (保持effective batch size = 32)"
echo "  - 继续训练，从checkpoint-3500"
echo ""

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=16
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

# 输出目录（已包含checkpoint）
OUTPUT_DIR="./output_4gpu_resume"

echo "📋 检查checkpoint..."
if [ -f "$OUTPUT_DIR/adapter_model.safetensors" ]; then
    echo "  ✓ 找到LoRA权重文件"
    echo "  ✓ 训练将从现有权重继续"
else
    echo "  ❌ 未找到checkpoint文件"
    exit 1
fi

# 启动分布式训练 - 内存优化参数
torchrun \
    --nproc_per_node=4 \
    --master_port=29501 \
    custom_lora_trainer.py \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path "OpenGVLab/InternVL3-8B" \
    --dataset_dir "/scratch/czr/Video-Guard/datasets" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 3 \
    --dataloader_num_workers 2 \
    --bf16 True \
    --gradient_checkpointing True \
    --use_lora True \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --max_samples 5000 10000 \
    --max_length 16384 \
    --trust_remote_code True \
    2>&1 | tee $OUTPUT_DIR/resume_optimized.log

echo ""
echo "✅ 训练完成！"
echo "模型保存在: $OUTPUT_DIR"
echo "训练日志: $OUTPUT_DIR/resume_optimized.log"