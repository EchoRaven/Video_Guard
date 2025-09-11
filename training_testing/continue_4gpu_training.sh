#!/bin/bash
# 4卡继续训练脚本 - 从8卡checkpoint继续
# 将checkpoint复制到输出目录，让训练器自动检测并继续

echo "🔄 使用4卡继续训练（从8卡checkpoint）..."
echo "配置:"
echo "  - GPUs: 4"
echo "  - Batch size per GPU: 2"
echo "  - Max sequence length: 24576"
echo "  - Gradient accumulation: 4 (effective batch size = 32)"
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

# 启动分布式训练
# 注意：不使用resume参数，让模型从OUTPUT_DIR加载现有权重
torchrun \
    --nproc_per_node=4 \
    --master_port=29501 \
    custom_lora_trainer.py \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path "OpenGVLab/InternVL3-8B" \
    --dataset_dir "/scratch/czr/Video-Guard/datasets" \
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
    2>&1 | tee $OUTPUT_DIR/continue_training.log

echo ""
echo "✅ 训练完成！"
echo "模型保存在: $OUTPUT_DIR"
echo "训练日志: $OUTPUT_DIR/continue_training.log"