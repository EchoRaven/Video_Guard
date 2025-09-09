#!/bin/bash
# 单卡测试脚本 - 用于快速测试
# 使用InternVL3-8B，小数据量

echo "🧪 启动单卡测试训练..."
echo "配置:"
echo "  - Model: InternVL3-8B"
echo "  - GPUs: 1"
echo "  - Batch size: 1"
echo "  - Max sequence length: 8192"
echo "  - Test samples: 10 Shot2Story + 10 SafeWatch"
echo ""

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# 创建测试输出目录
OUTPUT_DIR="./output_test_single_gpu"
mkdir -p $OUTPUT_DIR

# 单卡启动
python custom_lora_trainer.py \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path "OpenGVLab/InternVL3-8B" \
    --dataset_dir "/scratch/czr/Video-Guard/datasets" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_steps 1 \
    --save_steps 50 \
    --save_total_limit 1 \
    --dataloader_num_workers 2 \
    --bf16 True \
    --gradient_checkpointing True \
    --use_lora True \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --max_samples 10 10 \
    --max_length 8192 \
    --trust_remote_code True \
    2>&1 | tee $OUTPUT_DIR/training.log

echo ""
echo "✅ 测试完成！"
echo "模型保存在: $OUTPUT_DIR"
echo "训练日志: $OUTPUT_DIR/training.log"