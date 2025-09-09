#!/bin/bash
# 四卡训练启动脚本 - 备选方案
# 使用python -m torch.distributed.launch (适用于较老的PyTorch版本)

echo "🚀 启动4卡训练 (备选方案)..."
echo "配置:"
echo "  - GPUs: 4"
echo "  - Batch size per GPU: 2"  
echo "  - Max sequence length: 8192"
echo "  - Gradient accumulation: 2"
echo ""

# 设置要使用的GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 创建输出目录
OUTPUT_DIR="./output_4gpu_bs2_8k"
mkdir -p $OUTPUT_DIR

# 使用python -m torch.distributed.launch启动
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    custom_lora_trainer.py \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path "OpenGVLab/InternVL3-8B" \
    --dataset_dir "/scratch/czr/Video-Guard/datasets" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
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
    --max_samples 20000 20000 \
    --max_length 8192 \
    --trust_remote_code True \
    2>&1 | tee $OUTPUT_DIR/training.log

echo ""
echo "✅ 训练完成！"
echo "模型保存在: $OUTPUT_DIR"