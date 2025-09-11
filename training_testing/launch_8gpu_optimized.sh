#!/bin/bash
# 8卡优化训练启动脚本
# 使用更大的batch size和优化的参数

echo "🚀 启动8卡优化训练..."
echo "配置:"
echo "  - GPUs: 8"
echo "  - Batch size per GPU: 2"  
echo "  - Max sequence length: 8192"
echo "  - Gradient accumulation: 1 (effective batch size = 8*2*1 = 16)"
echo "  - LoRA rank: 64 (增大以提升模型容量)"
echo "  - Mixed precision: bfloat16"
echo "  - Gradient checkpointing: Enabled"
echo "  - Full dataset training with optimized settings"
echo ""

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=16

# NCCL优化设置
export NCCL_DEBUG=WARN  # 减少日志输出
export NCCL_IB_DISABLE=0  # 如果有InfiniBand，启用它
export NCCL_SOCKET_IFNAME=eth0  # 指定网络接口
export NCCL_P2P_DISABLE=0  # 启用P2P通信

# PyTorch优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # 优化显存分配

# 创建输出目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./output_8gpu_optimized_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

# 启动分布式训练
torchrun \
    --nproc_per_node=8 \
    --master_port=29501 \
    --nnodes=1 \
    --node_rank=0 \
    custom_lora_trainer.py \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path "OpenGVLab/InternVL3-8B" \
    --dataset_dir "/scratch/czr/Video-Guard/datasets" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 250 \
    --save_total_limit 5 \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --bf16 True \
    --bf16_full_eval True \
    --gradient_checkpointing True \
    --use_lora True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_target_modules "qkv_proj" "out_proj" "fc1" "fc2" \
    --max_samples -1 -1 \
    --max_length 8192 \
    --seed 42 \
    --trust_remote_code True \
    --ddp_find_unused_parameters False \
    --ddp_bucket_cap_mb 25 \
    --report_to tensorboard \
    --logging_dir $OUTPUT_DIR/logs \
    2>&1 | tee $OUTPUT_DIR/training.log

# 训练完成后的处理
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 训练成功完成！"
    echo "📊 模型保存在: $OUTPUT_DIR"
    echo "📝 训练日志: $OUTPUT_DIR/training.log"
    echo "📈 TensorBoard日志: $OUTPUT_DIR/logs"
    echo ""
    echo "查看TensorBoard:"
    echo "  tensorboard --logdir=$OUTPUT_DIR/logs"
else
    echo ""
    echo "❌ 训练过程中出现错误，请检查日志: $OUTPUT_DIR/training.log"
    exit 1
fi