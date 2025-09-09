#!/usr/bin/env python3
"""
测试参数解析是否正确
"""

import sys
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import List

# 从custom_lora_trainer.py导入dataclasses
sys.path.append('/scratch/czr/Video-Guard/training_testing')
from custom_lora_trainer import ModelArguments, DataArguments, LoRAArguments, TrainingArguments

def test_args():
    """测试参数解析"""
    
    # 模拟命令行参数
    test_args = [
        '--output_dir', './test_output',
        '--model_name_or_path', 'OpenGVLab/InternVL3-8B',
        '--dataset_dir', '/scratch/czr/Video-Guard/datasets',
        '--num_train_epochs', '3',
        '--per_device_train_batch_size', '2',
        '--gradient_accumulation_steps', '2',
        '--learning_rate', '1e-4',
        '--warmup_ratio', '0.05',
        '--weight_decay', '0.01',
        '--logging_steps', '10',
        '--save_steps', '500',
        '--save_total_limit', '3',
        '--dataloader_num_workers', '4',
        '--bf16', 'True',
        '--gradient_checkpointing', 'True',
        '--use_lora', 'True',
        '--lora_r', '128',
        '--lora_alpha', '256',
        '--lora_dropout', '0.1',
        '--max_samples', '200000', '20000',
        '--max_length', '8192',
        '--trust_remote_code', 'True',
    ]
    
    # 保存原始sys.argv
    original_argv = sys.argv
    
    try:
        # 设置测试参数
        sys.argv = ['test_script'] + test_args
        
        # 解析参数
        parser = HfArgumentParser((ModelArguments, DataArguments, LoRAArguments, TrainingArguments))
        model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()
        
        print("✅ 参数解析成功！")
        print("\n📊 解析的参数:")
        print("-"*40)
        
        print("\nModel Arguments:")
        print(f"  model_name_or_path: {model_args.model_name_or_path}")
        print(f"  trust_remote_code: {model_args.trust_remote_code}")
        
        print("\nData Arguments:")
        print(f"  dataset_dir: {data_args.dataset_dir}")
        print(f"  max_samples: {data_args.max_samples}")
        print(f"  max_length: {data_args.max_length}")
        
        print("\nLoRA Arguments:")
        print(f"  use_lora: {lora_args.use_lora}")
        print(f"  lora_r: {lora_args.lora_r}")
        print(f"  lora_alpha: {lora_args.lora_alpha}")
        print(f"  lora_dropout: {lora_args.lora_dropout}")
        
        print("\nTraining Arguments:")
        print(f"  output_dir: {training_args.output_dir}")
        print(f"  num_train_epochs: {training_args.num_train_epochs}")
        print(f"  per_device_train_batch_size: {training_args.per_device_train_batch_size}")
        print(f"  gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
        print(f"  learning_rate: {training_args.learning_rate}")
        print(f"  bf16: {training_args.bf16}")
        print(f"  gradient_checkpointing: {training_args.gradient_checkpointing}")
        
        return True
        
    except Exception as e:
        print(f"❌ 参数解析失败: {e}")
        return False
        
    finally:
        # 恢复原始sys.argv
        sys.argv = original_argv

if __name__ == "__main__":
    success = test_args()
    if success:
        print("\n🎉 所有参数都可以正确解析！")
        print("启动脚本已修复，可以重新运行训练。")
    else:
        print("\n⚠️ 参数解析仍有问题，请检查。")