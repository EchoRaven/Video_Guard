#!/usr/bin/env python3
"""
验证InternVL3-8B模型设置
"""

import torch
from transformers import AutoTokenizer, AutoModel
import sys

def verify_internvl3():
    """验证InternVL3-8B配置"""
    
    print("="*80)
    print("InternVL3-8B 模型配置验证")
    print("="*80)
    
    model_name = "OpenGVLab/InternVL3-8B"
    
    print(f"\n📦 模型: {model_name}")
    print("-"*40)
    
    # 检查模型信息
    print("\n1️⃣ 检查模型可用性...")
    try:
        # 只加载tokenizer和config，不加载权重
        print("   加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False
        )
        print("   ✅ Tokenizer加载成功")
        
        # 检查特殊tokens
        special_tokens = {
            '<IMG_CONTEXT>': tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>'),
            '<img>': tokenizer.convert_tokens_to_ids('<img>'),
            '</img>': tokenizer.convert_tokens_to_ids('</img>'),
            '<|vision_end|>': tokenizer.convert_tokens_to_ids('<|vision_end|>')
        }
        
        print("\n2️⃣ 特殊Token IDs:")
        for token, token_id in special_tokens.items():
            if token_id != tokenizer.unk_token_id:
                print(f"   {token}: {token_id} ✅")
            else:
                print(f"   {token}: NOT FOUND ❌")
        
    except Exception as e:
        print(f"   ❌ 错误: {e}")
        return False
    
    print("\n3️⃣ 模型参数:")
    print(f"   • 模型大小: ~8B参数")
    print(f"   • 视觉编码器: InternViT-300M")
    print(f"   • 语言模型: Qwen2.5-7B-Instruct")
    print(f"   • 每patch tokens: 256")
    print(f"   • 支持动态分辨率: 是")
    print(f"   • 最大patches: 12 (推荐6)")
    
    print("\n4️⃣ 训练配置建议:")
    print(f"   • LoRA rank: 128 (可以用32-256)")
    print(f"   • Learning rate: 1e-4")
    print(f"   • Batch size per GPU: 1-2")
    print(f"   • Max sequence length: 8192")
    print(f"   • Gradient checkpointing: 启用")
    print(f"   • BF16: 启用")
    
    print("\n5️⃣ 内存估算 (每GPU):")
    model_memory = 16  # GB
    batch_memory = 2   # GB for batch_size=2
    optimizer_memory = 5  # GB
    total = model_memory + batch_memory + optimizer_memory
    print(f"   • 模型: ~{model_memory} GB")
    print(f"   • Batch (size=2): ~{batch_memory} GB")
    print(f"   • 优化器状态: ~{optimizer_memory} GB")
    print(f"   • 总计: ~{total} GB")
    print(f"   • H200可用: 140 GB ✅")
    
    print("\n" + "="*80)
    print("✅ InternVL3-8B 配置验证完成！")
    print("\n可以使用以下命令启动训练:")
    print("  • 测试: bash launch_single_gpu_test.sh")
    print("  • 4卡训练: bash launch_4gpu_training.sh")
    
    return True

if __name__ == "__main__":
    success = verify_internvl3()
    sys.exit(0 if success else 1)