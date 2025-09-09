#!/usr/bin/env python3
"""
检查GPU状态和内存，确保可以进行4卡训练
"""

import torch
import subprocess
import sys

def check_gpu_status():
    """检查GPU是否准备好进行训练"""
    
    print("="*60)
    print("GPU状态检查")
    print("="*60)
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("❌ CUDA不可用！请检查驱动和PyTorch安装。")
        return False
    
    # 检查GPU数量
    gpu_count = torch.cuda.device_count()
    print(f"✅ 检测到 {gpu_count} 个GPU")
    
    if gpu_count < 4:
        print(f"⚠️  警告：只有{gpu_count}个GPU，需要4个GPU进行训练")
        print("   可以修改CUDA_VISIBLE_DEVICES来指定要使用的GPU")
        
    # 检查每个GPU的状态
    print("\n📊 GPU详细信息:")
    print("-"*40)
    
    total_memory_gb = 0
    available_memory_gb = 0
    
    for i in range(gpu_count):
        torch.cuda.set_device(i)
        props = torch.cuda.get_device_properties(i)
        
        # 获取内存信息
        total_mem = props.total_memory / (1024**3)  # 转换为GB
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        free = (props.total_memory - torch.cuda.memory_allocated(i)) / (1024**3)
        
        total_memory_gb += total_mem
        available_memory_gb += free
        
        print(f"\nGPU {i}: {props.name}")
        print(f"  总内存: {total_mem:.1f} GB")
        print(f"  已使用: {allocated:.1f} GB")
        print(f"  可用: {free:.1f} GB")
        
        # 检查内存是否足够
        if free < 20:  # 少于20GB可能不够
            print(f"  ⚠️ 内存可能不足，建议至少20GB可用内存")
    
    # 估算训练需求
    print("\n💡 训练配置估算:")
    print("-"*40)
    
    batch_size = 2
    max_length = 8192
    model_size_gb = 16  # InternVL2.5-8B大约16GB
    
    # 每个GPU的内存需求估算
    per_gpu_requirement = model_size_gb + (batch_size * max_length * 4 / (1024**3))  # 简单估算
    
    print(f"模型大小: ~{model_size_gb} GB")
    print(f"Batch size per GPU: {batch_size}")
    print(f"Max sequence length: {max_length}")
    print(f"预估每GPU内存需求: ~{per_gpu_requirement:.1f} GB")
    
    if gpu_count >= 4:
        print(f"\n✅ 4卡训练配置:")
        print(f"  - 总batch size: {4 * batch_size} (每卡{batch_size})")
        print(f"  - Gradient accumulation: 2")
        print(f"  - 有效batch size: {4 * batch_size * 2} = {4 * batch_size * 2}")
    
    # 检查nvidia-smi
    print("\n📈 nvidia-smi 输出:")
    print("-"*40)
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.free,memory.total', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 5:
                    idx, name, used, free, total = parts
                    print(f"GPU {idx}: {name} - 已用:{used}MB, 可用:{free}MB, 总计:{total}MB")
        else:
            print("无法运行nvidia-smi")
    except:
        print("nvidia-smi不可用")
    
    # 最终建议
    print("\n" + "="*60)
    print("建议:")
    print("="*60)
    
    if gpu_count >= 4 and available_memory_gb/gpu_count > 20:
        print("✅ GPU状态良好，可以开始4卡训练！")
        print("\n启动命令:")
        print("  bash launch_4gpu_training.sh")
        return True
    elif gpu_count < 4:
        print(f"⚠️ 只有{gpu_count}个GPU，可以调整训练配置:")
        print(f"  - 修改launch脚本中的--nproc_per_node={gpu_count}")
        print(f"  - 或设置CUDA_VISIBLE_DEVICES选择特定GPU")
        return False
    else:
        print("⚠️ GPU内存可能不足，建议:")
        print("  - 减小batch_size到1")
        print("  - 减小max_length")
        print("  - 启用gradient_checkpointing")
        print("  - 使用8bit量化")
        return False

if __name__ == "__main__":
    ready = check_gpu_status()
    sys.exit(0 if ready else 1)