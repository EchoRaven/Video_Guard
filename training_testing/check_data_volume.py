#!/usr/bin/env python3
"""
检查实际训练数据量
"""

import sys
sys.path.append('/scratch/czr/Video-Guard/training_testing')

from Dataloader import StreamingDataset
import json

def check_data_volume():
    """检查实际会加载的数据量"""
    
    print("="*80)
    print("训练数据量统计")
    print("="*80)
    
    # 检查原始数据量
    print("\n📊 原始数据量:")
    print("-"*40)
    
    # Shot2Story
    shot2story_path = '/scratch/czr/Video-Guard/datasets/shot2story/134k_full_train.json'
    with open(shot2story_path, 'r') as f:
        shot2story_data = json.load(f)
    
    total_shot2story_videos = len(shot2story_data)
    
    # 计算Shot2Story的clips数量
    total_shot2story_clips = 0
    for video in shot2story_data[:1000]:  # 采样计算平均值
        if 'video_names' in video:
            total_shot2story_clips += len(video['video_names'])
    avg_clips_per_video = total_shot2story_clips / min(1000, len(shot2story_data))
    estimated_total_clips = int(total_shot2story_videos * avg_clips_per_video)
    
    print(f"Shot2Story:")
    print(f"  • 总视频数: {total_shot2story_videos:,}")
    print(f"  • 平均每视频clips数: {avg_clips_per_video:.1f}")
    print(f"  • 预估总clips数: {estimated_total_clips:,}")
    
    # SafeWatch
    safewatch_path = '/scratch/czr/Video-Guard/datasets/safewatch_streaming_corrected.jsonl'
    total_safewatch_videos = 0
    total_safewatch_clips = 0
    
    with open(safewatch_path, 'r') as f:
        for line in f:
            total_safewatch_videos += 1
            data = json.loads(line)
            total_safewatch_clips += len(data.get('clip_video_paths', []))
    
    print(f"\nSafeWatch:")
    print(f"  • 总视频数: {total_safewatch_videos:,}")
    print(f"  • 总clips数: {total_safewatch_clips:,}")
    print(f"  • 平均每视频clips数: {total_safewatch_clips/total_safewatch_videos:.1f}")
    
    # 加载实际数据集看会使用多少
    print("\n📦 实际加载的数据量 (根据配置):")
    print("-"*40)
    
    # 使用配置的限制
    max_samples_config = [200000, 20000]  # [shot2story, safewatch]
    
    print(f"配置的限制:")
    print(f"  • Shot2Story: {max_samples_config[0]:,} 个视频")
    print(f"  • SafeWatch: {max_samples_config[1]:,} 个视频")
    
    # 实际能加载的数量
    actual_shot2story = min(max_samples_config[0], total_shot2story_videos)
    actual_safewatch = min(max_samples_config[1], total_safewatch_videos)
    
    print(f"\n实际会加载:")
    print(f"  • Shot2Story: {actual_shot2story:,} 个视频 (全部)")
    print(f"  • SafeWatch: {actual_safewatch:,} 个视频 (全部)")
    
    # 估算样本数量
    estimated_shot2story_samples = int(actual_shot2story * avg_clips_per_video) + actual_shot2story  # clips + final responses
    estimated_safewatch_samples = total_safewatch_clips + actual_safewatch  # clips + final responses
    
    print(f"\n预估训练样本数:")
    print(f"  • Shot2Story样本: ~{estimated_shot2story_samples:,}")
    print(f"  • SafeWatch样本: ~{estimated_safewatch_samples:,}")
    print(f"  • 总计: ~{estimated_shot2story_samples + estimated_safewatch_samples:,} 个样本")
    
    # 训练时间估算
    print("\n⏱️ 训练时间估算 (4卡, batch_size=2):")
    print("-"*40)
    
    total_samples = estimated_shot2story_samples + estimated_safewatch_samples
    batch_size_per_gpu = 2
    num_gpus = 4
    gradient_accumulation = 2
    effective_batch_size = batch_size_per_gpu * num_gpus * gradient_accumulation  # 16
    
    steps_per_epoch = total_samples // effective_batch_size
    num_epochs = 3
    total_steps = steps_per_epoch * num_epochs
    
    # 假设每步0.5-1秒
    min_time_hours = (total_steps * 0.5) / 3600
    max_time_hours = (total_steps * 1.0) / 3600
    
    print(f"  • 每epoch步数: {steps_per_epoch:,}")
    print(f"  • 总步数 (3 epochs): {total_steps:,}")
    print(f"  • 预估训练时间: {min_time_hours:.1f} - {max_time_hours:.1f} 小时")
    
    # 内存估算
    print("\n💾 内存使用估算:")
    print("-"*40)
    
    model_size_gb = 16  # InternVL2.5-8B
    per_sample_memory_mb = 100  # 粗略估算每个样本100MB
    batch_memory_gb = (batch_size_per_gpu * per_sample_memory_mb) / 1024
    
    total_memory_per_gpu = model_size_gb + batch_memory_gb + 5  # +5GB for optimizer states
    
    print(f"  • 模型大小: {model_size_gb} GB")
    print(f"  • Batch内存: ~{batch_memory_gb:.1f} GB")
    print(f"  • 每GPU总需求: ~{total_memory_per_gpu:.1f} GB")
    print(f"  • H200可用内存: 140 GB (充足)")
    
    print("\n" + "="*80)
    print("✅ 数据配置完成！")
    print(f"将使用全部 {actual_shot2story:,} 个Shot2Story视频和全部 {actual_safewatch:,} 个SafeWatch视频进行训练")

if __name__ == "__main__":
    check_data_volume()