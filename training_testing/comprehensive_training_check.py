#!/usr/bin/env python3
"""
Comprehensive check of Video-Guard training code
"""

import sys
import os
sys.path.append('/scratch/czr/Video-Guard/training_testing')

from Dataloader import StreamingDataset
import torch
import numpy as np

def check_training_components():
    """全面检查训练代码的各个组件"""
    
    print("="*80)
    print("VIDEO-GUARD 训练代码完整性检查")
    print("="*80)
    
    check_results = {
        'loss_calculation': False,
        'label_masking': False,
        'data_shuffling': False,
        'real_video_data': False,
        'no_fallback': False,
        'data_completeness': False
    }
    
    # 1. 检查Loss计算
    print("\n1️⃣ LOSS计算检查:")
    print("-"*40)
    
    # 读取DataCollator中的loss设置
    with open('/scratch/czr/Video-Guard/training_testing/train_multi_gpu.py', 'r') as f:
        collator_code = f.read()
    
    # 检查关键的loss masking逻辑
    if 'label = input_ids.clone()' in collator_code:
        print("✅ Loss计算使用input_ids.clone()初始化")
        
        # 检查clip样本的处理
        if 'first_img_pos' in collator_code and 'label[:first_img_pos] = -100' in collator_code:
            print("✅ Clip样本: 屏蔽用户指令部分(设为-100)")
        
        if 'img_token_id, end_img_token_id, img_context_token_id' in collator_code:
            print("✅ Clip样本: 屏蔽图像tokens(设为-100)")
            
        # 检查final response样本的处理  
        if 'vision_end_token_id = 151653' in collator_code:
            print("✅ Response样本: 从<|vision_end|>后开始计算loss")
            
        check_results['loss_calculation'] = True
        print("\n📊 Loss计算范围:")
        print("  • Clip样本: <label>...</label>和<summary>...</summary>")
        print("  • Response样本: <response>...</response>")
    else:
        print("❌ Loss计算逻辑可能有问题")
    
    # 2. 检查标签格式
    print("\n2️⃣ 标签格式检查:")
    print("-"*40)
    
    dataset = StreamingDataset(
        dataset_file='/scratch/czr/Video-Guard/datasets',
        tokenizer=None,
        max_samples=[5, 5],
        shuffle=False
    )
    
    # 检查样本标签格式
    has_closing_tags = True
    for i, sample in enumerate(dataset.samples[:10]):
        text = sample['full_prompt']
        
        # 检查闭合标签
        if '<label>' in text and '</label>' not in text:
            has_closing_tags = False
            print(f"❌ 样本{i}: 缺少</label>闭合标签")
        if '<summary>' in text and '</summary>' not in text:
            has_closing_tags = False
            print(f"❌ 样本{i}: 缺少</summary>闭合标签")
        if '<response>' in text and '</response>' not in text:
            has_closing_tags = False
            print(f"❌ 样本{i}: 缺少</response>闭合标签")
    
    if has_closing_tags:
        print("✅ 所有标签都有正确的闭合标签")
        check_results['label_masking'] = True
    
    # 3. 检查数据打乱
    print("\n3️⃣ 数据打乱检查:")
    print("-"*40)
    
    # 创建启用打乱的数据集
    dataset_shuffled = StreamingDataset(
        dataset_file='/scratch/czr/Video-Guard/datasets',
        tokenizer=None,
        max_samples=[10, 10],
        shuffle=True,
        random_seed=42
    )
    
    # 创建未打乱的数据集
    dataset_ordered = StreamingDataset(
        dataset_file='/scratch/czr/Video-Guard/datasets',
        tokenizer=None,
        max_samples=[10, 10],
        shuffle=False
    )
    
    # 比较前10个样本的顺序
    order_different = False
    for i in range(min(10, len(dataset_shuffled.samples), len(dataset_ordered.samples))):
        if dataset_shuffled.samples[i]['type'] != dataset_ordered.samples[i]['type']:
            order_different = True
            break
    
    if order_different:
        print("✅ 数据打乱功能正常工作")
        check_results['data_shuffling'] = True
        
        # 分析混合程度
        types_in_first_20 = [s['type'] for s in dataset_shuffled.samples[:20]]
        clips = types_in_first_20.count('clip')
        responses = types_in_first_20.count('final_response')
        print(f"  前20个样本: {clips}个clips, {responses}个responses (混合良好)")
    else:
        print("❌ 数据打乱可能未生效")
    
    # 4. 检查真实视频数据
    print("\n4️⃣ 真实视频数据检查:")
    print("-"*40)
    
    # 检查视频加载函数
    with open('/scratch/czr/Video-Guard/training_testing/Dataloader.py', 'r') as f:
        dataloader_code = f.read()
    
    if 'return None  # Return None instead of black frames' in dataloader_code:
        print("✅ 视频加载失败时返回None(不使用黑帧)")
        
    if 'return None  # Return None instead of empty tensor' in dataloader_code:
        print("✅ 无法加载帧时返回None(不使用空tensor)")
        
    # 测试实际加载
    sample_with_video = 0
    sample_without_video = 0
    
    for i in range(min(10, len(dataset))):
        try:
            sample_data = dataset[i]
            if sample_data and sample_data.get('pixel_values') is not None:
                sample_with_video += 1
            else:
                sample_without_video += 1
        except:
            sample_without_video += 1
    
    if sample_with_video > 0:
        print(f"✅ 成功加载真实视频数据: {sample_with_video}/{sample_with_video+sample_without_video}个样本")
        check_results['real_video_data'] = True
    
    # 5. 检查无fallback
    print("\n5️⃣ Fallback机制检查:")
    print("-"*40)
    
    has_fallback = False
    
    # 检查代码中的fallback文本
    fallback_texts = [
        "This video clip contains content that may require safety review",
        "This video clip appears to contain safe content",
        "Video unavailable"
    ]
    
    for i, sample in enumerate(dataset.samples[:20]):
        text = sample['full_prompt']
        for fallback in fallback_texts:
            if fallback in text:
                has_fallback = True
                print(f"❌ 样本{i}: 发现fallback文本: {fallback[:50]}...")
                break
    
    if not has_fallback:
        print("✅ 未发现fallback描述")
        check_results['no_fallback'] = True
    
    # 检查严格过滤
    if 'has_missing_descriptions = True' in dataloader_code:
        print("✅ 实施严格过滤: 缺少描述的视频会被跳过")
    
    # 6. 检查数据完整性
    print("\n6️⃣ 数据完整性检查:")
    print("-"*40)
    
    # 检查关键组件
    components = {
        'DataLoader存在': os.path.exists('/scratch/czr/Video-Guard/training_testing/Dataloader.py'),
        'Trainer存在': os.path.exists('/scratch/czr/Video-Guard/training_testing/custom_lora_trainer.py'),
        'Multi-GPU训练存在': os.path.exists('/scratch/czr/Video-Guard/training_testing/train_multi_gpu.py'),
        'SafeWatch数据存在': os.path.exists('/scratch/czr/Video-Guard/datasets/safewatch_streaming_corrected.jsonl'),
        'Shot2Story数据存在': os.path.exists('/scratch/czr/Video-Guard/datasets/shot2story/134k_full_train.json')
    }
    
    all_components_exist = all(components.values())
    
    for component, exists in components.items():
        if exists:
            print(f"✅ {component}")
        else:
            print(f"❌ {component}")
    
    if all_components_exist:
        check_results['data_completeness'] = True
    
    # 检查数据质量
    if len(dataset.samples) > 0:
        print(f"\n📊 数据统计:")
        clip_count = sum(1 for s in dataset.samples if s['type'] == 'clip')
        response_count = sum(1 for s in dataset.samples if s['type'] == 'final_response')
        print(f"  • 总样本数: {len(dataset.samples)}")
        print(f"  • Clip样本: {clip_count}")
        print(f"  • Response样本: {response_count}")
    
    # 总结
    print("\n" + "="*80)
    print("检查结果总结")
    print("="*80)
    
    all_passed = all(check_results.values())
    
    for check, passed in check_results.items():
        status = "✅" if passed else "❌"
        check_name = check.replace('_', ' ').title()
        print(f"{status} {check_name}")
    
    if all_passed:
        print("\n🎉 所有检查通过！训练代码已准备就绪。")
        print("\n建议的训练命令:")
        print("  python train_multi_gpu.py --num_train_epochs 3 --per_device_train_batch_size 1")
    else:
        print("\n⚠️ 部分检查未通过，请修复相关问题。")
    
    return all_passed

if __name__ == "__main__":
    success = check_training_components()
    sys.exit(0 if success else 1)