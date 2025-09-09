#!/usr/bin/env python3
"""
基础的DataLoader框架
支持多种数据格式和训练模式
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from typing import Dict, List, Optional, Union, Any
import logging
from PIL import Image
import numpy as np
from abc import ABC, abstractmethod
import cv2
import math
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    logging.warning("Decord not available, using cv2 for video loading")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Image preprocessing constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    """Build image transformation pipeline"""
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
        # Convert to bfloat16 to match model weights
        T.Lambda(lambda x: x.to(torch.bfloat16))
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio from target ratios"""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    """Dynamic preprocessing for images with variable aspect ratios"""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    """Load and preprocess a single image"""
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def load_video_frames(video_path, frame_indices, input_size=448, max_num=12):
    """Load specific frames from a video file
    Returns None if video cannot be loaded properly (no fallback to black frames)
    """
    frames = []
    
    try:
        if DECORD_AVAILABLE:
            # Use decord for efficient frame extraction
            vr = VideoReader(video_path, ctx=cpu(0))
            if len(vr) == 0:
                logger.warning(f"Video has no frames: {video_path}")
                return None
                
            for idx in frame_indices:
                idx = min(idx, len(vr) - 1)  # Ensure index is within bounds
                frame = vr[idx].asnumpy()
                frame = Image.fromarray(frame)
                frames.append(frame)
        else:
            # Use cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.warning(f"Cannot open video: {video_path}")
                return None  # Return None instead of black frames
            
            # Check if video has frames
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count == 0:
                logger.warning(f"Video has no frames: {video_path}")
                cap.release()
                return None
                
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    frames.append(frame)
                else:
                    # If any frame cannot be read, fail the entire video
                    logger.warning(f"Cannot read frame {idx} from {video_path}")
                    cap.release()
                    return None
            cap.release()
    except Exception as e:
        logger.warning(f"Error loading video {video_path}: {e}")
        return None
    
    # Verify we got all requested frames
    if len(frames) != len(frame_indices):
        logger.warning(f"Frame count mismatch: expected {len(frame_indices)}, got {len(frames)} for {video_path}")
        return None
    
    # Process each frame
    transform = build_transform(input_size=input_size)
    all_pixel_values = []
    
    for frame in frames:
        # Apply dynamic preprocessing to each frame
        images = dynamic_preprocess(frame, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)  # [num_patches, C, H, W]
        all_pixel_values.append(pixel_values)
    
    # Stack all frames: [num_frames, num_patches, C, H, W]
    if all_pixel_values:
        return torch.stack(all_pixel_values)
    else:
        return None  # Return None instead of empty tensor

def construct_clip_prompt(clip_info: dict, add_user_prompt: bool = True, add_image_placeholder: bool = True) -> str:
    """
    构建streaming分析的clip prompt
    支持多帧序列，每帧包含图像token和对应的响应
    使用新的标签格式：<label>...</label>, <summary>...</summary>
    """
    user_prompt = clip_info['user_prompt']
    # 获取每帧的实际patches数量列表，如果没有则使用默认值
    patches_per_frame_list = clip_info.get('patches_per_frame_list', None)
    default_patches = clip_info.get('num_patches_per_frame', 1)
    
    # InternVL3-8B: 每个patch需要256个IMG_CONTEXT tokens
    tokens_per_patch = 256
    
    video_path = clip_info['video_path']
    clip_labels = clip_info['clip_labels']
    sampled_frame_indices = clip_info['sampled_frame_indices']
    summary = clip_info.get('summary', '')
    
    # 开始构建完整的prompt序列
    clip_prompt = user_prompt if add_user_prompt else ''
    
    # 为每个采样帧构建prompt
    for frame_idx, frame_index in enumerate(sampled_frame_indices):
        # 获取当前帧的实际patches数量
        if patches_per_frame_list and frame_idx < len(patches_per_frame_list):
            current_patches = patches_per_frame_list[frame_idx]
        else:
            current_patches = default_patches
        
        total_tokens_current_frame = current_patches * tokens_per_patch
        
        # 使用<image>占位符，DataCollator会根据实际pixel_values替换为IMG_CONTEXT tokens
        if add_image_placeholder:
            clip_prompt += '<image>'
        else:
            pass  # 不添加图像占位符
        
        # 使用新的标签格式
        clip_prompt += '<label>'
        for label in clip_labels[frame_idx]:
            # 处理特殊的summary标签
            if label == '<summary>':
                clip_prompt += '</label><summary>' + summary + '</summary>'
            else:
                clip_prompt += label
        
        # 如果最后一个标签不是summary，需要关闭label标签
        if clip_labels[frame_idx][-1] != '<summary>':
            clip_prompt += '</label>'
        
        clip_prompt += '\n'
    return clip_prompt

def construct_final_response_prompt(final_response_info: dict) -> str:
    """construct final response prompt with proper closing tags"""
    user_prompt = final_response_info['user_prompt']
    clip_prompts = final_response_info['clip_prompts']
    final_response = final_response_info['final_response']
    
    # Add all clip prompts
    for clip_prompt in clip_prompts:
        user_prompt += clip_prompt
    
    # Insert video end label
    user_prompt += '<|vision_end|>'
    
    # Update final response format to use closing tag
    if final_response.startswith('<response>'):
        # Replace <response> with <response>...</response> format
        final_response = final_response.replace('<response>', '<response>', 1)
        if not final_response.endswith('</response>'):
            final_response += '</response>'
    else:
        # Wrap in response tags if not already present
        final_response = f'<response>{final_response}</response>'
    
    user_prompt += final_response
    return user_prompt


class StreamingDataset(Dataset):
    
    def __init__(self, 
                 dataset_file: str = '/scratch/czr/Video-Guard/datasets',
                 tokenizer = None,
                 max_samples: list[int] = [20000, 20000],  # [shot2story, safewatch] - 限制为20000个样本
                 max_length: int = 8192,  # 减少到8K以适应更大的batch size
                 input_size: int = 448,
                 max_num_patches: int = 6,  # 调整为6，避免token数量过多
                 shuffle: bool = True,  # 是否打乱samples
                 random_seed: int = 42,  # 随机种子，用于复现
                 **kwargs):

        self.dataset_file = dataset_file
        self.tokenizer = tokenizer
        self.max_samples = max_samples
        self.max_length = max_length
        self.input_size = input_size
        self.max_num_patches = max_num_patches
        self.num_image_tokens = 64
        self.shuffle = shuffle
        self.random_seed = random_seed
        
        # 加载数据
        self.samples = self.load_data()
        logger.info(f"📊 Loaded {len(self.samples)} samples from {dataset_file}")
    
    def get_video_info(self, video_path: str) -> tuple:
        """get fps and total frames - 优化版本，快速获取视频信息"""
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                return fps if fps > 0 else 30.0, total_frames if total_frames > 0 else 1800
            else:
                return 30.0, 1800  # 默认值，避免警告日志过多
        except Exception as e:
            return 30.0, 1800  # 默认值
    
    def load_data(self) -> List[Any]:
        # datas from shot 2 story or data from safeawatch
        # load full data first
        # shot 2 story
        shot_2_story_samples = []
        shot2story_path = os.path.join(self.dataset_file, 'shot2story/134k_full_train.json')
        
        if os.path.exists(shot2story_path) and self.max_samples[0] > 0:
            with open(shot2story_path, 'r') as f:
                shot_2_story_raw_data = json.load(f)
            
            # Limit to max_samples[0]
            actual_samples = min(len(shot_2_story_raw_data), self.max_samples[0])
            shot_2_story_raw_data = shot_2_story_raw_data[:actual_samples]
            logger.info(f"Loading Shot2Story data: {actual_samples:,} videos (from {self.max_samples[0]:,} requested)")
            video_root = os.path.join(self.dataset_file, 'shot2story-videos/release_134k_videos')
            
            shot_2_story_user_prompt = """<streaming_analysis> You are a video analyst. Analyze video frames to understand and describe content.
Watch each frame and respond with labels in <label>...</label> tags:
- <unsafe:C1>: Sexual content
- <unsafe:C2>: Harassment/bullying  
- <unsafe:C3>: Violence/harm
- <unsafe:C4>: Misinformation
- <unsafe:C5>: Illegal activities
- <unsafe:C6>: Hate speech/extremism
- <safe>: The frame is safe
- <continue>: Shot is not complete, keep analyzing
For the last frame, use <summary>...</summary> to provide a complete shot description"""
            
        valid_shot2story_count = 0
        logger.info(f"📦 处理Shot2Story数据: {len(shot_2_story_raw_data):,} 个视频")
        
        # 使用tqdm显示进度
        for data in tqdm(shot_2_story_raw_data, desc="Shot2Story加载", unit="video"):
            full_video = data['video']
            final_response = data['whole_caption']  # 修正拼写错误
            clips = data['video_names']
            clips_summaries = data['captions']
            
            # Quality filtering: skip videos without good summaries
            if (not final_response or len(final_response.strip()) < 50 or
                not clips_summaries or len(clips_summaries) == 0):
                continue
                
            # Filter clips with poor summaries
            valid_clips = []
            valid_summaries = []
            for i, (clip_name, summary) in enumerate(zip(clips, clips_summaries)):
                if (summary and len(summary.strip()) >= 20 and 
                    summary.strip().lower() not in ['clip analyzed.', 'no description', 'n/a'] and
                    'error' not in summary.lower()):
                    valid_clips.append(clip_name)
                    valid_summaries.append(summary)
            
            # Skip videos with no valid clips
            if len(valid_clips) == 0:
                continue
                
            clips = valid_clips
            clips_summaries = valid_summaries
            valid_shot2story_count += 1
            
            final_response_info = {'user_prompt': shot_2_story_user_prompt, 'clip_prompts': [], 'final_response': f'<response>{final_response}'}
            # gather frame num for each clip
            for i, video_name in enumerate(clips):
                # "W26nTWGbf3g.8_0_66.mp4" # start from 0 to 66
                parts = video_name.split('.')
                if len(parts) >= 2 and '_' in parts[1]:
                    frame_parts = parts[1].split('_')
                    if len(frame_parts) >= 3:
                        frame_start = int(frame_parts[1])
                        frame_end = int(frame_parts[2])
                        
                        # get video info
                        video_path = os.path.join(self.dataset_file, 'shot2story-videos/release_134k_videos', full_video)
                        fps, total_frames = self.get_video_info(video_path)
                        
                        # calculate clip duration
                        clip_duration = (frame_end - frame_start) / fps if fps > 0 else 1.0
                        
                        # decide sampling strategy based on duration
                        sampled_frame_indices = []
                        if clip_duration > 8:
                            # sample 8 frames equally
                            num_frames = 8
                            interval = (frame_end - frame_start) / num_frames
                            for j in range(num_frames):
                                frame_idx = frame_start + int(j * interval)
                                sampled_frame_indices.append(frame_idx)
                        else:
                            # sample 1 frame per second, but ensure we don't exceed clip bounds
                            num_frames = max(1, int(clip_duration))
                            if num_frames == 1:
                                # For very short clips, just sample the middle frame
                                sampled_frame_indices.append(frame_start + (frame_end - frame_start) // 2)
                            else:
                                # Sample evenly across the clip duration
                                interval = (frame_end - frame_start) / num_frames
                                for j in range(num_frames):
                                    frame_idx = frame_start + int(j * interval)
                                    if frame_idx <= frame_end:
                                        sampled_frame_indices.append(frame_idx)
                        
                        clip_labels = []
                        for j, frame_idx in enumerate(sampled_frame_indices):
                            if j == len(sampled_frame_indices) - 1:
                                clip_labels.append(['<safe>', '<summary>'])
                            else:
                                clip_labels.append(['<safe>', '<continue>'])

                        # 先使用默认patches数量，稍后根据实际加载的视频调整
                        patches_per_frame_list = [self.max_num_patches] * len(sampled_frame_indices)
                        
                        clip_info = {
                            'user_prompt': shot_2_story_user_prompt,
                            'patches_per_frame_list': patches_per_frame_list,  # 使用每帧的实际patches数量
                            'num_patches_per_frame': patches_per_frame_list[0] if patches_per_frame_list else 1,  # 兼容性
                            'video_path': os.path.join(video_root, full_video),
                            'clip_labels': clip_labels,
                            'sampled_frame_indices': sampled_frame_indices,
                            'summary': clips_summaries[i] if i < len(clips_summaries) else ""
                        }
                        final_response_info['clip_prompts'].append(construct_clip_prompt(clip_info, add_user_prompt=False, add_image_placeholder=False))

                        # 生成完整的clip prompt
                        clip_prompt = construct_clip_prompt(clip_info)
                        
                        shot_2_story_samples.append({
                            'type': 'clip',
                            'full_prompt': clip_prompt,
                            'info': clip_info
                        })
                        
            final_response_prompt = construct_final_response_prompt(final_response_info)

            shot_2_story_samples.append({
                'type': 'final_response',
                'full_prompt': final_response_prompt,
                'info': final_response_info
            })

        # Load SafeWatch data
        safewatch_user_prompt = """<streaming_analysis> You are a video analyst. Analyze video frames to understand and describe content.

Watch each frame and respond with labels in <label>...</label> tags:
- <unsafe:C1>: Sexual content
- <unsafe:C2>: Harassment/bullying  
- <unsafe:C3>: Violence/harm
- <unsafe:C4>: Misinformation
- <unsafe:C5>: Illegal activities
- <unsafe:C6>: Hate speech/extremism
- <safe>: The frame is safe
- <continue>: Shot is not complete, keep analyzing
For the last frame, use <summary>...</summary> to provide a complete shot description"""
        
        safewatch_samples = []
        safewatch_jsonl_path = os.path.join(self.dataset_file, 'safewatch_streaming_corrected.jsonl')
        
        if os.path.exists(safewatch_jsonl_path):
            logger.info(f"📦 处理SafeWatch数据 (最多 {self.max_samples[1]:,} 个视频)")
            valid_safewatch_count = 0
            
            # 获取总行数以显示准确的进度
            with open(safewatch_jsonl_path, 'r') as f:
                total_lines = sum(1 for _ in f)
            
            with open(safewatch_jsonl_path, 'r') as f:
                progress_bar = tqdm(f, total=min(total_lines, self.max_samples[1]), desc="SafeWatch加载", unit="video")
                for line_idx, line in enumerate(progress_bar):
                    if len(safewatch_samples) >= self.max_samples[1]:  # max_samples[1] for SafeWatch
                        break
                    
                    data = json.loads(line)
                    full_video_path = data['full_video_path']
                    clip_video_paths = data['clip_video_paths']
                    clip_video_labels = data['clip_video_labels']
                    clip_annotations = data['clip_video_annotations']
                    full_annotation = data['full_video_annotation']
                    
                    # Quality filtering: check final response quality
                    final_description = full_annotation.get('description', '')
                    if (not final_description or len(final_description.strip()) < 20 or
                        final_description.strip().lower() in ['video analyzed for safety.', 'no description', 'n/a']):
                        continue
                    
                    # Filter clips with valid annotations - STRICT: must have good descriptions
                    valid_clip_paths = []
                    valid_clip_labels = []
                    valid_clip_annotations = []
                    has_missing_descriptions = False
                    
                    for i, (clip_path, clip_labels, clip_annotation) in enumerate(zip(clip_video_paths, clip_video_labels, clip_annotations)):
                        # Check if clip has meaningful description
                        clip_desc = clip_annotation.get('description', '') if isinstance(clip_annotation, dict) else ''
                        
                        # STRICT: Require good description (no fallback)
                        if os.path.exists(clip_path) and clip_desc and len(clip_desc.strip()) >= 20:
                            valid_clip_paths.append(clip_path)
                            valid_clip_labels.append(clip_labels)
                            valid_clip_annotations.append(clip_annotation)
                        else:
                            # Mark that this video has missing clip descriptions
                            has_missing_descriptions = True
                    
                    # Skip videos if ANY clip is missing descriptions or if no valid clips
                    if has_missing_descriptions or len(valid_clip_paths) == 0:
                        continue
                    
                    # Also verify all clips have descriptions (double check)
                    if len(valid_clip_paths) != len(clip_video_paths):
                        # Not all clips are valid, skip this entire video
                        continue
                    
                    # Update with filtered data
                    clip_video_paths = valid_clip_paths
                    clip_video_labels = valid_clip_labels
                    clip_annotations = valid_clip_annotations
                    valid_safewatch_count += 1
                    
                    # Process final response
                    final_response_info = {
                        'user_prompt': safewatch_user_prompt,
                        'clip_prompts': [],
                        'final_response': f"<response>{final_description}"
                    }
                    
                    # Process each clip
                    all_clips_valid = True  # Track if all clips have valid descriptions
                    clip_infos = []  # Store clip infos temporarily
                    
                    for clip_idx, clip_path in enumerate(clip_video_paths):
                        # Get video info
                        fps, total_frames = self.get_video_info(clip_path)
                        
                        # Calculate actual clip duration from video
                        if fps > 0:
                            clip_duration = total_frames / fps
                        else:
                            clip_duration = 2.0  # fallback to 2 seconds if fps is invalid
                        
                        # Sample frames (1 per second for short clips, max 8 for longer)
                        sampled_frame_indices = []
                        if clip_duration > 8:
                            # Sample 8 frames equally spaced
                            num_frames = 8
                            interval = total_frames / num_frames
                            for j in range(num_frames):
                                frame_idx = int(j * interval)
                                sampled_frame_indices.append(frame_idx)
                        else:
                            # Sample 1 frame per second
                            num_frames = max(1, int(clip_duration))
                            for j in range(num_frames):
                                frame_idx = int(j * fps)
                                if frame_idx < total_frames:
                                    sampled_frame_indices.append(frame_idx)
                        
                        # Get labels for this clip
                        current_clip_labels = clip_video_labels[clip_idx] if clip_idx < len(clip_video_labels) else []
                        
                        # Try to get clip description from multiple sources (NO FALLBACK)
                        clip_description = ""
                        
                        # 1. First try clip_descriptions from full_video_annotation
                        clip_descriptions_list = full_annotation.get('clip_descriptions', [])
                        if clip_idx < len(clip_descriptions_list):
                            clip_description = clip_descriptions_list[clip_idx]
                        
                        # 2. If no description, try individual clip annotation
                        if not clip_description or len(clip_description.strip()) < 15:
                            clip_annotation = clip_annotations[clip_idx] if clip_idx < len(clip_annotations) else {}
                            if isinstance(clip_annotation, dict):
                                clip_description = clip_annotation.get('description', '')
                        
                        # 3. NO FALLBACK - if still no description, this clip will be invalid
                        if not clip_description or len(clip_description.strip()) < 20:
                            # Mark video as invalid and break
                            all_clips_valid = False
                            break
                        
                        # Build frame labels
                        frame_labels = []
                        for j, frame_idx in enumerate(sampled_frame_indices):
                            if j == len(sampled_frame_indices) - 1:
                                # Last frame of clip
                                if current_clip_labels:
                                    # If clip has unsafe labels, add them + summary
                                    labels = []
                                    for label in current_clip_labels:
                                        labels.append(f'<unsafe:{label}>')
                                    labels.append('<summary>')
                                    frame_labels.append(labels)
                                else:
                                    # Safe clip
                                    frame_labels.append(['<safe>', '<summary>'])
                            else:
                                # Not last frame
                                if current_clip_labels:
                                    # If clip has unsafe labels, add them + continue
                                    labels = []
                                    for label in current_clip_labels:
                                        labels.append(f'<unsafe:{label}>')
                                    labels.append('<continue>')
                                    frame_labels.append(labels)
                                else:
                                    # Safe clip
                                    frame_labels.append(['<safe>', '<continue>'])
                        
                        # 不再预加载视频，直接使用固定的patches数量
                        # 视频加载将在DataCollator中进行
                        patches_per_frame_list = [self.max_num_patches] * len(sampled_frame_indices)
                        
                        # Create clip info
                        clip_info = {
                            'user_prompt': safewatch_user_prompt,
                            'patches_per_frame_list': patches_per_frame_list,  # 使用每帧的实际patches数量
                            'num_patches_per_frame': patches_per_frame_list[0] if patches_per_frame_list else 1,  # 兼容性
                            'video_path': clip_path,
                            'clip_labels': frame_labels,
                            'sampled_frame_indices': sampled_frame_indices,
                            'summary': clip_description  # Must have valid description (no fallback)
                        }
                        
                        # Store clip info temporarily
                        clip_infos.append(clip_info)
                    
                    # Only add samples if ALL clips have valid descriptions
                    if all_clips_valid and len(clip_infos) == len(clip_video_paths):
                        # Now add all the clip samples and final response
                        for clip_info in clip_infos:
                            final_response_info['clip_prompts'].append(
                                construct_clip_prompt(clip_info, add_user_prompt=False, add_image_placeholder=False)
                            )
                            
                            # Generate complete clip prompt
                            clip_prompt = construct_clip_prompt(clip_info)
                            
                            safewatch_samples.append({
                                'type': 'clip',
                                'full_prompt': clip_prompt,
                                'info': clip_info
                            })
                        
                        # Add final response only if all clips are valid
                        final_response_prompt = construct_final_response_prompt(final_response_info)
                        safewatch_samples.append({
                            'type': 'final_response', 
                            'full_prompt': final_response_prompt,
                            'info': final_response_info
                        })
                    # else: Skip this video entirely since not all clips have descriptions
                
                # 关闭进度条
                progress_bar.close()
        
        # Combine all samples
        all_samples = shot_2_story_samples + safewatch_samples
        
        # Shuffle samples for better training if enabled
        if self.shuffle:
            import random
            random.seed(self.random_seed)  # 设置种子以便复现
            random.shuffle(all_samples)
            logger.info(f"✨ Shuffled all samples for mixed training (seed={self.random_seed})")
        else:
            logger.info("📝 Samples kept in original order (shuffle=False)")
        
        # Log detailed statistics
        logger.info(f"📊 Data Loading Summary:")
        logger.info(f"  Shot2Story: {len(shot_2_story_samples)} samples from {valid_shot2story_count} videos")
        logger.info(f"  SafeWatch: {len(safewatch_samples)} samples from {valid_safewatch_count} videos")
        logger.info(f"  Total: {len(all_samples)} high-quality samples" + (" (shuffled)" if self.shuffle else ""))
        
        return all_samples

    def process_sample(self, sample: Any, idx: int) -> Dict[str, Any]:
        """处理单个样本 - 按照MJ-Video方式：只返回文本和pixel_values，不tokenize"""
        full_prompt = sample['full_prompt']
        info = sample.get('info', {})
        
        # 加载视频数据
        pixel_values = None
        num_patches_list = []
        
        if 'video_path' in info and 'sampled_frame_indices' in info:
            video_path = info['video_path']
            frame_indices = info['sampled_frame_indices']
            
            if os.path.exists(video_path):
                try:
                    pixel_values = load_video_frames(
                        video_path, 
                        frame_indices,
                        input_size=self.input_size,
                        max_num=self.max_num_patches
                    )
                    
                    # Check if video was loaded successfully
                    if pixel_values is None:
                        logger.warning(f"Video loading returned None for {video_path}, skipping sample")
                        # Return None to indicate this sample should be skipped
                        return None
                    
                    # 计算每帧的patches数量 (参考MJ-Video)
                    if len(pixel_values.shape) == 5:
                        # pixel_values形状: [frames, patches, C, H, W]
                        frames, patches_per_frame, C, H, W = pixel_values.shape
                        
                        # 每帧的patches数量应该相同，所以num_patches_list是每帧的实际patches数
                        for frame_idx in range(frames):
                            num_patches_list.append(patches_per_frame)
                        
                        # 重新整形为MJ-Video格式: [total_patches, C, H, W]
                        pixel_values = pixel_values.view(frames * patches_per_frame, C, H, W)
                        
                except Exception as e:
                    logger.warning(f"Failed to load video {video_path}: {e}")
                    # Return None to skip this sample
                    return None
            else:
                logger.warning(f"Video file does not exist: {video_path}")
                return None
        
        # 按照MJ-Video方式：只返回文本和pixel_values，让DataCollator处理tokenization
        return {
            'text': full_prompt,  # 原始文本，包含<image>占位符
            'pixel_values': pixel_values,
            'num_patches_list': num_patches_list,  # 使用实际计算的patches数量，不是预设值
            'type': sample['type']
        }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Try to process the sample
        result = self.process_sample(self.samples[idx], idx)
        
        # If video loading failed, try next samples until we find a valid one
        attempts = 0
        while result is None and attempts < 10:
            # Try next sample (wrap around if needed)
            idx = (idx + 1) % len(self.samples)
            result = self.process_sample(self.samples[idx], idx)
            attempts += 1
        
        # If still None after multiple attempts, return a minimal valid sample
        if result is None:
            logger.error(f"Could not find valid sample after {attempts} attempts")
            # Return a minimal text-only sample to avoid crashing
            return {
                'text': 'Video unavailable.',
                'pixel_values': None,
                'num_patches_list': [],
                'type': 'error'
            }
        
        return result