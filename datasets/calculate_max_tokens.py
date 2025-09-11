#!/usr/bin/env python3
"""
Calculate maximum possible token count for video data
"""

import json
import os

print("Calculating maximum token count for videos...")
print("="*80)

# InternVL3-8B uses 256 tokens per image patch
TOKENS_PER_PATCH = 256

# From Dataloader settings
MAX_NUM_PATCHES = 6  # max_num_patches parameter in Dataloader

print("Model: InternVL3-8B")
print(f"Tokens per patch: {TOKENS_PER_PATCH}")
print(f"Max patches per frame: {MAX_NUM_PATCHES}")
print()

# Analyze SafeWatch data
print("1. SafeWatch Analysis:")
print("-"*40)

safewatch_file = '/scratch/czr/Video-Guard/datasets/safewatch_streaming_fixed_v2.jsonl'
max_clips = 0
max_frames_per_clip = 0
total_max_frames = 0
examples = []

with open(safewatch_file, 'r') as f:
    for i, line in enumerate(f):
        if i >= 100:  # Sample first 100
            break
        
        data = json.loads(line)
        clip_paths = data.get('clip_video_paths', [])
        num_clips = len(clip_paths)
        
        if num_clips > max_clips:
            max_clips = num_clips
            
        # Estimate frames per clip based on filename patterns
        frames_in_video = 0
        for clip_path in clip_paths:
            # Extract clip duration from filename (e.g., 000000_000006.mp4 = 6 seconds)
            filename = os.path.basename(clip_path)
            if '_' in filename:
                parts = filename.replace('.mp4', '').split('_')
                if len(parts) == 2:
                    try:
                        start = int(parts[0])
                        end = int(parts[1])
                        duration = end - start
                        
                        # From Dataloader: if duration > 8, sample 8 frames; else 1 frame per second
                        if duration > 8:
                            frames = 8
                        else:
                            frames = max(1, duration)
                        
                        frames_in_video += frames
                        if frames > max_frames_per_clip:
                            max_frames_per_clip = frames
                    except:
                        frames_in_video += 4  # Default estimate
        
        if frames_in_video > total_max_frames:
            total_max_frames = frames_in_video
            examples.append({
                'video': data['full_video_path'].split('/')[-1],
                'num_clips': num_clips,
                'total_frames': frames_in_video
            })

print(f"Max clips per video: {max_clips}")
print(f"Max frames per clip: {max_frames_per_clip}")
print(f"Max total frames per video: {total_max_frames}")
print()

# Calculate tokens for SafeWatch
safewatch_clip_tokens = max_frames_per_clip * MAX_NUM_PATCHES * TOKENS_PER_PATCH
safewatch_total_tokens = total_max_frames * MAX_NUM_PATCHES * TOKENS_PER_PATCH

print("SafeWatch token calculations:")
print(f"  Per clip (worst case): {max_frames_per_clip} frames × {MAX_NUM_PATCHES} patches × {TOKENS_PER_PATCH} tokens")
print(f"  = {safewatch_clip_tokens:,} tokens per clip")
print(f"  Total per video (worst case): {total_max_frames} frames × {MAX_NUM_PATCHES} patches × {TOKENS_PER_PATCH} tokens")
print(f"  = {safewatch_total_tokens:,} tokens for video")
print()

# Add text tokens (prompts, responses)
print("2. Text Token Estimates:")
print("-"*40)
print("User prompt: ~100 tokens")
print("Frame labels + summaries: ~50 tokens per frame")
print(f"Total text for {total_max_frames} frames: ~{total_max_frames * 50} tokens")
print("Final response: ~200 tokens")
print()

total_text_tokens = 100 + (total_max_frames * 50) + 200
grand_total = safewatch_total_tokens + total_text_tokens

print("="*80)
print("TOTAL TOKEN COUNT (WORST CASE):")
print("="*80)
print(f"Video tokens: {safewatch_total_tokens:,}")
print(f"Text tokens: {total_text_tokens:,}")
print(f"GRAND TOTAL: {grand_total:,} tokens")
print()

# Check against max_length setting
max_length = 16384  # From launch script
print(f"Current max_length setting: {max_length:,}")
if grand_total > max_length:
    print(f"⚠️  WARNING: Maximum tokens ({grand_total:,}) exceeds max_length ({max_length:,})!")
    print(f"   Overflow: {grand_total - max_length:,} tokens")
    print("   This will cause training errors!")
    
    # Calculate safe settings
    safe_frames = max_length // (MAX_NUM_PATCHES * TOKENS_PER_PATCH + 50)
    print(f"\n   Recommended: Limit to ~{safe_frames} frames total per video")
    
    safe_patches = max_length // (total_max_frames * TOKENS_PER_PATCH + total_max_frames * 50)
    print(f"   Or reduce max_num_patches to {safe_patches}")
else:
    print(f"✅ OK: Maximum tokens ({grand_total:,}) fits within max_length ({max_length:,})")
    
print()
print("Example videos with high frame counts:")
for ex in examples[:3]:
    print(f"  - {ex['video']}: {ex['num_clips']} clips, {ex['total_frames']} frames")