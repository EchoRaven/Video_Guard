#!/usr/bin/env python3
"""
Fix video paths in safewatch_streaming_corrected.jsonl to match actual file structure
"""

import json
import os
from pathlib import Path
from tqdm import tqdm

print("Fixing video paths in SafeWatch JSONL...")
print("="*80)

# First, let's understand the actual directory structure
base_dir = '/scratch/czr/Video-Guard/datasets/SafeWatch-Bench-200K-720P'
clip_dir = f'{base_dir}/clip'
full_dir = f'{base_dir}/full'

# Get all actual directories in clip and full
clip_categories = set(os.listdir(clip_dir))
full_categories = set(os.listdir(full_dir))

print(f"Found {len(clip_categories)} clip categories: {sorted(clip_categories)}")
print(f"Found {len(full_categories)} full categories: {sorted(full_categories)}")

# Create mapping from incorrect to correct category names
category_mapping = {
    'benign_child': 'child',
    'benign_animal': 'benign',
    'unsafe_sexual': 'sexual_2',  # Need to check which one
    'unsafe_violence': 'violence_1',  # Need to check which one
    'unsafe_abuse': 'abuse_1',
    'unsafe_crash': 'crash_1',
    'unsafe_extremism': 'extremism',
    'unsafe_illegal': 'illegal',
    'unsafe_misinformation': 'misinformation_1',
    'unsafe_religion': 'religion_1',
}

# Let's check a few videos to understand the mapping better
print("\nChecking path patterns...")
input_file = '/scratch/czr/Video-Guard/datasets/safewatch_streaming_corrected.jsonl'
sample_paths = []

with open(input_file, 'r') as f:
    for i, line in enumerate(f):
        if i >= 20:  # Sample first 20
            break
        data = json.loads(line)
        full_path = data['full_video_path']
        clip_paths = data.get('clip_video_paths', [])
        sample_paths.append({
            'full': full_path,
            'clips': clip_paths[:2] if clip_paths else []  # First 2 clips
        })

print("\nSample original paths:")
for i, sample in enumerate(sample_paths[:5]):
    print(f"\nVideo {i+1}:")
    print(f"  Full: {sample['full']}")
    if sample['clips']:
        print(f"  Clip: {sample['clips'][0]}")

def fix_video_path(path):
    """Fix a single video path to match actual directory structure"""
    
    # Extract parts of the path
    parts = path.split('/')
    
    # Find where SafeWatch-Bench starts
    if 'SafeWatch-Bench-200K-720P' in path:
        idx = parts.index('SafeWatch-Bench-200K-720P')
        
        if idx + 2 < len(parts):
            path_type = parts[idx + 1]  # 'clip' or 'full'
            category = parts[idx + 2]    # e.g., 'benign_child'
            
            # Map category to correct name
            if category in category_mapping:
                parts[idx + 2] = category_mapping[category]
            # Check if category exists in actual directories
            elif path_type == 'clip' and category not in clip_categories:
                # Try to find a matching category
                for actual_cat in clip_categories:
                    if category.replace('_', '') in actual_cat or actual_cat in category:
                        parts[idx + 2] = actual_cat
                        break
            elif path_type == 'full' and category not in full_categories:
                # Try to find a matching category
                for actual_cat in full_categories:
                    if category.replace('_', '') in actual_cat or actual_cat in category:
                        parts[idx + 2] = actual_cat
                        break
    
    return '/'.join(parts)

# Now fix all paths in the JSONL file
print("\n" + "="*80)
print("Fixing all video paths...")
print("="*80)

output_file = '/scratch/czr/Video-Guard/datasets/safewatch_streaming_fixed.jsonl'
total_videos = 0
fixed_videos = 0
videos_with_all_clips_found = 0
videos_with_some_clips_missing = 0

# Count total lines first
with open(input_file, 'r') as f:
    total_lines = sum(1 for _ in f)

with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    for line in tqdm(f_in, total=total_lines, desc="Processing videos"):
        total_videos += 1
        data = json.loads(line)
        
        # Fix full video path
        original_full = data['full_video_path']
        fixed_full = fix_video_path(original_full)
        data['full_video_path'] = fixed_full
        
        if fixed_full != original_full:
            fixed_videos += 1
        
        # Fix clip paths
        original_clips = data.get('clip_video_paths', [])
        fixed_clips = []
        all_clips_found = True
        
        for clip_path in original_clips:
            fixed_clip = fix_video_path(clip_path)
            fixed_clips.append(fixed_clip)
            
            # Check if fixed path exists
            if not os.path.exists(fixed_clip):
                all_clips_found = False
        
        data['clip_video_paths'] = fixed_clips
        
        if all_clips_found and len(fixed_clips) > 0:
            videos_with_all_clips_found += 1
        elif len(fixed_clips) > 0:
            videos_with_some_clips_missing += 1
        
        # Write fixed data
        f_out.write(json.dumps(data) + '\n')

print(f"\nProcessing complete!")
print(f"Total videos processed: {total_videos}")
print(f"Videos with paths fixed: {fixed_videos}")
print(f"Videos with all clips found: {videos_with_all_clips_found}")
print(f"Videos with some clips missing: {videos_with_some_clips_missing}")
print(f"\nFixed JSONL saved to: {output_file}")

# Verify the fixes
print("\n" + "="*80)
print("Verifying fixed paths...")
print("="*80)

verified_found = 0
verified_missing = 0
sample_size = min(1000, total_videos)

with open(output_file, 'r') as f:
    for i, line in enumerate(f):
        if i >= sample_size:
            break
        
        data = json.loads(line)
        
        # Check full video
        if os.path.exists(data['full_video_path']):
            verified_found += 1
        else:
            verified_missing += 1
        
        # Check clips
        for clip_path in data.get('clip_video_paths', []):
            if os.path.exists(clip_path):
                verified_found += 1
            else:
                verified_missing += 1

print(f"Verification (first {sample_size} videos):")
print(f"  Paths found: {verified_found}")
print(f"  Paths still missing: {verified_missing}")
print(f"  Success rate: {verified_found*100/(verified_found+verified_missing):.1f}%")