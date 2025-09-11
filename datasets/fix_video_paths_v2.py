#!/usr/bin/env python3
"""
Fix video paths in safewatch_streaming_corrected.jsonl to match actual file structure
Version 2: More comprehensive mapping
"""

import json
import os
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

print("Fixing video paths in SafeWatch JSONL (V2)...")
print("="*80)

base_dir = '/scratch/czr/Video-Guard/datasets/SafeWatch-Bench-200K-720P'

# Build a comprehensive mapping by checking actual paths
print("Building path mapping from actual files...")

# For clips: map video_id -> actual category
clip_mapping = {}
clip_dir = f'{base_dir}/clip'
for category in os.listdir(clip_dir):
    category_path = os.path.join(clip_dir, category)
    if os.path.isdir(category_path):
        for video_id in os.listdir(category_path):
            video_id_path = os.path.join(category_path, video_id)
            if os.path.isdir(video_id_path):
                clip_mapping[video_id] = category

print(f"Found {len(clip_mapping)} video IDs in clip directories")

# For full videos: map video_filename -> (category, subdir)
full_mapping = {}
full_dir = f'{base_dir}/full'
for category in os.listdir(full_dir):
    category_path = os.path.join(full_dir, category)
    if os.path.isdir(category_path):
        # Check for subdirectories like 'target'
        for item in os.listdir(category_path):
            item_path = os.path.join(category_path, item)
            if os.path.isdir(item_path):
                # This is a subdirectory, check for videos inside
                for video_file in os.listdir(item_path):
                    if video_file.endswith('.mp4'):
                        video_name = video_file.replace('.mp4', '')
                        full_mapping[video_name] = (category, item)
            elif item.endswith('.mp4'):
                # Video directly in category folder
                video_name = item.replace('.mp4', '')
                full_mapping[video_name] = (category, None)

print(f"Found {len(full_mapping)} videos in full directories")

def fix_clip_path(path):
    """Fix a clip video path using the mapping"""
    parts = path.split('/')
    
    # Find the video ID (it's the directory name before the clip filename)
    for i, part in enumerate(parts):
        if part.endswith('.mp4') and i > 0:
            video_id = parts[i-1]
            if video_id in clip_mapping:
                # Found the correct category
                correct_category = clip_mapping[video_id]
                # Find where to replace the category in the path
                if 'SafeWatch-Bench-200K-720P' in path:
                    idx = parts.index('SafeWatch-Bench-200K-720P')
                    if idx + 2 < len(parts):
                        parts[idx + 2] = correct_category
                        return '/'.join(parts)
    
    return path  # Return original if no mapping found

def fix_full_path(path):
    """Fix a full video path using the mapping"""
    # Extract video filename
    video_file = os.path.basename(path)
    if video_file.endswith('.mp4'):
        video_name = video_file.replace('.mp4', '')
        
        if video_name in full_mapping:
            category, subdir = full_mapping[video_name]
            # Construct the correct path
            if subdir:
                return f"{base_dir}/full/{category}/{subdir}/{video_file}"
            else:
                return f"{base_dir}/full/{category}/{video_file}"
    
    return path  # Return original if no mapping found

# Process the JSONL file
print("\n" + "="*80)
print("Processing JSONL file...")
print("="*80)

input_file = '/scratch/czr/Video-Guard/datasets/safewatch_streaming_corrected.jsonl'
output_file = '/scratch/czr/Video-Guard/datasets/safewatch_streaming_fixed_v2.jsonl'

# Count total lines
with open(input_file, 'r') as f:
    total_lines = sum(1 for _ in f)

stats = {
    'total': 0,
    'full_fixed': 0,
    'clips_fixed': 0,
    'full_found': 0,
    'clips_found': 0,
    'clips_missing': 0
}

with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    for line in tqdm(f_in, total=total_lines, desc="Processing videos"):
        stats['total'] += 1
        data = json.loads(line)
        
        # Fix full video path
        original_full = data['full_video_path']
        fixed_full = fix_full_path(original_full)
        data['full_video_path'] = fixed_full
        
        if fixed_full != original_full:
            stats['full_fixed'] += 1
        
        if os.path.exists(fixed_full):
            stats['full_found'] += 1
        
        # Fix clip paths
        original_clips = data.get('clip_video_paths', [])
        fixed_clips = []
        
        for clip_path in original_clips:
            fixed_clip = fix_clip_path(clip_path)
            fixed_clips.append(fixed_clip)
            
            if fixed_clip != clip_path:
                stats['clips_fixed'] += 1
            
            if os.path.exists(fixed_clip):
                stats['clips_found'] += 1
            else:
                stats['clips_missing'] += 1
        
        data['clip_video_paths'] = fixed_clips
        
        # Write fixed data
        f_out.write(json.dumps(data) + '\n')

print("\n" + "="*80)
print("RESULTS:")
print("="*80)
print(f"Total videos processed: {stats['total']}")
print(f"Full videos fixed: {stats['full_fixed']}")
print(f"Full videos found after fixing: {stats['full_found']} ({stats['full_found']*100/stats['total']:.1f}%)")
print(f"Clip paths fixed: {stats['clips_fixed']}")
print(f"Clip paths found after fixing: {stats['clips_found']}")
print(f"Clip paths still missing: {stats['clips_missing']}")
if stats['clips_found'] + stats['clips_missing'] > 0:
    print(f"Clip success rate: {stats['clips_found']*100/(stats['clips_found']+stats['clips_missing']):.1f}%")

print(f"\nFixed JSONL saved to: {output_file}")

# Quick verification
print("\n" + "="*80)
print("Quick verification (first 100 videos)...")
print("="*80)

verify_stats = defaultdict(int)
with open(output_file, 'r') as f:
    for i, line in enumerate(f):
        if i >= 100:
            break
        
        data = json.loads(line)
        
        # Check full video
        if os.path.exists(data['full_video_path']):
            verify_stats['full_found'] += 1
        else:
            verify_stats['full_missing'] += 1
            if i < 5:  # Print first few missing
                print(f"Missing full: {data['full_video_path']}")
        
        # Check clips
        for clip_path in data.get('clip_video_paths', []):
            if os.path.exists(clip_path):
                verify_stats['clip_found'] += 1
            else:
                verify_stats['clip_missing'] += 1
                if verify_stats['clip_missing'] <= 5:  # Print first few missing
                    print(f"Missing clip: {clip_path}")

print(f"\nVerification results (first 100 videos):")
print(f"  Full videos found: {verify_stats['full_found']}/100")
print(f"  Clips found: {verify_stats['clip_found']}")
print(f"  Clips missing: {verify_stats['clip_missing']}")
total_paths = verify_stats['full_found'] + verify_stats['full_missing'] + verify_stats['clip_found'] + verify_stats['clip_missing']
total_found = verify_stats['full_found'] + verify_stats['clip_found']
print(f"  Overall success rate: {total_found*100/total_paths:.1f}%")