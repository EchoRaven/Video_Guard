#!/usr/bin/env python3
"""
Analyze SafeWatch labels from the actual annotations file
"""

import json
from collections import Counter
from pathlib import Path

# Load the SafeWatch annotations
annotations_file = '/scratch/czr/Video-Guard/datasets/safewatch_live_annotations.json'
with open(annotations_file, 'r') as f:
    annotations = json.load(f)

print("Analyzing SafeWatch Live annotations...")
print("="*60)

total_videos = len(annotations)
safe_videos = 0
unsafe_videos = 0
category_counter = Counter()
annotation_counter = Counter()

# Analyze each video
for entry in annotations:
    video_name = entry['video_name']
    video_path = entry['video_path']
    unsafe_category = entry.get('unsafe_category', '')
    all_categories = entry.get('all_categories', [])
    
    if unsafe_category and unsafe_category != 'safe':
        unsafe_videos += 1
        # Extract category (e.g., "unsafe:C1" -> "C1")
        if ':' in unsafe_category:
            cat = unsafe_category.split(':')[1]
            category_counter[cat] += 1
        
        # Count all categories
        for cat in all_categories:
            if ':' in cat:
                cat_code = cat.split(':')[1]
                category_counter[cat_code] += 1
    else:
        safe_videos += 1
    
    # Count specific annotations
    annotations_list = entry.get('annotations', [])
    for ann in annotations_list:
        annotation_counter[ann] += 1

print(f"Total videos: {total_videos}")
print(f"Safe videos: {safe_videos} ({safe_videos*100/total_videos:.1f}%)")
print(f"Unsafe videos: {unsafe_videos} ({unsafe_videos*100/total_videos:.1f}%)")
print()

print("Unsafe category distribution:")
for cat, count in category_counter.most_common():
    print(f"  {cat}: {count} occurrences")
print()

print("Top 10 specific annotations:")
for ann, count in annotation_counter.most_common(10):
    print(f"  {ann}: {count} occurrences")
print()

# Check the distribution in the corrected JSONL file
print("="*60)
print("Now checking safewatch_streaming_corrected.jsonl...")
print("="*60)

jsonl_file = '/scratch/czr/Video-Guard/datasets/safewatch_streaming_corrected.jsonl'
jsonl_safe = 0
jsonl_unsafe = 0
jsonl_category_counter = Counter()

# Create a mapping from video paths to ground truth
gt_mapping = {}
for entry in annotations:
    video_path = entry['video_path']
    video_name = Path(video_path).name
    gt_mapping[video_name] = entry.get('unsafe_category', 'safe')

# Check JSONL file
with open(jsonl_file, 'r') as f:
    for line_idx, line in enumerate(f):
        if line_idx >= 1000:  # Check first 1000
            break
        
        data = json.loads(line)
        full_video_path = data['full_video_path']
        video_name = Path(full_video_path).name
        
        # Get ground truth from mapping
        if video_name in gt_mapping:
            gt_category = gt_mapping[video_name]
            if gt_category and gt_category != 'safe':
                jsonl_unsafe += 1
                if ':' in gt_category:
                    cat = gt_category.split(':')[1]
                    jsonl_category_counter[cat] += 1
            else:
                jsonl_safe += 1
        else:
            # If not in mapping, check the labels field
            labels = data['full_video_annotation'].get('labels', [])
            if labels:
                jsonl_unsafe += 1
                for label in labels:
                    jsonl_category_counter[label] += 1
            else:
                jsonl_safe += 1

print(f"JSONL file (first 1000 videos):")
print(f"  Safe videos: {jsonl_safe} ({jsonl_safe*100/(jsonl_safe+jsonl_unsafe):.1f}%)")
print(f"  Unsafe videos: {jsonl_unsafe} ({jsonl_unsafe*100/(jsonl_safe+jsonl_unsafe):.1f}%)")
print()
print("JSONL unsafe category distribution:")
for cat, count in jsonl_category_counter.most_common():
    print(f"  {cat}: {count} videos")