#!/usr/bin/env python3
"""
Check the ground truth labels from full_gt.json
"""

import json
from collections import Counter

safewatch_file = '/scratch/czr/Video-Guard/datasets/safewatch_streaming_corrected.jsonl'

total_videos = 0
safe_videos = 0
unsafe_videos = 0
label_counter = Counter()
guardrail_counter = Counter()

print("Checking ground truth labels from full_gt.json...")
print("="*60)

# First, let's see the structure of the data
print("\nChecking data structure (first 3 videos):")
with open(safewatch_file, 'r') as f:
    for line_idx, line in enumerate(f):
        if line_idx >= 3:
            break
        
        data = json.loads(line)
        video_name = data['full_video_path'].split('/')[-1]
        
        print(f"\nVideo {line_idx + 1}: {video_name}")
        
        # Check what keys are available
        print(f"  Top-level keys: {list(data.keys())}")
        
        # Check if there's a full_gt or full_video_annotation_gt
        if 'full_video_annotation_gt' in data:
            print(f"  Has 'full_video_annotation_gt'")
            gt_annotation = data['full_video_annotation_gt']
            print(f"    GT keys: {list(gt_annotation.keys())}")
            if 'labels' in gt_annotation:
                print(f"    GT labels: {gt_annotation['labels']}")
        
        # Check the regular full_video_annotation
        full_annotation = data['full_video_annotation']
        if 'labels' in full_annotation:
            print(f"  Full annotation labels: {full_annotation['labels']}")

print("\n" + "="*60)
print("Analyzing label distribution from available ground truth...")
print("="*60)

# Now analyze the actual distribution
with open(safewatch_file, 'r') as f:
    for line_idx, line in enumerate(f):
        if line_idx >= 1000:  # Check first 1000 videos
            break
            
        total_videos += 1
        data = json.loads(line)
        
        # Try to get ground truth labels from different possible locations
        labels = []
        
        # Option 1: Check full_video_annotation_gt if it exists
        if 'full_video_annotation_gt' in data:
            gt_annotation = data['full_video_annotation_gt']
            labels = gt_annotation.get('labels', [])
        # Option 2: Use labels from full_video_annotation
        else:
            full_annotation = data['full_video_annotation']
            labels = full_annotation.get('labels', [])
        
        # Also check guardrail for ground truth
        guardrail = {}
        if 'full_video_annotation_gt' in data:
            guardrail = data['full_video_annotation_gt'].get('guardrail', {})
        else:
            guardrail = data['full_video_annotation'].get('guardrail', {})
        
        # Count based on labels
        if labels:
            unsafe_videos += 1
            for label in labels:
                label_counter[label] += 1
        else:
            safe_videos += 1
        
        # Count guardrail violations
        for category, is_flagged in guardrail.items():
            if is_flagged:
                guardrail_counter[category] += 1

print(f"Total videos analyzed: {total_videos}")
print(f"Safe videos (no labels): {safe_videos} ({safe_videos*100/total_videos:.1f}%)")
print(f"Unsafe videos (has labels): {unsafe_videos} ({unsafe_videos*100/total_videos:.1f}%)")
print()

print("Label distribution in unsafe videos:")
for label, count in label_counter.most_common():
    print(f"  {label}: {count} videos")
print()

print("Guardrail violations distribution:")
for category, count in guardrail_counter.most_common():
    print(f"  {category}: {count} videos")