#!/usr/bin/env python3
"""
Check the 'labels' field in full_annotation which should indicate safe/unsafe
"""

import json
from collections import Counter

safewatch_file = '/scratch/czr/Video-Guard/datasets/safewatch_streaming_corrected.jsonl'

total_videos = 0
safe_videos = 0  # videos with empty labels list
unsafe_videos = 0  # videos with non-empty labels list
label_counter = Counter()

print("Checking 'labels' field in full_annotation...")
print("="*60)

with open(safewatch_file, 'r') as f:
    for line_idx, line in enumerate(f):
        if line_idx >= 1000:  # Check first 1000 videos
            break
            
        total_videos += 1
        data = json.loads(line)
        
        # Get labels from full_annotation
        full_annotation = data['full_video_annotation']
        labels = full_annotation.get('labels', [])
        
        if labels:  # Non-empty labels mean unsafe
            unsafe_videos += 1
            for label in labels:
                label_counter[label] += 1
        else:  # Empty labels mean safe
            safe_videos += 1

print(f"Total videos analyzed: {total_videos}")
print(f"Safe videos (empty labels): {safe_videos} ({safe_videos*100/total_videos:.1f}%)")
print(f"Unsafe videos (has labels): {unsafe_videos} ({unsafe_videos*100/total_videos:.1f}%)")
print()

print("Label distribution in unsafe videos:")
for label, count in label_counter.most_common():
    print(f"  {label}: {count} videos")
print()

# Also check consistency with guardrail
print("Checking consistency between 'labels' and 'guardrail'...")
consistent = 0
inconsistent = 0
examples = []

with open(safewatch_file, 'r') as f:
    for line_idx, line in enumerate(f):
        if line_idx >= 100:  # Check first 100
            break
            
        data = json.loads(line)
        full_annotation = data['full_video_annotation']
        
        # Get both labels and guardrail
        labels = full_annotation.get('labels', [])
        guardrail = full_annotation.get('guardrail', {})
        
        has_labels = bool(labels)
        has_guardrail_violations = any(guardrail.values())
        
        if has_labels == has_guardrail_violations:
            consistent += 1
        else:
            inconsistent += 1
            if len(examples) < 5:
                video_name = data['full_video_path'].split('/')[-1]
                examples.append({
                    'video': video_name,
                    'labels': labels,
                    'guardrail_violations': [k for k, v in guardrail.items() if v]
                })

print(f"Consistent: {consistent}")
print(f"Inconsistent: {inconsistent}")

if examples:
    print("\nInconsistent examples:")
    for ex in examples:
        print(f"  Video: {ex['video']}")
        print(f"    Labels: {ex['labels']}")
        print(f"    Guardrail violations: {ex['guardrail_violations']}")