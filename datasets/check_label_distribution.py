#!/usr/bin/env python3
"""
Check the actual label distribution in SafeWatch data
"""

import json
from collections import Counter

safewatch_file = '/scratch/czr/Video-Guard/datasets/safewatch_streaming_corrected.jsonl'

total_videos = 0
safe_videos = 0
unsafe_videos = 0
label_counter = Counter()
guardrail_counter = Counter()

print("Checking SafeWatch label distribution...")
print("="*60)

with open(safewatch_file, 'r') as f:
    for line_idx, line in enumerate(f):
        if line_idx >= 500:  # Check first 500 videos
            break
            
        total_videos += 1
        data = json.loads(line)
        
        # Check clip labels
        clip_video_labels = data['clip_video_labels']
        full_annotation = data['full_video_annotation']
        
        # Check if video has any unsafe clips
        has_unsafe = False
        for clip_labels in clip_video_labels:
            if clip_labels:  # Non-empty list means unsafe
                has_unsafe = True
                for label in clip_labels:
                    label_counter[label] += 1
        
        if has_unsafe:
            unsafe_videos += 1
        else:
            safe_videos += 1
        
        # Check guardrail flags
        guardrail = full_annotation.get('guardrail', {})
        for category, is_flagged in guardrail.items():
            if is_flagged:
                guardrail_counter[category] += 1

print(f"Total videos analyzed: {total_videos}")
print(f"Safe videos: {safe_videos} ({safe_videos*100/total_videos:.1f}%)")
print(f"Unsafe videos: {unsafe_videos} ({unsafe_videos*100/total_videos:.1f}%)")
print()

print("Label distribution in clips:")
for label, count in label_counter.most_common():
    print(f"  {label}: {count} occurrences")
print()

print("Guardrail flags distribution:")
for category, count in guardrail_counter.most_common():
    print(f"  {category}: {count} videos flagged")

# Check consistency between labels and guardrails
print("\n" + "="*60)
print("CHECKING CONSISTENCY:")
print("="*60)

# Re-check for consistency
consistent = 0
inconsistent = 0
examples = []

with open(safewatch_file, 'r') as f:
    for line_idx, line in enumerate(f):
        if line_idx >= 100:  # Check first 100
            break
            
        data = json.loads(line)
        video_name = data['full_video_path'].split('/')[-1]
        
        # Get clip labels
        clip_video_labels = data['clip_video_labels']
        has_unsafe_clips = any(clip_labels for clip_labels in clip_video_labels)
        
        # Get guardrail
        guardrail = data['full_video_annotation'].get('guardrail', {})
        has_guardrail_flags = any(guardrail.values())
        
        # Check consistency
        if has_unsafe_clips == has_guardrail_flags:
            consistent += 1
        else:
            inconsistent += 1
            if len(examples) < 5:
                examples.append({
                    'video': video_name,
                    'has_unsafe_clips': has_unsafe_clips,
                    'has_guardrail_flags': has_guardrail_flags,
                    'clip_labels': clip_video_labels,
                    'guardrail': guardrail
                })

print(f"Consistent videos: {consistent}")
print(f"Inconsistent videos: {inconsistent}")

if examples:
    print("\nInconsistent examples:")
    for ex in examples:
        print(f"  Video: {ex['video']}")
        print(f"    Has unsafe clips: {ex['has_unsafe_clips']}")
        print(f"    Has guardrail flags: {ex['has_guardrail_flags']}")
        print(f"    Clip labels: {ex['clip_labels'][:3]}...")  # Show first 3