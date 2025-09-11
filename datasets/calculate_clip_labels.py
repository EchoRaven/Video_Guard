#!/usr/bin/env python3
"""
Calculate clip-level label statistics for SafeWatch-720P
"""

import json
from collections import Counter

jsonl_file = '/scratch/czr/Video-Guard/datasets/safewatch_streaming_corrected.jsonl'

print("SafeWatch-720P Clip-Level Label Statistics")
print("="*80)

# Initialize counters
total_videos = 0
total_clips = 0
safe_clips = 0
unsafe_clips = 0
clip_label_counter = Counter()

# Also track video-level stats
videos_all_safe_clips = 0
videos_all_unsafe_clips = 0
videos_mixed_clips = 0

# Process each video
with open(jsonl_file, 'r') as f:
    for line_idx, line in enumerate(f):
        total_videos += 1
        data = json.loads(line)
        
        # Get clip labels
        clip_video_labels = data.get('clip_video_labels', [])
        
        # Count clips in this video
        num_clips_in_video = len(clip_video_labels)
        total_clips += num_clips_in_video
        
        # Track safe/unsafe clips in this video
        safe_clips_in_video = 0
        unsafe_clips_in_video = 0
        
        # Process each clip
        for clip_idx, clip_labels in enumerate(clip_video_labels):
            if clip_labels:  # Non-empty list means unsafe
                unsafe_clips += 1
                unsafe_clips_in_video += 1
                # Count each label
                for label in clip_labels:
                    clip_label_counter[label] += 1
            else:  # Empty list means safe
                safe_clips += 1
                safe_clips_in_video += 1
        
        # Classify video based on clip composition
        if safe_clips_in_video == num_clips_in_video:
            videos_all_safe_clips += 1
        elif unsafe_clips_in_video == num_clips_in_video:
            videos_all_unsafe_clips += 1
        else:
            videos_mixed_clips += 1

print(f"TOTAL VIDEOS: {total_videos}")
print(f"TOTAL CLIPS: {total_clips}")
print(f"Average clips per video: {total_clips/total_videos:.1f}")
print()

print("CLIP-LEVEL STATISTICS:")
print("-"*40)
print(f"Safe clips (empty labels []): {safe_clips} ({safe_clips*100/total_clips:.1f}%)")
print(f"Unsafe clips (has labels): {unsafe_clips} ({unsafe_clips*100/total_clips:.1f}%)")
print()

print("Unsafe clip label distribution:")
for label, count in clip_label_counter.most_common():
    print(f"  {label}: {count} clips ({count*100/total_clips:.2f}% of all clips)")
print()

print("VIDEO-LEVEL CLIP COMPOSITION:")
print("-"*40)
print(f"Videos with ALL safe clips: {videos_all_safe_clips} ({videos_all_safe_clips*100/total_videos:.1f}%)")
print(f"Videos with ALL unsafe clips: {videos_all_unsafe_clips} ({videos_all_unsafe_clips*100/total_videos:.1f}%)")
print(f"Videos with MIXED safe/unsafe clips: {videos_mixed_clips} ({videos_mixed_clips*100/total_videos:.1f}%)")
print()

# Sample some mixed videos to understand the pattern
print("ANALYZING MIXED VIDEOS (first 10):")
print("-"*40)
mixed_examples = []
with open(jsonl_file, 'r') as f:
    for line_idx, line in enumerate(f):
        if len(mixed_examples) >= 10:
            break
            
        data = json.loads(line)
        clip_video_labels = data.get('clip_video_labels', [])
        
        safe_count = sum(1 for labels in clip_video_labels if not labels)
        unsafe_count = sum(1 for labels in clip_video_labels if labels)
        
        if safe_count > 0 and unsafe_count > 0:
            video_name = data['full_video_path'].split('/')[-1]
            mixed_examples.append({
                'name': video_name,
                'total_clips': len(clip_video_labels),
                'safe_clips': safe_count,
                'unsafe_clips': unsafe_count,
                'pattern': ['S' if not labels else 'U' for labels in clip_video_labels]
            })

for ex in mixed_examples:
    print(f"  {ex['name'][:30]:30} | {ex['safe_clips']} safe, {ex['unsafe_clips']} unsafe | Pattern: {''.join(ex['pattern'])}")

print()
print("="*80)
print("SUMMARY:")
print("="*80)
print(f"• {safe_clips*100/total_clips:.1f}% of clips are SAFE")
print(f"• {unsafe_clips*100/total_clips:.1f}% of clips are UNSAFE")
print(f"• {videos_mixed_clips*100/total_videos:.1f}% of videos have MIXED safe/unsafe segments")
print(f"• Most common unsafe label: {clip_label_counter.most_common(1)[0][0] if clip_label_counter else 'N/A'}")