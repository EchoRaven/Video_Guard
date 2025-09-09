#!/usr/bin/env python3
"""
Correct statistics for SafeWatch data labels
"""

import json
import os
from tqdm import tqdm

def analyze_safewatch():
    jsonl_path = '/scratch/czr/Video-Guard/datasets/safewatch_streaming_final.jsonl'
    
    total_videos = 0
    videos_with_labels = 0
    videos_without_labels = 0
    
    total_clips = 0
    clips_with_labels = 0
    clips_without_labels = 0
    clips_with_empty_desc = 0
    
    label_distribution = {}
    
    print("Analyzing SafeWatch data...")
    with open(jsonl_path, 'r') as f:
        for line in tqdm(f, desc="Processing"):
            data = json.loads(line)
            total_videos += 1
            
            # Check full video labels
            full_labels = data['full_video_annotation'].get('labels', [])
            if full_labels:
                videos_with_labels += 1
            else:
                videos_without_labels += 1
            
            # Check each clip
            clip_labels_list = data.get('clip_video_labels', [])
            clip_annotations = data.get('clip_video_annotations', [])
            
            for i, clip_labels in enumerate(clip_labels_list):
                total_clips += 1
                
                if clip_labels:  # Has labels like ['C2'], ['C4']
                    clips_with_labels += 1
                    for label in clip_labels:
                        label_distribution[label] = label_distribution.get(label, 0) + 1
                else:
                    clips_without_labels += 1
                
                # Check if clip has description
                if i < len(clip_annotations):
                    clip_ann = clip_annotations[i]
                    desc = clip_ann.get('description', '')
                    if not desc or len(desc.strip()) < 15:
                        clips_with_empty_desc += 1
    
    print(f"\n📊 SafeWatch Statistics:")
    print(f"{'='*50}")
    print(f"Total videos: {total_videos}")
    print(f"  - With labels: {videos_with_labels} ({videos_with_labels/total_videos*100:.1f}%)")
    print(f"  - Without labels: {videos_without_labels} ({videos_without_labels/total_videos*100:.1f}%)")
    
    print(f"\nTotal clips: {total_clips}")
    print(f"  - With labels: {clips_with_labels} ({clips_with_labels/total_clips*100:.1f}%)")
    print(f"  - Without labels (safe): {clips_without_labels} ({clips_without_labels/total_clips*100:.1f}%)")
    print(f"  - With empty descriptions: {clips_with_empty_desc} ({clips_with_empty_desc/total_clips*100:.1f}%)")
    
    print(f"\nLabel distribution:")
    for label, count in sorted(label_distribution.items()):
        print(f"  {label}: {count} clips")
    
    # Calculate how many would use fallback descriptions
    print(f"\n⚠️  Fallback Impact:")
    print(f"  Clips with empty descriptions that would use fallback: {clips_with_empty_desc}")
    print(f"  This is {clips_with_empty_desc/total_clips*100:.1f}% of all clips")
    
    return clips_with_empty_desc / total_clips * 100

if __name__ == "__main__":
    fallback_rate = analyze_safewatch()
    
    if fallback_rate < 10:
        print("\n✅ Good: Low fallback rate, data quality is acceptable")
    elif fallback_rate < 30:
        print("\n⚠️  Warning: Moderate fallback rate, some training data uses generic descriptions")
    else:
        print("\n❌ Problem: High fallback rate, many clips lack proper descriptions")