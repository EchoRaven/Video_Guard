#!/usr/bin/env python3
"""
Analyze the impact of current strict filtering on SafeWatch data
"""

import json
import os

# Analyze SafeWatch data
safewatch_file = '/scratch/czr/Video-Guard/datasets/safewatch_streaming_corrected.jsonl'

total_videos = 0
videos_with_empty_summaries = 0
videos_with_no_labels = 0
videos_skipped_by_filter = 0
videos_kept = 0

empty_summary_examples = []
no_label_examples = []

print("Analyzing SafeWatch data filtering impact...")
print("="*60)

with open(safewatch_file, 'r') as f:
    for line_idx, line in enumerate(f):
        if line_idx >= 1000:  # Analyze first 1000 for speed
            break
            
        total_videos += 1
        data = json.loads(line)
        
        full_video_path = data['full_video_path']
        clip_video_paths = data['clip_video_paths']
        clip_video_labels = data['clip_video_labels']
        clip_annotations = data['clip_video_annotations']
        full_annotation = data['full_video_annotation']
        
        # Check final description
        final_description = full_annotation.get('description', '')
        if not final_description or len(final_description.strip()) < 20:
            videos_skipped_by_filter += 1
            continue
            
        # Check if any clips have no labels
        has_no_labels = False
        for clip_labels in clip_video_labels:
            if not clip_labels:  # Empty list means safe
                has_no_labels = True
        
        if has_no_labels:
            videos_with_no_labels += 1
            if len(no_label_examples) < 3:
                no_label_examples.append(full_video_path.split('/')[-1])
        
        # Check clips for missing descriptions
        has_missing_descriptions = False
        for i, clip_annotation in enumerate(clip_annotations):
            # Check both sources of descriptions
            clip_desc = ""
            
            # Source 1: clip_descriptions from full_annotation
            clip_descriptions_list = full_annotation.get('clip_descriptions', [])
            if i < len(clip_descriptions_list):
                clip_desc = clip_descriptions_list[i]
            
            # Source 2: individual clip annotation
            if not clip_desc or len(clip_desc.strip()) < 15:
                if isinstance(clip_annotation, dict):
                    clip_desc = clip_annotation.get('description', '')
            
            # Check if description is missing/too short
            if not clip_desc or len(clip_desc.strip()) < 20:
                has_missing_descriptions = True
                if len(empty_summary_examples) < 3:
                    empty_summary_examples.append({
                        'video': full_video_path.split('/')[-1],
                        'clip_idx': i,
                        'desc_length': len(clip_desc.strip()) if clip_desc else 0
                    })
                break
        
        if has_missing_descriptions:
            videos_with_empty_summaries += 1
            videos_skipped_by_filter += 1
        else:
            videos_kept += 1

print(f"Total videos analyzed: {total_videos}")
print(f"Videos kept after filtering: {videos_kept} ({videos_kept*100/total_videos:.1f}%)")
print(f"Videos skipped by filter: {videos_skipped_by_filter} ({videos_skipped_by_filter*100/total_videos:.1f}%)")
print()

print("Breakdown of issues:")
print(f"  - Videos with empty/short clip summaries: {videos_with_empty_summaries}")
print(f"  - Videos with no labels (all safe): {videos_with_no_labels}")
print()

print("Examples of videos with empty summaries:")
for ex in empty_summary_examples[:3]:
    print(f"  - {ex['video']}, clip {ex['clip_idx']}, desc length: {ex['desc_length']}")
print()

print("Examples of videos with no labels (safe videos):")
for ex in no_label_examples[:3]:
    print(f"  - {ex}")

print("\n" + "="*60)
print("ANALYSIS SUMMARY:")
print("="*60)
print("\nCurrent filtering logic:")
print("1. SKIPS videos if ANY clip has description < 20 chars")
print("2. KEEPS videos with no labels (they're safe)")
print("3. Labels: empty list [] means SAFE, ['C1'] means unsafe:C1")
print("\nPotential issues:")
print("- Many videos might be skipped due to missing clip descriptions")
print("- This is VERY strict and may reduce training data significantly")
print("\nRecommendation:")
print("- Consider using a fallback: 'Video clip showing [frame X to Y]'")
print("- Or use the final description for clips without individual descriptions")