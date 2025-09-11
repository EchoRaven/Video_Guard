#!/usr/bin/env python3
"""
Calculate exact statistics for SafeBench-720P dataset
"""

import json
from collections import Counter

jsonl_file = '/scratch/czr/Video-Guard/datasets/safewatch_streaming_corrected.jsonl'

print("SafeBench-720P (safewatch_streaming_corrected.jsonl) Statistics")
print("="*80)

# Initialize counters
total_videos = 0

# Label statistics
videos_with_labels = 0
videos_without_labels = 0
label_counter = Counter()

# Summary/Description statistics
videos_with_final_description = 0
videos_without_final_description = 0
videos_with_short_final_description = 0

# Clip description statistics
videos_with_all_clip_descriptions = 0
videos_missing_some_clip_descriptions = 0
videos_with_no_clip_descriptions = 0

# Guardrail statistics
videos_with_guardrail_violations = 0
videos_without_guardrail_violations = 0
guardrail_counter = Counter()

# Process each video
with open(jsonl_file, 'r') as f:
    for line_idx, line in enumerate(f):
        total_videos += 1
        data = json.loads(line)
        
        # Get full annotation
        full_annotation = data['full_video_annotation']
        
        # 1. Check labels
        labels = full_annotation.get('labels', [])
        if labels:
            videos_with_labels += 1
            for label in labels:
                label_counter[label] += 1
        else:
            videos_without_labels += 1
        
        # 2. Check final description
        final_description = full_annotation.get('description', '')
        if final_description:
            videos_with_final_description += 1
            if len(final_description.strip()) < 20:
                videos_with_short_final_description += 1
        else:
            videos_without_final_description += 1
        
        # 3. Check clip descriptions
        clip_descriptions = full_annotation.get('clip_descriptions', [])
        num_clips = len(data.get('clip_video_paths', []))
        
        if clip_descriptions:
            # Count how many clips have valid descriptions
            valid_clip_descs = 0
            for desc in clip_descriptions:
                if desc and len(desc.strip()) >= 20:
                    valid_clip_descs += 1
            
            if valid_clip_descs == num_clips:
                videos_with_all_clip_descriptions += 1
            elif valid_clip_descs == 0:
                videos_with_no_clip_descriptions += 1
            else:
                videos_missing_some_clip_descriptions += 1
        else:
            videos_with_no_clip_descriptions += 1
        
        # 4. Check guardrail
        guardrail = full_annotation.get('guardrail', {})
        has_violation = False
        for category, is_flagged in guardrail.items():
            if is_flagged:
                has_violation = True
                guardrail_counter[category] += 1
        
        if has_violation:
            videos_with_guardrail_violations += 1
        else:
            videos_without_guardrail_violations += 1

# Print results
print(f"TOTAL VIDEOS: {total_videos}")
print()

print("1. LABEL STATISTICS:")
print("-"*40)
print(f"Videos WITH labels (unsafe): {videos_with_labels} ({videos_with_labels*100/total_videos:.1f}%)")
print(f"Videos WITHOUT labels (safe): {videos_without_labels} ({videos_without_labels*100/total_videos:.1f}%)")
print()
print("Label distribution in unsafe videos:")
for label, count in label_counter.most_common():
    print(f"  {label}: {count} videos ({count*100/total_videos:.1f}% of all videos)")
print()

print("2. FINAL DESCRIPTION STATISTICS:")
print("-"*40)
print(f"Videos WITH final description: {videos_with_final_description} ({videos_with_final_description*100/total_videos:.1f}%)")
print(f"Videos WITHOUT final description: {videos_without_final_description} ({videos_without_final_description*100/total_videos:.1f}%)")
print(f"Videos with SHORT (<20 chars) final description: {videos_with_short_final_description} ({videos_with_short_final_description*100/total_videos:.1f}%)")
print()

print("3. CLIP DESCRIPTION STATISTICS:")
print("-"*40)
print(f"Videos with ALL clip descriptions (≥20 chars): {videos_with_all_clip_descriptions} ({videos_with_all_clip_descriptions*100/total_videos:.1f}%)")
print(f"Videos MISSING SOME clip descriptions: {videos_missing_some_clip_descriptions} ({videos_missing_some_clip_descriptions*100/total_videos:.1f}%)")
print(f"Videos with NO valid clip descriptions: {videos_with_no_clip_descriptions} ({videos_with_no_clip_descriptions*100/total_videos:.1f}%)")
print()

print("4. GUARDRAIL STATISTICS:")
print("-"*40)
print(f"Videos WITH guardrail violations: {videos_with_guardrail_violations} ({videos_with_guardrail_violations*100/total_videos:.1f}%)")
print(f"Videos WITHOUT guardrail violations: {videos_without_guardrail_violations} ({videos_without_guardrail_violations*100/total_videos:.1f}%)")
print()
print("Guardrail violation distribution:")
for category, count in guardrail_counter.most_common():
    print(f"  {category}: {count} videos ({count*100/total_videos:.1f}% of all videos)")
print()

print("="*80)
print("SUMMARY:")
print("="*80)
print(f"Safe videos (no labels): {videos_without_labels} ({videos_without_labels*100/total_videos:.1f}%)")
print(f"Unsafe videos (has labels): {videos_with_labels} ({videos_with_labels*100/total_videos:.1f}%)")
print(f"Videos with usable descriptions: {videos_with_final_description} ({videos_with_final_description*100/total_videos:.1f}%)")
print(f"Videos with complete clip descriptions: {videos_with_all_clip_descriptions} ({videos_with_all_clip_descriptions*100/total_videos:.1f}%)")