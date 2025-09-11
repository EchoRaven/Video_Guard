#!/usr/bin/env python3
"""
Verify how labels are stored and extracted in SafeWatch data
"""

import json

jsonl_file = '/scratch/czr/Video-Guard/datasets/safewatch_streaming_corrected.jsonl'

print("Verifying label extraction from SafeWatch data...")
print("="*80)
print("\nChecking first 10 videos in detail:\n")

with open(jsonl_file, 'r') as f:
    for line_idx, line in enumerate(f):
        if line_idx >= 10:
            break
        
        data = json.loads(line)
        video_name = data['full_video_path'].split('/')[-1]
        
        print(f"Video {line_idx + 1}: {video_name}")
        print("-"*60)
        
        # Check all available fields
        print("Available top-level keys:", list(data.keys()))
        
        # 1. Check full_video_annotation (the response/description)
        full_annotation = data.get('full_video_annotation', {})
        print(f"\nfull_video_annotation keys: {list(full_annotation.keys())}")
        
        # Check labels in full_annotation
        full_labels = full_annotation.get('labels', [])
        print(f"  - labels: {full_labels}")
        
        # Check guardrail
        guardrail = full_annotation.get('guardrail', {})
        guardrail_violations = [k for k, v in guardrail.items() if v]
        print(f"  - guardrail violations: {guardrail_violations if guardrail_violations else 'None (all False)'}")
        
        # 2. Check if there's a separate ground truth field
        if 'full_video_annotation_gt' in data:
            print("\nfull_video_annotation_gt found!")
            gt_annotation = data['full_video_annotation_gt']
            print(f"  GT keys: {list(gt_annotation.keys())}")
            gt_labels = gt_annotation.get('labels', [])
            print(f"  GT labels: {gt_labels}")
        
        # 3. Check clip-level labels
        clip_labels = data.get('clip_video_labels', [])
        print(f"\nclip_video_labels (total {len(clip_labels)} clips):")
        for i, labels in enumerate(clip_labels):
            if labels:  # Only show non-empty
                print(f"  Clip {i}: {labels}")
        
        # Count safe vs unsafe clips
        safe_clips = sum(1 for labels in clip_labels if not labels)
        unsafe_clips = sum(1 for labels in clip_labels if labels)
        print(f"  Summary: {safe_clips} safe clips, {unsafe_clips} unsafe clips")
        
        # 4. Check clip_video_annotations for ground truth
        clip_annotations = data.get('clip_video_annotations', [])
        if clip_annotations and isinstance(clip_annotations[0], dict):
            print(f"\nclip_video_annotations (first clip):")
            first_clip = clip_annotations[0]
            print(f"  Keys: {list(first_clip.keys())}")
            if 'labels' in first_clip:
                print(f"  Labels in annotation: {first_clip['labels']}")
            if 'gt_labels' in first_clip:
                print(f"  GT labels in annotation: {first_clip['gt_labels']}")
        
        print("\n")

# Now do a statistical check
print("="*80)
print("STATISTICAL VERIFICATION:")
print("="*80)

total_videos = 0
videos_labels_match_guardrail = 0
videos_labels_mismatch_guardrail = 0
videos_clips_match_full = 0
videos_clips_mismatch_full = 0

with open(jsonl_file, 'r') as f:
    for line_idx, line in enumerate(f):
        if line_idx >= 1000:  # Check first 1000
            break
        
        total_videos += 1
        data = json.loads(line)
        
        # Get full video labels and guardrail
        full_annotation = data['full_video_annotation']
        full_labels = full_annotation.get('labels', [])
        guardrail = full_annotation.get('guardrail', {})
        has_guardrail_violation = any(guardrail.values())
        
        # Check consistency between labels and guardrail
        if bool(full_labels) == has_guardrail_violation:
            videos_labels_match_guardrail += 1
        else:
            videos_labels_mismatch_guardrail += 1
        
        # Get clip labels
        clip_labels = data.get('clip_video_labels', [])
        has_unsafe_clips = any(labels for labels in clip_labels)
        
        # Check consistency between clip labels and full labels
        if bool(full_labels) == has_unsafe_clips:
            videos_clips_match_full += 1
        else:
            videos_clips_mismatch_full += 1

print(f"Checked {total_videos} videos:")
print(f"  Labels match guardrail: {videos_labels_match_guardrail} ({videos_labels_match_guardrail*100/total_videos:.1f}%)")
print(f"  Labels mismatch guardrail: {videos_labels_mismatch_guardrail}")
print(f"  Clip labels match full labels: {videos_clips_match_full} ({videos_clips_match_full*100/total_videos:.1f}%)")
print(f"  Clip labels mismatch full labels: {videos_clips_mismatch_full}")

if videos_clips_mismatch_full > 0:
    print("\nWARNING: Some videos have inconsistent clip vs full labels!")
    print("This might indicate the labels are not properly aligned.")