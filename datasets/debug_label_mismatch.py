#!/usr/bin/env python3
"""
Debug why labels in JSONL don't match ground truth annotations
"""

import json
from pathlib import Path

# Load the SafeWatch annotations
annotations_file = '/scratch/czr/Video-Guard/datasets/safewatch_live_annotations.json'
with open(annotations_file, 'r') as f:
    annotations = json.load(f)

# Create ground truth mapping
gt_mapping = {}
for entry in annotations:
    video_path = entry['video_path']
    video_name = Path(video_path).name
    gt_mapping[video_name] = {
        'category': entry.get('unsafe_category', 'safe'),
        'annotations': entry.get('annotations', [])
    }

print("Checking label mismatches in JSONL file...")
print("="*60)

# Check JSONL file
jsonl_file = '/scratch/czr/Video-Guard/datasets/safewatch_streaming_corrected.jsonl'
mismatches = []
correct_matches = []
not_found = []

with open(jsonl_file, 'r') as f:
    for line_idx, line in enumerate(f):
        if line_idx >= 20:  # Check first 20 for detailed analysis
            break
        
        data = json.loads(line)
        full_video_path = data['full_video_path']
        video_name = Path(full_video_path).name
        
        # Get labels from JSONL
        jsonl_labels = data['full_video_annotation'].get('labels', [])
        jsonl_guardrail = data['full_video_annotation'].get('guardrail', {})
        
        # Get ground truth
        if video_name in gt_mapping:
            gt_info = gt_mapping[video_name]
            gt_category = gt_info['category']
            
            # Determine if JSONL matches GT
            if gt_category == 'safe' or not gt_category:
                # Should be safe
                if not jsonl_labels:
                    correct_matches.append(video_name)
                else:
                    mismatches.append({
                        'video': video_name,
                        'gt': 'safe',
                        'jsonl_labels': jsonl_labels,
                        'jsonl_guardrail': [k for k, v in jsonl_guardrail.items() if v]
                    })
            elif 'C1' in gt_category:
                # Should have C1 label
                if 'C1' in jsonl_labels:
                    correct_matches.append(video_name)
                else:
                    mismatches.append({
                        'video': video_name,
                        'gt': gt_category,
                        'jsonl_labels': jsonl_labels,
                        'jsonl_guardrail': [k for k, v in jsonl_guardrail.items() if v]
                    })
        else:
            not_found.append(video_name)

print(f"Analyzed {line_idx} videos:")
print(f"  Correct matches: {len(correct_matches)}")
print(f"  Mismatches: {len(mismatches)}")
print(f"  Not found in GT: {len(not_found)}")
print()

if mismatches:
    print("Examples of mismatches:")
    for mm in mismatches[:5]:
        print(f"  Video: {mm['video']}")
        print(f"    Ground truth: {mm['gt']}")
        print(f"    JSONL labels: {mm['jsonl_labels']}")
        print(f"    JSONL guardrail flags: {mm['jsonl_guardrail']}")
        print()

# Check if the issue is with the full_video_annotation field
print("="*60)
print("Checking if labels might be in clip_video_labels instead...")
print("="*60)

with open(jsonl_file, 'r') as f:
    for line_idx, line in enumerate(f):
        if line_idx >= 10:
            break
        
        data = json.loads(line)
        video_name = Path(data['full_video_path']).name
        
        # Check clip labels
        clip_labels = data.get('clip_video_labels', [])
        full_labels = data['full_video_annotation'].get('labels', [])
        
        # Flatten clip labels
        all_clip_labels = []
        for clip in clip_labels:
            if clip:
                all_clip_labels.extend(clip)
        unique_clip_labels = list(set(all_clip_labels))
        
        print(f"Video {line_idx + 1}: {video_name}")
        print(f"  Full annotation labels: {full_labels}")
        print(f"  Unique clip labels: {unique_clip_labels}")
        
        if video_name in gt_mapping:
            gt_category = gt_mapping[video_name]['category']
            print(f"  Ground truth: {gt_category}")
        print()