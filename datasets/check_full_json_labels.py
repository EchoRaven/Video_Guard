#!/usr/bin/env python3
"""
Check how safe/unsafe labels are actually stored in full.json
"""

import json

safewatch_file = '/scratch/czr/Video-Guard/datasets/safewatch_streaming_corrected.jsonl'

print("Checking how safe/unsafe labels are stored in full.json...")
print("="*60)

# Check first few examples to understand the structure
with open(safewatch_file, 'r') as f:
    for line_idx, line in enumerate(f):
        if line_idx >= 5:  # Check first 5 videos
            break
            
        data = json.loads(line)
        video_name = data['full_video_path'].split('/')[-1]
        
        print(f"\nVideo {line_idx + 1}: {video_name}")
        
        # Check full_video_annotation structure
        full_annotation = data['full_video_annotation']
        
        # Look for safe/unsafe label in full annotation
        print(f"  Keys in full_annotation: {list(full_annotation.keys())}")
        
        # Check if there's a direct safe/unsafe field
        if 'safe' in full_annotation:
            print(f"  Has 'safe' field: {full_annotation['safe']}")
        if 'unsafe' in full_annotation:
            print(f"  Has 'unsafe' field: {full_annotation['unsafe']}")
        if 'label' in full_annotation:
            print(f"  Has 'label' field: {full_annotation['label']}")
        if 'is_safe' in full_annotation:
            print(f"  Has 'is_safe' field: {full_annotation['is_safe']}")
            
        # Check guardrail
        guardrail = full_annotation.get('guardrail', {})
        has_any_violation = any(guardrail.values())
        print(f"  Guardrail violations: {has_any_violation}")
        if has_any_violation:
            violations = [k for k, v in guardrail.items() if v]
            print(f"    Violated categories: {violations}")
        
        # Check clip labels
        clip_labels = data.get('clip_video_labels', [])
        has_any_clip_labels = any(labels for labels in clip_labels)
        print(f"  Has clip labels: {has_any_clip_labels}")
        if has_any_clip_labels:
            all_labels = []
            for labels in clip_labels:
                if labels:
                    all_labels.extend(labels)
            unique_labels = list(set(all_labels))
            print(f"    Unique labels in clips: {unique_labels}")
        
        # Print full annotation to see all fields
        print(f"  Full annotation keys and sample values:")
        for key in full_annotation.keys():
            value = full_annotation[key]
            if isinstance(value, str) and len(value) > 100:
                print(f"    {key}: {value[:50]}...")
            elif isinstance(value, dict):
                print(f"    {key}: <dict with {len(value)} keys>")
            elif isinstance(value, list):
                print(f"    {key}: <list with {len(value)} items>")
            else:
                print(f"    {key}: {value}")