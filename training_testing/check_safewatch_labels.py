#!/usr/bin/env python3
"""Check SafeWatch data labels"""

import json

# Check multiple samples
with open('/scratch/czr/Video-Guard/datasets/safewatch_streaming_final.jsonl', 'r') as f:
    for i, line in enumerate(f):
        if i >= 10:  # Check first 10 samples
            break
        
        data = json.loads(line)
        path_parts = data["full_video_path"].split("/")
        category = path_parts[-3] if len(path_parts) >= 3 else "unknown"
        filename = path_parts[-1]
        
        print(f"\n{'='*60}")
        print(f"Sample {i+1}: {category}/{filename}")
        print(f"{'='*60}")
        
        # Check full video annotation
        full_ann = data["full_video_annotation"]
        print(f"Full video labels: {full_ann.get('labels', [])}")
        print(f"Full video label_ids: {full_ann.get('label_ids', [])}")
        
        # Check guardrail flags
        guardrail = full_ann.get('guardrail', {})
        unsafe_categories = [k for k, v in guardrail.items() if v]
        print(f"Guardrail unsafe flags: {unsafe_categories if unsafe_categories else 'All False (Safe)'}")
        
        # Check clip_video_labels field
        clip_labels = data.get('clip_video_labels', [])
        print(f"\nclip_video_labels field: {clip_labels}")
        
        # Check individual clip annotations
        print(f"\nIndividual clip annotations:")
        for j, clip_ann in enumerate(data.get('clip_video_annotations', [])[:3]):
            print(f"  Clip {j+1}:")
            print(f"    - labels: {clip_ann.get('labels', [])}")
            print(f"    - label_ids: {clip_ann.get('label_ids', [])}")
            print(f"    - subcategories: {clip_ann.get('subcategories', [])}")
        
        # Check if there's a mismatch
        if category.startswith('abuse') or category.startswith('violence'):
            print(f"\n⚠️  WARNING: This is from '{category}' folder but has no labels!")