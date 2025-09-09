#!/usr/bin/env python3
"""Debug why descriptions are missing"""

import json

with open('/scratch/czr/Video-Guard/datasets/safewatch_streaming_final.jsonl', 'r') as f:
    has_desc = 0
    no_desc = 0
    
    for i, line in enumerate(f):
        if i >= 100:  # Check first 100
            break
            
        data = json.loads(line)
        
        # Check clip_descriptions in full_video_annotation
        clip_descs = data['full_video_annotation'].get('clip_descriptions', [])
        
        if i < 5:  # Print details for first 5
            print(f"\nVideo {i+1}:")
            print(f"  Has clip_descriptions field: {'clip_descriptions' in data['full_video_annotation']}")
            print(f"  Number of clip_descriptions: {len(clip_descs)}")
            print(f"  Number of clips: {len(data['clip_video_paths'])}")
            
            if clip_descs:
                print(f"  First description: {clip_descs[0][:50]}...")
        
        if clip_descs and any(d for d in clip_descs if d and len(d.strip()) >= 15):
            has_desc += 1
        else:
            no_desc += 1
    
    print(f"\n📊 Results from first 100 videos:")
    print(f"  Videos with descriptions: {has_desc}")
    print(f"  Videos without descriptions: {no_desc}")