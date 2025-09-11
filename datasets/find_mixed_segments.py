#!/usr/bin/env python3
"""
Find C1 videos with mixed safe/unsafe segments
"""

import json
from pathlib import Path

# Load annotations
with open('/scratch/czr/Video-Guard/datasets/safewatch_live_simple.json', 'r') as f:
    data = json.load(f)

# Find C1 videos with interesting segment patterns
print("Searching for C1 videos with mixed safe/unsafe segments...")
print("="*60)

mixed_videos = []
for item in data:
    if item['label'] == 'unsafe:C1' and item.get('segments'):
        # Calculate total video duration and unsafe duration
        video_path = item['path']
        video_name = Path(video_path).name
        segments = item['segments']
        
        # Check if video has multiple segments or gaps (implying safe parts)
        if len(segments) > 1:
            # Multiple unsafe segments with gaps
            mixed_videos.append({
                'path': video_path,
                'name': video_name,
                'segments': segments,
                'num_segments': len(segments),
                'type': 'multiple_unsafe_segments'
            })
        elif len(segments) == 1:
            # Single segment - check if it doesn't cover the whole video
            seg = segments[0]
            # If segment doesn't start at 0 or is short, there are safe parts
            if seg['start'] > 2.0 or (seg['end'] - seg['start']) < 5.0:
                mixed_videos.append({
                    'path': video_path,
                    'name': video_name,
                    'segments': segments,
                    'num_segments': 1,
                    'type': 'partial_unsafe'
                })

# Sort by number of segments
mixed_videos.sort(key=lambda x: x['num_segments'], reverse=True)

# Print top candidates
print("\nTop videos with mixed safe/unsafe content:\n")

for i, video in enumerate(mixed_videos[:10], 1):
    print(f"{i}. {video['name']}")
    print(f"   Path: {video['path']}")
    print(f"   Type: {video['type']}")
    print(f"   Segments: {video['segments']}")
    
    # Calculate timeline
    total_unsafe = sum(seg['end'] - seg['start'] for seg in video['segments'])
    print(f"   Total unsafe duration: {total_unsafe:.1f}s")
    
    # Describe timeline
    if len(video['segments']) > 1:
        print("   Timeline:")
        prev_end = 0
        for j, seg in enumerate(video['segments']):
            if seg['start'] > prev_end:
                print(f"     [0.0-{seg['start']:.1f}s]: SAFE")
            print(f"     [{seg['start']:.1f}-{seg['end']:.1f}s]: UNSAFE (segment {j+1})")
            prev_end = seg['end']
        # Assume video continues after last segment
        print(f"     [{prev_end:.1f}s-end]: SAFE (if video continues)")
    else:
        seg = video['segments'][0]
        if seg['start'] > 0:
            print(f"   Timeline: SAFE [0-{seg['start']:.1f}s] → UNSAFE [{seg['start']:.1f}-{seg['end']:.1f}s] → SAFE [after {seg['end']:.1f}s]")
        else:
            print(f"   Timeline: UNSAFE [{seg['start']:.1f}-{seg['end']:.1f}s] → SAFE [after {seg['end']:.1f}s]")
    
    print()

# Recommend best candidates
print("\n" + "="*60)
print("RECOMMENDED VIDEOS FOR TESTING:")
print("="*60)

if mixed_videos:
    # Best candidate with multiple segments
    multi_seg = [v for v in mixed_videos if v['num_segments'] > 1]
    if multi_seg:
        best_multi = multi_seg[0]
        print("\n1. Best video with multiple unsafe segments:")
        print(f"   video_path = '{best_multi['path']}'")
        print(f"   # {len(best_multi['segments'])} unsafe segments: {best_multi['segments']}")
    
    # Best candidate with single partial segment
    single_seg = [v for v in mixed_videos if v['num_segments'] == 1 and v['segments'][0]['start'] > 2.0]
    if single_seg:
        best_single = single_seg[0]
        print("\n2. Best video with safe start, then unsafe:")
        print(f"   video_path = '{best_single['path']}'")
        print(f"   # Unsafe from {best_single['segments'][0]['start']:.1f}s to {best_single['segments'][0]['end']:.1f}s")
    
    # Short unsafe segment (good for testing)
    short_unsafe = [v for v in mixed_videos 
                    if any((seg['end'] - seg['start']) < 3.0 for seg in v['segments'])]
    if short_unsafe:
        best_short = short_unsafe[0]
        print("\n3. Video with short unsafe segment:")
        print(f"   video_path = '{best_short['path']}'")
        print(f"   # Short unsafe segments: {best_short['segments']}")

print("\n" + "="*60)
print("Usage in test_full_video_streaming.py:")
print("="*60)
print("Copy one of the video_path values above and use it in the script")