#!/usr/bin/env python3
"""
Find C1 videos with very short unsafe segments (mostly safe)
"""

import json
from pathlib import Path

# Load annotations
with open('/scratch/czr/Video-Guard/datasets/safewatch_live_simple.json', 'r') as f:
    data = json.load(f)

print("Searching for C1 videos with SHORT unsafe segments (mostly safe)...")
print("="*60)

short_unsafe_videos = []

for item in data:
    if item['label'] == 'unsafe:C1' and item.get('segments'):
        video_path = item['path']
        video_name = Path(video_path).name
        segments = item['segments']
        
        # Calculate total unsafe duration
        total_unsafe = sum(seg['end'] - seg['start'] for seg in segments)
        
        # Look for videos with very short unsafe segments (< 5 seconds total)
        if total_unsafe < 5.0:
            # Estimate video length (assume videos are at least 2x the last segment end)
            last_segment_end = max(seg['end'] for seg in segments)
            estimated_length = max(last_segment_end * 1.5, last_segment_end + 10)
            unsafe_percentage = (total_unsafe / estimated_length) * 100
            
            short_unsafe_videos.append({
                'path': video_path,
                'name': video_name,
                'segments': segments,
                'total_unsafe': total_unsafe,
                'last_segment_end': last_segment_end,
                'estimated_length': estimated_length,
                'unsafe_percentage': unsafe_percentage
            })

# Sort by unsafe duration (shortest first)
short_unsafe_videos.sort(key=lambda x: x['total_unsafe'])

print(f"Found {len(short_unsafe_videos)} videos with short unsafe segments\n")

# Print top candidates
print("TOP VIDEOS WITH VERY SHORT UNSAFE SEGMENTS:")
print("(These are mostly safe with brief unsafe moments)\n")

for i, video in enumerate(short_unsafe_videos[:15], 1):
    print(f"{i}. {video['name']}")
    print(f"   Path: {video['path']}")
    print(f"   Unsafe segments: {video['segments']}")
    print(f"   Total unsafe: {video['total_unsafe']:.1f}s")
    print(f"   Estimated video length: >{video['last_segment_end']:.1f}s")
    print(f"   Unsafe percentage: <{video['unsafe_percentage']:.1f}%")
    
    # Describe timeline
    print("   Timeline:")
    for j, seg in enumerate(video['segments']):
        if seg['start'] > 0:
            print(f"     SAFE: [0.0-{seg['start']:.1f}s]")
        print(f"     UNSAFE: [{seg['start']:.1f}-{seg['end']:.1f}s] ({seg['end']-seg['start']:.1f}s)")
        if j == len(video['segments']) - 1:
            print(f"     SAFE: [{seg['end']:.1f}s onwards]")
    print()

# Special categories
print("\n" + "="*60)
print("SPECIAL RECOMMENDATIONS:")
print("="*60)

# 1. Ultra-short unsafe (1-2 seconds)
ultra_short = [v for v in short_unsafe_videos if v['total_unsafe'] <= 2.0]
if ultra_short:
    print("\n1. ULTRA-SHORT unsafe segments (1-2 seconds):")
    for v in ultra_short[:3]:
        print(f"   video_path = '{v['path']}'")
        print(f"   # Only {v['total_unsafe']:.1f}s unsafe: {v['segments']}")
        print()

# 2. Short unsafe with long safe start
late_unsafe = [v for v in short_unsafe_videos 
                if v['segments'][0]['start'] > 10.0 and v['total_unsafe'] < 5.0]
if late_unsafe:
    print("2. Short unsafe after LONG safe start:")
    for v in late_unsafe[:3]:
        print(f"   video_path = '{v['path']}'")
        print(f"   # Safe for {v['segments'][0]['start']:.1f}s, then {v['total_unsafe']:.1f}s unsafe")
        print()

# 3. Very brief unsafe in middle
middle_unsafe = [v for v in short_unsafe_videos 
                 if v['segments'][0]['start'] > 5.0 and 
                 v['total_unsafe'] < 3.0]
if middle_unsafe:
    print("3. VERY BRIEF unsafe in middle (good for testing detection):")
    for v in middle_unsafe[:3]:
        print(f"   video_path = '{v['path']}'")
        print(f"   # {v['total_unsafe']:.1f}s unsafe starting at {v['segments'][0]['start']:.1f}s")
        print()

print("="*60)
print("These videos are perfect for testing if the model can:")
print("1. Correctly identify long safe segments")
print("2. Detect brief unsafe content without false positives")
print("3. Return to safe classification after unsafe content ends")
print("="*60)