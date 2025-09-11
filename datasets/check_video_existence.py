#!/usr/bin/env python3
"""
Check if all videos referenced in the datasets actually exist
"""

import json
import os
from pathlib import Path
from collections import Counter

print("Checking video file existence in datasets...")
print("="*80)

# Check Shot2Story videos
print("\n1. Checking Shot2Story videos...")
print("-"*40)

shot2story_base = '/scratch/czr/Shot2Story/data'
shot2story_missing = []
shot2story_found = 0
shot2story_total = 0

# Check the JSON files
for split in ['train', 'val', 'test']:
    json_file = f'{shot2story_base}/{split}.json'
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
            for item in data[:100]:  # Check first 100 per split
                shot2story_total += 1
                video_path = item.get('video_path', '')
                if video_path:
                    full_path = f'{shot2story_base}/{video_path}'
                    if os.path.exists(full_path):
                        shot2story_found += 1
                    else:
                        shot2story_missing.append(full_path)

print(f"Shot2Story videos checked: {shot2story_total}")
print(f"  Found: {shot2story_found}")
print(f"  Missing: {len(shot2story_missing)}")
if shot2story_missing:
    print(f"  First 5 missing:")
    for path in shot2story_missing[:5]:
        print(f"    - {path}")

# Check SafeWatch-720P videos
print("\n2. Checking SafeWatch-720P videos...")
print("-"*40)

safewatch_file = '/scratch/czr/Video-Guard/datasets/safewatch_streaming_corrected.jsonl'
safewatch_missing = []
safewatch_found = 0
safewatch_total = 0
missing_by_type = Counter()

with open(safewatch_file, 'r') as f:
    for line_idx, line in enumerate(f):
        if line_idx >= 1000:  # Check first 1000 videos
            break
        
        data = json.loads(line)
        
        # Check full video
        full_video_path = data['full_video_path']
        safewatch_total += 1
        
        if os.path.exists(full_video_path):
            safewatch_found += 1
        else:
            safewatch_missing.append(full_video_path)
            missing_by_type['full_video'] += 1
        
        # Check clip videos
        clip_paths = data.get('clip_video_paths', [])
        for clip_path in clip_paths:
            safewatch_total += 1
            if os.path.exists(clip_path):
                safewatch_found += 1
            else:
                safewatch_missing.append(clip_path)
                missing_by_type['clip'] += 1

print(f"SafeWatch videos checked (full + clips): {safewatch_total}")
print(f"  Found: {safewatch_found}")
print(f"  Missing: {len(safewatch_missing)}")
print(f"  Missing by type:")
for video_type, count in missing_by_type.items():
    print(f"    - {video_type}: {count}")

if safewatch_missing:
    print(f"\n  First 10 missing:")
    for path in safewatch_missing[:10]:
        print(f"    - {path}")

# Check SafeWatch-Live videos
print("\n3. Checking SafeWatch-Live videos...")
print("-"*40)

live_annotations_file = '/scratch/czr/Video-Guard/datasets/safewatch_live_annotations.json'
live_missing = []
live_found = 0
live_total = 0

if os.path.exists(live_annotations_file):
    with open(live_annotations_file, 'r') as f:
        annotations = json.load(f)
        for entry in annotations[:100]:  # Check first 100
            video_path = entry.get('video_path', '')
            if video_path:
                live_total += 1
                if os.path.exists(video_path):
                    live_found += 1
                else:
                    live_missing.append(video_path)

print(f"SafeWatch-Live videos checked: {live_total}")
print(f"  Found: {live_found}")
print(f"  Missing: {len(live_missing)}")
if live_missing:
    print(f"  First 5 missing:")
    for path in live_missing[:5]:
        print(f"    - {path}")

# Summary
print("\n" + "="*80)
print("SUMMARY:")
print("="*80)

total_checked = shot2story_total + safewatch_total + live_total
total_found = shot2story_found + safewatch_found + live_found
total_missing = len(shot2story_missing) + len(safewatch_missing) + len(live_missing)

print(f"Total videos checked: {total_checked}")
print(f"Total found: {total_found} ({total_found*100/total_checked if total_checked > 0 else 0:.1f}%)")
print(f"Total missing: {total_missing} ({total_missing*100/total_checked if total_checked > 0 else 0:.1f}%)")

if total_missing > 0:
    print("\n⚠️  WARNING: Some video files are missing!")
    print("This may cause training errors when the dataloader tries to load them.")
else:
    print("\n✅ All checked video files exist!")