#!/usr/bin/env python3
"""
Check how clip annotations are structured in SafeWatch
"""

import json

# Read full.json
annotation_file = '/scratch/czr/Video-Guard/datasets/SafeWatch-Bench-200K-720P/annotation/abuse_1/full.json'

with open(annotation_file, 'r') as f:
    data = json.load(f)

print("Analyzing full.json structure...")
print("="*60)
print(f"Total entries: {len(data)}")

# Separate full videos from clips
full_videos = []
clip_videos = []

for item in data:
    video_path = item['video']
    if '/full/' in video_path:
        full_videos.append(item)
    elif '/clip/' in video_path:
        clip_videos.append(item)

print(f"Full videos: {len(full_videos)}")
print(f"Clip videos: {len(clip_videos)}")

# Check first full video
if full_videos:
    first_full = full_videos[0]
    video_name = first_full['video'].split('/')[-1]
    print(f"\nFirst full video: {video_name}")
    
    # Get GPT response
    gpt_response = first_full['conversations'][1]['value']
    lines = gpt_response.split('\n')
    
    # Find DESCRIPTION
    for line in lines:
        if line.startswith("DESCRIPTION:"):
            print(f"  Description: {line[12:80]}...")
            break

# Check corresponding clips
print(f"\nChecking clips for the same video...")
video_base = video_name.replace('.mp4', '')

matching_clips = []
for clip in clip_videos:
    clip_path = clip['video']
    if video_base in clip_path:
        matching_clips.append(clip)

print(f"Found {len(matching_clips)} matching clips")

# Show first 3 clips
for i, clip in enumerate(matching_clips[:3]):
    clip_name = clip['video'].split('/')[-1]
    print(f"\nClip {i+1}: {clip_name}")
    
    # Get GPT response for clip
    gpt_response = clip['conversations'][1]['value']
    lines = gpt_response.split('\n')
    
    # Find DESCRIPTION
    for line in lines:
        if line.startswith("DESCRIPTION:"):
            desc = line[12:].strip()
            print(f"  Description: {desc[:100]}...")
            break
    
    # Find GUARDRAIL flags
    for j, line in enumerate(lines):
        if line.startswith("GUARDRAIL:"):
            unsafe_flags = []
            # Check next few lines for flags
            for k in range(j, min(j+10, len(lines))):
                if "true" in lines[k]:
                    if "C1" in lines[k]:
                        unsafe_flags.append("C1")
                    if "C2" in lines[k]:
                        unsafe_flags.append("C2")
                    if "C3" in lines[k]:
                        unsafe_flags.append("C3")
                    if "C4" in lines[k]:
                        unsafe_flags.append("C4")
                    if "C5" in lines[k]:
                        unsafe_flags.append("C5")
                    if "C6" in lines[k]:
                        unsafe_flags.append("C6")
            print(f"  Unsafe flags: {unsafe_flags if unsafe_flags else 'None (safe)'}")
            break

print("\n" + "="*60)
print("CONCLUSION:")
print("The full.json file contains BOTH:")
print("1. Full video annotations (with clip summaries in EXPLANATION)")  
print("2. Individual clip annotations (each with its own DESCRIPTION)")
print("\nThe extraction script should use the individual clip annotations")
print("instead of trying to parse the EXPLANATION field!")