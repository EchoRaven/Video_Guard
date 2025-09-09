#!/usr/bin/env python3
"""
Generate correct SafeWatch streaming data by matching full videos with their clip annotations
"""

import json
import os
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

def parse_gpt_response(response_text):
    """Parse GPT response to extract description, guardrail, and explanation"""
    lines = response_text.split('\n')
    
    description = ""
    guardrail = {}
    explanation = ""
    
    for i, line in enumerate(lines):
        if line.startswith("DESCRIPTION:"):
            description = line[12:].strip()
        elif line.startswith("GUARDRAIL:"):
            # Parse JSON guardrail
            guardrail_text = ""
            for j in range(i, len(lines)):
                guardrail_text += lines[j]
                if '}' in lines[j]:
                    break
            
            # Extract flags
            guardrail_str = guardrail_text[10:].strip()
            guardrail = {
                "C1(Sexual Content)": "\"C1(Sexual Content)\": true" in guardrail_str,
                "C2(Harassment & Bullying)": "\"C2(Harassment & Bullying)\": true" in guardrail_str,
                "C3(Threats, Violence & Harm)": "\"C3(Threats, Violence & Harm)\": true" in guardrail_str,
                "C4(False & Deceptive Information)": "\"C4(False & Deceptive Information)\": true" in guardrail_str,
                "C5(Illegal/Regulated Activities)": "\"C5(Illegal/Regulated Activities)\": true" in guardrail_str,
                "C6(Hateful Content & Extremism)": "\"C6(Hateful Content & Extremism)\": true" in guardrail_str
            }
        elif line.startswith("EXPLANATION:"):
            explanation = '\n'.join(lines[i:]).replace("EXPLANATION:", "").strip()
    
    return description, guardrail, explanation

def extract_labels_from_guardrail(guardrail):
    """Convert guardrail dict to label list"""
    labels = []
    label_ids = []
    
    for key, value in guardrail.items():
        if value:
            # Extract category number
            if "C1" in key:
                labels.append("C1")
                label_ids.append(1)
            elif "C2" in key:
                labels.append("C2")
                label_ids.append(2)
            elif "C3" in key:
                labels.append("C3")
                label_ids.append(3)
            elif "C4" in key:
                labels.append("C4")
                label_ids.append(4)
            elif "C5" in key:
                labels.append("C5")
                label_ids.append(5)
            elif "C6" in key:
                labels.append("C6")
                label_ids.append(6)
    
    return labels, label_ids

def process_category_folder(category_path):
    """Process one category folder (e.g., abuse_1)"""
    full_json_path = os.path.join(category_path, 'full.json')
    
    if not os.path.exists(full_json_path):
        return []
    
    print(f"Processing {category_path.name}...")
    
    with open(full_json_path, 'r') as f:
        data = json.load(f)
    
    # Separate full videos and clips
    full_videos = {}
    clip_annotations = defaultdict(list)
    
    for item in data:
        video_path = item['video']
        
        if '/full/' in video_path:
            # Full video annotation
            video_name = video_path.split('/')[-1]
            base_name = video_name.replace('.mp4', '')
            
            # Parse GPT response
            gpt_response = item['conversations'][1]['value']
            description, guardrail, explanation = parse_gpt_response(gpt_response)
            labels, label_ids = extract_labels_from_guardrail(guardrail)
            
            full_videos[base_name] = {
                'video_path': video_path,
                'description': description,
                'guardrail': guardrail,
                'labels': labels,
                'label_ids': label_ids,
                'explanation': explanation
            }
            
        elif '/clip/' in video_path:
            # Clip annotation
            # Extract base video name from clip path
            # Example: dataset/clip/abuse_1/-0nDuJdCniyuJCgx/000000_000002.mp4
            parts = video_path.split('/')
            if len(parts) >= 2:
                base_name = parts[-2]  # Get folder name (video base name)
                clip_name = parts[-1]   # Get clip file name
                
                # Parse GPT response for clip
                gpt_response = item['conversations'][1]['value']
                description, guardrail, explanation = parse_gpt_response(gpt_response)
                labels, label_ids = extract_labels_from_guardrail(guardrail)
                
                # Extract time range from clip name
                time_range = clip_name.replace('.mp4', '')
                
                clip_annotations[base_name].append({
                    'clip_path': video_path,
                    'clip_name': clip_name,
                    'time_range': time_range,
                    'description': description,
                    'guardrail': guardrail,
                    'labels': labels,
                    'label_ids': label_ids,
                    'subcategories': []  # Can be filled if needed
                })
    
    # Create output entries
    output_entries = []
    
    for base_name, full_info in full_videos.items():
        # Get corresponding clips
        clips = clip_annotations.get(base_name, [])
        
        if not clips:
            print(f"  Warning: No clips found for {base_name}")
            continue
        
        # Sort clips by time range
        clips.sort(key=lambda x: x['time_range'])
        
        # Build clip paths, labels, and annotations
        clip_paths = []
        clip_labels = []
        clip_video_annotations = []
        clip_descriptions = []
        
        for clip in clips:
            # Build actual clip path
            actual_clip_path = f"/scratch/czr/Video-Guard/datasets/SafeWatch-Bench-200K-720P/clip/{category_path.name}/{base_name}/{clip['clip_name']}"
            clip_paths.append(actual_clip_path)
            clip_labels.append(clip['labels'])
            
            clip_video_annotations.append({
                'clip_name': clip['clip_name'],
                'time_range': clip['time_range'],
                'labels': clip['labels'],
                'label_ids': clip['label_ids'],
                'subcategories': clip['subcategories'],
                'description': clip['description']  # Use the actual clip description!
            })
            
            clip_descriptions.append(clip['description'])
        
        # Build full video path
        actual_full_path = f"/scratch/czr/Video-Guard/datasets/SafeWatch-Bench-200K-720P/full/{category_path.name}/target/{base_name}.mp4"
        
        # Create entry
        entry = {
            'full_video_path': actual_full_path,
            'clip_video_paths': clip_paths,
            'clip_video_labels': clip_labels,
            'full_video_annotation': {
                'labels': full_info['labels'],
                'label_ids': full_info['label_ids'],
                'subcategories': [],
                'description': full_info['description'],
                'guardrail': full_info['guardrail'],
                'clip_descriptions': clip_descriptions  # Individual clip descriptions
            },
            'clip_video_annotations': clip_video_annotations
        }
        
        output_entries.append(entry)
    
    return output_entries

def main():
    """Process all SafeWatch categories and generate corrected data file"""
    
    base_path = Path('/scratch/czr/Video-Guard/datasets/SafeWatch-Bench-200K-720P/annotation')
    output_file = '/scratch/czr/Video-Guard/datasets/safewatch_streaming_corrected.jsonl'
    
    all_entries = []
    
    # Get all category folders
    category_folders = sorted([d for d in base_path.iterdir() if d.is_dir()])
    
    print(f"Found {len(category_folders)} category folders")
    
    # Process each category
    for category_path in tqdm(category_folders, desc="Processing categories"):
        entries = process_category_folder(category_path)
        all_entries.extend(entries)
    
    # Write output
    print(f"\nWriting {len(all_entries)} entries to {output_file}")
    
    with open(output_file, 'w') as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + '\n')
    
    # Statistics
    total_clips = sum(len(e['clip_video_paths']) for e in all_entries)
    clips_with_desc = sum(
        sum(1 for desc in e['full_video_annotation']['clip_descriptions'] if desc and len(desc) > 20)
        for e in all_entries
    )
    
    print("\n" + "="*60)
    print("STATISTICS:")
    print(f"Total videos: {len(all_entries)}")
    print(f"Total clips: {total_clips}")
    print(f"Clips with descriptions: {clips_with_desc} ({clips_with_desc/total_clips*100:.1f}%)")
    
    # Sample check
    print("\n" + "="*60)
    print("SAMPLE CHECK (first entry):")
    if all_entries:
        first = all_entries[0]
        print(f"Video: {first['full_video_path'].split('/')[-1]}")
        print(f"Number of clips: {len(first['clip_video_paths'])}")
        print(f"Full video labels: {first['full_video_annotation']['labels']}")
        print("\nFirst 3 clips:")
        for i in range(min(3, len(first['clip_video_annotations']))):
            clip = first['clip_video_annotations'][i]
            print(f"  Clip {i+1}:")
            print(f"    Labels: {clip['labels']}")
            print(f"    Description: {clip['description'][:80]}...")

if __name__ == "__main__":
    main()