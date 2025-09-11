#!/usr/bin/env python3
"""
Merge annotation files and match with SafeWatch-Bench-Live videos
Convert Chinese annotations to English JSON format
"""

import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import re

# Annotation type mapping from Chinese to English
ANNOTATION_MAPPING = {
    '正常': 'safe',
    '打不开': 'corrupted',
    '身体暴露：乳房': 'body_exposure:breast',
    '身体暴露：隐私部位': 'body_exposure:private_parts',
    '身体暴露：私密部位': 'body_exposure:private_parts',
    '身体暴露：臀部': 'body_exposure:buttocks',
    '明显性行为：性交': 'explicit_sexual:intercourse',
    '明显性行为：性行为': 'explicit_sexual:sexual_activity',
    '性暗示动作：具有明显性暗示的动作': 'sexual_suggestion:suggestive_motion',
    '性暗示动作：揉捏胸部': 'sexual_suggestion:breast_touching',
    '性暗示动作：挑逗姿势': 'sexual_suggestion:seductive_pose',
    '自慰示意或器具或抚摸私处': 'masturbation_or_touching_private',
    '不当触摸：触摸私处': 'inappropriate_touching:private_parts',
    '不当触摸：揉捏胸部': 'inappropriate_touching:breast',
    '不当触摸：揉捏胸部和触摸私处': 'inappropriate_touching:breast_and_private',
    '性暗示动作：湿身挑逗': 'sexual_suggestion:wet_seduction',
    '身体暴露：乳房并跳舞挑逗': 'body_exposure:breast_with_dance',
    '身体暴露：乳房和隐私部位': 'body_exposure:breast_and_private',
    '身体暴露：乳房，隐私部位': 'body_exposure:breast_and_private',
    '身体暴露：乳房，私密部位': 'body_exposure:breast_and_private',
    '身体暴露：乳房，臀部': 'body_exposure:breast_and_buttocks',
}

# Map to C1-C6 categories
def map_to_unsafe_category(annotation: str) -> str:
    """Map detailed annotation to C1-C6 unsafe categories"""
    if annotation == 'safe' or annotation == 'corrupted':
        return annotation
    
    # All these are sexual content - C1
    return 'unsafe:C1'

def parse_time_range(time_str: str) -> List[Dict[str, float]]:
    """Parse time range string like '4-7秒' or '1-2秒，5-8秒' """
    if pd.isna(time_str) or time_str == '':
        return []
    
    segments = []
    # Remove '秒' and split by comma or semicolon
    time_str = time_str.replace('秒', '').replace('；', ',')
    
    for segment in time_str.split(','):
        segment = segment.strip()
        if '-' in segment:
            parts = segment.split('-')
            if len(parts) == 2:
                try:
                    start = float(parts[0])
                    end = float(parts[1])
                    segments.append({'start': start, 'end': end})
                except ValueError:
                    continue
    
    return segments

def find_video_path(video_name: str, base_dir: str = '/scratch/czr/SafeWatch-Bench-Live/unsafe') -> str:
    """Find the actual video file path"""
    # Extract the base name (e.g., 'aishe8864_1' -> 'aishe8864')
    base_name = video_name.rsplit('_', 1)[0]
    
    # Check if directory exists
    dir_path = os.path.join(base_dir, base_name)
    if not os.path.exists(dir_path):
        return None
    
    # List all video files
    video_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.mp4')])
    
    # Extract number from video_name
    try:
        video_num = int(video_name.split('_')[-1])
        if video_num <= len(video_files):
            return os.path.join(dir_path, video_files[video_num - 1])
    except (ValueError, IndexError):
        pass
    
    return None

def process_annotations():
    """Process both annotation files and create unified JSON"""
    
    # Read ann1.csv (aishe8864 series)
    df1 = pd.read_csv('/scratch/czr/Video-Guard/datasets/ann1.csv', encoding='gbk')
    
    # Read ann2.xlsx (yp66yp series)
    df2 = pd.read_excel('/scratch/czr/Video-Guard/datasets/ann2.xlsx')
    # Rename column for consistency
    df2 = df2.rename(columns={'视频名称（按视频名称升序排列）': '视频名称'})
    
    # Combine both dataframes
    all_annotations = []
    
    for df in [df1, df2]:
        for _, row in df.iterrows():
            video_name = row['视频名称']
            annotation = row['标注']
            time_range = row.get('时间段', '')
            description = row.get('说明', '')
            
            # Skip if no annotation
            if pd.isna(annotation):
                continue
            
            # Find video path
            video_path = find_video_path(video_name)
            if not video_path:
                print(f"Warning: Could not find video file for {video_name}")
                continue
            
            # Map annotation to English
            english_annotation = ANNOTATION_MAPPING.get(annotation, annotation)
            unsafe_category = map_to_unsafe_category(english_annotation)
            
            # Parse time segments
            time_segments = parse_time_range(time_range)
            
            # Create entry
            entry = {
                'video_name': video_name,
                'video_path': video_path,
                'annotation': english_annotation,
                'unsafe_category': unsafe_category,
                'time_segments': time_segments,
                'description': description if not pd.isna(description) else '',
                'original_annotation': annotation  # Keep original for reference
            }
            
            all_annotations.append(entry)
    
    # Group by video to merge multiple annotations for same video
    video_dict = {}
    for entry in all_annotations:
        video_path = entry['video_path']
        if video_path not in video_dict:
            video_dict[video_path] = {
                'video_name': entry['video_name'],
                'video_path': video_path,
                'annotations': [],
                'time_segments': [],
                'unsafe_categories': set(),
                'descriptions': []
            }
        
        video_dict[video_path]['annotations'].append(entry['annotation'])
        video_dict[video_path]['time_segments'].extend(entry['time_segments'])
        video_dict[video_path]['unsafe_categories'].add(entry['unsafe_category'])
        if entry['description']:
            video_dict[video_path]['descriptions'].append(entry['description'])
    
    # Convert to final format
    final_annotations = []
    for video_path, data in video_dict.items():
        # Merge overlapping time segments
        merged_segments = merge_time_segments(data['time_segments'])
        
        # Determine primary unsafe category
        unsafe_cats = list(data['unsafe_categories'])
        primary_category = 'safe' if 'safe' in unsafe_cats else unsafe_cats[0] if unsafe_cats else 'unknown'
        
        final_entry = {
            'video_name': data['video_name'],
            'video_path': data['video_path'],
            'unsafe_category': primary_category,
            'all_categories': unsafe_cats,
            'annotations': list(set(data['annotations'])),  # Unique annotations
            'time_segments': merged_segments,
            'descriptions': data['descriptions'],
            'total_unsafe_duration': sum(seg['end'] - seg['start'] for seg in merged_segments)
        }
        
        final_annotations.append(final_entry)
    
    # Sort by video name
    final_annotations.sort(key=lambda x: x['video_name'])
    
    # Save to JSON
    output_path = '/scratch/czr/Video-Guard/datasets/safewatch_live_annotations.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_annotations, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(final_annotations)} annotated videos to {output_path}")
    
    # Print statistics
    print("\n=== Annotation Statistics ===")
    print(f"Total videos: {len(final_annotations)}")
    
    category_counts = {}
    for entry in final_annotations:
        cat = entry['unsafe_category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("\nCategory distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} ({count*100/len(final_annotations):.1f}%)")
    
    # Save a simplified version for easy loading
    simplified = []
    for entry in final_annotations:
        simplified.append({
            'path': entry['video_path'],
            'label': entry['unsafe_category'],
            'segments': entry['time_segments']
        })
    
    simple_output = '/scratch/czr/Video-Guard/datasets/safewatch_live_simple.json'
    with open(simple_output, 'w') as f:
        json.dump(simplified, f, indent=2)
    
    print(f"\nSaved simplified version to {simple_output}")
    
    return final_annotations

def merge_time_segments(segments: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Merge overlapping time segments"""
    if not segments:
        return []
    
    # Sort by start time
    segments = sorted(segments, key=lambda x: x['start'])
    
    merged = [segments[0]]
    for seg in segments[1:]:
        last = merged[-1]
        if seg['start'] <= last['end']:
            # Overlapping, merge
            last['end'] = max(last['end'], seg['end'])
        else:
            # Non-overlapping, add new
            merged.append(seg)
    
    return merged

if __name__ == '__main__':
    annotations = process_annotations()
    
    # Print some examples
    print("\n=== Example Entries ===")
    for entry in annotations[:3]:
        print(f"\nVideo: {entry['video_name']}")
        print(f"  Path: {entry['video_path']}")
        print(f"  Category: {entry['unsafe_category']}")
        print(f"  Segments: {entry['time_segments']}")
        print(f"  Annotations: {entry['annotations']}")