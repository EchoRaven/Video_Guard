#!/usr/bin/env python3
"""
Verify the quality of corrected SafeWatch data
"""

import json
from tqdm import tqdm

def verify_corrected_data():
    jsonl_path = '/scratch/czr/Video-Guard/datasets/safewatch_streaming_corrected.jsonl'
    
    total_videos = 0
    total_clips = 0
    clips_with_good_desc = 0
    clips_with_empty_desc = 0
    clips_with_fallback = 0
    
    # Sample some good descriptions
    good_descriptions = []
    
    print("Verifying corrected SafeWatch data...")
    with open(jsonl_path, 'r') as f:
        for line in tqdm(f, desc="Processing"):
            data = json.loads(line)
            total_videos += 1
            
            clips = data.get('clip_video_annotations', [])
            total_clips += len(clips)
            
            for clip in clips:
                desc = clip.get('description', '')
                
                if not desc or len(desc.strip()) < 15:
                    clips_with_empty_desc += 1
                elif "This video clip" in desc and ("safety review" in desc or "safe content" in desc):
                    # These are likely fallback descriptions
                    clips_with_fallback += 1
                else:
                    clips_with_good_desc += 1
                    # Collect some examples
                    if len(good_descriptions) < 10 and len(desc) > 30:
                        good_descriptions.append(desc[:150])
    
    print("\n📊 Corrected SafeWatch Data Quality:")
    print("="*60)
    print(f"Total videos: {total_videos:,}")
    print(f"Total clips: {total_clips:,}")
    print(f"Clips with good descriptions: {clips_with_good_desc:,} ({clips_with_good_desc/total_clips*100:.1f}%)")
    print(f"Clips with empty descriptions: {clips_with_empty_desc:,} ({clips_with_empty_desc/total_clips*100:.1f}%)")
    print(f"Clips with fallback descriptions: {clips_with_fallback:,} ({clips_with_fallback/total_clips*100:.1f}%)")
    
    print("\n📝 Sample Good Descriptions:")
    print("-"*60)
    for i, desc in enumerate(good_descriptions, 1):
        print(f"{i}. {desc}...")
    
    print("\n✅ Summary:")
    if clips_with_good_desc/total_clips > 0.9:
        print(f"EXCELLENT: {clips_with_good_desc/total_clips*100:.1f}% of clips have meaningful descriptions!")
        print("The corrected file is much better than the original final file.")
    elif clips_with_good_desc/total_clips > 0.7:
        print(f"GOOD: {clips_with_good_desc/total_clips*100:.1f}% of clips have meaningful descriptions.")
    else:
        print(f"NEEDS IMPROVEMENT: Only {clips_with_good_desc/total_clips*100:.1f}% have good descriptions.")

if __name__ == "__main__":
    verify_corrected_data()