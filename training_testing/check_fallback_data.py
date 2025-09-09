#!/usr/bin/env python3
"""
Check how many training samples use fallback descriptions
"""

import json
import os
from tqdm import tqdm

# Fallback descriptions that indicate a problem
FALLBACK_DESCRIPTIONS = [
    "This video clip contains content that may require safety review.",
    "This video clip appears to contain safe content with no harmful material detected.",
    "Incomplete shot (video ended)",
    "Shot analyzed and processed."
]

def check_safewatch_data():
    """Check SafeWatch data for fallback descriptions"""
    safewatch_jsonl_path = '/scratch/czr/Video-Guard/datasets/safewatch_streaming_final.jsonl'
    
    total_clips = 0
    fallback_clips = 0
    empty_descriptions = 0
    short_descriptions = 0
    
    if os.path.exists(safewatch_jsonl_path):
        print("Analyzing SafeWatch data...")
        with open(safewatch_jsonl_path, 'r') as f:
            for line_idx, line in enumerate(tqdm(f, desc="Processing SafeWatch")):
                if line_idx >= 1000:  # Limit to first 1000 for quick analysis
                    break
                    
                data = json.loads(line)
                clip_annotations = data.get('clip_video_annotations', [])
                full_annotation = data.get('full_video_annotation', {})
                clip_descriptions_list = full_annotation.get('clip_descriptions', [])
                
                # Check each clip
                for clip_idx, clip_annotation in enumerate(clip_annotations):
                    total_clips += 1
                    
                    # Try to get description from various sources
                    clip_description = ""
                    
                    # 1. From clip_descriptions list
                    if clip_idx < len(clip_descriptions_list):
                        clip_description = clip_descriptions_list[clip_idx]
                    
                    # 2. From individual clip annotation
                    if not clip_description or len(clip_description.strip()) < 15:
                        if isinstance(clip_annotation, dict):
                            clip_description = clip_annotation.get('description', '')
                    
                    # Check if it's empty or too short
                    if not clip_description:
                        empty_descriptions += 1
                        fallback_clips += 1
                    elif len(clip_description.strip()) < 15:
                        short_descriptions += 1
                        fallback_clips += 1
                    elif any(fallback in clip_description for fallback in FALLBACK_DESCRIPTIONS):
                        fallback_clips += 1
    
    return total_clips, fallback_clips, empty_descriptions, short_descriptions

def check_shot2story_data():
    """Check Shot2Story data for quality issues"""
    shot2story_path = '/scratch/czr/Video-Guard/datasets/shot2story/134k_full_train.json'
    
    total_videos = 0
    videos_with_bad_summaries = 0
    total_clips = 0
    clips_with_bad_summaries = 0
    
    if os.path.exists(shot2story_path):
        print("\nAnalyzing Shot2Story data...")
        with open(shot2story_path, 'r') as f:
            data = json.load(f)
            
        # Limit to first 1000 videos for quick analysis
        for video_idx, video_data in enumerate(tqdm(data[:1000], desc="Processing Shot2Story")):
            total_videos += 1
            
            final_response = video_data.get('whole_caption', '')
            clips_summaries = video_data.get('captions', [])
            
            # Check final response quality
            if not final_response or len(final_response.strip()) < 50:
                videos_with_bad_summaries += 1
                continue
            
            # Check each clip summary
            for summary in clips_summaries:
                total_clips += 1
                if (not summary or 
                    len(summary.strip()) < 20 or 
                    summary.strip().lower() in ['clip analyzed.', 'no description', 'n/a'] or
                    'error' in summary.lower()):
                    clips_with_bad_summaries += 1
    
    return total_videos, videos_with_bad_summaries, total_clips, clips_with_bad_summaries

def main():
    print("="*80)
    print("TRAINING DATA FALLBACK ANALYSIS")
    print("="*80)
    
    # Check SafeWatch data
    total_sw_clips, fallback_sw_clips, empty_sw, short_sw = check_safewatch_data()
    
    print(f"\n📊 SafeWatch Data Statistics:")
    print(f"  Total clips analyzed: {total_sw_clips}")
    print(f"  Clips using fallback: {fallback_sw_clips} ({fallback_sw_clips/total_sw_clips*100:.1f}%)")
    print(f"  - Empty descriptions: {empty_sw}")
    print(f"  - Short descriptions (<15 chars): {short_sw}")
    print(f"  - Actual fallback text: {fallback_sw_clips - empty_sw - short_sw}")
    
    # Check Shot2Story data
    total_s2s_videos, bad_s2s_videos, total_s2s_clips, bad_s2s_clips = check_shot2story_data()
    
    print(f"\n📊 Shot2Story Data Statistics:")
    print(f"  Total videos analyzed: {total_s2s_videos}")
    print(f"  Videos with bad summaries: {bad_s2s_videos} ({bad_s2s_videos/total_s2s_videos*100:.1f}%)")
    print(f"  Total clips analyzed: {total_s2s_clips}")
    print(f"  Clips with bad summaries: {bad_s2s_clips} ({bad_s2s_clips/total_s2s_clips*100:.1f}%)")
    
    print("\n" + "="*80)
    print("⚠️  FALLBACK IMPACT SUMMARY:")
    print("="*80)
    
    if fallback_sw_clips > 0:
        print(f"\n❌ SafeWatch: {fallback_sw_clips/total_sw_clips*100:.1f}% of clips will use fallback descriptions!")
        print("   This means the model is trained on generic fallback text instead of real descriptions.")
    
    if bad_s2s_clips > 0:
        print(f"\n❌ Shot2Story: {bad_s2s_clips/total_s2s_clips*100:.1f}% of clips have poor quality summaries!")
        print("   These clips may be filtered out or use poor quality text.")
    
    print("\n💡 RECOMMENDATION:")
    print("   The high fallback rate explains why the model doesn't generate proper summaries.")
    print("   The training data contains too many generic/fallback descriptions.")
    print("   Consider filtering the training data more strictly or using better annotations.")

if __name__ == "__main__":
    main()