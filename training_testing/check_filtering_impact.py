#!/usr/bin/env python3
"""
Check the impact of strict filtering on dataset size
"""

import json
from tqdm import tqdm

def check_filtering_impact():
    """Compare dataset size before and after strict filtering"""
    
    corrected_path = '/scratch/czr/Video-Guard/datasets/safewatch_streaming_corrected.jsonl'
    
    # Analyze the corrected file
    total_videos = 0
    videos_with_complete_descs = 0
    videos_with_missing_descs = 0
    
    print("Analyzing impact of strict filtering...")
    print("="*80)
    
    with open(corrected_path, 'r') as f:
        for line in tqdm(f, desc="Analyzing videos"):
            data = json.loads(line)
            total_videos += 1
            
            # Check if ALL clips have valid descriptions
            clips = data.get('clip_video_annotations', [])
            all_clips_valid = True
            
            for clip in clips:
                desc = clip.get('description', '')
                if not desc or len(desc.strip()) < 20:
                    all_clips_valid = False
                    break
            
            if all_clips_valid:
                videos_with_complete_descs += 1
            else:
                videos_with_missing_descs += 1
    
    print("\n📊 Filtering Impact Analysis:")
    print(f"Total videos in corrected file: {total_videos:,}")
    print(f"Videos with complete descriptions: {videos_with_complete_descs:,} ({videos_with_complete_descs/total_videos*100:.1f}%)")
    print(f"Videos filtered out (missing descriptions): {videos_with_missing_descs:,} ({videos_with_missing_descs/total_videos*100:.1f}%)")
    
    # Estimate training data reduction
    avg_clips_per_video = 8  # Approximate
    original_clips = total_videos * avg_clips_per_video
    remaining_clips = videos_with_complete_descs * avg_clips_per_video
    
    print("\n📉 Training Data Impact (estimated):")
    print(f"Original clip samples: ~{original_clips:,}")
    print(f"Remaining clip samples: ~{remaining_clips:,}")
    print(f"Reduction: ~{original_clips - remaining_clips:,} clips ({(original_clips - remaining_clips)/original_clips*100:.1f}%)")
    
    print("\n💡 Recommendation:")
    if videos_with_missing_descs / total_videos < 0.1:
        print("✅ Impact is minimal (<10% data loss). The strict filtering is acceptable.")
    elif videos_with_missing_descs / total_videos < 0.2:
        print("⚠️ Moderate impact (10-20% data loss). Consider if quality improvement justifies the reduction.")
    else:
        print("❌ Significant impact (>20% data loss). May need to reconsider the filtering strategy.")
    
    return videos_with_complete_descs, videos_with_missing_descs

if __name__ == "__main__":
    complete, missing = check_filtering_impact()
    
    print(f"\n📝 Summary:")
    print(f"The strict filtering ensures high-quality training data by excluding {missing:,} videos")
    print(f"that have incomplete clip descriptions, keeping {complete:,} high-quality videos.")