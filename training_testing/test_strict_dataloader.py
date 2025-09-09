#!/usr/bin/env python3
"""
Test the modified Dataloader with strict filtering
"""

import sys
import os
sys.path.append('/scratch/czr/Video-Guard/training_testing')

from Dataloader import StreamingDataset
import logging

logging.basicConfig(level=logging.INFO)

def test_strict_dataloader():
    """Test that the dataloader correctly filters out videos with missing descriptions"""
    
    print("Testing Strict Dataloader...")
    print("="*80)
    
    # Create dataset with small sample size for testing
    dataset = StreamingDataset(
        dataset_file='/scratch/czr/Video-Guard/datasets',
        tokenizer=None,  # Not needed for this test
        max_samples=[100, 100],  # Small sample for testing
        max_length=16384,
        input_size=448,
        max_num_patches=6
    )
    
    print(f"\n📊 Dataset Statistics:")
    print(f"Total samples loaded: {len(dataset.samples)}")
    
    # Analyze samples
    clip_samples = 0
    final_response_samples = 0
    
    # Check that no clips have empty descriptions
    empty_descriptions = 0
    fallback_descriptions = 0
    good_descriptions = 0
    
    for sample in dataset.samples:
        if sample['type'] == 'clip':
            clip_samples += 1
            
            # Check the description in the clip info
            info = sample.get('info', {})
            summary = info.get('summary', '')
            
            if not summary or len(summary.strip()) < 20:
                empty_descriptions += 1
                print(f"  ❌ Found empty description in clip: {info.get('video_path', 'unknown')}")
            elif "This video clip" in summary and ("safety review" in summary or "safe content" in summary):
                fallback_descriptions += 1
                print(f"  ⚠️ Found fallback description in clip: {info.get('video_path', 'unknown')}")
            else:
                good_descriptions += 1
                
        elif sample['type'] == 'final_response':
            final_response_samples += 1
    
    print(f"\n📈 Sample Breakdown:")
    print(f"  Clip samples: {clip_samples}")
    print(f"  Final response samples: {final_response_samples}")
    
    print(f"\n🔍 Description Quality Check:")
    print(f"  Good descriptions: {good_descriptions} ({good_descriptions/clip_samples*100:.1f}%)" if clip_samples > 0 else "No clips")
    print(f"  Empty descriptions: {empty_descriptions} ({empty_descriptions/clip_samples*100:.1f}%)" if clip_samples > 0 else "No clips")
    print(f"  Fallback descriptions: {fallback_descriptions} ({fallback_descriptions/clip_samples*100:.1f}%)" if clip_samples > 0 else "No clips")
    
    print("\n✅ Test Results:")
    if empty_descriptions == 0 and fallback_descriptions == 0:
        print("  SUCCESS: No empty or fallback descriptions found!")
        print("  All clips have meaningful descriptions as required.")
    else:
        print(f"  FAILED: Found {empty_descriptions} empty and {fallback_descriptions} fallback descriptions")
        print("  The strict filtering is not working correctly.")
    
    # Additional check: verify that videos are properly paired with final responses
    print(f"\n📝 Video Integrity Check:")
    # Rough estimate: each video should have ~8 clips
    estimated_videos = final_response_samples
    avg_clips_per_video = clip_samples / estimated_videos if estimated_videos > 0 else 0
    print(f"  Estimated videos: {estimated_videos}")
    print(f"  Average clips per video: {avg_clips_per_video:.1f}")
    
    if avg_clips_per_video >= 3:  # Reasonable threshold
        print("  ✅ Videos appear to have complete clip sets")
    else:
        print("  ⚠️ Some videos may have been filtered out due to missing descriptions")
    
    return empty_descriptions == 0 and fallback_descriptions == 0

if __name__ == "__main__":
    success = test_strict_dataloader()
    
    if success:
        print("\n🎉 All tests passed! The Dataloader correctly filters out incomplete data.")
    else:
        print("\n❌ Tests failed. Please check the Dataloader implementation.")