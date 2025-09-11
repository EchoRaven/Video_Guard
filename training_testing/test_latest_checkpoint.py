#!/usr/bin/env python3
"""
Test with the latest checkpoint (3500 steps)
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_streaming_model import StreamingVideoGuardTester
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def test_with_latest_checkpoint():
    # Use latest checkpoint
    checkpoint_path = "/scratch/czr/Video-Guard/training_testing/output_8gpu_full/checkpoint-3500"
    test_video = "/scratch/czr/Video-Guard/datasets/shot2story-videos/release_134k_videos/--5gVbAaF_A.9.mp4"
    
    print("="*80)
    print("Testing with LATEST checkpoint (3500 steps)")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Video: {test_video}")
    print("="*80)
    
    # Initialize tester
    print("\n🔧 Initializing model...")
    tester = StreamingVideoGuardTester(
        base_model_path="OpenGVLab/InternVL3-8B",
        checkpoint_path=checkpoint_path,
        device="cuda:0",  # Will use GPU 7 with CUDA_VISIBLE_DEVICES
        input_size=448,
        max_num_patches=6
    )
    print("✅ Model loaded successfully!")
    
    # Test the video
    print("\n📹 Analyzing video...")
    results = tester.analyze_video_streaming(
        video_path=test_video,
        fps_sample=30,
        max_frames=4,  # Test with fewer frames first
        save_visualization=True,
        output_dir="./latest_checkpoint_results"
    )
    
    print("\n📊 RESULTS:")
    print("-"*40)
    
    for frame_result in results['frame_results']:
        frame_num = frame_result['frame_number']
        labels = frame_result['labels']
        raw_response = frame_result['raw_response']
        has_continue = frame_result.get('has_continue', False)
        has_summary = frame_result.get('has_summary', False)
        
        print(f"\n🎞️ Frame {frame_num}:")
        print(f"   Labels: {labels}")
        print(f"   Has continue: {has_continue}")
        print(f"   Has summary: {has_summary}")
        
        # Show full raw response for debugging
        if raw_response:
            print(f"   Raw response: '{raw_response}'")
        else:
            print(f"   Raw response: [EMPTY]")
        
        if frame_result['summary']:
            print(f"   Summary: {frame_result['summary'][:100]}...")
    
    # Check if model is generating proper streaming format
    print("\n" + "="*40)
    print("📝 ANALYSIS:")
    print("-"*40)
    
    # Count how many frames had proper responses
    frames_with_continue = sum(1 for r in results['frame_results'] if r.get('has_continue', False))
    frames_with_summary = sum(1 for r in results['frame_results'] if r.get('has_summary', False))
    frames_with_labels = sum(1 for r in results['frame_results'] if len(r.get('labels', [])) > 0)
    empty_responses = sum(1 for r in results['frame_results'] if not r.get('raw_response', ''))
    
    print(f"Total frames: {len(results['frame_results'])}")
    print(f"Frames with labels: {frames_with_labels}")
    print(f"Frames with continue: {frames_with_continue}")
    print(f"Frames with summary: {frames_with_summary}")
    print(f"Empty responses: {empty_responses}")
    
    if empty_responses == len(results['frame_results']):
        print("\n⚠️ WARNING: Model is not generating any text!")
        print("This could mean:")
        print("1. The model needs more training")
        print("2. The generation parameters need adjustment")
        print("3. The prompt format doesn't match training")
    elif frames_with_continue == 0 and frames_with_summary == 0:
        print("\n⚠️ WARNING: Model is not generating proper streaming format!")
        print("Expected: <safe><continue></label> or <safe></label><summary>...</summary>")
    else:
        print("\n✅ Model is generating streaming format correctly!")
    
    print("="*80)

if __name__ == "__main__":
    test_with_latest_checkpoint()