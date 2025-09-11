#!/usr/bin/env python3
"""
Quick test script to verify the trained model
"""

import sys
import os
sys.path.append('/scratch/czr/Video-Guard/training_testing')

from test_trained_model import VideoGuardTester
import json

def main():
    print("="*80)
    print("QUICK MODEL TEST")
    print("="*80)
    
    # Configuration
    checkpoint_path = "/scratch/czr/Video-Guard/training_testing/output_8gpu_full/checkpoint-3500"  # Latest checkpoint
    sample_video = "/scratch/czr/Video-Guard/datasets/shot2story-videos/release_134k_videos/--5gVbAaF_A.9.mp4"
    
    print(f"\n📍 Configuration:")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Test video: {os.path.basename(sample_video)}")
    print()
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Create tester
    print("🔧 Loading model...")
    tester = VideoGuardTester(
        base_model_path="OpenGVLab/InternVL3-8B",
        checkpoint_path=checkpoint_path,
        device="cuda:0"
    )
    
    # Run test
    print("\n🎬 Analyzing video...")
    results = tester.analyze_video_streaming(
        sample_video,
        fps_sample=60,  # Sample every 2 seconds
        max_frames=5,   # Process 5 frames
        save_visualization=True,
        output_dir="./quick_test_results"
    )
    
    # Display results
    print("\n" + "="*80)
    print("ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\nVideo: {os.path.basename(sample_video)}")
    print(f"Frames processed: {results.get('total_frames_processed', 0)}")
    
    print("\n📊 Frame-by-frame analysis:")
    for frame_result in results.get('frame_results', []):
        labels = frame_result.get('labels', ['safe'])
        frame_num = frame_result.get('frame_number', 0)
        print(f"  Frame {frame_num}: {', '.join(labels)}")
        
        if frame_result.get('summary'):
            summary = frame_result['summary'][:100]
            if len(frame_result['summary']) > 100:
                summary += "..."
            print(f"    Summary: {summary}")
    
    print("\n📝 Final Video Summary:")
    final_summary = results.get('final_summary', 'No summary generated')
    if len(final_summary) > 200:
        final_summary = final_summary[:200] + "..."
    print(f"  {final_summary}")
    
    print("\n" + "="*80)
    print("✅ Test complete!")
    print(f"📁 Results saved to: ./quick_test_results/")
    print(f"🎨 Visualization: ./quick_test_results/*_visualization.png")
    print("="*80)

if __name__ == "__main__":
    main()