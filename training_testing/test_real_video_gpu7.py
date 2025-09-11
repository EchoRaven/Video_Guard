#!/usr/bin/env python3
"""
Real Video Testing Example for Video-Guard Streaming Model using GPU 7
Shows detailed frame-by-frame analysis with actual video
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_streaming_model import StreamingVideoGuardTester
import logging

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def test_real_video():
    """Test on a real video with detailed output"""
    
    # Configuration
    base_model = "OpenGVLab/InternVL3-8B"
    checkpoint_path = "/scratch/czr/Video-Guard/training_testing/output_8gpu_full/checkpoint-500"
    
    # Select a real video from the dataset
    test_video = "/scratch/czr/Video-Guard/datasets/shot2story-videos/release_134k_videos/--5gVbAaF_A.9.mp4"
    
    # Test parameters
    fps_sample = 30  # Sample every 30 frames (1 second at 30fps)
    max_frames = 6   # Process 6 frames to save memory
    output_dir = "./real_video_test_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("Video-Guard Streaming Model - Real Video Testing (GPU 7)")
    print("="*80)
    print(f"Base Model: {base_model}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"GPU: cuda:7")
    print(f"Output Directory: {output_dir}")
    print("="*80)
    
    # Initialize the tester with GPU 7
    print("\n🔧 Initializing model on GPU 7...")
    tester = StreamingVideoGuardTester(
        base_model_path=base_model,
        checkpoint_path=checkpoint_path,
        device="cuda:0",  # Use cuda:0 when CUDA_VISIBLE_DEVICES=7 is set
        input_size=448,
        max_num_patches=6  # Reduce to save memory
    )
    print("✅ Model initialized successfully on GPU 7!")
    
    video_name = Path(test_video).name
    print(f"\n{'='*80}")
    print(f"Testing Video: {video_name}")
    print(f"{'='*80}")
    
    if not os.path.exists(test_video):
        print(f"❌ Video not found: {test_video}")
        return
    
    print(f"📹 Video path: {test_video}")
    print(f"⚙️  Settings: fps_sample={fps_sample}, max_frames={max_frames}")
    print("\n🎬 Starting streaming analysis...")
    print("-"*40)
    
    # Perform analysis
    results = tester.analyze_video_streaming(
        video_path=test_video,
        fps_sample=fps_sample,
        max_frames=max_frames,
        save_visualization=True,
        output_dir=output_dir
    )
    
    # Display detailed results
    print("\n📊 FRAME-BY-FRAME STREAMING ANALYSIS:")
    print("-"*40)
    
    for frame_result in results['frame_results']:
        frame_num = frame_result['frame_number']
        labels = frame_result['labels']
        summary = frame_result['summary']
        raw_response = frame_result.get('raw_response', '')
        
        print(f"\n🎞️  Frame {frame_num}:")
        print(f"   Patches: {frame_result['num_patches']}")
        print(f"   Labels: {', '.join(labels)}")
        
        # Show label interpretations
        print("   Interpretations:")
        for label in labels:
            if label == 'safe':
                print("      ✅ Safe content")
            elif label == 'continue':
                print("      ➡️  Shot continues")
            elif label.startswith('unsafe:'):
                unsafe_type = label.split(':')[1]
                unsafe_meanings = {
                    'C1': '🔴 Sexual content detected',
                    'C2': '🟠 Harassment/bullying detected',
                    'C3': '🔴 Violence/harm detected',
                    'C4': '🟣 Misinformation detected',
                    'C5': '🟤 Illegal activities detected',
                    'C6': '🔴 Hate speech/extremism detected'
                }
                print(f"      {unsafe_meanings.get(unsafe_type, '⚠️  Unknown unsafe content')}")
        
        if summary:
            print(f"   📝 Summary: {summary[:150]}...")
            if len(summary) > 150:
                print(f"      (Full summary: {len(summary)} characters)")
        
        # Show raw response for debugging (first 150 chars)
        if raw_response and len(raw_response) > 0:
            print(f"   🔍 Raw response preview: {raw_response[:150]}...")
    
    # Final summary
    print("\n" + "="*40)
    print("📋 FINAL ANALYSIS SUMMARY:")
    print("="*40)
    print(f"Total frames processed: {results['total_frames_processed']}")
    print(f"Timestamp: {results['timestamp']}")
    
    if results['final_summary']:
        print(f"\n📖 Final Video Summary:")
        print(f"   {results['final_summary'][:300]}")
        if len(results['final_summary']) > 300:
            print(f"   ... (Total: {len(results['final_summary'])} characters)")
    else:
        print("\n⚠️  No final summary generated")
    
    # Show content safety analysis
    print("\n🛡️  CONTENT SAFETY ANALYSIS:")
    print("-"*40)
    
    # Count label occurrences
    label_counts = {}
    for frame_result in results['frame_results']:
        for label in frame_result['labels']:
            label_counts[label] = label_counts.get(label, 0) + 1
    
    # Determine overall safety
    unsafe_labels = [l for l in label_counts.keys() if l.startswith('unsafe:')]
    if unsafe_labels:
        print("⚠️  UNSAFE CONTENT DETECTED:")
        for label in unsafe_labels:
            count = label_counts[label]
            unsafe_type = label.split(':')[1]
            unsafe_meanings = {
                'C1': 'Sexual content',
                'C2': 'Harassment/bullying',
                'C3': 'Violence/harm',
                'C4': 'Misinformation',
                'C5': 'Illegal activities',
                'C6': 'Hate speech/extremism'
            }
            meaning = unsafe_meanings.get(unsafe_type, 'Unknown')
            print(f"   - {label} ({meaning}): {count} frame(s)")
    else:
        print("✅ Video appears to be SAFE")
        print(f"   - Safe frames: {label_counts.get('safe', 0)}")
        print(f"   - Continue frames: {label_counts.get('continue', 0)}")
    
    # Save detailed report
    report_path = os.path.join(output_dir, f"{Path(video_name).stem}_detailed_report.json")
    detailed_report = {
        "video_name": video_name,
        "video_path": test_video,
        "analysis_timestamp": datetime.now().isoformat(),
        "settings": {
            "fps_sample": fps_sample,
            "max_frames": max_frames,
            "model": base_model,
            "checkpoint": checkpoint_path,
            "device": "cuda:7"
        },
        "results": results,
        "safety_analysis": {
            "label_counts": label_counts,
            "has_unsafe_content": len(unsafe_labels) > 0,
            "unsafe_categories": unsafe_labels
        }
    }
    
    with open(report_path, 'w') as f:
        json.dump(detailed_report, f, indent=2)
    
    print(f"\n💾 Detailed report saved to: {report_path}")
    print(f"🖼️  Visualization saved to: {os.path.join(output_dir, Path(video_name).stem + '_streaming_visualization.png')}")
    
    print("\n" + "="*80)
    print("✅ Testing Complete!")
    print("="*80)
    
    # Show example of how streaming works
    print("\n📚 HOW STREAMING WORKS IN THIS TEST:")
    print("-"*40)
    print("1. Frame 1 is processed → Model generates response based on first frame only")
    print("2. Frame 2 is processed → Model sees Frame 1's analysis + Frame 2")
    print("3. Frame 3 is processed → Model sees Frame 1&2's analysis + Frame 3")
    print("   ... and so on ...")
    print("This mimics the training process where the model learns to maintain context!")
    print("="*80)

if __name__ == "__main__":
    test_real_video()