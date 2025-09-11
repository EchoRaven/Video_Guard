#!/usr/bin/env python3
"""
Real Video Testing Example for Video-Guard Streaming Model
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
    test_videos = [
        "/scratch/czr/Video-Guard/datasets/shot2story-videos/release_134k_videos/--5gVbAaF_A.9.mp4",
        "/scratch/czr/Video-Guard/datasets/shot2story-videos/release_134k_videos/--69fZu7c9w.4.mp4",
        "/scratch/czr/Video-Guard/datasets/shot2story-videos/release_134k_videos/--7wvrXfl0g.5.mp4"
    ]
    
    # Test parameters
    fps_sample = 30  # Sample every 30 frames (1 second at 30fps)
    max_frames = 8   # Process maximum 8 frames (matching training)
    output_dir = "./real_video_test_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("Video-Guard Streaming Model - Real Video Testing")
    print("="*80)
    print(f"Base Model: {base_model}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output Directory: {output_dir}")
    print("="*80)
    
    # Initialize the tester
    print("\n🔧 Initializing model...")
    tester = StreamingVideoGuardTester(
        base_model_path=base_model,
        checkpoint_path=checkpoint_path,
        device="cuda:0",
        input_size=448,
        max_num_patches=12
    )
    print("✅ Model initialized successfully!")
    
    # Test each video
    for video_idx, video_path in enumerate(test_videos[:1], 1):  # Test first video for now
        video_name = Path(video_path).name
        print(f"\n{'='*80}")
        print(f"Testing Video {video_idx}: {video_name}")
        print(f"{'='*80}")
        
        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            continue
        
        print(f"📹 Video path: {video_path}")
        print(f"⚙️  Settings: fps_sample={fps_sample}, max_frames={max_frames}")
        print("\n🎬 Starting streaming analysis...")
        print("-"*40)
        
        # Perform analysis
        results = tester.analyze_video_streaming(
            video_path=video_path,
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
                print(f"   📝 Summary: {summary[:100]}...")
                if len(summary) > 100:
                    print(f"      (Full summary: {len(summary)} characters)")
            
            # Show raw response for debugging
            if raw_response and len(raw_response) > 0:
                print(f"   🔍 Raw response preview: {raw_response[:80]}...")
        
        # Final summary
        print("\n" + "="*40)
        print("📋 FINAL ANALYSIS SUMMARY:")
        print("="*40)
        print(f"Total frames processed: {results['total_frames_processed']}")
        print(f"Timestamp: {results['timestamp']}")
        
        if results['final_summary']:
            print(f"\n📖 Final Video Summary:")
            print(f"   {results['final_summary']}")
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
                print(f"   - {label}: {count} frame(s)")
        else:
            print("✅ Video appears to be SAFE")
        
        # Save detailed report
        report_path = os.path.join(output_dir, f"{Path(video_name).stem}_detailed_report.json")
        detailed_report = {
            "video_name": video_name,
            "video_path": video_path,
            "analysis_timestamp": datetime.now().isoformat(),
            "settings": {
                "fps_sample": fps_sample,
                "max_frames": max_frames,
                "model": base_model,
                "checkpoint": checkpoint_path
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

def create_test_comparison():
    """Create a comparison between batch and streaming inference"""
    
    print("\n" + "="*80)
    print("STREAMING vs BATCH INFERENCE COMPARISON")
    print("="*80)
    
    print("\n📌 KEY DIFFERENCES:")
    print("-"*40)
    
    print("\n1️⃣  BATCH INFERENCE (Original test_trained_model.py):")
    print("   • Processes all frames at once")
    print("   • Concatenates all image tokens together")
    print("   • Single forward pass through the model")
    print("   • Cannot maintain conversation context between frames")
    print("   • Example prompt structure:")
    print("     <streaming_analysis>...")
    print("     Frame 1: <image><label>")
    print("     Frame 2: <image><label>")
    print("     Frame 3: <image><label>")
    print("     [All processed together]")
    
    print("\n2️⃣  STREAMING INFERENCE (New test_streaming_model.py):")
    print("   • Processes frames one by one")
    print("   • Maintains conversation history")
    print("   • Multiple forward passes (one per frame)")
    print("   • Each frame sees previous frame's analysis")
    print("   • Example prompt structure:")
    print("     Frame 1: <image><label> → Generate → <safe><continue></label>")
    print("     Frame 2: [includes Frame 1 context] <image><label> → Generate → <safe><continue></label>")
    print("     Frame 3: [includes Frame 1&2 context] <image><label> → Generate → <safe></label><summary>...</summary>")
    
    print("\n3️⃣  WHY STREAMING IS CORRECT:")
    print("   ✅ Matches training data format exactly")
    print("   ✅ Allows model to build understanding over time")
    print("   ✅ Can detect shot boundaries naturally")
    print("   ✅ Generates coherent summaries based on accumulated context")
    print("   ✅ More similar to how humans watch videos (frame by frame)")
    
    print("\n4️⃣  EXPECTED BEHAVIOR:")
    print("   • Early frames: Often '<safe><continue>' as shot develops")
    print("   • Middle frames: May detect specific content issues")
    print("   • Final frames: Should include summary when shot completes")
    print("   • Shot boundaries: Model may reset and start new analysis")
    
    print("="*80)

if __name__ == "__main__":
    # First show the comparison
    create_test_comparison()
    
    # Then run the real video test
    test_real_video()