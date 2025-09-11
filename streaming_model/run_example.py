#!/usr/bin/env python3
"""
Run the example from test_full_video_streaming.py using the new StreamingModel
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import StreamingModel
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from streaming_inference import StreamingModel

def main():
    # Configuration - same as test_full_video_streaming.py
    checkpoint = '/scratch/czr/Video-Guard/training_testing/output_4gpu_bs2_16k/checkpoint-4000'
    
    # Example unsafe video from SafeWatch dataset - SEXUAL CONTENT (should trigger unsafe:C1)
    video_path = '/scratch/czr/Video-Guard/datasets/SafeWatch-Bench/real/videos/C4/aigc content_benchmark/3.mp4'
    # Sexual 4:
    # video_path = "/scratch/czr/Video-Guard/datasets/SafeWatch-Bench-200K-720P/full/sexual_4/target/[filename]"
    
    # Sexual 5:
    # video_path = "/scratch/czr/Video-Guard/datasets/SafeWatch-Bench-200K-720P/full/sexual_5/target/[filename]"
    
    # Violence:
    # video_path = "/scratch/czr/Video-Guard/datasets/SafeWatch-Bench-200K-720P/full/violence_1_abuse/target/Abuse001_x264.mp4_merged.mp4"
    
    # Abuse:
    # video_path = "/scratch/czr/Video-Guard/datasets/SafeWatch-Bench-200K-720P/full/abuse_1/target/-0nDuJdCniyuJCgx.mp4"
    
    # Extremism:
    # video_path = "/scratch/czr/Video-Guard/datasets/SafeWatch-Bench-200K-720P/full/extremism/target/00h.05mn__7179721456656452869.mp4"
    
    # Safe video from shot2story:
    # video_path = "/scratch/czr/Video-Guard/datasets/shot2story-videos/release_134k_videos/--5gVbAaF_A.9.mp4"
    
    print("="*80)
    print("STREAMING MODEL - FULL VIDEO ANALYSIS")
    print("Using new StreamingModel class")
    print("="*80)
    
    # Initialize StreamingModel
    print(f"\nInitializing StreamingModel...")
    print(f"Checkpoint: {checkpoint}")
    print(f"Video: {video_path}")
    
    model = StreamingModel(
        base_model_path="OpenGVLab/InternVL3-8B",
        checkpoint_path=checkpoint,
        device="cuda:0"
    )
    
    # Process video with 1 fps sampling (same as test file)
    print("\nProcessing video...")
    results = model.process_video_streaming(video_path, fps=1.0)
    
    # Save results
    video_name = Path(video_path).stem
    output_dir = "./streaming_model_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON results
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{video_name}_streaming_{timestamp}.json")
    model.save_results(results, output_path)
    
    # Also save a latest version for easy access
    latest_path = os.path.join(output_dir, f"{video_name}_streaming_latest.json")
    model.save_results(results, latest_path)
    
    # Print summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    if results['status'] == 'completed':
        # Extract metadata
        meta = results.get('analysis_metadata', {})
        print(f"\nAnalysis Summary:")
        print(f"  Total frames processed: {meta.get('total_frames_processed', 0)}")
        print(f"  Total clips (after merge): {meta.get('total_clips_after_merge', 0)}")
        print(f"  Max patches per frame: {meta.get('max_patches_per_frame', 2)}")
        print(f"  Vision end token used: {meta.get('vision_end_token_used', False)}")
        
        # Print clip summaries
        print(f"\nClip Breakdown:")
        for clip in results.get('clips', []):
            print(f"\n  Clip {clip['clip_idx']}:")
            print(f"    Frames: {clip['num_frames']}")
            summary = clip.get('clip_summary', 'No summary')
            if len(summary) > 100:
                print(f"    Summary: {summary[:100]}...")
            else:
                print(f"    Summary: {summary}")
            
            # Count labels in this clip
            label_counts = {}
            for frame_result in clip.get('frame_results', []):
                for label in frame_result.get('labels', []):
                    label_counts[label] = label_counts.get(label, 0) + 1
            if label_counts:
                print(f"    Labels: {', '.join(f'{k}({v})' for k, v in label_counts.items())}")
        
        # Print final response
        final_resp = results.get('final_response', {})
        if isinstance(final_resp, dict):
            content = final_resp.get('cleaned', 'No final response')
            print(f"\nFinal Video Summary:")
            if len(content) > 500:
                print(f"  {content[:500]}...")
            else:
                print(f"  {content}")
            print(f"\n  Token count: {final_resp.get('token_count', 0)}")
            print(f"  Truncated: {final_resp.get('truncated', False)}")
        
        print(f"\nResults saved to:")
        print(f"  Full: {output_path}")
        print(f"  Latest: {latest_path}")
    else:
        print(f"\nError: {results.get('error', 'Unknown error')}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()