#!/usr/bin/env python3
"""Show detailed inference example with full output"""

import json
from streaming_inference import StreamingVideoAnalyzer

# Test video path
VIDEO_PATH = '/scratch/czr/Video-Guard/datasets/shot2story-videos/release_134k_videos/--5gVbAaF_A.9.mp4'

print("="*80)
print("STREAMING VIDEO ANALYSIS EXAMPLE")
print("="*80)

# Create analyzer
print("\n1. Loading model...")
analyzer = StreamingVideoAnalyzer(
    'OpenGVLab/InternVL3-8B',
    '/scratch/czr/Video-Guard/training_testing/output_full_lora_streaming/checkpoint-990',
    device_id=5
)

# Analyze video with more frames
print("\n2. Analyzing video (processing 5 frames)...")
results = analyzer.analyze_video_streaming(
    VIDEO_PATH,
    fps_sample=90,  # Sample every ~3-4 seconds
    max_frames=5    # Process 5 frames total
)

# Display detailed results
print("\n" + "="*80)
print("ANALYSIS RESULTS")
print("="*80)

print(f"\nVideo: {VIDEO_PATH.split('/')[-1]}")
print(f"Total frames processed: {results['total_frames_processed']}")
print(f"Number of shots detected: {len(results['shots'])}")

print("\n" + "-"*80)
print("SHOT-BY-SHOT ANALYSIS:")
print("-"*80)

for i, shot in enumerate(results['shots'], 1):
    print(f"\n>>> Shot #{i}")
    print(f"    Frame indices: {shot['frame_indices']}")
    print(f"    Labels per frame:")
    for j, labels in enumerate(shot['frame_labels']):
        print(f"      Frame {j+1}: {labels if labels else '[No labels]'}")
    print(f"    Summary: {shot['summary']}")

print("\n" + "-"*80)
print("FINAL VIDEO SUMMARY:")
print("-"*80)
print(f"\n{results['final_summary']}")

# Save full results to JSON
output_file = 'inference_example_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Full results saved to: {output_file}")
print("="*80)