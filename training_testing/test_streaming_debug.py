#!/usr/bin/env python3
"""Debug streaming inference to see what model actually generates"""

import logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

from streaming_inference import StreamingVideoAnalyzer

# Create analyzer
analyzer = StreamingVideoAnalyzer(
    'OpenGVLab/InternVL3-8B',
    '/scratch/czr/Video-Guard/training_testing/output_full_lora_streaming/checkpoint-990',
    device_id=5
)

# Test with just 2 frames
results = analyzer.analyze_video_streaming(
    '/scratch/czr/Video-Guard/datasets/shot2story-videos/release_134k_videos/--5gVbAaF_A.9.mp4',
    fps_sample=60,
    max_frames=2
)

print('\n' + '='*80)
print('Results:')
for shot in results['shots']:
    print(f"Shot: {shot['summary']}")
print(f"Final: {results['final_summary'][:200]}...")
print('='*80)