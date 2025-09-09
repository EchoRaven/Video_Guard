#!/usr/bin/env python3
"""Test different checkpoints to find the best one"""

from streaming_inference import StreamingVideoAnalyzer
import numpy as np
from PIL import Image

# Checkpoints to test
checkpoints = [
    990,    # Early
    3000,   # Medium early
    5000,   # Medium
    8000,   # Medium late
    10000,  # Late
    12000,  # Very late
]

# Create test image
test_image = Image.fromarray(np.zeros((448, 448, 3), dtype=np.uint8))

print("Testing different checkpoints...")
print("="*80)

for ckpt in checkpoints:
    ckpt_path = f'/scratch/czr/Video-Guard/training_testing/output_full_lora_streaming/checkpoint-{ckpt}'
    
    try:
        print(f"\nCheckpoint-{ckpt}:")
        print("-"*40)
        
        # Load model
        analyzer = StreamingVideoAnalyzer(
            'OpenGVLab/InternVL3-8B',
            checkpoint_path=ckpt_path,
            device_id=5
        )
        
        # Process image
        pixel_values, num_patches = analyzer.process_frame(test_image)
        
        # Build prompt
        prompt = analyzer.streaming_prompt + '\n<image>'
        tokens_per_patch = 256
        total_tokens = num_patches * tokens_per_patch
        IMG_START_TOKEN = '<img>'
        IMG_END_TOKEN = '</img>'
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * total_tokens + IMG_END_TOKEN
        prompt_with_tokens = prompt.replace('<image>', image_tokens)
        
        # Generate response
        response = analyzer.generate_response(prompt_with_tokens, pixel_values)
        
        # Parse
        labels, action, summary = analyzer.parse_response(response)
        
        # Print results
        print(f"  Raw (first 100 chars): {response[:100]}")
        print(f"  Labels: {labels}")
        print(f"  Action: {action}")
        if summary:
            print(f"  Summary: {summary[:50]}...")
        
        # Clean up model to free memory
        del analyzer
        import torch
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "="*80)
print("Test complete!")