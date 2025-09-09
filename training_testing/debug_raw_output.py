#!/usr/bin/env python3
"""Debug raw model output"""

import torch
from streaming_inference import StreamingVideoAnalyzer
import numpy as np
from PIL import Image

# Create analyzer
analyzer = StreamingVideoAnalyzer(
    'OpenGVLab/InternVL3-8B',
    '/scratch/czr/Video-Guard/training_testing/output_full_lora_streaming/checkpoint-990',
    device_id=5
)

# Create a simple test with black image
test_image = Image.fromarray(np.zeros((448, 448, 3), dtype=np.uint8))
pixel_values, num_patches = analyzer.process_frame(test_image)

# Build simple prompt
prompt = analyzer.streaming_prompt + '\n<image>'

# Replace <image> with tokens
tokens_per_patch = 256
total_tokens = num_patches * tokens_per_patch
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * total_tokens + IMG_END_TOKEN
prompt_with_tokens = prompt.replace('<image>', image_tokens)

print("Testing raw model output...")
print("="*80)
print("Input prompt (truncated):")
print(prompt[:200] + "...")
print("\nNumber of patches:", num_patches)
print("Number of image tokens:", total_tokens)

# Generate response
response = analyzer.generate_response(prompt_with_tokens, pixel_values)

print("\n" + "="*80)
print("RAW MODEL OUTPUT:")
print("="*80)
print(response)

# Parse it
labels, action, summary = analyzer.parse_response(response)
print("\n" + "="*80)
print("PARSED:")
print(f"Labels: {labels}")
print(f"Action: {action}")
print(f"Summary: {summary}")