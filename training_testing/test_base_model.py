#!/usr/bin/env python3
"""Test base model without LoRA to see its default behavior"""

from streaming_inference import StreamingVideoAnalyzer
import numpy as np
from PIL import Image

# Create analyzer WITHOUT LoRA checkpoint
print("Loading BASE model (no LoRA)...")
analyzer = StreamingVideoAnalyzer(
    'OpenGVLab/InternVL3-8B',
    checkpoint_path=None,  # No LoRA
    device_id=5
)

# Create a test image
test_image = Image.fromarray(np.zeros((448, 448, 3), dtype=np.uint8))
pixel_values, num_patches = analyzer.process_frame(test_image)

# Build prompt
prompt = analyzer.streaming_prompt + '\n<image>'

# Replace <image> with tokens
tokens_per_patch = 256
total_tokens = num_patches * tokens_per_patch
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * total_tokens + IMG_END_TOKEN
prompt_with_tokens = prompt.replace('<image>', image_tokens)

print("\n" + "="*80)
print("BASE MODEL OUTPUT (without LoRA):")
print("="*80)

# Generate response
response = analyzer.generate_response(prompt_with_tokens, pixel_values)
print("Raw output:", response[:200])

# Parse
labels, action, summary = analyzer.parse_response(response)
print(f"\nParsed:")
print(f"  Labels: {labels}")
print(f"  Action: {action}")
print(f"  Summary: {summary[:100] if summary else None}")

print("\n" + "="*80)
print("Now testing WITH LoRA checkpoint:")
print("="*80)

# Load with LoRA
analyzer_lora = StreamingVideoAnalyzer(
    'OpenGVLab/InternVL3-8B',
    '/scratch/czr/Video-Guard/training_testing/output_full_lora_streaming/checkpoint-990',
    device_id=5
)

response_lora = analyzer_lora.generate_response(prompt_with_tokens, pixel_values)
print("Raw output:", response_lora[:200])

labels_lora, action_lora, summary_lora = analyzer_lora.parse_response(response_lora)
print(f"\nParsed:")
print(f"  Labels: {labels_lora}")
print(f"  Action: {action_lora}")
print(f"  Summary: {summary_lora[:100] if summary_lora else None}")