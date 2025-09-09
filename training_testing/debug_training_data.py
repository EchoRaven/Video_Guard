#!/usr/bin/env python3
"""Debug training data to see what model actually learns"""

from Dataloader import StreamingDataset
from train_multi_gpu import DataCollatorForStreaming
from transformers import AutoTokenizer
import torch

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/InternVL3-8B', trust_remote_code=True)

# Create dataset
dataset = StreamingDataset(
    dataset_file='/scratch/czr/Video-Guard/datasets',
    tokenizer=tokenizer,
    max_samples=[10, 10]  # Small sample
)

# Get a sample
sample = dataset[0]

print("="*80)
print("SAMPLE KEYS:", sample.keys())
print("="*80)

# Show the text
text = sample['text']
print("\nTEXT (first 500 chars):")
print(text[:500])

# Check what labels are in the data
print("\n" + "="*80)
print("CHECKING LABELS IN PROMPT:")
print("="*80)

# Look for our key tokens
key_patterns = ['<safe>', '<unsafe:', '<continue>', '<summary>', '<|vision_end|>']
for pattern in key_patterns:
    if pattern in text:
        # Find context around first occurrence
        idx = text.find(pattern)
        start = max(0, idx - 30)
        end = min(len(text), idx + 50)
        context = text[start:end]
        print(f"\n{pattern}:")
        print(f"  Context: ...{context}...")

# Test DataCollator
print("\n" + "="*80)
print("TESTING DATACOLLATOR:")
print("="*80)

collator = DataCollatorForStreaming(tokenizer)

# Create a simple batch
batch = collator([sample])

print(f"Input IDs shape: {batch['input_ids'].shape}")
print(f"Labels shape: {batch['labels'].shape}")
print(f"Pixel values: {batch['pixel_values'].shape if batch['pixel_values'] is not None else 'None'}")

# Check which parts are being trained (labels != -100)
labels = batch['labels'][0]
input_ids = batch['input_ids'][0]

# Find where training starts
train_start = -1
for i, label in enumerate(labels):
    if label != -100:
        train_start = i
        break

if train_start >= 0:
    print(f"\nTraining starts at position: {train_start}")
    print(f"Total sequence length: {len(input_ids)}")
    
    # Decode the part being trained
    trained_ids = input_ids[train_start:train_start+100]  # First 100 tokens being trained
    trained_text = tokenizer.decode(trained_ids, skip_special_tokens=False)
    print(f"\nFirst part being trained:")
    print(trained_text[:200])
else:
    print("\nNo training labels found (all -100)!")