#!/usr/bin/env python3
"""Quick test of model behavior"""

import torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

print("Testing model outputs...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/InternVL3-8B', trust_remote_code=True)

# Test prompt
test_prompt = """<streaming_analysis> You are a video analyst. Analyze video frames to understand and describe content.
Watch each frame and decide:
- <unsafe:C1>: Sexual content
- <unsafe:C2>: Harassment/bullying  
- <unsafe:C3>: Violence/harm
- <unsafe:C4>: Misinformation
- <unsafe:C5>: Illegal activities
- <unsafe:C6>: Hate speech/extremism
- <safe>: The frame is safe
- <continue>: Shot is not complete, keep analyzing
- <summary>: Shot is complete, provide a complete shot description
[Imagine there is a black image here]
"""

# Tokenize
inputs = tokenizer(test_prompt, return_tensors='pt', truncation=False)

print("\n" + "="*80)
print("Test prompt tokens:", inputs['input_ids'].shape)
print("="*80)

# Check what tokens are generated for our key phrases
for phrase in ['<safe>', '<continue>', '<summary>', '<unsafe:C1>']:
    tokens = tokenizer(phrase, add_special_tokens=False)['input_ids']
    decoded = tokenizer.decode(tokens)
    print(f"{phrase:15} -> tokens: {tokens[:5]}... -> decoded: '{decoded}'")