#!/usr/bin/env python3
"""Check special token IDs"""

from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/InternVL3-8B', trust_remote_code=True)

# Check special tokens
special_tokens = {
    '<img>': None,
    '</img>': None,
    '<IMG_CONTEXT>': None,
    '<|vision_end|>': None,
    '<safe>': None,
    '<unsafe:C1>': None,
    '<continue>': None,
    '<summary>': None,
}

print("Checking special token IDs:")
print("="*50)
for token in special_tokens:
    token_id = tokenizer.convert_tokens_to_ids(token)
    special_tokens[token] = token_id
    print(f"{token:20} -> {token_id}")

print("\n" + "="*50)
print("Hardcoded values in training code:")
print("img_token_id = 151665  # <img>")
print("end_img_token_id = 151666  # </img>")
print("img_context_token_id = 151667  # <IMG_CONTEXT>")
print("vision_end_token_id = 151653  # <|vision_end|>")

print("\n" + "="*50)
print("Comparison:")
print(f"<img>: Actual={special_tokens['<img>']}, Hardcoded=151665, Match={special_tokens['<img>']==151665}")
print(f"</img>: Actual={special_tokens['</img>']}, Hardcoded=151666, Match={special_tokens['</img>']==151666}")
print(f"<IMG_CONTEXT>: Actual={special_tokens['<IMG_CONTEXT>']}, Hardcoded=151667, Match={special_tokens['<IMG_CONTEXT>']==151667}")
print(f"<|vision_end|>: Actual={special_tokens['<|vision_end|>']}, Hardcoded=151653, Match={special_tokens['<|vision_end|>']==151653}")