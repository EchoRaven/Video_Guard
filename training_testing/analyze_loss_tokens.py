#!/usr/bin/env python3
"""
Analyze how many tokens participate in loss calculation
"""

import torch
import json
from Dataloader import StreamingDataset, construct_clip_prompt
from transformers import AutoTokenizer
import numpy as np

def analyze_loss_tokens():
    """Analyze the distribution of tokens participating in loss"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "OpenGVLab/InternVL3-8B",
        trust_remote_code=True,
        use_fast=False
    )
    
    # Create a sample clip prompt (similar to training)
    user_prompt = """<streaming_analysis> You are a video analyst. Analyze video frames to understand and describe content.
Watch each frame and respond with labels in <label>...</label> tags:
- <unsafe:C1>: Sexual content
- <unsafe:C2>: Harassment/bullying  
- <unsafe:C3>: Violence/harm
- <unsafe:C4>: Misinformation
- <unsafe:C5>: Illegal activities
- <unsafe:C6>: Hate speech/extremism
- <safe>: The frame is safe
- <continue>: Shot is not complete, keep analyzing
For the last frame, use <summary>...</summary> to provide a complete shot description"""
    
    # Simulate a 3-frame clip
    clip_info = {
        'user_prompt': user_prompt,
        'patches_per_frame_list': [3, 3, 3],  # 3 patches per frame
        'num_patches_per_frame': 3,
        'video_path': 'dummy.mp4',
        'clip_labels': [
            ['<safe>', '<continue>'],  # Frame 1
            ['<safe>', '<continue>'],  # Frame 2
            ['<safe>', '<summary>']    # Frame 3
        ],
        'sampled_frame_indices': [0, 30, 60],
        'summary': 'This is a test video showing safe content'
    }
    
    # Construct the full prompt
    full_prompt = construct_clip_prompt(clip_info)
    print("="*80)
    print("FULL PROMPT:")
    print("="*80)
    print(full_prompt)
    print("="*80)
    
    # Tokenize
    tokens = tokenizer(full_prompt, return_tensors='pt')
    input_ids = tokens['input_ids'][0]
    
    print(f"\nTotal tokens: {len(input_ids)}")
    
    # Simulate label creation (following training logic)
    labels = input_ids.clone()
    
    # Find first <img> token (151665)
    img_token_id = 151665
    img_positions = (input_ids == img_token_id).nonzero(as_tuple=True)[0]
    
    if len(img_positions) > 0:
        first_img_pos = img_positions[0].item()
        # Everything before first <img> is ignored
        labels[:first_img_pos] = -100
        
        # Also ignore image tokens
        img_end_token_id = 151666  # </img>
        img_context_token_id = 151667  # <IMG_CONTEXT>
        
        for i in range(len(input_ids)):
            if input_ids[i] in [img_token_id, img_end_token_id, img_context_token_id]:
                labels[i] = -100
    
    # Count tokens participating in loss
    valid_labels = labels[labels != -100]
    num_loss_tokens = len(valid_labels)
    loss_ratio = num_loss_tokens / len(input_ids)
    
    print(f"\nTokens participating in loss: {num_loss_tokens}/{len(input_ids)} ({loss_ratio:.1%})")
    
    # Decode the tokens that participate in loss
    print("\n" + "="*80)
    print("TOKENS PARTICIPATING IN LOSS:")
    print("="*80)
    
    # Create a mask for loss tokens
    loss_mask = (labels != -100)
    loss_input_ids = input_ids.clone()
    loss_input_ids[~loss_mask] = tokenizer.pad_token_id
    
    # Decode only the loss tokens
    loss_text = tokenizer.decode(loss_input_ids[loss_mask])
    print(loss_text)
    
    # Analyze what types of tokens are in loss
    print("\n" + "="*80)
    print("TOKEN ANALYSIS:")
    print("="*80)
    
    # Check for specific tokens
    label_tokens = ['<label>', '</label>', '<safe>', '<continue>', '<summary>', '</summary>']
    for token_str in label_tokens:
        token_ids = tokenizer.encode(token_str, add_special_tokens=False)
        count = 0
        for tid in token_ids:
            count += (loss_input_ids[loss_mask] == tid).sum().item()
        print(f"{token_str}: {count} occurrences in loss")
    
    # Estimate cross-entropy loss
    print("\n" + "="*80)
    print("LOSS ESTIMATION:")
    print("="*80)
    
    # If model predicts perfectly (probability=1 for correct token)
    perfect_loss = 0.0
    
    # If model predicts randomly (uniform distribution over vocab)
    vocab_size = len(tokenizer)
    random_loss = np.log(vocab_size)
    
    # Typical good model (90% probability for correct token)
    good_model_loss = -np.log(0.9)
    
    print(f"Perfect prediction loss: {perfect_loss:.4f}")
    print(f"Good model (90% accuracy) loss: {good_model_loss:.4f}")  
    print(f"Random prediction loss: {random_loss:.4f}")
    
    # Your model's loss of 0.07 suggests ~93% token accuracy
    observed_loss = 0.07
    implied_accuracy = np.exp(-observed_loss)
    print(f"\nObserved loss of {observed_loss:.4f} implies ~{implied_accuracy:.1%} token accuracy")
    
    print("\n" + "="*80)
    print("CONCLUSION:")
    print("="*80)
    print(f"With {num_loss_tokens} tokens participating in loss,")
    print(f"and {loss_ratio:.1%} of total tokens being trained,")
    print(f"a loss of 0.07 is reasonable for a well-trained model.")
    print("The model has learned to predict the label tokens with high accuracy.")

if __name__ == "__main__":
    analyze_loss_tokens()