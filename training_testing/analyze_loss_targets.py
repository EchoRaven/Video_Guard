#!/usr/bin/env python3
"""
Analyze what content should contribute to loss calculation
"""

import sys
sys.path.append('/scratch/czr/Video-Guard/training_testing')

from Dataloader import StreamingDataset

def analyze_loss_targets():
    """Analyze the structure of prompts to understand loss targets"""
    
    print("="*80)
    print("LOSS CALCULATION TARGET ANALYSIS")
    print("="*80)
    
    # Create small dataset
    dataset = StreamingDataset(
        dataset_file='/scratch/czr/Video-Guard/datasets',
        tokenizer=None,
        max_samples=[2, 2],
        max_length=16384,
        input_size=448,
        max_num_patches=6
    )
    
    print(f"\nDataset loaded: {len(dataset.samples)} samples")
    
    # Analyze different sample types
    clip_sample = None
    response_sample = None
    
    for sample in dataset.samples:
        if sample['type'] == 'clip' and clip_sample is None:
            clip_sample = sample
        elif sample['type'] == 'final_response' and response_sample is None:
            response_sample = sample
        
        if clip_sample and response_sample:
            break
    
    print("\n" + "="*60)
    print("CURRENT PROMPT STRUCTURE:")
    print("="*60)
    
    if clip_sample:
        print("\n📹 CLIP SAMPLE STRUCTURE:")
        prompt = clip_sample['full_prompt']
        
        # Find key sections
        lines = prompt.split('\n')[:10]  # First 10 lines
        
        print("\nPrompt components:")
        print("1. User instruction (system prompt)")
        print("2. Frame inputs with labels")
        print("3. Model responses")
        
        # Show example frame
        frame_start = prompt.find('<image>')
        if frame_start != -1:
            frame_end = prompt.find('\n', frame_start)
            if frame_end != -1:
                example_frame = prompt[frame_start:frame_end]
                print(f"\nExample frame output: {example_frame[:200]}")
    
    if response_sample:
        print("\n📊 FINAL RESPONSE STRUCTURE:")
        prompt = response_sample['full_prompt']
        
        # Find response section
        response_start = prompt.find('<response>')
        vision_end = prompt.find('<|vision_end|>')
        
        if vision_end != -1:
            print("\nComponents:")
            print("1. User instruction")
            print("2. All clip prompts with frames")
            print("3. <|vision_end|> marker")
            print("4. <response>...</response> final summary")
    
    print("\n" + "="*60)
    print("LOSS CALCULATION RECOMMENDATIONS:")
    print("="*60)
    
    print("\n🎯 SHOULD CONTRIBUTE TO LOSS (Model Outputs):")
    print("  ✅ <label>...</label> - Model's safety assessment")
    print("  ✅ <summary>...</summary> - Model's clip description")
    print("  ✅ <response>...</response> - Model's final video summary")
    
    print("\n❌ SHOULD NOT CONTRIBUTE TO LOSS (Inputs/Instructions):")
    print("  ❌ User prompt/instructions")
    print("  ❌ <image> tokens and pixel values")
    print("  ❌ <streaming_analysis> system prompt")
    print("  ❌ <|vision_end|> marker")
    
    print("\n📝 LOSS MASKING STRATEGY:")
    print("  1. For clip samples:")
    print("     - Calculate loss ONLY on <label>...</label> and <summary>...</summary>")
    print("     - Mask out everything before <label> (including <image> and instructions)")
    print("  2. For final response samples:")
    print("     - Calculate loss ONLY on <response>...</response>")
    print("     - Mask out entire prompt before <response>")
    
    print("\n⚠️  CURRENT ISSUE:")
    print("  If loss is calculated on the entire sequence, the model is being trained")
    print("  to predict the input instructions and image tokens, which is wrong.")
    print("  We need to mask these parts and only calculate loss on model outputs.")

if __name__ == "__main__":
    analyze_loss_targets()