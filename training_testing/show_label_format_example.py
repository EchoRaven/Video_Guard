#!/usr/bin/env python3
"""
Show examples of the new label format
"""

import sys
sys.path.append('/scratch/czr/Video-Guard/training_testing')

from Dataloader import StreamingDataset

def show_format_examples():
    """Display examples of the new label format"""
    
    print("New Label Format Examples")
    print("="*80)
    
    # Create dataset
    dataset = StreamingDataset(
        dataset_file='/scratch/czr/Video-Guard/datasets',
        tokenizer=None,
        max_samples=[1, 1],  # Just one sample of each
        max_length=16384,
        input_size=448,
        max_num_patches=6
    )
    
    # Find one clip sample and one final response sample
    clip_sample = None
    response_sample = None
    
    for sample in dataset.samples:
        if sample['type'] == 'clip' and clip_sample is None:
            clip_sample = sample
        elif sample['type'] == 'final_response' and response_sample is None:
            response_sample = sample
        
        if clip_sample and response_sample:
            break
    
    if clip_sample:
        print("\n📹 CLIP SAMPLE FORMAT:")
        print("-"*60)
        
        # Extract the part after the user prompt to show the actual format
        prompt = clip_sample['full_prompt']
        
        # Find where the actual frames start (after the user prompt)
        frame_start = prompt.find('<image>')
        if frame_start != -1:
            # Show first 3 frames
            frames = prompt[frame_start:].split('\n')[:3]
            
            for i, frame in enumerate(frames, 1):
                if frame:
                    print(f"\nFrame {i}:")
                    # Clean up for display
                    display = frame.replace('<image>', '[IMAGE_TOKENS]')
                    
                    # Limit length for readability
                    if len(display) > 200:
                        display = display[:200] + '...'
                    
                    print(f"  {display}")
        
        print("\n💡 Key Format Points for Clips:")
        print("  • Each frame: [IMAGE_TOKENS]<label><safe><continue></label>")
        print("  • Last frame: [IMAGE_TOKENS]<label></label><summary>description...</summary>")
        print("  • Clear boundaries with closing tags")
    
    if response_sample:
        print("\n\n📊 FINAL RESPONSE FORMAT:")
        print("-"*60)
        
        prompt = response_sample['full_prompt']
        
        # Find the response part
        response_start = prompt.find('<response>')
        vision_end = prompt.find('<|vision_end|>')
        
        if vision_end != -1:
            print("\nStructure:")
            print("  [User prompt]")
            print("  [All clip prompts with frames]")
            print("  <|vision_end|>")
            
            if response_start != -1:
                response_part = prompt[response_start:]
                # Show first and last parts
                if len(response_part) > 300:
                    print(f"  {response_part[:150]}...")
                    print(f"  ...{response_part[-50:]}")
                else:
                    print(f"  {response_part}")
        
        print("\n💡 Key Format Points for Final Response:")
        print("  • Wrapped in <response>...</response> tags")
        print("  • Clear end marker for inference")
        print("  • Follows after <|vision_end|> marker")
    
    print("\n" + "="*80)
    print("🎯 INFERENCE BENEFITS:")
    print("  1. Model knows when to stop generating after </label>")
    print("  2. Model knows when summary is complete after </summary>")
    print("  3. Model knows when final response is done after </response>")
    print("  4. Clear structure for streaming inference")

if __name__ == "__main__":
    show_format_examples()