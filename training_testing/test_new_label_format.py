#!/usr/bin/env python3
"""
Test the new label format with closing tags
"""

import sys
import os
sys.path.append('/scratch/czr/Video-Guard/training_testing')

from Dataloader import StreamingDataset
import re

def check_tag_balance(text, tag_name):
    """Check if opening and closing tags are balanced"""
    open_pattern = f'<{tag_name}>'
    close_pattern = f'</{tag_name}>'
    
    open_count = len(re.findall(open_pattern, text))
    close_count = len(re.findall(close_pattern, text))
    
    return open_count, close_count, open_count == close_count

def test_new_label_format():
    """Test that the new label format is correctly applied"""
    
    print("Testing New Label Format with Closing Tags")
    print("="*80)
    
    # Create dataset with small sample size for testing
    dataset = StreamingDataset(
        dataset_file='/scratch/czr/Video-Guard/datasets',
        tokenizer=None,
        max_samples=[10, 10],  # Very small sample for detailed testing
        max_length=16384,
        input_size=448,
        max_num_patches=6
    )
    
    print(f"\n📊 Dataset loaded: {len(dataset.samples)} samples")
    
    # Test different types of samples
    clip_samples_tested = 0
    final_response_samples_tested = 0
    
    errors = []
    
    for i, sample in enumerate(dataset.samples[:20]):  # Test first 20 samples
        full_prompt = sample['full_prompt']
        sample_type = sample['type']
        
        if sample_type == 'clip':
            clip_samples_tested += 1
            
            # Check for label tags
            label_open, label_close, label_balanced = check_tag_balance(full_prompt, 'label')
            if not label_balanced:
                errors.append(f"Clip {i}: Unbalanced <label> tags ({label_open} open, {label_close} close)")
            
            # Check for summary tags
            summary_open, summary_close, summary_balanced = check_tag_balance(full_prompt, 'summary')
            if not summary_balanced:
                errors.append(f"Clip {i}: Unbalanced <summary> tags ({summary_open} open, {summary_close} close)")
            
            # Verify the format structure
            # Should have patterns like: <label>...<safe><continue></label> or <label>...</label><summary>...</summary>
            label_pattern = r'<label>.*?</label>'
            summary_pattern = r'<summary>.*?</summary>'
            
            label_matches = re.findall(label_pattern, full_prompt, re.DOTALL)
            summary_matches = re.findall(summary_pattern, full_prompt, re.DOTALL)
            
            # Print sample output for inspection
            if clip_samples_tested <= 3:  # Show first 3 clips
                print(f"\n📝 Clip Sample {clip_samples_tested}:")
                print("  First 500 chars of prompt:")
                print("  " + full_prompt[:500].replace('\n', '\n  '))
                print(f"  Label tags found: {len(label_matches)}")
                print(f"  Summary tags found: {len(summary_matches)}")
                
                # Show one label example
                if label_matches:
                    print(f"  Example label: {label_matches[0][:100]}...")
                if summary_matches:
                    print(f"  Example summary: {summary_matches[0][:100]}...")
        
        elif sample_type == 'final_response':
            final_response_samples_tested += 1
            
            # Check for response tags
            response_open, response_close, response_balanced = check_tag_balance(full_prompt, 'response')
            if not response_balanced:
                errors.append(f"Final response {i}: Unbalanced <response> tags ({response_open} open, {response_close} close)")
            
            # Check that response has closing tag
            if '<response>' in full_prompt and not '</response>' in full_prompt:
                errors.append(f"Final response {i}: Missing </response> closing tag")
            
            # Print sample for inspection
            if final_response_samples_tested <= 2:  # Show first 2 final responses
                print(f"\n📝 Final Response Sample {final_response_samples_tested}:")
                # Find the response part
                response_start = full_prompt.find('<response>')
                if response_start != -1:
                    response_part = full_prompt[response_start:response_start+200]
                    print(f"  Response start: {response_part}...")
                    
                    # Check if it ends properly
                    if '</response>' in full_prompt:
                        response_end = full_prompt.rfind('</response>')
                        ending = full_prompt[max(0, response_end-50):response_end+11]
                        print(f"  Response end: ...{ending}")
    
    # Summary
    print("\n" + "="*80)
    print("📊 Test Summary:")
    print(f"  Clip samples tested: {clip_samples_tested}")
    print(f"  Final response samples tested: {final_response_samples_tested}")
    
    if errors:
        print(f"\n❌ Found {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
    else:
        print("\n✅ All tags are properly balanced!")
    
    # Verify tag patterns for inference
    print("\n🔍 Inference Pattern Verification:")
    print("  The new format allows clear detection of:")
    print("  1. Label end: </label> marks when frame labels are complete")
    print("  2. Summary end: </summary> marks when summary is complete")  
    print("  3. Response end: </response> marks when final response is complete")
    print("  This enables proper streaming inference control.")
    
    return len(errors) == 0

if __name__ == "__main__":
    success = test_new_label_format()
    
    if success:
        print("\n🎉 Success! The new label format is correctly implemented.")
    else:
        print("\n⚠️ Issues found. Please review the tag formatting.")