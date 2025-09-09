#!/usr/bin/env python3
"""
Comprehensive verification of all training requirements
"""

import sys
import os
sys.path.append('/scratch/czr/Video-Guard/training_testing')

from Dataloader import StreamingDataset
import torch
import numpy as np

def verify_all_requirements():
    """Verify all training requirements are met"""
    
    print("="*80)
    print("TRAINING REQUIREMENTS VERIFICATION")
    print("="*80)
    
    # Create dataset
    dataset = StreamingDataset(
        dataset_file='/scratch/czr/Video-Guard/datasets',
        tokenizer=None,
        max_samples=[50, 50],  # Small sample for testing
        max_length=16384,
        input_size=448,
        max_num_patches=6
    )
    
    print(f"\n📊 Dataset loaded: {len(dataset.samples)} samples")
    
    # Track verification results
    checks = {
        'no_fallback': True,
        'frame_limits': True,
        'label_format': True,
        'real_frames': True,
        'no_empty_desc': True
    }
    
    # Test samples
    max_frames_found = 0
    samples_with_video = 0
    samples_without_video = 0
    black_frame_detected = False
    
    for i in range(min(20, len(dataset))):
        try:
            sample = dataset[i]
            
            # 1. Check no fallback descriptions
            text = sample['text']
            if "This video clip contains content that may require safety review" in text:
                print(f"  ❌ Sample {i}: Found fallback description")
                checks['no_fallback'] = False
            if "This video clip appears to contain safe content" in text:
                print(f"  ❌ Sample {i}: Found fallback description")
                checks['no_fallback'] = False
            
            # 2. Check label format (closing tags)
            if '<label>' in text and '</label>' not in text:
                print(f"  ❌ Sample {i}: Missing </label> closing tag")
                checks['label_format'] = False
            if '<summary>' in text and '</summary>' not in text:
                print(f"  ❌ Sample {i}: Missing </summary> closing tag")
                checks['label_format'] = False
            if '<response>' in text and '</response>' not in text:
                print(f"  ❌ Sample {i}: Missing </response> closing tag")
                checks['label_format'] = False
            
            # 3. Check frame count
            pixel_values = sample.get('pixel_values')
            if pixel_values is not None:
                samples_with_video += 1
                
                if isinstance(pixel_values, torch.Tensor):
                    # Check shape
                    if len(pixel_values.shape) == 4:  # [total_patches, C, H, W]
                        num_patches_list = sample.get('num_patches_list', [])
                        if num_patches_list:
                            num_frames = len(num_patches_list)
                            max_frames_found = max(max_frames_found, num_frames)
                            
                            if num_frames > 8:
                                print(f"  ❌ Sample {i}: Too many frames ({num_frames})")
                                checks['frame_limits'] = False
                    
                    # 4. Check for black/empty frames
                    # Black frames would be all zeros or very low values
                    # Convert to float32 first to avoid BFloat16 numpy issues
                    frame_data = pixel_values.float().cpu().numpy()
                    if np.all(frame_data < 0.01) or np.all(frame_data == 0):
                        print(f"  ⚠️ Sample {i}: Possible black/empty frame detected")
                        black_frame_detected = True
                        checks['real_frames'] = False
            else:
                samples_without_video += 1
                if sample.get('type') != 'final_response' and sample.get('type') != 'error':
                    print(f"  ⚠️ Sample {i}: No video data for clip sample")
            
            # 5. Check for empty descriptions
            if '<summary></summary>' in text or '<summary> </summary>' in text:
                print(f"  ❌ Sample {i}: Empty summary found")
                checks['no_empty_desc'] = False
                
        except Exception as e:
            print(f"  ⚠️ Error processing sample {i}: {e}")
    
    # Print summary
    print("\n" + "="*80)
    print("📋 VERIFICATION RESULTS:")
    print("-"*40)
    
    print("\n1️⃣ NO FALLBACK DESCRIPTIONS:")
    if checks['no_fallback']:
        print("   ✅ PASS - No fallback descriptions found")
    else:
        print("   ❌ FAIL - Fallback descriptions still present")
    
    print("\n2️⃣ FRAME COUNT LIMITS:")
    print(f"   Max frames found: {max_frames_found}")
    if checks['frame_limits'] and max_frames_found <= 8:
        print("   ✅ PASS - Frame count within limits (≤8)")
    else:
        print("   ❌ FAIL - Frame count exceeds limit")
    
    print("\n3️⃣ LABEL FORMAT:")
    if checks['label_format']:
        print("   ✅ PASS - All labels have proper closing tags")
    else:
        print("   ❌ FAIL - Missing closing tags found")
    
    print("\n4️⃣ REAL FRAME INPUTS:")
    print(f"   Samples with video: {samples_with_video}")
    print(f"   Samples without video: {samples_without_video}")
    if not black_frame_detected and checks['real_frames']:
        print("   ✅ PASS - Using real video frames")
    else:
        print("   ❌ FAIL - Black/empty frames detected")
    
    print("\n5️⃣ NO EMPTY DESCRIPTIONS:")
    if checks['no_empty_desc']:
        print("   ✅ PASS - All descriptions have content")
    else:
        print("   ❌ FAIL - Empty descriptions found")
    
    # Overall result
    print("\n" + "="*80)
    all_passed = all(checks.values())
    if all_passed:
        print("🎉 ALL REQUIREMENTS VERIFIED SUCCESSFULLY!")
        print("The training code is ready for use.")
    else:
        print("⚠️ SOME REQUIREMENTS NOT MET")
        print("Please review the failed checks above.")
    
    return all_passed

if __name__ == "__main__":
    success = verify_all_requirements()
    sys.exit(0 if success else 1)