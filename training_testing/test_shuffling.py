#!/usr/bin/env python3
"""
Test sample shuffling in Dataloader
"""

import sys
sys.path.append('/scratch/czr/Video-Guard/training_testing')

from Dataloader import StreamingDataset

def test_shuffling():
    """Test that shuffling works correctly"""
    
    print("="*80)
    print("TESTING SAMPLE SHUFFLING")
    print("="*80)
    
    # Create two datasets with same seed - should get same order
    print("\n1️⃣ Testing reproducibility with same seed...")
    
    dataset1 = StreamingDataset(
        dataset_file='/scratch/czr/Video-Guard/datasets',
        tokenizer=None,
        max_samples=[10, 10],  # Small sample
        shuffle=True,
        random_seed=42
    )
    
    dataset2 = StreamingDataset(
        dataset_file='/scratch/czr/Video-Guard/datasets',
        tokenizer=None,
        max_samples=[10, 10],  # Same sample size
        shuffle=True,
        random_seed=42  # Same seed
    )
    
    # Check first 5 samples are in same order
    same_order = True
    for i in range(min(5, len(dataset1.samples), len(dataset2.samples))):
        if dataset1.samples[i]['type'] != dataset2.samples[i]['type']:
            same_order = False
            break
    
    if same_order:
        print("   ✅ Same seed produces same order - PASS")
    else:
        print("   ❌ Same seed produces different order - FAIL")
    
    # Create dataset with different seed - should get different order
    print("\n2️⃣ Testing different seeds produce different orders...")
    
    dataset3 = StreamingDataset(
        dataset_file='/scratch/czr/Video-Guard/datasets',
        tokenizer=None,
        max_samples=[10, 10],
        shuffle=True,
        random_seed=123  # Different seed
    )
    
    different_order = False
    for i in range(min(5, len(dataset1.samples), len(dataset3.samples))):
        if dataset1.samples[i]['type'] != dataset3.samples[i]['type']:
            different_order = True
            break
    
    if different_order:
        print("   ✅ Different seeds produce different orders - PASS")
    else:
        print("   ❌ Different seeds produce same order - FAIL")
    
    # Test without shuffling
    print("\n3️⃣ Testing shuffle=False keeps original order...")
    
    dataset_no_shuffle = StreamingDataset(
        dataset_file='/scratch/czr/Video-Guard/datasets',
        tokenizer=None,
        max_samples=[10, 10],
        shuffle=False  # No shuffling
    )
    
    # Count how many clips appear before final_responses
    clips_before_responses = 0
    found_response = False
    for sample in dataset_no_shuffle.samples[:20]:
        if sample['type'] == 'final_response':
            found_response = True
        elif sample['type'] == 'clip' and not found_response:
            clips_before_responses += 1
    
    print(f"   Without shuffle: {clips_before_responses} clips before first response")
    
    # With shuffle, clips and responses should be mixed
    clips_before_responses_shuffled = 0
    found_response = False
    for sample in dataset1.samples[:20]:
        if sample['type'] == 'final_response':
            found_response = True
        elif sample['type'] == 'clip' and not found_response:
            clips_before_responses_shuffled += 1
    
    print(f"   With shuffle: {clips_before_responses_shuffled} clips before first response")
    
    # Analyze sample distribution
    print("\n4️⃣ Sample type distribution analysis...")
    
    def analyze_distribution(samples, name):
        """Analyze the distribution of sample types"""
        clip_count = sum(1 for s in samples if s['type'] == 'clip')
        response_count = sum(1 for s in samples if s['type'] == 'final_response')
        
        print(f"\n   {name}:")
        print(f"   - Clips: {clip_count}")
        print(f"   - Final responses: {response_count}")
        
        # Check mixing in first 10 samples
        first_10_types = [s['type'] for s in samples[:10]]
        clip_in_first_10 = first_10_types.count('clip')
        response_in_first_10 = first_10_types.count('final_response')
        
        print(f"   - First 10 samples: {clip_in_first_10} clips, {response_in_first_10} responses")
        
        return clip_count, response_count
    
    analyze_distribution(dataset_no_shuffle.samples, "Without shuffle")
    analyze_distribution(dataset1.samples, "With shuffle (seed=42)")
    analyze_distribution(dataset3.samples, "With shuffle (seed=123)")
    
    print("\n" + "="*80)
    print("✅ SHUFFLING TEST COMPLETE")
    print("Shuffling allows better mixing of different sample types during training,")
    print("preventing the model from overfitting to sequential patterns.")

if __name__ == "__main__":
    test_shuffling()