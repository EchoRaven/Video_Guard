#!/usr/bin/env python3
"""Test with different generation parameters"""

import torch
from streaming_inference import StreamingVideoAnalyzer

class TestAnalyzer(StreamingVideoAnalyzer):
    """Modified analyzer for testing different parameters"""
    
    def generate_response(self, prompt_with_tokens, pixel_values=None):
        """Modified generation with different parameters"""
        # Tokenize
        inputs = self.tokenizer(
            prompt_with_tokens,
            return_tensors='pt',
            truncation=False
        )
        
        device = next(self.model.parameters()).device
        input_ids = inputs['input_ids'].to(device)
        
        if pixel_values is not None:
            pixel_values = pixel_values.to(device).to(torch.bfloat16)
        
        # Test different generation strategies
        with torch.no_grad():
            if pixel_values is not None:
                # Use model's generate method instead of manual loop
                outputs = self.model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    max_new_tokens=100,
                    temperature=0.1,  # Lower temperature for more focused output
                    do_sample=False,  # Greedy decoding
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            else:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=100,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        
        # Decode only the generated part
        generated_ids = outputs[0][input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        
        return generated_text


# Test
print("Testing with greedy decoding (temperature=0.1, do_sample=False)...")
analyzer = TestAnalyzer(
    'OpenGVLab/InternVL3-8B',
    '/scratch/czr/Video-Guard/training_testing/output_full_lora_streaming/checkpoint-990',
    device_id=5
)

results = analyzer.analyze_video_streaming(
    '/scratch/czr/Video-Guard/datasets/shot2story-videos/release_134k_videos/--5gVbAaF_A.9.mp4',
    fps_sample=120,
    max_frames=3
)

print("\n" + "="*80)
print("RESULTS WITH GREEDY DECODING:")
print("="*80)
for i, shot in enumerate(results['shots'], 1):
    print(f"\nShot {i}:")
    print(f"  Labels: {shot['frame_labels']}")
    print(f"  Summary: {shot['summary'][:100]}...")
print(f"\nFinal: {results['final_summary'][:200]}...")