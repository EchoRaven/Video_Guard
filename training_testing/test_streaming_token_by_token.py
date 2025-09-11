#!/usr/bin/env python3
"""
Token-by-token streaming inference - generate one token at a time
"""

import os
import sys
import torch
import json
from pathlib import Path
from typing import List, Tuple
import logging
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Image preprocessing constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        T.Lambda(lambda x: x.to(torch.bfloat16))
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) 
        for i in range(1, n + 1) for j in range(1, n + 1) 
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    
    return processed_images

def load_video_frames(video_path: str, max_frames: int = 4) -> List[Image.Image]:
    frames = []
    
    if DECORD_AVAILABLE:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        if total_frames == 0:
            return []
        
        indices = [int(i * total_frames / max_frames) for i in range(min(max_frames, total_frames))]
        for idx in indices:
            frame = vr[idx].asnumpy()
            frames.append(Image.fromarray(frame))
    else:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = [int(i * total_frames / max_frames) for i in range(min(max_frames, total_frames))]
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
        cap.release()
    
    return frames

class TokenByTokenTester:
    def __init__(
        self,
        base_model_path: str = "OpenGVLab/InternVL3-8B",
        checkpoint_path: str = None,
        device: str = "cuda:0"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.input_size = 448
        self.max_num_patches = 6
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            use_fast=False
        )
        
        # Load model with low memory usage
        self.model = AutoModel.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map={'': self.device}
        )
        
        # Load LoRA if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading LoRA from {checkpoint_path}")
            self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
            self.model = self.model.merge_and_unload()
        
        self.model.eval()
        
        # Set img_context_token_id
        if not hasattr(self.model, 'img_context_token_id') or self.model.img_context_token_id is None:
            self.model.img_context_token_id = 151667
        
        # Build transform
        self.transform = build_transform(self.input_size)
        
        # User prompt
        self.user_prompt = """<streaming_analysis> You are a video analyst. Analyze video frames to understand and describe content.
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
    
    def process_frame(self, frame: Image.Image) -> torch.Tensor:
        images = dynamic_preprocess(
            frame,
            min_num=1,
            max_num=self.max_num_patches,
            image_size=self.input_size,
            use_thumbnail=True
        )
        
        pixel_values = [self.transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    
    @torch.no_grad()
    def generate_token_by_token(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        max_tokens: int = 50
    ) -> str:
        """Generate tokens one by one"""
        generated_ids = input_ids.clone()
        
        for step in range(max_tokens):
            # Forward pass to get logits
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=generated_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    use_cache=True
                )
            
            # Get next token probabilities
            next_token_logits = outputs.logits[0, -1, :]
            
            # Sample next token (greedy for now)
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            
            # Decode the token to see what we're generating
            token_str = self.tokenizer.decode([next_token_id])
            print(f"Generated token: '{token_str}' (id: {next_token_id})")
            
            # Add to generated sequence
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones(1, 1).to(self.device)], dim=1)
            
            # Check for stopping conditions
            generated_text = self.tokenizer.decode(generated_ids[0][input_ids.shape[1]:])
            
            # Look for complete patterns
            if '</label>' in generated_text:
                if '<continue>' in generated_text:
                    print("Found complete: <continue></label>")
                    break
                elif not '<continue>' in generated_text and '<safe>' in generated_text:
                    # Continue generating for potential summary
                    if '</summary>' in generated_text:
                        print("Found complete: </summary>")
                        break
                    elif step > 30:  # Give more space for summary
                        print("Generated enough for summary")
                        break
            
            # Fallback: stop after enough tokens
            if step > 40:
                print("Maximum tokens reached")
                break
        
        # Decode final response
        generated_text = self.tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        return generated_text
    
    @torch.no_grad()
    def streaming_inference_token_by_token(self, frames: List[Image.Image]):
        """Token-by-token streaming inference"""
        context = self.user_prompt
        all_pixel_values = []
        results = []
        
        for frame_idx, frame in enumerate(frames):
            print(f"\n{'='*60}")
            print(f"Processing Frame {frame_idx + 1}/{len(frames)}")
            print(f"{'='*60}")
            
            # Process frame
            pixel_values = self.process_frame(frame)
            all_pixel_values.append(pixel_values)
            
            # Add frame prompt - only add image, let model generate everything else
            context += f"\n<image>"
            
            # Replace <image> placeholders with actual tokens
            prompt_with_tokens = self._replace_image_tokens(context, all_pixel_values)
            
            # Tokenize
            inputs = self.tokenizer(prompt_with_tokens, return_tensors='pt')
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # Prepare pixel values
            pixel_values_cat = torch.cat(all_pixel_values, dim=0)
            pixel_values_cat = pixel_values_cat.to(self.device).to(torch.bfloat16)
            
            print(f"Input tokens: {input_ids.shape[1]}")
            print(f"Generating token by token...")
            
            # Generate token by token
            response = self.generate_token_by_token(
                input_ids, 
                attention_mask, 
                pixel_values_cat,
                max_tokens=50
            )
            
            print(f"\nFull response: '{response}'")
            
            # Parse response
            has_continue = '<continue>' in response
            has_summary = '</summary>' in response
            
            # Extract labels
            labels = []
            for label in ['unsafe:C1', 'unsafe:C2', 'unsafe:C3', 'unsafe:C4', 'unsafe:C5', 'unsafe:C6', 'safe', 'continue']:
                if f'<{label}>' in response:
                    labels.append(label)
            
            print(f"Parsed labels: {labels}")
            
            # Store result
            results.append({
                'frame': frame_idx + 1,
                'labels': labels,
                'response': response,
                'has_continue': has_continue
            })
            
            # Update context with the response
            context += response
            
            # If no continue or has summary, stop
            if not has_continue or has_summary:
                print(f"\nStopping at frame {frame_idx + 1}")
                break
        
        return results
    
    def _replace_image_tokens(self, prompt: str, pixel_values_list: List[torch.Tensor]) -> str:
        """Replace <image> with actual image tokens"""
        IMG_START_TOKEN = '<img>'
        IMG_END_TOKEN = '</img>'
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        num_image_token = 256
        
        result = prompt
        for pv in pixel_values_list:
            if '<image>' in result:
                num_patches = pv.shape[0]
                total_tokens = num_image_token * num_patches
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * total_tokens + IMG_END_TOKEN
                result = result.replace('<image>', image_tokens, 1)
        
        return result


def main():
    # Configuration
    checkpoint = "/scratch/czr/Video-Guard/training_testing/output_8gpu_full/checkpoint-3500"
    video_path = "/scratch/czr/Video-Guard/datasets/shot2story-videos/release_134k_videos/--5gVbAaF_A.9.mp4"
    
    print("="*80)
    print("TOKEN-BY-TOKEN STREAMING TEST")
    print("="*80)
    
    # Initialize
    tester = TokenByTokenTester(
        base_model_path="OpenGVLab/InternVL3-8B",
        checkpoint_path=checkpoint,
        device="cuda:0"
    )
    
    # Load video
    frames = load_video_frames(video_path, max_frames=3)  # Test with fewer frames
    print(f"Loaded {len(frames)} frames")
    
    # Run inference
    results = tester.streaming_inference_token_by_token(frames)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for r in results:
        print(f"Frame {r['frame']}: {r['labels']}")
        print(f"  Response: {r['response']}")


if __name__ == "__main__":
    main()