#!/usr/bin/env python3
"""
Fixed Streaming Inference for Video-Guard Model
Aligned with training configuration
"""

import torch
import os
import logging
import json
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
import cv2
from typing import List, Dict, Any, Optional
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from dataclasses import dataclass
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Image preprocessing constants (must match training)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

@dataclass
class Shot:
    """Represents a video shot/clip"""
    frame_indices: List[int]
    frame_labels: List[List[str]]  # Labels for each frame
    summary: Optional[str] = None

@dataclass
class StreamingState:
    """Maintains the streaming analysis state"""
    accumulated_prompt: str
    current_shot_frames: List[int]
    current_shot_labels: List[List[str]]
    shots: List[Shot]
    clip_prompts: List[str]  # For final response
    frame_patches: List[int]  # Number of patches for each frame

def build_transform(input_size):
    """Build image transformation pipeline (matching training)"""
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        T.Lambda(lambda x: x.to(torch.bfloat16))
    ])
    return transform

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
    """Dynamic preprocessing for images (matching training EXACTLY)"""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = orig_width * orig_height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    # Calculate the target width and height
    target_width = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]

    # Resize the image
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
    
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    
    return processed_images

class StreamingVideoAnalyzer:
    """Fixed streaming video analyzer aligned with training"""
    
    def __init__(self, base_model_path: str, checkpoint_path: Optional[str] = None, device_id: int = 0):
        """Initialize the streaming analyzer"""
        logger.info("Loading model and tokenizer...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            use_fast=False
        )
        
        # Set CUDA device
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
        
        # Load model on specified device
        self.model = AutoModel.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map={"": 0}  # Device 0 after CUDA_VISIBLE_DEVICES
        )
        
        # Load LoRA checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading LoRA checkpoint from: {checkpoint_path}")
            self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
            self.model = self.model.merge_and_unload()
        
        self.model.eval()
        
        # Set img_context_token_id
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        self.model.img_context_token_id = self.img_context_token_id
        
        # Image processing - MUST match training configuration
        self.input_size = 448
        self.max_num_patches = 6  # Changed from 12 to match training
        self.transform = build_transform(self.input_size)
        
        # Streaming prompt template (exact match with training)
        self.streaming_prompt = """<streaming_analysis> You are a video analyst. Analyze video frames to understand and describe content.
Watch each frame and decide:
- <unsafe:C1>: Sexual content
- <unsafe:C2>: Harassment/bullying  
- <unsafe:C3>: Violence/harm
- <unsafe:C4>: Misinformation
- <unsafe:C5>: Illegal activities
- <unsafe:C6>: Hate speech/extremism
- <safe>: The frame is safe
- <continue>: Shot is not complete, keep analyzing
- <summary>: Shot is complete, provide a complete shot description"""
        
        logger.info("Model loaded successfully!")
    
    def process_frame(self, frame: Image.Image) -> tuple:
        """Process a single frame and return pixel values and patch count"""
        # Apply dynamic preprocessing (matching training exactly)
        images = dynamic_preprocess(
            frame, 
            image_size=self.input_size, 
            use_thumbnail=True,  # Match training configuration
            max_num=self.max_num_patches  # Use 6 instead of 12
        )
        
        # Transform all patches
        pixel_values = [self.transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)  # [num_patches, C, H, W]
        
        return pixel_values, len(images)
    
    def generate_response(self, prompt_with_tokens: str, pixel_values: Optional[torch.Tensor] = None) -> str:
        """Generate model response using forward pass"""
        # Tokenize
        inputs = self.tokenizer(
            prompt_with_tokens,
            return_tensors='pt',
            truncation=False
        )
        
        # Get device
        device = next(self.model.parameters()).device
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        if pixel_values is not None:
            pixel_values = pixel_values.to(device).to(torch.bfloat16)
        
        # Create labels (ignore for inference)
        labels = torch.full_like(input_ids, -100)
        
        # Autoregressive generation with adjusted parameters
        generated_ids = input_ids.clone()
        generated_text = ""
        max_new_tokens = 200  # Increased to allow multiple labels
        
        for step in range(max_new_tokens):
            with torch.no_grad():
                if step == 0 and pixel_values is not None:
                    # First pass with image
                    outputs = self.model(
                        input_ids=generated_ids,
                        attention_mask=torch.ones_like(generated_ids),
                        pixel_values=pixel_values,
                        labels=torch.full_like(generated_ids, -100),
                        return_dict=True
                    )
                else:
                    # Subsequent passes without image
                    outputs = self.model.language_model(
                        input_ids=generated_ids,
                        attention_mask=torch.ones_like(generated_ids),
                        return_dict=True
                    )
            
            # Get next token
            next_token_logits = outputs.logits[0, -1, :]
            
            # Use higher temperature for more diverse generation (matching training)
            temperature = 0.7  # Increased from 0.3
            next_token_logits = next_token_logits / temperature
            
            # Apply top-p sampling with higher threshold
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Top-p filtering with higher threshold
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > 0.95  # Increased from 0.9
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[indices_to_remove] = 0
            probs = probs / probs.sum()
            
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            # Add to sequence
            generated_ids = torch.cat([generated_ids, torch.tensor([[next_token_id]]).to(device)], dim=1)
            token = self.tokenizer.decode([next_token_id])
            generated_text += token
            
            # Stop conditions - more flexible
            if '<continue>' in generated_text:
                break
            if '<summary>' in generated_text:
                # For summary, continue to get more content
                summary_part = generated_text.split('<summary>')[-1]
                if len(summary_part) < 30:  # Reduced from 50 for more flexibility
                    continue
                break
            if next_token_id == self.tokenizer.eos_token_id:
                break
        
        return generated_text
    
    def parse_response(self, response: str) -> tuple:
        """Parse model response to extract all labels and action (more flexible)"""
        labels = []
        action = None
        summary = None
        
        # Clean the response
        cleaned = response.strip()
        
        # Extract ALL safety labels (can be multiple)
        if '<safe>' in cleaned:
            labels.append('<safe>')
        
        # Check for all unsafe categories
        for i in range(1, 7):
            if f'<unsafe:C{i}>' in cleaned:
                labels.append(f'<unsafe:C{i}>')
        
        # If no explicit labels found, assume safe
        if not labels:
            labels.append('<safe>')
        
        # Extract action (more flexible detection)
        if '<continue>' in cleaned:
            action = 'continue'
        elif '<summary>' in cleaned:
            action = 'summary'
            # Extract summary text after <summary> tag
            parts = cleaned.split('<summary>')
            if len(parts) > 1:
                summary_text = parts[1]
                # Clean up summary
                for marker in ['<continue>', '<safe>', '<unsafe:', '\n<', '<error', '</']:
                    if marker in summary_text:
                        summary_text = summary_text[:summary_text.find(marker)]
                summary = summary_text.strip()
                
                # If summary is too short, use a default
                if not summary or len(summary) < 10:
                    summary = "Shot analyzed and processed."
        
        # Default action if not found
        if action is None:
            action = 'continue'  # Default to continue if unclear
        
        return labels, action, summary
    
    def analyze_video_streaming(self, video_path: str, fps_sample: int = 1, max_frames: int = None) -> Dict[str, Any]:
        """
        Analyze video in streaming mode, frame by frame
        
        Args:
            video_path: Path to video file
            fps_sample: Sample rate (1 = every frame, 30 = 1 per second for 30fps video)
            max_frames: Maximum frames to process (None = all)
        """
        logger.info(f"Analyzing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video info: {fps:.2f} fps, {total_frames} total frames")
        
        # Initialize streaming state
        state = StreamingState(
            accumulated_prompt=self.streaming_prompt,
            current_shot_frames=[],
            current_shot_labels=[],
            shots=[],
            clip_prompts=[],
            frame_patches=[]
        )
        
        # Store all pixel values for the current clip
        current_clip_pixels = []
        
        # Calculate frames to sample
        frame_indices = list(range(0, total_frames, fps_sample))
        if max_frames:
            frame_indices = frame_indices[:max_frames]
        
        logger.info(f"Processing {len(frame_indices)} frames...")
        
        # Process each frame
        for idx, frame_idx in enumerate(tqdm(frame_indices, desc="Processing frames")):
            # Read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame {frame_idx}")
                continue
            
            # Convert to PIL Image
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame)
            
            # Process frame
            pixel_values, num_patches = self.process_frame(pil_frame)
            current_clip_pixels.append(pixel_values)
            state.frame_patches.append(num_patches)
            
            # Build current prompt with all frames in current clip
            prompt_with_images = self.streaming_prompt
            
            # Add image tokens for all frames in current clip
            for i in range(len(current_clip_pixels)):
                prompt_with_images += '\n<image>'
            
            # Replace <image> tags with proper tokens
            for i in range(len(current_clip_pixels)):
                if i < len(state.frame_patches):
                    patches = state.frame_patches[i]
                    tokens_per_patch = 256
                    total_tokens = patches * tokens_per_patch
                    IMG_START_TOKEN = '<img>'
                    IMG_END_TOKEN = '</img>'
                    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
                    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * total_tokens + IMG_END_TOKEN
                    prompt_with_images = prompt_with_images.replace('<image>', image_tokens, 1)
            
            # Add accumulated responses for this clip
            accumulated_responses = state.accumulated_prompt[len(self.streaming_prompt):]
            prompt_with_images += accumulated_responses
            
            # Generate response for this frame
            logger.debug(f"Generating response for frame {idx+1}/{len(frame_indices)}")
            
            # Stack all pixel values for current clip
            all_clip_pixels = torch.cat(current_clip_pixels, dim=0)
            
            # Generate response for this frame
            response = self.generate_response(prompt_with_images, all_clip_pixels)
            
            # Parse response to get labels and action
            labels, action, summary = self.parse_response(response)
            
            # Build formatted response matching training format
            formatted_response = ""
            for label in labels:
                formatted_response += label
            if action == 'continue':
                formatted_response += '<continue>'
            elif action == 'summary':
                formatted_response += '<summary>'
                if summary:
                    formatted_response += summary
            
            # Update accumulated prompt with formatted response
            state.accumulated_prompt += formatted_response + '\n'
            
            # Store frame info
            state.current_shot_frames.append(frame_idx)
            state.current_shot_labels.append(labels)
            
            # Log frame analysis
            logger.info(f"Frame {idx+1}: Labels={labels}, Action={action}")
            logger.debug(f"Raw response: {response[:200]}")
            
            # Check if shot is complete
            if action == 'summary':
                logger.info(f"Shot complete with summary: {summary}")
                
                # Save completed shot
                shot = Shot(
                    frame_indices=state.current_shot_frames.copy(),
                    frame_labels=state.current_shot_labels.copy(),
                    summary=summary
                )
                state.shots.append(shot)
                
                # Save clip prompt for final response
                clip_content = state.accumulated_prompt[len(self.streaming_prompt):].lstrip('\n')
                state.clip_prompts.append(clip_content)
                
                # Reset for next clip
                state.accumulated_prompt = self.streaming_prompt
                state.current_shot_frames = []
                state.current_shot_labels = []
                state.frame_patches = []
                current_clip_pixels = []  # Clear pixel values for next clip
        
        cap.release()
        
        # If there's an incomplete shot, save it
        if state.current_shot_frames:
            shot = Shot(
                frame_indices=state.current_shot_frames,
                frame_labels=state.current_shot_labels,
                summary="Incomplete shot (video ended)"
            )
            state.shots.append(shot)
            clip_content = state.accumulated_prompt[len(self.streaming_prompt):].lstrip('\n')
            state.clip_prompts.append(clip_content)
        
        # Generate final response
        logger.info("Generating final video summary...")
        
        # Build final prompt exactly as in training
        final_prompt = self.streaming_prompt
        for clip_prompt in state.clip_prompts:
            final_prompt += clip_prompt
        final_prompt += '<|vision_end|>'
        
        # Generate final response (text-only, no images)
        final_response = self.generate_response(final_prompt, pixel_values=None)
        
        # Extract final summary
        if '<response>' in final_response:
            start = final_response.find('<response>') + len('<response>')
            end = final_response.find('</response>', start)
            if end == -1:
                final_summary = final_response[start:].strip()
            else:
                final_summary = final_response[start:end].strip()
        else:
            final_summary = final_response.strip()
        
        # Compile results
        results = {
            'video_path': video_path,
            'total_frames_processed': len(frame_indices),
            'shots': [
                {
                    'frame_indices': shot.frame_indices,
                    'frame_labels': shot.frame_labels,
                    'summary': shot.summary
                }
                for shot in state.shots
            ],
            'final_summary': final_summary
        }
        
        return results


def main():
    """Main function to test streaming inference"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fixed Streaming Video Analysis")
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--checkpoint', type=str, 
                       default='/scratch/czr/Video-Guard/training_testing/output_full_lora_streaming/checkpoint-990',
                       help='Path to LoRA checkpoint')
    parser.add_argument('--model', type=str, default='OpenGVLab/InternVL3-8B',
                       help='Base model path')
    parser.add_argument('--fps-sample', type=int, default=30,
                       help='Frame sampling rate (30 = 1fps for 30fps video)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to process')
    parser.add_argument('--device', type=int, default=5,
                       help='GPU device ID')
    parser.add_argument('--output', type=str, default='fixed_streaming_results.json',
                       help='Output file path')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = StreamingVideoAnalyzer(
        base_model_path=args.model,
        checkpoint_path=args.checkpoint,
        device_id=args.device
    )
    
    # Analyze video
    results = analyzer.analyze_video_streaming(
        video_path=args.video,
        fps_sample=args.fps_sample,
        max_frames=args.max_frames
    )
    
    # Print results
    print("\n" + "="*80)
    print(f"Video Analysis Results")
    print("="*80)
    print(f"Video: {os.path.basename(args.video)}")
    print(f"Frames processed: {results['total_frames_processed']}")
    print(f"Number of shots: {len(results['shots'])}")
    
    for i, shot in enumerate(results['shots']):
        print(f"\n--- Shot {i+1} ---")
        print(f"Frames: {shot['frame_indices'][:5]}..." if len(shot['frame_indices']) > 5 else f"Frames: {shot['frame_indices']}")
        unique_labels = set()
        for labels in shot['frame_labels']:
            unique_labels.update(labels)
        print(f"Labels: {unique_labels}")
        print(f"Summary: {shot['summary']}")
    
    print(f"\n--- Final Video Summary ---")
    print(results['final_summary'])
    print("="*80)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()