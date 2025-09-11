#!/usr/bin/env python3
"""
Full video streaming inference - handles multiple clips and final response
Mimics the complete training process
"""

import os
import sys
import torch
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec

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

def load_video_frames(video_path: str, max_frames: int = 20, fps_sample: int = None) -> Tuple[List[Image.Image], List[int]]:
    """Load frames from video with sampling - if fps_sample is None, sample 1 frame per second"""
    frames = []
    frame_indices = []
    
    if DECORD_AVAILABLE:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        if total_frames == 0:
            return [], []
        
        # Get video FPS if fps_sample is None (sample 1 frame per second)
        if fps_sample is None:
            fps = vr.get_avg_fps()
            fps_sample = int(fps)  # Sample every fps frames = 1 frame per second
            print(f"Video FPS: {fps:.2f}, sampling every {fps_sample} frames (1 per second)")
        
        # Sample frames at specified interval
        for i in range(0, min(total_frames, max_frames * fps_sample), fps_sample):
            frame_indices.append(i)
            frame = vr[i].asnumpy()
            frames.append(Image.fromarray(frame))
            if len(frames) >= max_frames:
                break
    else:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Get video FPS if fps_sample is None (sample 1 frame per second)
        if fps_sample is None:
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps_sample = int(fps)  # Sample every fps frames = 1 frame per second
            print(f"Video FPS: {fps:.2f}, sampling every {fps_sample} frames (1 per second)")
        
        for i in range(0, min(total_frames, max_frames * fps_sample), fps_sample):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame_indices.append(i)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
                if len(frames) >= max_frames:
                    break
        cap.release()
    
    return frames, frame_indices

class FullVideoStreamingTester:
    def __init__(
        self,
        base_model_path: str = "OpenGVLab/InternVL3-8B",
        checkpoint_path: str = None,
        device: str = "cuda:0"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.input_size = 448
        self.max_num_patches = 2
        self.checkpoint_path = checkpoint_path  # Store for metadata
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            use_fast=False
        )
        
        # Load model
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
        
        # User prompt for clips - MUST match training format exactly
        self.clip_prompt = """<streaming_analysis> You are a video analyst. Analyze video frames to understand and describe content.

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
        
        # Special tokens
        self.vision_end_token = '<|vision_end|>'
    
    def process_frame(self, frame: Image.Image) -> torch.Tensor:
        """Process a single frame"""
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
    
    @torch.no_grad()
    def generate_frame_response(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        max_tokens: int = 500  # Increased limit for full responses
    ) -> str:
        """Generate response token by token for a single frame"""
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
            
            # Sample next token (greedy for consistency)
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            
            # Debug: print generated token
            token_str = self.tokenizer.decode([next_token_id])
            print(f"Token {step}: '{token_str}' (id: {next_token_id})", end='')
            
            # Add to generated sequence
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones(1, 1).to(self.device)], dim=1)
            
            # Check for stopping conditions
            generated_text = self.tokenizer.decode(generated_ids[0][input_ids.shape[1]:])
            
            # IMPORTANT: Check for </response> FIRST (for final response generation)
            if '</response>' in generated_text:
                print("\n→ Found: </response> - stopping generation")
                break
            
            # Look for complete patterns for frame responses
            if '</label>' in generated_text:
                if '<continue>' in generated_text:
                    print("\n→ Found: <continue></label>")
                    break
                elif not '<continue>' in generated_text:
                    # This is the last frame, must generate summary
                    # Keep generating until we find </summary>
                    if '</summary>' in generated_text:
                        print("\n→ Found: </summary>")
                        break
                    # Don't stop early - let it generate the full summary
            
            # Check for EOS token
            if next_token_id == self.tokenizer.eos_token_id:
                print(f"\n→ Found: EOS token (id: {next_token_id})")
                break
            
            # Only stop if we've generated way too many tokens
            if step >= max_tokens - 1:
                print(f"\n→ Maximum tokens reached ({max_tokens})")
                break
        
        # Decode final response
        generated_text = self.tokenizer.decode(generated_ids[0][input_ids.shape[1]:])
        
        # Clean up: Remove any text after </response> if present
        if '</response>' in generated_text:
            end_pos = generated_text.find('</response>') + len('</response>')
            generated_text = generated_text[:end_pos]
        
        # Remove special tokens for clean output
        generated_text = self.tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        return generated_text
    
    def save_frame_samples(self, frames: List[Image.Image], results: Dict[str, Any], output_dir: str, video_name: str) -> str:
        """Save sample frames with safety labels as a grid visualization"""
        if not frames:
            return None
        
        # Create a grid of sample frames
        num_samples = min(len(frames), 12)  # Max 12 frames in grid
        sample_indices = np.linspace(0, len(frames)-1, num_samples, dtype=int)
        
        # Create figure
        cols = 4
        rows = (num_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Video Frame Samples - {video_name}', fontsize=14, fontweight='bold')
        
        # Get frame safety labels from results
        frame_safety = {}
        for clip in results.get('clips', []):
            for frame_result in clip['frame_results']:
                idx = frame_result['frame_idx']
                labels = frame_result['labels']
                if any(l == 'unsafe:C1' for l in labels):  # Only C1 is considered unsafe
                    frame_safety[idx] = 'unsafe'
                elif any(l == 'safe' or (l.startswith('unsafe:') and l != 'unsafe:C1') for l in labels):  # All others are safe
                    frame_safety[idx] = 'safe'
                else:
                    frame_safety[idx] = 'unknown'
        
        # Plot each sample frame
        for i, idx in enumerate(sample_indices):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            # Display frame
            frame = frames[idx]
            ax.imshow(frame)
            ax.axis('off')
            
            # Add frame info and safety label
            safety = frame_safety.get(idx, 'unknown')
            color = '#2ecc71' if safety == 'safe' else '#e74c3c' if safety == 'unsafe' else '#95a5a6'
            ax.set_title(f'Frame {idx+1}\\n[{safety.upper()}]', fontsize=10, color=color, fontweight='bold')
            
            # Add colored border
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
        
        # Hide empty subplots
        for i in range(num_samples, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frames_path = os.path.join(output_dir, f"{video_name}_frames_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(frames_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return frames_path
    
    @torch.no_grad()
    def analyze_full_video(
        self,
        frames: List[Image.Image],
        frame_indices: List[int],
        video_path: str = None
    ) -> Dict[str, Any]:
        """
        Analyze full video with multiple clips and final response
        Following the exact training process
        """
        all_clips = []  # Store all clip results
        all_clip_prompts = []  # Store clip prompts for final response
        frame_idx = 0
        clip_idx = 0
        
        print("\n" + "="*80)
        print("STARTING FULL VIDEO ANALYSIS")
        print("="*80)
        
        # Process clips until all frames are consumed
        while frame_idx < len(frames):
            clip_idx += 1
            print(f"\n{'='*60}")
            print(f"CLIP {clip_idx} PROCESSING")
            print(f"{'='*60}")
            
            # Start new clip with fresh context
            clip_context = self.clip_prompt
            clip_pixel_values = []
            clip_frames = []
            clip_results = []
            
            # Process frames for this clip
            while frame_idx < len(frames):
                current_frame = frames[frame_idx]
                print(f"\nProcessing frame {frame_idx + 1}/{len(frames)} (Clip {clip_idx})")
                
                # Process frame
                pixel_values = self.process_frame(current_frame)
                clip_pixel_values.append(pixel_values)
                
                # Add frame to context
                clip_context += f"\n<image>"
                
                # Prepare input with token limit check
                prompt_with_tokens = self._replace_image_tokens(clip_context, clip_pixel_values)
                
                # Check token length and truncate if needed
                test_inputs = self.tokenizer(prompt_with_tokens, return_tensors='pt', truncation=False)
                if test_inputs['input_ids'].shape[1] > 11000:  # Leave room for generation
                    print(f"WARNING: Context too long ({test_inputs['input_ids'].shape[1]} tokens), truncating...")
                    # Keep only the last N frames worth of context
                    max_frames_in_context = 5  # Adjust as needed
                    if len(clip_pixel_values) > max_frames_in_context:
                        # Keep initial prompt and last N frames
                        clip_pixel_values = clip_pixel_values[-max_frames_in_context:]
                        # Rebuild context with fewer frames
                        lines = clip_context.split('\n')
                        # Find where images start
                        image_start_idx = next((i for i, line in enumerate(lines) if '<image>' in line), 0)
                        # Keep prompt and last N image/response pairs
                        kept_lines = lines[:image_start_idx]
                        image_response_pairs = lines[image_start_idx:]
                        # Estimate pairs (image + response lines)
                        if len(image_response_pairs) > max_frames_in_context * 10:  # Rough estimate
                            image_response_pairs = image_response_pairs[-(max_frames_in_context * 10):]
                        clip_context = '\n'.join(kept_lines + image_response_pairs)
                        prompt_with_tokens = self._replace_image_tokens(clip_context, clip_pixel_values)
                
                inputs = self.tokenizer(prompt_with_tokens, return_tensors='pt', truncation=True, max_length=11000)
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)
                
                # Prepare pixel values
                all_pixels = torch.cat(clip_pixel_values, dim=0)
                all_pixels = all_pixels.to(self.device).to(torch.bfloat16)
                
                # Generate response with unlimited token length for complete summaries
                response = self.generate_frame_response(input_ids, attention_mask, all_pixels, max_tokens=1000)
                print(f"Response: {response[:150]}...")
                
                # Parse response
                has_continue = '<continue>' in response
                has_summary = '</summary>' in response
                
                # Extract labels
                labels = []
                for label in ['unsafe:C1', 'unsafe:C2', 'unsafe:C3', 'unsafe:C4', 'unsafe:C5', 'unsafe:C6', 'safe']:
                    if f'<{label}>' in response:
                        labels.append(label)
                
                # Extract summary if present
                summary = None
                if '<summary>' in response and '</summary>' in response:
                    start = response.find('<summary>') + len('<summary>')
                    end = response.find('</summary>')
                    summary = response[start:end].strip()
                
                clip_results.append({
                    'frame_idx': frame_idx,
                    'frame_number': frame_idx + 1,
                    'video_frame_index': frame_indices[frame_idx] if frame_idx < len(frame_indices) else frame_idx,
                    'labels': labels,
                    'summary': summary,
                    'raw_response': response,
                    'has_continue': has_continue,
                    'has_summary': has_summary,
                    'response_length': len(response),
                    'num_patches': pixel_values.shape[0] if pixel_values is not None else 0
                })
                
                # Update context with response
                clip_context += response
                clip_frames.append(frame_idx)
                frame_idx += 1
                
                # Check if clip is complete
                # A clip is only complete when we get a summary
                if has_summary:
                    print(f"\nClip {clip_idx} complete with {len(clip_frames)} frames")
                    if summary:
                        print(f"Summary: {summary[:100]}...")
                    break
                
                # If no continue tag, we should keep generating until we get a summary
                # The model might need more context to generate the summary
                if not has_continue:
                    print(f"\nNo continue tag, expecting summary next...")
                    # Don't break here - let the model continue to generate summary
                
                # Special handling for last frame of video
                if frame_idx == len(frames) and not has_summary:
                    print(f"\nLast frame reached without summary, forcing summary generation...")
                    # Add <summary> tag to force model to generate summary
                    clip_context += "<summary>"
                    
                    # Generate summary with updated context
                    prompt_with_tokens = self._replace_image_tokens(clip_context, clip_pixel_values)
                    
                    # Apply same token limit check
                    test_inputs = self.tokenizer(prompt_with_tokens, return_tensors='pt', truncation=False)
                    if test_inputs['input_ids'].shape[1] > 11000:
                        print(f"WARNING: Summary context too long ({test_inputs['input_ids'].shape[1]} tokens), truncating...")
                        max_frames_in_context = 5
                        if len(clip_pixel_values) > max_frames_in_context:
                            clip_pixel_values = clip_pixel_values[-max_frames_in_context:]
                            lines = clip_context.split('\n')
                            image_start_idx = next((i for i, line in enumerate(lines) if '<image>' in line), 0)
                            kept_lines = lines[:image_start_idx]
                            image_response_pairs = lines[image_start_idx:]
                            if len(image_response_pairs) > max_frames_in_context * 10:
                                image_response_pairs = image_response_pairs[-(max_frames_in_context * 10):]
                            clip_context = '\n'.join(kept_lines + image_response_pairs)
                            prompt_with_tokens = self._replace_image_tokens(clip_context, clip_pixel_values)
                    
                    inputs = self.tokenizer(prompt_with_tokens, return_tensors='pt', truncation=True, max_length=11000)
                    input_ids = inputs['input_ids'].to(self.device)
                    attention_mask = inputs['attention_mask'].to(self.device)
                    
                    # Generate forced summary with unlimited length
                    forced_response = self.generate_frame_response(input_ids, attention_mask, all_pixels, max_tokens=1000)
                    print(f"Forced summary: {forced_response[:150]}...")
                    
                    # Extract the summary
                    if '</summary>' in forced_response:
                        end = forced_response.find('</summary>')
                        summary = forced_response[:end].strip()
                    else:
                        summary = forced_response.strip()
                    
                    # Update the last frame's result with summary
                    if clip_results:
                        clip_results[-1]['summary'] = summary
                        clip_results[-1]['raw_response'] = clip_results[-1].get('raw_response', '') + f"<summary>{summary}</summary>"
                    
                    break
                
                # No frame limit - let the model decide when to end the clip
                # The model will generate summary when it thinks the clip is complete
            
            # Store clip information with detailed metadata
            all_clips.append({
                'clip_idx': clip_idx,
                'clip_number': clip_idx,
                'frames': clip_frames,
                'num_frames': len(clip_frames),
                'frame_results': clip_results,
                'clip_summary': summary if summary else "No summary generated",
                'clip_ended_with_summary': bool(summary),
                'total_response_tokens': sum(r['response_length'] for r in clip_results),
                'clip_prompt_length': len(clip_context)
            })
            
            # Store clip prompt for final response
            # According to training: we keep EVERYTHING except the initial user prompt
            # This includes all <image>, <label>, labels, </label>, <summary> tags
            clip_prompt_for_final = clip_context.replace(self.clip_prompt, '')
            all_clip_prompts.append(clip_prompt_for_final)
            
            # IMPORTANT: Clear context for next clip (as per training)
            # Each clip starts fresh with only the user prompt
        
        # Post-process: Merge clips with identical summaries
        print(f"\n{'='*60}")
        print(f"POST-PROCESSING: Merging clips with identical summaries")
        print(f"{'='*60}")
        
        merged_clips = []
        merged_clip_prompts = []
        i = 0
        while i < len(all_clips):
            current_clip = all_clips[i]
            current_prompt = all_clip_prompts[i]
            
            # Check if next clip has identical summary
            while i + 1 < len(all_clips):
                next_clip = all_clips[i + 1]
                if (current_clip['clip_summary'] == next_clip['clip_summary'] and 
                    current_clip['clip_summary'] != "No summary generated"):
                    print(f"Merging clip {current_clip['clip_idx']} and {next_clip['clip_idx']} (same summary)")
                    # Merge frames and results
                    current_clip['frames'].extend(next_clip['frames'])
                    current_clip['frame_results'].extend(next_clip['frame_results'])
                    # Merge prompts
                    current_prompt += all_clip_prompts[i + 1]
                    i += 1
                else:
                    break
            
            merged_clips.append(current_clip)
            merged_clip_prompts.append(current_prompt)
            i += 1
        
        print(f"Merged {len(all_clips)} clips into {len(merged_clips)} clips")
        all_clips = merged_clips
        all_clip_prompts = merged_clip_prompts
        
        print(f"\n{'='*60}")
        print(f"GENERATING FINAL RESPONSE")
        print(f"{'='*60}")
        
        # Construct final response prompt (matching training)
        # Start with user prompt, then add ALL accumulated clip content
        final_prompt = self.clip_prompt
        for clip_prompt in all_clip_prompts:
            final_prompt += clip_prompt
        
        # CRITICAL: Add vision_end token before final response generation
        final_prompt += self.vision_end_token
        print(f"\nFinal prompt includes {len(all_clip_prompts)} clips")
        print(f"Added vision_end token: {self.vision_end_token}")
        
        # Generate final response
        print("Generating final video summary...")
        
        # For final response, we don't add new images - just use the accumulated text
        # The model should generate based on all the clip summaries
        # Increase max_length to handle multiple clips while staying under model limit
        inputs = self.tokenizer(final_prompt, return_tensors='pt', truncation=True, max_length=11000)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Log if truncation occurred
        if len(self.tokenizer.encode(final_prompt)) > 11000:
            print(f"WARNING: Final prompt truncated from {len(self.tokenizer.encode(final_prompt))} to 11000 tokens")
        
        # Generate final response token by token
        generated_ids = input_ids.clone()
        
        print("Generating tokens for final response...")
        max_final_tokens = 2000  # No limit on final response length
        for step in range(max_final_tokens):
            # Forward pass to get logits
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=generated_ids,
                    attention_mask=attention_mask,
                    # No pixel_values for final response - it's text-only based on accumulated summaries
                    use_cache=True
                )
            
            # Get next token probabilities
            next_token_logits = outputs.logits[0, -1, :]
            
            # Sample next token (greedy)
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            
            # Add to generated sequence
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones(1, 1).to(self.device)], dim=1)
            
            # Check for stopping conditions
            generated_text = self.tokenizer.decode(generated_ids[0][input_ids.shape[1]:])
            
            # Look for </response> to stop
            if '</response>' in generated_text:
                print(f"\nGenerated {step+1} tokens for final response - found </response>")
                break
            
            # Check for EOS token
            if next_token_id == self.tokenizer.eos_token_id:
                print(f"\nFound EOS token at step {step+1}")
                break
            
            # Fallback: stop after enough tokens
            if step >= max_final_tokens - 1:
                print(f"\nReached max tokens for final response ({max_final_tokens})")
                break
        
        # First decode without removing special tokens to clean up
        raw_final = self.tokenizer.decode(generated_ids[0][input_ids.shape[1]:])
        
        # Remove anything after </response> if it exists
        if '</response>' in raw_final:
            raw_final = raw_final[:raw_final.find('</response>') + len('</response>')]
        
        # Now decode with special tokens removed for clean output
        final_response = self.tokenizer.decode(self.tokenizer.encode(raw_final, add_special_tokens=False), skip_special_tokens=True)
        
        # Extract content from <response> tags if present
        if '<response>' in final_response and '</response>' in final_response:
            start = final_response.find('<response>') + len('<response>')
            end = final_response.find('</response>')
            final_response_content = final_response[start:end].strip()
        else:
            final_response_content = final_response
        
        print(f"\nFinal Response: {final_response_content[:200]}...")
        
        # Compile comprehensive results with analysis metadata
        results = {
            'video_path': video_path if video_path else 'unknown',
            'model_checkpoint': self.checkpoint_path if hasattr(self, 'checkpoint_path') else 'base_model',
            'analysis_metadata': {
                'total_frames_loaded': len(frames),
                'total_frames_processed': frame_idx,
                'fps_sample': 'auto_1_per_second',  # Automatically determined based on video FPS
                'input_size': self.input_size,
                'max_patches_per_frame': self.max_num_patches,
                'total_clips_before_merge': len(all_clip_prompts),
                'total_clips_after_merge': len(all_clips),
                'vision_end_token_used': True,
                'final_prompt_length': len(final_prompt),
                'final_response_token_count': len(final_response.split())
            },
            'clips': all_clips,
            'clip_merging': {
                'clips_merged': len(all_clip_prompts) > len(all_clips),
                'merge_reason': 'identical_summaries' if len(all_clip_prompts) > len(all_clips) else 'no_merge_needed'
            },
            'final_response': {
                'content': final_response_content,
                'raw_response': final_response,
                'response_length': len(final_response_content),
                'has_response_tags': '<response>' in final_response and '</response>' in final_response
            },
            'safety_analysis': {
                'total_unsafe_frames': sum(1 for clip in all_clips 
                    for result in clip['frame_results'] 
                    if any(label == 'unsafe:C1' for label in result['labels'])),  # Only C1 is considered unsafe
                'total_safe_frames': sum(1 for clip in all_clips 
                    for result in clip['frame_results'] 
                    if not any(label == 'unsafe:C1' for label in result['labels']) and 
                       any(label == 'safe' or (label.startswith('unsafe:') and label != 'unsafe:C1') for label in result['labels'])),  # Safe only if no C1
                'unsafe_categories_detected': list(set(
                    label for clip in all_clips 
                    for result in clip['frame_results'] 
                    for label in result['labels'] 
                    if label == 'unsafe:C1'  # Only C1 is considered unsafe
                ))
            },
            'timestamp': datetime.now().isoformat(),
            'processing_time_estimate': 'Not tracked'  # Could add timing later
        }
        
        return results
    
    def generate_visualization(self, results: Dict[str, Any], output_path: str) -> str:
        """Generate a comprehensive visualization of the analysis results"""
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Video Streaming Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. Frame Safety Distribution (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        safety = results.get('safety_analysis', {})
        safe_frames = safety.get('total_safe_frames', 0)
        unsafe_frames = safety.get('total_unsafe_frames', 0)
        
        if safe_frames + unsafe_frames > 0:
            sizes = [safe_frames, unsafe_frames]
            labels = [f'Safe ({safe_frames})', f'Unsafe ({unsafe_frames})']
            colors = ['#2ecc71', '#e74c3c']
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        else:
            ax1.text(0.5, 0.5, 'No frames analyzed', ha='center', va='center')
        ax1.set_title('Frame Safety Distribution')
        
        # 2. Unsafe Categories Breakdown (Top Middle)
        ax2 = fig.add_subplot(gs[0, 1])
        unsafe_cats = safety.get('unsafe_categories_detected', [])
        if unsafe_cats:
            # Count occurrences of each category
            cat_counts = {}
            for clip in results.get('clips', []):
                for frame_result in clip['frame_results']:
                    for label in frame_result['labels']:
                        if 'unsafe' in label:
                            cat_counts[label] = cat_counts.get(label, 0) + 1
            
            if cat_counts:
                # Only show C1 as unsafe, filter out others
                filtered_cat_counts = {k: v for k, v in cat_counts.items() if k == 'unsafe:C1'}
                if filtered_cat_counts:
                    categories = list(filtered_cat_counts.keys())
                    counts = list(filtered_cat_counts.values())
                    colors_map = {
                        'unsafe:C1': '#e74c3c',  # Red - Sexual (only unsafe category)
                    }
                    bar_colors = [colors_map.get(cat, '#95a5a6') for cat in categories]
                    ax2.bar(range(len(categories)), counts, color=bar_colors)
                    ax2.set_xticks(range(len(categories)))
                    ax2.set_xticklabels([c.replace('unsafe:', 'C') for c in categories], rotation=45)
                    ax2.set_ylabel('Count')
                    ax2.set_title('Unsafe Categories Detected (Only C1)')
                else:
                    ax2.text(0.5, 0.5, 'No unsafe:C1 detected', ha='center', va='center')
                    ax2.set_title('Unsafe Categories (Only C1)')
            else:
                ax2.text(0.5, 0.5, 'No unsafe categories', ha='center', va='center')
        else:
            ax2.text(0.5, 0.5, 'No unsafe content detected', ha='center', va='center')
            ax2.set_title('Unsafe Categories')
        
        # 3. Clip Analysis Summary (Top Right)
        ax3 = fig.add_subplot(gs[0, 2])
        meta = results.get('analysis_metadata', {})
        info_text = [
            f"Total Frames: {meta.get('total_frames_loaded', 0)}",
            f"Processed: {meta.get('total_frames_processed', 0)}",
            f"FPS Sample: {meta.get('fps_sample', 'auto_1_per_second')}",
            f"Clips: {meta.get('total_clips_after_merge', 0)}",
            f"Merged from: {meta.get('total_clips_before_merge', 0)}"
        ]
        ax3.text(0.1, 0.8, '\n'.join(info_text), fontsize=11, 
                verticalalignment='top', fontfamily='monospace')
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        ax3.set_title('Analysis Metadata')
        
        # 4. Frame Timeline (Middle - Full Width)
        ax4 = fig.add_subplot(gs[1, :])
        clips = results.get('clips', [])
        if clips:
            # Create timeline visualization
            y_pos = 0
            colors_timeline = []
            positions = []
            
            for clip in clips:
                for frame_result in clip['frame_results']:
                    frame_idx = frame_result['frame_idx']
                    is_safe = any('safe' == label or (label.startswith('unsafe:') and label != 'unsafe:C1') for label in frame_result['labels'])
                    is_unsafe = any('unsafe:C1' == label for label in frame_result['labels'])  # Only C1 is unsafe
                    
                    if is_unsafe:
                        color = '#e74c3c'
                    elif is_safe:
                        color = '#2ecc71'
                    else:
                        color = '#95a5a6'
                    
                    colors_timeline.append(color)
                    positions.append(frame_idx)
            
            if positions:
                ax4.scatter(positions, [y_pos]*len(positions), c=colors_timeline, s=100, marker='s')
                ax4.set_xlim(-1, max(positions) + 1)
                ax4.set_ylim(-0.5, 0.5)
                ax4.set_xlabel('Frame Index')
                ax4.set_yticks([])
                ax4.set_title('Frame Safety Timeline')
                ax4.grid(True, alpha=0.3)
                
                # Add legend
                safe_patch = mpatches.Patch(color='#2ecc71', label='Safe')
                unsafe_patch = mpatches.Patch(color='#e74c3c', label='Unsafe')
                ax4.legend(handles=[safe_patch, unsafe_patch], loc='upper right')
        else:
            ax4.text(0.5, 0.5, 'No timeline data', ha='center', va='center')
            ax4.set_title('Frame Timeline')
        
        # 5. Clip Summaries (Bottom Left - 2 cells wide)
        ax5 = fig.add_subplot(gs[2, :2])
        clip_text = "Clip Summaries:\n\n"
        for i, clip in enumerate(clips[:3]):  # Show first 3 clips
            summary = clip.get('clip_summary', 'No summary')
            if len(summary) > 80:
                summary = summary[:77] + "..."
            clip_text += f"Clip {i+1}: {summary}\n\n"
        
        ax5.text(0.05, 0.95, clip_text, fontsize=9, 
                verticalalignment='top', wrap=True, 
                transform=ax5.transAxes)
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.axis('off')
        ax5.set_title('Clip Summaries', loc='left')
        
        # 6. Final Response Preview (Bottom Right)
        ax6 = fig.add_subplot(gs[2, 2])
        final_resp = results.get('final_response', {})
        if isinstance(final_resp, dict):
            response_text = final_resp.get('content', 'No response')
        else:
            response_text = str(final_resp)
        
        if len(response_text) > 200:
            response_text = response_text[:197] + "..."
        
        ax6.text(0.05, 0.95, f"Final Response:\n\n{response_text}", 
                fontsize=9, verticalalignment='top', wrap=True,
                transform=ax6.transAxes)
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        ax6.set_title('Final Response', loc='left')
        
        # Add footer with metadata
        fig.text(0.5, 0.02, f"Generated: {results.get('timestamp', 'Unknown')} | Model: {results.get('model_checkpoint', 'Base')}",
                ha='center', fontsize=8, style='italic')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_analysis_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable analysis report"""
        report = []
        report.append("="*80)
        report.append("VIDEO STREAMING ANALYSIS REPORT")
        report.append("="*80)
        
        # Basic info
        report.append(f"\nVideo: {results.get('video_path', 'Unknown')}")
        report.append(f"Checkpoint: {results.get('model_checkpoint', 'Base model')}")
        report.append(f"Timestamp: {results.get('timestamp', 'Unknown')}")
        
        # Analysis metadata
        meta = results.get('analysis_metadata', {})
        report.append(f"\nAnalysis Details:")
        report.append(f"  • Frames processed: {meta.get('total_frames_processed', 0)}/{meta.get('total_frames_loaded', 0)}")
        report.append(f"  • Clips identified: {meta.get('total_clips_after_merge', 0)}")
        report.append(f"  • Clips before merge: {meta.get('total_clips_before_merge', 0)}")
        report.append(f"  • FPS sample rate: {meta.get('fps_sample', 'auto_1_per_second')}")
        
        # Safety analysis
        safety = results.get('safety_analysis', {})
        report.append(f"\nSafety Analysis:")
        report.append(f"  • Safe frames: {safety.get('total_safe_frames', 0)}")
        report.append(f"  • Unsafe frames: {safety.get('total_unsafe_frames', 0)}")
        if safety.get('unsafe_categories_detected'):
            report.append(f"  • Unsafe categories: {', '.join(safety['unsafe_categories_detected'])}")
        
        # Clip details
        report.append(f"\nClip Breakdown:")
        for clip in results.get('clips', []):
            report.append(f"\n  Clip {clip['clip_idx']}:")
            report.append(f"    • Frames: {clip['num_frames']}")
            report.append(f"    • Summary: {clip['clip_summary'][:100]}..." if len(clip['clip_summary']) > 100 else f"    • Summary: {clip['clip_summary']}")
            
            # Count labels in this clip
            label_counts = {}
            for frame_result in clip['frame_results']:
                for label in frame_result['labels']:
                    label_counts[label] = label_counts.get(label, 0) + 1
            if label_counts:
                report.append(f"    • Labels: {', '.join(f'{k}({v})' for k, v in label_counts.items())}")
        
        # Final response
        final_resp = results.get('final_response', {})
        if isinstance(final_resp, dict):
            report.append(f"\nFinal Video Summary:")
            content = final_resp.get('content', '')
            if len(content) > 500:
                report.append(f"  {content[:500]}...")
            else:
                report.append(f"  {content}")
            report.append(f"\n  (Length: {final_resp.get('response_length', 0)} chars)")
        
        report.append("\n" + "="*80)
        return "\n".join(report)
    
    def save_results(self, results: Dict[str, Any], output_dir: str, video_name: str, video_path: str = None, frames: List[Image.Image] = None):
        """Save results to JSON file with enhanced metadata"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Add video path to results if not already present
        if video_path and 'video_path' not in results:
            results['video_path'] = video_path
        
        # Create detailed filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"{video_name}_full_analysis_{timestamp}.json")
        
        # Also save a latest version for easy access
        latest_path = os.path.join(output_dir, f"{video_name}_latest.json")
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        with open(latest_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Generate and save human-readable report
        report = self.generate_analysis_report(results)
        report_path = os.path.join(output_dir, f"{video_name}_report_{timestamp}.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Generate visualization
        viz_path = os.path.join(output_dir, f"{video_name}_analysis_{timestamp}.png")
        self.generate_visualization(results, viz_path)
        
        # Also save latest visualization
        viz_latest_path = os.path.join(output_dir, f"{video_name}_latest.png")
        self.generate_visualization(results, viz_latest_path)
        
        # Generate frame samples visualization if frames provided
        frames_viz_path = None
        if frames:
            frames_viz_path = self.save_frame_samples(frames, results, output_dir, video_name)
        
        print(f"\nResults saved to:")
        print(f"  JSON Full: {output_path}")
        print(f"  JSON Latest: {latest_path}")
        print(f"  Report: {report_path}")
        print(f"  Visualization: {viz_path}")
        print(f"  Viz Latest: {viz_latest_path}")
        if frames_viz_path:
            print(f"  Frame Samples: {frames_viz_path}")
        
        # Print report to console as well
        print("\n" + report)
        
        return output_path


def main():
    # Configuration
    checkpoint = '/scratch/czr/Video-Guard/training_testing/output_4gpu_bs2_16k/checkpoint-8000'
    
    # Example unsafe video from SafeWatch dataset - SEXUAL CONTENT (should trigger unsafe:C1)
    video_path = "/scratch/czr/SafeWatch-Bench-Live/unsafe/aishe8864/20231012_072105_1721197962532212868.mp4"    # - Sexual 4: "/scratch/czr/Video-Guard/datasets/SafeWatch-Bench-200K-720P/full/sexual_4/target/" + filename
    # - Sexual 5: "/scratch/czr/Video-Guard/datasets/SafeWatch-Bench-200K-720P/full/sexual_5/target/" + filename
    # - Violence: "/scratch/czr/Video-Guard/datasets/SafeWatch-Bench-200K-720P/full/violence_1_abuse/target/Abuse001_x264.mp4_merged.mp4"
    # - Abuse: "/scratch/czr/Video-Guard/datasets/SafeWatch-Bench-200K-720P/full/abuse_1/target/-0nDuJdCniyuJCgx.mp4"
    # - Extremism: "/scratch/czr/Video-Guard/datasets/SafeWatch-Bench-200K-720P/full/extremism/target/00h.05mn__7179721456656452869.mp4"
    
    # Alternative: safe video from shot2story
    # video_path = "/scratch/czr/Video-Guard/datasets/shot2story-videos/release_134k_videos/--5gVbAaF_A.9.mp4"
    
    print("="*80)
    print("FULL VIDEO STREAMING ANALYSIS")
    print("Complete process: Multiple clips + Final response")
    print("="*80)
    
    # Initialize tester
    tester = FullVideoStreamingTester(
        base_model_path="OpenGVLab/InternVL3-8B",
        checkpoint_path=checkpoint,
        device="cuda:0"
    )
    
    # Load video frames - sample 1 frame per second
    # For a 1min 10sec video, we should get ~70 frames
    frames, frame_indices = load_video_frames(video_path, max_frames=120, fps_sample=None)
    print(f"Loaded {len(frames)} frames from video")
    
    # Analyze full video with video path for metadata
    results = tester.analyze_full_video(frames, frame_indices, video_path=video_path)
    
    # Save results with enhanced metadata and frame samples
    video_name = Path(video_path).stem
    output_path = tester.save_results(results, "./full_video_results", video_name, video_path=video_path, frames=frames)
    
    # Print summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    # Extract metadata from new structure
    meta = results.get('analysis_metadata', {})
    print(f"Total frames processed: {meta.get('total_frames_processed', 0)}")
    print(f"Total clips identified: {meta.get('total_clips_after_merge', 0)}")
    
    # Print clip summaries
    for clip in results.get('clips', []):
        print(f"\nClip {clip['clip_idx']}:")
        print(f"  Frames: {clip.get('frames', [])}")
        summary = clip.get('clip_summary', 'No summary')
        if len(summary) > 100:
            print(f"  Summary: {summary[:100]}...")
        else:
            print(f"  Summary: {summary}")
    
    # Print final response
    final_resp = results.get('final_response', {})
    if isinstance(final_resp, dict):
        content = final_resp.get('content', 'No final response')
        print(f"\nFinal Video Summary:")
        if len(content) > 300:
            print(f"  {content[:300]}...")
        else:
            print(f"  {content}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()