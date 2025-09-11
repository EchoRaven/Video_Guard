#!/usr/bin/env python3
"""
StreamingModel class for video safety inference
Exactly matches the inference logic from test_full_video_streaming.py
"""

import os
import torch
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from PIL import Image
import numpy as np
from datetime import datetime

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
    """Build image transformation pipeline"""
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        T.Lambda(lambda x: x.to(torch.bfloat16))
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio from target ratios"""
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

def dynamic_preprocess(image, min_num=1, max_num=2, image_size=448, use_thumbnail=False):
    """Dynamic preprocessing for images with patch splitting"""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    
    # Target ratios for patch splitting
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) 
        for i in range(1, n + 1) 
        for j in range(1, n + 1) 
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    
    # Find best aspect ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    
    # Calculate target dimensions
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    
    # Resize and split image
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


class StreamingModel:
    """
    Streaming video safety model - exact match to test_full_video_streaming.py
    """
    
    def __init__(
        self,
        base_model_path: str = "OpenGVLab/InternVL3-8B",
        checkpoint_path: Optional[str] = None,
        device: str = "cuda:0"
    ):
        """
        Initialize the streaming model
        
        Args:
            base_model_path: Path to base model
            checkpoint_path: Path to LoRA checkpoint
            device: Device to run model on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.input_size = 448
        self.max_num_patches = 2  # Reduced as requested
        self.checkpoint_path = checkpoint_path
        
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
        
        logger.info(f"StreamingModel initialized on {self.device}")
        logger.info(f"Max patches: {self.max_num_patches}")
    
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
        max_tokens: int = 500
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
            
            # Add to generated sequence
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones(1, 1).to(self.device)], dim=1)
            
            # Check for stopping conditions
            generated_text = self.tokenizer.decode(generated_ids[0][input_ids.shape[1]:])
            
            # IMPORTANT: Check for </response> FIRST (for final response generation)
            if '</response>' in generated_text:
                break
            
            # Look for complete patterns for frame responses
            if '</label>' in generated_text:
                if '<continue>' in generated_text:
                    break
                elif not '<continue>' in generated_text:
                    # This is the last frame, must generate summary
                    # Keep generating until we find </summary>
                    if '</summary>' in generated_text:
                        break
                    # Don't stop early - let it generate the full summary
            
            # Check for EOS token
            if next_token_id == self.tokenizer.eos_token_id:
                break
            
            # Only stop if we've generated way too many tokens
            if step >= max_tokens - 1:
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
    
    def load_video(self, video_path: str, fps: float = 1.0) -> Tuple[List[Image.Image], List[int]]:
        """Load video and extract frames at specified FPS"""
        if not DECORD_AVAILABLE:
            logger.error("Decord not available for video loading")
            return None, None
            
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            video_fps = vr.get_avg_fps()
            total_frames = len(vr)
            duration = total_frames / video_fps
            
            logger.info(f"Video: {total_frames} frames, {video_fps:.1f} fps, {duration:.1f}s duration")
            
            # Calculate frame indices based on desired FPS
            frame_interval = int(video_fps / fps)
            frame_indices = list(range(0, total_frames, frame_interval))
            
            logger.info(f"Extracting {len(frame_indices)} frames at {fps} fps (every {frame_interval} frames)")
            
            # Extract frames
            frames = []
            for idx in frame_indices:
                frame = Image.fromarray(vr[idx].asnumpy())
                frames.append(frame)
            
            return frames, frame_indices
            
        except Exception as e:
            logger.error(f"Failed to load video: {e}")
            return None, None
    
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
        
        logger.info("Starting full video analysis...")
        
        # Process clips until all frames are consumed
        while frame_idx < len(frames):
            clip_idx += 1
            logger.info(f"Processing clip {clip_idx}...")
            
            # Start new clip with fresh context
            clip_context = self.clip_prompt
            clip_pixel_values = []
            clip_frames = []
            clip_results = []
            
            # Process frames for this clip
            while frame_idx < len(frames):
                current_frame = frames[frame_idx]
                # Processing frame
                
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
                    logger.warning(f"Context too long ({test_inputs['input_ids'].shape[1]} tokens), truncating...")
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
                # Frame response generated
                
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
                    logger.info(f"Clip {clip_idx} complete with {len(clip_frames)} frames")
                    break
                
                # If no continue tag, we should keep generating until we get a summary
                # The model might need more context to generate the summary
                if not has_continue:
                    # No continue tag, expecting summary
                    # Don't break here - let the model continue to generate summary
                    pass
                
                # Special handling for last frame of video
                if frame_idx == len(frames) and not has_summary:
                    logger.debug("Last frame reached without summary, forcing summary generation...")
                    # Add <summary> tag to force model to generate summary
                    clip_context += "<summary>"
                    
                    # Generate summary with updated context
                    prompt_with_tokens = self._replace_image_tokens(clip_context, clip_pixel_values)
                    
                    # Apply same token limit check
                    test_inputs = self.tokenizer(prompt_with_tokens, return_tensors='pt', truncation=False)
                    if test_inputs['input_ids'].shape[1] > 11000:
                        logger.warning(f"Summary context too long ({test_inputs['input_ids'].shape[1]} tokens), truncating...")
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
                    # Forced summary generated
                    
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
        logger.info("Post-processing: Merging clips with identical summaries...")
        
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
                    logger.debug(f"Merging clip {current_clip['clip_idx']} and {next_clip['clip_idx']} (same summary)")
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
        
        logger.info(f"Merged {len(all_clips)} clips into {len(merged_clips)} clips")
        all_clips = merged_clips
        all_clip_prompts = merged_clip_prompts
        
        logger.info("Generating final response...")
        
        # Construct final response prompt (matching training)
        # Start with user prompt, then add ALL accumulated clip content
        final_prompt = self.clip_prompt
        for clip_prompt in all_clip_prompts:
            final_prompt += clip_prompt
        
        # CRITICAL: Add vision_end token before final response generation
        final_prompt += self.vision_end_token
        logger.debug(f"Final prompt includes {len(all_clip_prompts)} clips with vision_end token")
        
        # Generate final response
        # Generating final video summary
        
        # For final response, we don't add new images - just use the accumulated text
        # The model should generate based on all the clip summaries
        # Increase max_length to handle multiple clips while staying under model limit
        inputs = self.tokenizer(final_prompt, return_tensors='pt', truncation=True, max_length=11000)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Log if truncation occurred
        if len(self.tokenizer.encode(final_prompt)) > 11000:
            logger.warning(f"Final prompt truncated from {len(self.tokenizer.encode(final_prompt))} to 11000 tokens")
        
        # Generate final response token by token
        generated_ids = input_ids.clone()
        
        # Generating tokens for final response
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
                logger.debug(f"Generated {step+1} tokens for final response")
                break
            
            # Check for EOS token
            if next_token_id == self.tokenizer.eos_token_id:
                logger.debug(f"Found EOS token at step {step+1}")
                break
            
            # Fallback: stop after enough tokens
            if step >= max_final_tokens - 1:
                logger.debug(f"Reached max tokens for final response ({max_final_tokens})")
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
        
        logger.info(f"Final response generated: {len(final_response_content)} characters")
        
        # Compile comprehensive results with analysis metadata
        results = {
            'video_path': video_path if video_path else 'unknown',
            'model_checkpoint': self.checkpoint_path if hasattr(self, 'checkpoint_path') else 'base_model',
            'analysis_metadata': {
                'total_frames_loaded': len(frames),
                'total_frames_processed': frame_idx,
                'fps_sample': 'auto_1_per_second',
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
                'merged': len(all_clip_prompts) != len(all_clips),
                'original_clips': len(all_clip_prompts),
                'merged_clips': len(all_clips)
            },
            'final_response': {
                'raw': raw_final,
                'cleaned': final_response_content,
                'token_count': step + 1 if 'step' in locals() else 0,
                'truncated': len(self.tokenizer.encode(final_prompt)) > 11000
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def process_video_streaming(self, video_path: str, fps: float = 1.0) -> Dict[str, Any]:
        """
        Process video in streaming fashion
        
        Args:
            video_path: Path to video file
            fps: Frames per second to extract
            
        Returns:
            Dictionary with streaming results and final decision
        """
        # Load video and extract frames
        frames, frame_indices = self.load_video(video_path, fps)
        if frames is None:
            return {
                'video_path': video_path,
                'status': 'error',
                'error': 'Failed to load video',
                'timestamp': datetime.now().isoformat()
            }
        
        # Analyze video
        results = self.analyze_full_video(frames, frame_indices, video_path)
        results['status'] = 'completed'
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save inference results to JSON"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")


def main():
    """Example usage of StreamingModel"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Streaming Video Safety Inference')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--checkpoint', type=str, help='Path to LoRA checkpoint')
    parser.add_argument('--base-model', type=str, default='OpenGVLab/InternVL3-8B',
                       help='Base model path')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--output', type=str, help='Output JSON path')
    parser.add_argument('--fps', type=float, default=1.0, help='Frames per second to extract')
    
    args = parser.parse_args()
    
    # Initialize model
    model = StreamingModel(
        base_model_path=args.base_model,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # Process video
    results = model.process_video_streaming(args.video, args.fps)
    
    # Save results if output path provided
    if args.output:
        model.save_results(results, args.output)
    
    # Print summary
    if results['status'] == 'completed':
        print(f"\nSTREAMING INFERENCE RESULTS")
        print(f"Video: {args.video}")
        print(f"Processed: {results['analysis_metadata']['total_frames_processed']} frames in {results['analysis_metadata']['total_clips_after_merge']} clips")
        print(f"\nFinal Response:")
        print(results['final_response']['cleaned'][:500])
    else:
        print(f"\nError processing video: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()