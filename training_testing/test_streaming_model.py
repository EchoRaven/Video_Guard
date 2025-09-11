#!/usr/bin/env python3
"""
Streaming Video-Guard Model Testing Script
Correctly implements streaming inference matching the training process
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from PIL import Image
import cv2
from tqdm import tqdm
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Import transformers and PEFT
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# Try to import decord
try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    print("Decord not available, using cv2 for video loading")

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Image preprocessing constants (matching training)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Label definitions
LABEL_DEFINITIONS = {
    'unsafe:C1': 'Sexual content',
    'unsafe:C2': 'Harassment/bullying',
    'unsafe:C3': 'Violence/harm', 
    'unsafe:C4': 'Misinformation',
    'unsafe:C5': 'Illegal activities',
    'unsafe:C6': 'Hate speech/extremism',
    'safe': 'Safe content',
    'continue': 'Shot continues',
    'summary': 'Shot summary'
}

def build_transform(input_size):
    """Build image transformation pipeline (exactly matching training)"""
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        # Convert to bfloat16 to match model weights
        T.Lambda(lambda x: x.to(torch.bfloat16))
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio from target ratios (matching training)"""
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

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    """Dynamic preprocessing for images with variable aspect ratios (matching training)"""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) 
        for i in range(1, n + 1) for j in range(1, n + 1) 
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

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
        # Split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    
    return processed_images

def load_video_frames(video_path: str, fps_sample: int = 30, max_frames: int = 10) -> Tuple[List[Image.Image], List[int]]:
    """Load frames from video at specified sampling rate"""
    frames = []
    frame_indices = []
    
    try:
        if DECORD_AVAILABLE:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            
            if total_frames == 0:
                logger.warning(f"Video has no frames: {video_path}")
                return [], []
            
            # Calculate frame indices to sample
            for i in range(0, min(total_frames, max_frames * fps_sample), fps_sample):
                frame_indices.append(i)
                frame = vr[i].asnumpy()
                frame = Image.fromarray(frame)
                frames.append(frame)
                
                if len(frames) >= max_frames:
                    break
        else:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return [], []
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate frame indices to sample
            for i in range(0, min(total_frames, max_frames * fps_sample), fps_sample):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frame_indices.append(i)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    frames.append(frame)
                    
                    if len(frames) >= max_frames:
                        break
            
            cap.release()
            
    except Exception as e:
        logger.error(f"Error loading video {video_path}: {e}")
        return [], []
    
    return frames, frame_indices

class StreamingVideoGuardTester:
    """Streaming tester class that correctly implements the training streaming process"""
    
    def __init__(
        self, 
        base_model_path: str = "OpenGVLab/InternVL3-8B",
        checkpoint_path: Optional[str] = None,
        device: str = "cuda:0",
        input_size: int = 448,
        max_num_patches: int = 12
    ):
        """Initialize the tester with model and tokenizer"""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.max_num_patches = max_num_patches
        
        logger.info(f"Loading model from {base_model_path}")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer (matching training)
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            use_fast=False
        )
        
        # Load base model (matching training)
        self.model = AutoModel.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=None  # We'll handle device placement
        )
        
        # Load LoRA checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading LoRA checkpoint from {checkpoint_path}")
            self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
            # Merge LoRA weights for inference
            self.model = self.model.merge_and_unload()
            logger.info("LoRA weights merged successfully")
        
        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Set special tokens (matching training)
        if not hasattr(self.tokenizer, 'img_context_token_id') or self.tokenizer.img_context_token_id is None:
            self.tokenizer.img_context_token_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        
        if not hasattr(self.model, 'img_context_token_id') or self.model.img_context_token_id is None:
            self.model.img_context_token_id = self.tokenizer.img_context_token_id
        
        logger.info(f"img_context_token_id: {self.model.img_context_token_id}")
        
        # Build transform
        self.transform = build_transform(self.input_size)
        
        # Define streaming prompt template (matching training)
        self.streaming_prompt_template = """<streaming_analysis> You are a video analyst. Analyze video frames to understand and describe content.

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
        
        logger.info("Model initialization complete!")
    
    def process_frame(self, frame: Image.Image) -> Tuple[torch.Tensor, int]:
        """Process a single frame (matching training preprocessing)"""
        # Apply dynamic preprocessing
        images = dynamic_preprocess(
            frame,
            image_size=self.input_size,
            use_thumbnail=True,
            max_num=self.max_num_patches
        )
        
        # Transform all patches
        pixel_values = [self.transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)  # [num_patches, C, H, W]
        
        return pixel_values, len(images)
    
    @torch.no_grad()
    def streaming_inference(
        self,
        frames: List[Image.Image],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> Dict[str, Any]:
        """
        Perform true streaming inference following the exact training logic:
        1. Process frame with <image><label> prompt
        2. Model generates labels, check if </label> appears
        3. If <continue> in labels, add next frame
        4. If no <continue>, let model continue generating until </summary>
        """
        # Initialize context with the system prompt
        current_context = self.streaming_prompt_template
        frame_results = []
        accumulated_pixel_values = []
        frame_idx = 0
        
        while frame_idx < len(frames):
            logger.info(f"Processing frame {frame_idx + 1}/{len(frames)}")
            
            # Process current frame
            frame = frames[frame_idx]
            pixel_values, num_patches = self.process_frame(frame)
            accumulated_pixel_values.append(pixel_values)
            
            # Build prompt for current frame
            current_context += f"\n<image><label>"
            
            # Prepare the prompt with image tokens
            prompt_with_tokens = self.prepare_prompt_for_streaming(
                current_context, 
                accumulated_pixel_values
            )
            
            # Tokenize
            inputs = self.tokenizer(
                prompt_with_tokens,
                return_tensors='pt',
                truncation=False
            )
            
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # Prepare all accumulated pixel values
            all_pixels = torch.cat(accumulated_pixel_values, dim=0)
            all_pixels = all_pixels.to(self.device).to(torch.bfloat16)
            
            # Generate response - let model generate until it produces </label> or </summary>
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=all_pixels,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    # Add more generation params to encourage output
                    min_new_tokens=10,  # Force at least some generation
                    repetition_penalty=1.1,
                    # Important: don't stop at first token, let model complete its response
                    stopping_criteria=None
                )
            
            # Decode only the new tokens
            generated_tokens = outputs[0][input_ids.shape[1]:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            logger.info(f"Frame {frame_idx + 1} response: {response[:100]}...")
            
            # Parse response to check what was generated
            has_continue = '<continue>' in response
            has_label_close = '</label>' in response
            has_summary = '<summary>' in response or '</summary>' in response
            
            # Extract labels from response
            labels = []
            for label_key in ['unsafe:C1', 'unsafe:C2', 'unsafe:C3', 'unsafe:C4', 'unsafe:C5', 'unsafe:C6', 'safe', 'continue']:
                if f'<{label_key}>' in response:
                    labels.append(label_key)
            
            # Extract summary if present
            summary = None
            if '<summary>' in response and '</summary>' in response:
                summary_start = response.find('<summary>') + len('<summary>')
                summary_end = response.find('</summary>')
                summary = response[summary_start:summary_end].strip()
            
            # Store frame result
            frame_results.append({
                'frame_index': frame_idx,
                'frame_number': frame_idx + 1,
                'num_patches': num_patches,
                'labels': labels if labels else ['safe'],
                'summary': summary,
                'raw_response': response,
                'has_continue': has_continue,
                'has_summary': has_summary
            })
            
            # Update context with the complete response
            current_context += response
            
            # Decide next action based on response
            if has_continue and has_label_close:
                # Model wants to continue with next frame
                frame_idx += 1
            elif has_summary or (has_label_close and not has_continue):
                # Shot is complete, either with summary or just ending labels
                frame_idx += 1
                
                # If there are more frames and model completed a shot, start new shot
                if frame_idx < len(frames):
                    logger.info(f"Shot complete, starting new shot at frame {frame_idx + 1}")
                    current_context = self.streaming_prompt_template
                    accumulated_pixel_values = []
            else:
                # Unexpected response format, move to next frame anyway
                logger.warning(f"Unexpected response format at frame {frame_idx + 1}, continuing...")
                frame_idx += 1
        
        return {
            'frame_results': frame_results,
            'final_context': current_context
        }
    
    def prepare_prompt_for_streaming(
        self, 
        prompt: str, 
        accumulated_pixel_values: List[torch.Tensor]
    ) -> str:
        """Prepare prompt with correct image tokens for all accumulated frames"""
        IMG_START_TOKEN = '<img>'
        IMG_END_TOKEN = '</img>'
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        
        # Each patch needs 256 tokens (InternVL3-8B)
        num_image_token = 256
        
        # Count <image> placeholders
        image_count = prompt.count('<image>')
        
        # Replace each <image> with appropriate tokens
        modified_prompt = prompt
        for i, pixel_values in enumerate(accumulated_pixel_values):
            if '<image>' in modified_prompt:
                num_patches = pixel_values.shape[0]
                total_image_tokens = num_image_token * num_patches
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * total_image_tokens + IMG_END_TOKEN
                modified_prompt = modified_prompt.replace('<image>', image_tokens, 1)
        
        return modified_prompt
    
    def parse_frame_response(self, response: str) -> Tuple[List[str], Optional[str]]:
        """Parse the model's response to extract labels and summary"""
        labels = []
        summary = None
        
        # Debug: print raw response
        if response:
            print(f"DEBUG - Raw response: {response[:200]}")
        
        # The model should generate in format: <safe><continue></label> or just text
        # Sometimes model may not include the tags properly
        
        # Check for each label type in the response
        for label_key in ['unsafe:C1', 'unsafe:C2', 'unsafe:C3', 'unsafe:C4', 'unsafe:C5', 'unsafe:C6', 'safe', 'continue']:
            # Check both with and without angle brackets
            if f'<{label_key}>' in response or label_key in response:
                labels.append(label_key)
        
        # Extract summary if present
        if '<summary>' in response and '</summary>' in response:
            summary_start = response.find('<summary>') + len('<summary>')
            summary_end = response.find('</summary>')
            summary = response[summary_start:summary_end].strip()
        elif 'summary>' in response:  # Sometimes model might not close tag properly
            # Try to extract any text after summary
            summary_idx = response.find('summary>') + len('summary>')
            potential_summary = response[summary_idx:].strip()
            if len(potential_summary) > 10:  # If there's meaningful content
                summary = potential_summary[:500]  # Limit length
        
        # Default labels based on frame position if nothing found
        if not labels:
            # Assume safe and continue for non-final frames
            labels = ['safe', 'continue']
        
        return labels, summary
    
    def analyze_video_streaming(
        self,
        video_path: str,
        fps_sample: int = 30,
        max_frames: int = 10,
        save_visualization: bool = True,
        output_dir: str = "./test_results"
    ) -> Dict[str, Any]:
        """Analyze video using true streaming approach"""
        logger.info(f"Analyzing video: {video_path}")
        
        # Load video frames
        frames, frame_indices = load_video_frames(video_path, fps_sample, max_frames)
        
        if not frames:
            logger.error("No frames loaded from video")
            return {"error": "Failed to load video frames"}
        
        logger.info(f"Loaded {len(frames)} frames from video")
        
        # Perform streaming inference
        streaming_results = self.streaming_inference(frames)
        
        # Extract final summary from the last frame
        final_summary = None
        for result in reversed(streaming_results['frame_results']):
            if result['summary']:
                final_summary = result['summary']
                break
        
        # Compile results
        results = {
            'video_path': video_path,
            'total_frames_processed': len(frames),
            'fps_sample': fps_sample,
            'frame_results': streaming_results['frame_results'],
            'final_summary': final_summary,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save visualization if requested
        if save_visualization:
            self._save_visualization(frames, streaming_results['frame_results'], results, output_dir)
        
        # Save JSON results
        os.makedirs(output_dir, exist_ok=True)
        video_name = Path(video_path).stem
        json_path = os.path.join(output_dir, f"{video_name}_streaming_results.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {json_path}")
        
        return results
    
    def _save_visualization(
        self,
        frames: List[Image.Image],
        frame_results: List[Dict],
        results: Dict,
        output_dir: str
    ):
        """Save visualization of analysis results"""
        video_name = Path(results['video_path']).stem
        
        # Create figure with subplots
        n_frames = len(frames)
        n_cols = min(5, n_frames)
        n_rows = (n_frames + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(n_cols * 4, n_rows * 5))
        gs = GridSpec(n_rows + 1, n_cols, height_ratios=[1] * n_rows + [0.3])
        
        # Define colors for different label types
        label_colors = {
            'unsafe:C1': 'red',
            'unsafe:C2': 'orange',
            'unsafe:C3': 'darkred',
            'unsafe:C4': 'purple',
            'unsafe:C5': 'brown',
            'unsafe:C6': 'maroon',
            'safe': 'green',
            'continue': 'blue',
            'summary': 'gray'
        }
        
        # Plot each frame with its analysis
        for idx, (frame, result) in enumerate(zip(frames, frame_results)):
            row = idx // n_cols
            col = idx % n_cols
            
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(frame)
            ax.axis('off')
            
            # Add frame info
            title = f"Frame {result['frame_number']}"
            ax.set_title(title, fontsize=10, fontweight='bold')
            
            # Add labels as colored boxes
            labels = result['labels'] if result['labels'] else ['safe']
            label_text = []
            for label in labels:
                color = label_colors.get(label, 'black')
                label_desc = LABEL_DEFINITIONS.get(label, label)
                label_text.append(f"{label_desc}")
            
            # Add text below image
            label_str = '\n'.join(label_text)
            ax.text(0.5, -0.1, label_str, transform=ax.transAxes,
                   ha='center', va='top', fontsize=8, wrap=True)
            
            # Add summary if present
            if result['summary']:
                summary_text = f"Summary: {result['summary'][:50]}..."
                ax.text(0.5, -0.2, summary_text, transform=ax.transAxes,
                       ha='center', va='top', fontsize=7, style='italic', wrap=True)
        
        # Add legend
        legend_ax = fig.add_subplot(gs[-1, :])
        legend_ax.axis('off')
        
        # Create legend patches
        patches = []
        for label, color in label_colors.items():
            if label in LABEL_DEFINITIONS:
                patches.append(mpatches.Patch(color=color, label=LABEL_DEFINITIONS[label]))
        
        legend_ax.legend(handles=patches, loc='center', ncol=min(3, len(patches)), 
                        frameon=False, fontsize=9)
        
        # Add final summary
        fig.suptitle(f"Streaming Video Analysis: {video_name}", fontsize=14, fontweight='bold')
        
        # Add final summary text at bottom
        if results['final_summary']:
            summary_text = f"Final Summary: {results['final_summary'][:200]}..."
            fig.text(0.5, 0.02, summary_text, ha='center', fontsize=10, wrap=True)
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        viz_path = os.path.join(output_dir, f"{video_name}_streaming_visualization.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {viz_path}")


def main():
    parser = argparse.ArgumentParser(description="Test Video-Guard streaming model")
    
    # Model arguments
    parser.add_argument('--base_model', type=str, default='OpenGVLab/InternVL3-8B',
                       help='Base model path')
    parser.add_argument('--checkpoint', type=str, 
                       default='/scratch/czr/Video-Guard/training_testing/output_8gpu_full/checkpoint-500',
                       help='Path to LoRA checkpoint')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    
    # Video arguments
    parser.add_argument('--video_path', type=str, 
                       help='Path to single video file to test')
    parser.add_argument('--dataset_dir', type=str,
                       default='/scratch/czr/Video-Guard/datasets/shot2story-videos/release_134k_videos',
                       help='Directory containing videos to test')
    
    # Processing arguments
    parser.add_argument('--fps_sample', type=int, default=30,
                       help='Sample every N frames')
    parser.add_argument('--max_frames', type=int, default=10,
                       help='Maximum frames to process per video')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./streaming_test_results',
                       help='Directory to save results')
    parser.add_argument('--no_visualization', action='store_true',
                       help='Skip saving visualizations')
    
    args = parser.parse_args()
    
    # Create tester
    logger.info("Initializing Streaming Video-Guard Tester...")
    tester = StreamingVideoGuardTester(
        base_model_path=args.base_model,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # Test on video
    if args.video_path:
        # Test single video
        logger.info(f"Testing single video: {args.video_path}")
        results = tester.analyze_video_streaming(
            args.video_path,
            fps_sample=args.fps_sample,
            max_frames=args.max_frames,
            save_visualization=not args.no_visualization,
            output_dir=args.output_dir
        )
        
        # Print results
        print("\n" + "="*80)
        print("STREAMING ANALYSIS RESULTS")
        print("="*80)
        print(f"Video: {args.video_path}")
        print(f"Frames processed: {results['total_frames_processed']}")
        print("\nFrame-by-frame streaming analysis:")
        for frame_result in results['frame_results']:
            labels = frame_result['labels'] if frame_result['labels'] else ['safe']
            print(f"  Frame {frame_result['frame_number']}: {labels}")
            if frame_result['summary']:
                print(f"    Summary: {frame_result['summary']}")
        print(f"\nFinal Summary: {results['final_summary']}")
        print("="*80)
    else:
        # Find a sample video from dataset
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        video_files = []
        for ext in video_extensions:
            video_files.extend(Path(args.dataset_dir).rglob(f'*{ext}'))
        
        if video_files:
            test_video = str(video_files[0])
            logger.info(f"Testing sample video: {test_video}")
            results = tester.analyze_video_streaming(
                test_video,
                fps_sample=args.fps_sample,
                max_frames=args.max_frames,
                save_visualization=not args.no_visualization,
                output_dir=args.output_dir
            )
            
            # Print results
            print("\n" + "="*80)
            print("STREAMING ANALYSIS RESULTS")
            print("="*80)
            print(f"Video: {test_video}")
            print(f"Frames processed: {results['total_frames_processed']}")
            print("\nFrame-by-frame streaming analysis:")
            for frame_result in results['frame_results']:
                labels = frame_result['labels'] if frame_result['labels'] else ['safe']
                print(f"  Frame {frame_result['frame_number']}: {labels}")
                if frame_result['summary']:
                    print(f"    Summary: {frame_result['summary']}")
            print(f"\nFinal Summary: {results['final_summary']}")
            print("="*80)
        else:
            logger.error(f"No video files found in {args.dataset_dir}")
    
    logger.info("Streaming testing complete!")


if __name__ == "__main__":
    main()