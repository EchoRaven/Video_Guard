#!/usr/bin/env python3
"""
Complete Testing Script for Trained Video-Guard Model
Fully mimics the training code structure for accurate evaluation
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

class VideoGuardTester:
    """Tester class that mimics the training process exactly"""
    
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
    
    def construct_streaming_prompt(
        self, 
        frame_idx: int,
        total_patches: int,
        accumulated_context: str = ""
    ) -> str:
        """Construct prompt for streaming analysis (matching training format)"""
        # Match training format: use <image> placeholder that will be replaced
        image_placeholder = '<image>'
        
        if frame_idx == 0:
            # First frame
            prompt = f"{self.streaming_prompt_template}\nFrame 1:\n{image_placeholder}<label>"
        else:
            # Subsequent frames with accumulated context
            prompt = f"{accumulated_context}\nFrame {frame_idx + 1}:\n{image_placeholder}<label>"
        
        return prompt
    
    def prepare_prompt_with_image_tokens(self, prompt: str, num_patches: int) -> str:
        """Replace <image> placeholder with actual image tokens (matching training)"""
        IMG_START_TOKEN = '<img>'
        IMG_END_TOKEN = '</img>'
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        
        # Each patch needs 256 tokens (InternVL3-8B)
        num_image_token = 256
        
        # Replace <image> placeholder with actual tokens
        if '<image>' in prompt:
            total_image_tokens = num_image_token * num_patches
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * total_image_tokens + IMG_END_TOKEN
            prompt = prompt.replace('<image>', image_tokens)
        
        return prompt
    
    def prepare_prompt_with_multiple_images(self, prompt: str, num_patches_list: List[int]) -> str:
        """Replace multiple <image> placeholders with actual image tokens"""
        IMG_START_TOKEN = '<img>'
        IMG_END_TOKEN = '</img>'
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        
        # Each patch needs 256 tokens (InternVL3-8B)
        num_image_token = 256
        
        # Replace each <image> placeholder with corresponding tokens
        for num_patches in num_patches_list:
            if '<image>' in prompt:
                total_image_tokens = num_image_token * num_patches
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * total_image_tokens + IMG_END_TOKEN
                prompt = prompt.replace('<image>', image_tokens, 1)  # Replace one at a time
        
        return prompt
    
    @torch.no_grad()
    def generate_response_batch(
        self, 
        prompt: str,
        pixel_values: Optional[torch.Tensor] = None,
        num_patches_list: Optional[List[int]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> str:
        """Generate model response for multiple frames (matching training)"""
        # Replace all <image> placeholders with actual image tokens
        if num_patches_list:
            prompt = self.prepare_prompt_with_multiple_images(prompt, num_patches_list)
        
        # Tokenize the prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=False
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Prepare pixel values
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device).to(torch.bfloat16)
        
        # Generate response
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        return response
    
    @torch.no_grad()
    def generate_response(
        self, 
        prompt: str,
        pixel_values: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> str:
        """Generate model response (matching training inference)"""
        # Get number of patches from pixel_values
        if pixel_values is not None:
            if len(pixel_values.shape) == 3:  # [patches, C, H, W]
                num_patches = pixel_values.shape[0]
            elif len(pixel_values.shape) == 4:  # [1, patches, C, H, W]
                num_patches = pixel_values.shape[1]
            else:
                num_patches = 1
            
            # Replace <image> placeholder with actual image tokens
            prompt = self.prepare_prompt_with_image_tokens(prompt, num_patches)
        
        # Tokenize the prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=False
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Prepare pixel values
        if pixel_values is not None:
            # Flatten pixel values for batch processing
            if len(pixel_values.shape) == 3:  # [patches, C, H, W]
                pixel_values = pixel_values.unsqueeze(0)  # [1, patches, C, H, W]
            
            pixel_values_flat = pixel_values.view(-1, pixel_values.shape[-3], 
                                                  pixel_values.shape[-2], pixel_values.shape[-1])
            pixel_values_flat = pixel_values_flat.to(self.device).to(torch.bfloat16)
        else:
            pixel_values_flat = None
        
        # Generate response
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values_flat,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        return response
    
    def analyze_video_streaming(
        self,
        video_path: str,
        fps_sample: int = 30,
        max_frames: int = 10,
        save_visualization: bool = True,
        output_dir: str = "./test_results"
    ) -> Dict[str, Any]:
        """Analyze video using streaming approach (matching training)"""
        logger.info(f"Analyzing video: {video_path}")
        
        # Load video frames
        frames, frame_indices = load_video_frames(video_path, fps_sample, max_frames)
        
        if not frames:
            logger.error("No frames loaded from video")
            return {"error": "Failed to load video frames"}
        
        logger.info(f"Loaded {len(frames)} frames from video")
        
        # Process all frames first to get pixel values
        all_pixel_values = []
        all_num_patches = []
        
        for frame in frames:
            pixel_values, num_patches = self.process_frame(frame)
            all_pixel_values.append(pixel_values)
            all_num_patches.append(num_patches)
        
        # Concatenate all pixel values (matching training)
        if all_pixel_values:
            combined_pixel_values = torch.cat(all_pixel_values, dim=0)  # [total_patches, C, H, W]
        else:
            combined_pixel_values = None
        
        # Build complete prompt with all frames (matching training format)
        full_prompt = self.streaming_prompt_template
        for idx in range(len(frames)):
            full_prompt += f"\nFrame {idx + 1}:\n<image><label>"
        
        # Generate response for all frames at once
        response = self.generate_response_batch(full_prompt, combined_pixel_values, all_num_patches)
        
        # Parse response to extract labels for each frame
        frame_results = self.parse_streaming_response(response, frame_indices, all_num_patches)
        
        # Log results
        for idx, result in enumerate(frame_results):
            labels = result['labels'] if result['labels'] else ['safe']
            logger.info(f"Frame {idx + 1}: {labels}")
        
        # Extract final summary from response
        final_summary = self.extract_final_summary(response)
        
        # Compile results
        results = {
            'video_path': video_path,
            'total_frames_processed': len(frames),
            'fps_sample': fps_sample,
            'frame_results': frame_results,
            'final_summary': final_summary,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save visualization if requested
        if save_visualization:
            self._save_visualization(frames, frame_results, results, output_dir)
        
        # Save JSON results
        os.makedirs(output_dir, exist_ok=True)
        video_name = Path(video_path).stem
        json_path = os.path.join(output_dir, f"{video_name}_results.json")
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
            title = f"Frame {result['frame_number']} (idx: {result['frame_index']})"
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
        fig.suptitle(f"Video Analysis: {video_name}", fontsize=14, fontweight='bold')
        
        # Add final summary text at bottom
        if results['final_summary']:
            summary_text = f"Final Summary: {results['final_summary'][:200]}..."
            fig.text(0.5, 0.02, summary_text, ha='center', fontsize=10, wrap=True)
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        viz_path = os.path.join(output_dir, f"{video_name}_visualization.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {viz_path}")
    
    def test_on_dataset(
        self,
        dataset_dir: str,
        video_list: Optional[List[str]] = None,
        fps_sample: int = 30,
        max_frames: int = 10,
        max_videos: int = 10,
        output_dir: str = "./test_results"
    ) -> Dict[str, Any]:
        """Test on multiple videos from dataset"""
        # Get video files
        if video_list:
            video_files = video_list
        else:
            # Find all video files in directory
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
            video_files = []
            for ext in video_extensions:
                video_files.extend(Path(dataset_dir).rglob(f'*{ext}'))
            video_files = [str(f) for f in video_files[:max_videos]]
        
        logger.info(f"Testing on {len(video_files)} videos")
        
        all_results = []
        
        for video_path in tqdm(video_files, desc="Testing videos"):
            try:
                result = self.analyze_video_streaming(
                    video_path,
                    fps_sample=fps_sample,
                    max_frames=max_frames,
                    save_visualization=True,
                    output_dir=output_dir
                )
                all_results.append(result)
            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")
                all_results.append({
                    'video_path': video_path,
                    'error': str(e)
                })
        
        # Save summary
        summary = {
            'total_videos': len(video_files),
            'successful': len([r for r in all_results if 'error' not in r]),
            'failed': len([r for r in all_results if 'error' in r]),
            'results': all_results,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = os.path.join(output_dir, 'test_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Test summary saved to {summary_path}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Test Video-Guard trained model")
    
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
    parser.add_argument('--video_list', type=str, nargs='+',
                       help='List of specific video paths to test')
    
    # Processing arguments
    parser.add_argument('--fps_sample', type=int, default=30,
                       help='Sample every N frames')
    parser.add_argument('--max_frames', type=int, default=10,
                       help='Maximum frames to process per video')
    parser.add_argument('--max_videos', type=int, default=5,
                       help='Maximum number of videos to test')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./test_results',
                       help='Directory to save results')
    parser.add_argument('--no_visualization', action='store_true',
                       help='Skip saving visualizations')
    
    args = parser.parse_args()
    
    # Create tester
    logger.info("Initializing Video-Guard Tester...")
    tester = VideoGuardTester(
        base_model_path=args.base_model,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # Test on video(s)
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
        print("ANALYSIS RESULTS")
        print("="*80)
        print(f"Video: {args.video_path}")
        print(f"Frames processed: {results['total_frames_processed']}")
        print("\nFrame-by-frame analysis:")
        for frame_result in results['frame_results']:
            labels = frame_result['labels'] if frame_result['labels'] else ['safe']
            print(f"  Frame {frame_result['frame_number']}: {labels}")
            if frame_result['summary']:
                print(f"    Summary: {frame_result['summary']}")
        print(f"\nFinal Summary: {results['final_summary']}")
        print("="*80)
        
    elif args.video_list:
        # Test specific videos
        logger.info(f"Testing {len(args.video_list)} specified videos")
        summary = tester.test_on_dataset(
            dataset_dir=None,
            video_list=args.video_list,
            fps_sample=args.fps_sample,
            max_frames=args.max_frames,
            output_dir=args.output_dir
        )
        
        print(f"\nTested {summary['total_videos']} videos")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        
    else:
        # Test on dataset
        logger.info(f"Testing videos from dataset: {args.dataset_dir}")
        summary = tester.test_on_dataset(
            dataset_dir=args.dataset_dir,
            fps_sample=args.fps_sample,
            max_frames=args.max_frames,
            max_videos=args.max_videos,
            output_dir=args.output_dir
        )
        
        print(f"\nTested {summary['total_videos']} videos")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
    
    logger.info("Testing complete!")


if __name__ == "__main__":
    main()