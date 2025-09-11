#!/usr/bin/env python3
"""
Enhanced model comparison between GPT-4 and Video-Guard
- Includes benign videos from SafeWatch-Live
- Tests 10 unsafe and 10 safe videos
- Implements early stopping for Video-Guard
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from PIL import Image

# Set environment variable for GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# Add path for imports
sys.path.append('/scratch/czr/Video-Guard/training_testing')
sys.path.append('/scratch/czr/Video-Guard/datasets')

from compare_models_unsafe_detection import ModelComparison
from load_annotations import SafeWatchLiveAnnotations
from get_benign_videos import get_benign_videos, get_mixed_safe_videos


class EnhancedModelComparison(ModelComparison):
    """Enhanced comparison with early stopping for Video-Guard"""
    
    def sample_frames_adaptive(self, video_info: Dict[str, Any], max_duration: int = 30) -> Tuple[List[Image.Image], List[int]]:
        """
        Sample frames with adaptive FPS based on video source
        - Benign videos: 1 fps
        - Other videos: 2 fps
        """
        video_path = video_info['path']
        source = video_info.get('source', 'regular')
        
        # Set FPS based on source
        if source == 'benign':
            sample_fps = 1  # 1 frame per second for benign videos
            print(f"    Using 1 fps for benign video")
        else:
            sample_fps = 2  # 2 frames per second for potentially unsafe videos
            print(f"    Using 2 fps for {source} video")
        
        # Temporarily override the instance fps
        original_fps = self.sample_fps
        self.sample_fps = sample_fps
        
        # Sample frames
        frames, frame_indices = self.sample_frames_from_video(video_path, max_duration)
        
        # Restore original fps
        self.sample_fps = original_fps
        
        return frames, frame_indices
    
    def test_video_guard_with_early_stop(self, video_path: str, frames: List[Image.Image], 
                                         frame_indices: List[int]) -> Dict[str, Any]:
        """
        Test Video-Guard model with early stopping on first unsafe frame
        """
        print(f"  Testing Video-Guard with early stopping...")
        
        start_time = time.time()
        unsafe_detected = False
        first_unsafe_frame = None
        unsafe_categories = []
        frames_processed = 0
        
        # Process frames one by one (or in small batches)
        batch_size = 4  # Process 4 frames at a time for efficiency
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            batch_indices = frame_indices[i:i+batch_size]
            frames_processed += len(batch_frames)
            
            # Analyze batch
            results = self.video_guard.analyze_full_video(
                batch_frames, 
                batch_indices, 
                video_path=video_path
            )
            
            # Check for unsafe content
            safety_analysis = results.get('safety_analysis', {})
            if safety_analysis.get('total_unsafe_frames', 0) > 0:
                unsafe_detected = True
                first_unsafe_frame = i
                unsafe_categories = safety_analysis.get('unsafe_categories_detected', [])
                print(f"    ⚠️ Unsafe content detected at frame batch {i//batch_size + 1}")
                print(f"    Categories: {unsafe_categories}")
                break
        
        inference_time = time.time() - start_time
        
        return {
            'model': 'Video-Guard',
            'video_path': video_path,
            'total_frames_available': len(frames),
            'frames_processed': frames_processed,
            'is_unsafe': unsafe_detected,
            'first_unsafe_frame': first_unsafe_frame,
            'unsafe_categories': unsafe_categories,
            'inference_time': inference_time,
            'early_stopped': unsafe_detected  # True if stopped early due to unsafe content
        }
    
    def test_gpt4_enhanced(self, video_path: str, frames: List[Image.Image]) -> Dict[str, Any]:
        """Enhanced GPT-4 testing with better error handling"""
        print(f"  Testing GPT-4...")
        
        try:
            # Use parent class method but with additional error handling
            result = self.test_gpt4(video_path, frames)
            
            if result.get('error'):
                print(f"    ❌ GPT-4 error: {result['error']}")
            else:
                print(f"    GPT-4 prediction: {'UNSAFE' if result['is_unsafe'] else 'SAFE'}")
                if result.get('unsafe_categories'):
                    print(f"    Categories: {result['unsafe_categories']}")
            
            return result
            
        except Exception as e:
            print(f"    ❌ GPT-4 API error: {str(e)}")
            return {
                'model': 'GPT-4',
                'video_path': video_path,
                'error': str(e),
                'is_unsafe': None,
                'inference_time': 0
            }


def run_enhanced_comparison():
    """Run enhanced comparison with benign videos and early stopping"""
    
    print("="*60)
    print("ENHANCED MODEL COMPARISON: Video-Guard vs GPT-4")
    print("Dataset: SafeWatch-Bench-Live (including benign videos)")
    print("="*60)
    
    # Load annotations
    annotations = SafeWatchLiveAnnotations()
    
    # Get videos: 10 unsafe, 10 safe (mix of regular safe and benign)
    print("\n📹 Selecting Videos for Testing...")
    
    # Get unsafe videos
    unsafe_samples = annotations.get_random_unsafe(10)
    print(f"  Selected {len(unsafe_samples)} unsafe videos")
    
    # Get safe videos: 5 from benign folder, 5 from regular safe
    benign_videos = get_benign_videos(n_videos=5)
    regular_safe = annotations.get_random_safe(5)
    safe_samples = benign_videos + regular_safe
    print(f"  Selected {len(benign_videos)} benign videos")
    print(f"  Selected {len(regular_safe)} regular safe videos")
    
    # Combine all videos
    all_videos = unsafe_samples + safe_samples
    
    print("\n📊 Video Distribution:")
    print(f"  Total videos: {len(all_videos)}")
    print(f"  Unsafe: {len(unsafe_samples)}")
    print(f"  Safe: {len(safe_samples)} (Benign: {len(benign_videos)}, Regular: {len(regular_safe)})")
    
    # Display selected videos
    print("\nUnsafe Videos:")
    for i, v in enumerate(unsafe_samples[:5], 1):  # Show first 5
        print(f"  {i}. {Path(v['path']).name}")
    if len(unsafe_samples) > 5:
        print(f"  ... and {len(unsafe_samples) - 5} more")
    
    print("\nSafe Videos (Benign):")
    for i, v in enumerate(benign_videos[:3], 1):  # Show first 3
        print(f"  {i}. {Path(v['path']).name}")
    if len(benign_videos) > 3:
        print(f"  ... and {len(benign_videos) - 3} more")
    
    print("\nSafe Videos (Regular):")
    for i, v in enumerate(regular_safe[:3], 1):  # Show first 3
        print(f"  {i}. {Path(v['path']).name}")
    if len(regular_safe) > 3:
        print(f"  ... and {len(regular_safe) - 3} more")
    
    print("\n" + "="*60)
    print("Starting Comparison Test...")
    print("="*60)
    
    # Initialize comparison
    comparison = EnhancedModelComparison(
        video_guard_checkpoint="/scratch/czr/Video-Guard/training_testing/output_8gpu_full/checkpoint-3500",
        sample_fps=2,  # Default 2 fps, but will use 1 fps for benign videos
        max_frames_gpt=30,  # Limit for GPT-4
        device="cuda:0"
    )
    
    # Test all videos
    results = []
    
    for i, video_info in enumerate(all_videos):
        print(f"\n[{i+1}/{len(all_videos)}] Processing: {Path(video_info['path']).name}")
        source = video_info.get('source', 'regular')
        print(f"  Source: {source}")
        print(f"  Ground truth: {video_info['label']}")
        
        # Sample frames with adaptive FPS
        frames, frame_indices = comparison.sample_frames_adaptive(video_info)
        if not frames:
            print("  ❌ Could not load video")
            continue
        
        print(f"  ✓ Loaded {len(frames)} frames")
        
        # Test Video-Guard with early stopping
        vg_result = comparison.test_video_guard_with_early_stop(
            video_info['path'], frames, frame_indices
        )
        
        # Test GPT-4
        gpt_result = comparison.test_gpt4_enhanced(video_info['path'], frames)
        
        # Store result
        result = {
            'video_path': video_info['path'],
            'video_name': Path(video_info['path']).name,
            'source': source,
            'ground_truth': video_info['label'],
            'ground_truth_is_unsafe': video_info['label'].startswith('unsafe'),
            'video_guard': {
                'prediction': 'unsafe' if vg_result['is_unsafe'] else 'safe',
                'correct': vg_result['is_unsafe'] == video_info['label'].startswith('unsafe'),
                'categories': vg_result.get('unsafe_categories', []),
                'frames_processed': vg_result.get('frames_processed', 0),
                'total_frames': vg_result.get('total_frames_available', 0),
                'early_stopped': vg_result.get('early_stopped', False),
                'inference_time': vg_result['inference_time']
            },
            'gpt4': {
                'prediction': 'unsafe' if gpt_result.get('is_unsafe') else 'safe' if gpt_result.get('is_unsafe') is not None else 'error',
                'correct': gpt_result.get('is_unsafe') == video_info['label'].startswith('unsafe') if gpt_result.get('is_unsafe') is not None else None,
                'categories': gpt_result.get('unsafe_categories', []),
                'unsafe_frames': gpt_result.get('unsafe_frames', 0),
                'total_frames': gpt_result.get('total_frames', 0),
                'inference_time': gpt_result.get('inference_time', 0),
                'error': gpt_result.get('error')
            }
        }
        results.append(result)
    
    # Calculate and display statistics
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    # Video-Guard statistics
    vg_correct = sum(1 for r in results if r['video_guard']['correct'])
    vg_tp = sum(1 for r in results if r['ground_truth_is_unsafe'] and r['video_guard']['prediction'] == 'unsafe')
    vg_fp = sum(1 for r in results if not r['ground_truth_is_unsafe'] and r['video_guard']['prediction'] == 'unsafe')
    vg_tn = sum(1 for r in results if not r['ground_truth_is_unsafe'] and r['video_guard']['prediction'] == 'safe')
    vg_fn = sum(1 for r in results if r['ground_truth_is_unsafe'] and r['video_guard']['prediction'] == 'safe')
    
    print(f"\n🤖 Video-Guard Performance:")
    print(f"  Accuracy: {vg_correct}/{len(results)} ({vg_correct/len(results)*100:.1f}%)")
    print(f"  Confusion Matrix:")
    print(f"    True Positives:  {vg_tp} (correctly identified unsafe)")
    print(f"    False Positives: {vg_fp} (incorrectly marked as unsafe)")
    print(f"    True Negatives:  {vg_tn} (correctly identified safe)")
    print(f"    False Negatives: {vg_fn} (missed unsafe content)")
    
    if vg_tp + vg_fp > 0:
        precision = vg_tp / (vg_tp + vg_fp)
        print(f"  Precision: {precision:.2%}")
    if vg_tp + vg_fn > 0:
        recall = vg_tp / (vg_tp + vg_fn)
        print(f"  Recall: {recall:.2%}")
    
    avg_vg_time = np.mean([r['video_guard']['inference_time'] for r in results])
    print(f"  Avg inference time: {avg_vg_time:.2f}s")
    
    # Early stopping statistics
    early_stopped = sum(1 for r in results if r['video_guard']['early_stopped'])
    print(f"  Videos with early stopping: {early_stopped}/{len(results)}")
    
    # GPT-4 statistics (if available)
    gpt_valid = [r for r in results if r['gpt4']['correct'] is not None]
    if gpt_valid:
        gpt_correct = sum(1 for r in gpt_valid if r['gpt4']['correct'])
        gpt_tp = sum(1 for r in results if r['ground_truth_is_unsafe'] and r['gpt4']['prediction'] == 'unsafe')
        gpt_fp = sum(1 for r in results if not r['ground_truth_is_unsafe'] and r['gpt4']['prediction'] == 'unsafe')
        gpt_tn = sum(1 for r in results if not r['ground_truth_is_unsafe'] and r['gpt4']['prediction'] == 'safe')
        gpt_fn = sum(1 for r in results if r['ground_truth_is_unsafe'] and r['gpt4']['prediction'] == 'safe')
        
        print(f"\n🤖 GPT-4 Performance:")
        print(f"  Accuracy: {gpt_correct}/{len(gpt_valid)} ({gpt_correct/len(gpt_valid)*100:.1f}%)")
        print(f"  Confusion Matrix:")
        print(f"    True Positives:  {gpt_tp} (correctly identified unsafe)")
        print(f"    False Positives: {gpt_fp} (incorrectly marked as unsafe)")
        print(f"    True Negatives:  {gpt_tn} (correctly identified safe)")
        print(f"    False Negatives: {gpt_fn} (missed unsafe content)")
        
        if gpt_tp + gpt_fp > 0:
            precision = gpt_tp / (gpt_tp + gpt_fp)
            print(f"  Precision: {precision:.2%}")
        if gpt_tp + gpt_fn > 0:
            recall = gpt_tp / (gpt_tp + gpt_fn)
            print(f"  Recall: {recall:.2%}")
        
        avg_gpt_time = np.mean([r['gpt4']['inference_time'] for r in gpt_valid if r['gpt4']['inference_time'] > 0])
        print(f"  Avg inference time: {avg_gpt_time:.2f}s")
        
        gpt_errors = len(results) - len(gpt_valid)
        if gpt_errors > 0:
            print(f"  API Errors: {gpt_errors}")
    else:
        print(f"\n⚠️ GPT-4 results not available (API issues)")
    
    # Save results
    output_dir = Path("/scratch/czr/Video-Guard/datasets/comparison_results")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"enhanced_comparison_{timestamp}.json"
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_unsafe': len(unsafe_samples),
            'n_safe_regular': len(regular_safe),
            'n_safe_benign': len(benign_videos),
            'sample_fps': 'adaptive (benign: 1fps, others: 2fps)',
            'max_frames_gpt': 30,
            'early_stopping_enabled': True
        },
        'summary': {
            'total_videos': len(results),
            'video_guard': {
                'accuracy': vg_correct / len(results) if len(results) > 0 else 0,
                'precision': vg_tp / (vg_tp + vg_fp) if (vg_tp + vg_fp) > 0 else 0,
                'recall': vg_tp / (vg_tp + vg_fn) if (vg_tp + vg_fn) > 0 else 0,
                'confusion_matrix': {
                    'true_positives': vg_tp,
                    'false_positives': vg_fp,
                    'true_negatives': vg_tn,
                    'false_negatives': vg_fn
                },
                'avg_inference_time': avg_vg_time,
                'early_stopped_count': early_stopped
            }
        },
        'detailed_results': results
    }
    
    # Add GPT-4 summary if available
    if gpt_valid:
        report['summary']['gpt4'] = {
            'accuracy': gpt_correct / len(gpt_valid) if len(gpt_valid) > 0 else 0,
            'precision': gpt_tp / (gpt_tp + gpt_fp) if (gpt_tp + gpt_fp) > 0 else 0,
            'recall': gpt_tp / (gpt_tp + gpt_fn) if (gpt_tp + gpt_fn) > 0 else 0,
            'confusion_matrix': {
                'true_positives': gpt_tp,
                'false_positives': gpt_fp,
                'true_negatives': gpt_tn,
                'false_negatives': gpt_fn
            },
            'avg_inference_time': avg_gpt_time if 'avg_gpt_time' in locals() else 0,
            'errors': len(results) - len(gpt_valid)
        }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Results saved to: {output_file}")
    
    # Print errors if any
    vg_errors = [r for r in results if not r['video_guard']['correct']]
    if vg_errors:
        print(f"\n⚠️ Video-Guard Errors ({len(vg_errors)}):")
        for e in vg_errors[:5]:  # Show first 5
            source = e.get('source', 'regular')
            print(f"  {e['video_name']} ({source}): GT={e['ground_truth']} → Pred={e['video_guard']['prediction']}")
        if len(vg_errors) > 5:
            print(f"  ... and {len(vg_errors) - 5} more")
    
    return report


if __name__ == "__main__":
    print("Starting enhanced model comparison test...\n")
    
    # Run enhanced comparison
    results = run_enhanced_comparison()
    
    print("\n✅ Test completed!")