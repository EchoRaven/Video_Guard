#!/usr/bin/env python3
"""
Simplified batch testing for Video-Guard model on SafeWatch-Bench-Live dataset
Focus on Video-Guard performance with detailed frame-level analysis
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import logging
from tqdm import tqdm
import cv2
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Add parent directory to path
sys.path.append('/scratch/czr/Video-Guard/training_testing')

# Import Video-Guard components
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from test_full_video_streaming import FullVideoStreamingTester, load_video_frames

# Import annotation loader
from load_annotations import SafeWatchLiveAnnotations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoGuardTester:
    """Test Video-Guard model on SafeWatch-Bench-Live dataset"""
    
    def __init__(
        self,
        checkpoint_path: str = "/scratch/czr/Video-Guard/training_testing/output_8gpu_full/checkpoint-3500",
        sample_fps: int = 2,
        max_frames: int = 60,
        device: str = "cuda:0"
    ):
        self.checkpoint_path = checkpoint_path
        self.sample_fps = sample_fps
        self.max_frames = max_frames
        self.device = device
        
        # Initialize Video-Guard model
        logger.info(f"Loading Video-Guard model from {checkpoint_path}")
        self.model = FullVideoStreamingTester(
            base_model_path="OpenGVLab/InternVL3-8B",
            checkpoint_path=checkpoint_path,
            device=device
        )
        
        # Load annotations
        logger.info("Loading SafeWatch-Bench-Live annotations...")
        self.annotations = SafeWatchLiveAnnotations()
        
        # Results storage
        self.results = []
        self.frame_predictions = []
        self.ground_truths = []
    
    def sample_frames_from_video(
        self, 
        video_path: str, 
        max_duration: int = 30,
        target_segments: Optional[List[Dict[str, float]]] = None
    ) -> Tuple[List[Image.Image], List[int], List[bool]]:
        """
        Sample frames from video at specified FPS
        Returns: (frames, frame_indices, ground_truth_labels)
        """
        frames = []
        frame_indices = []
        ground_truth_labels = []  # True if frame is in unsafe segment
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Cannot open video: {video_path}")
            return [], [], []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Limit duration
        duration = min(duration, max_duration)
        
        # Calculate frame interval for sampling
        frame_interval = int(fps / self.sample_fps) if fps > self.sample_fps else 1
        
        # Sample frames
        for i in range(0, int(duration * fps), frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
                frame_indices.append(i)
                
                # Check if this frame is in an unsafe segment
                current_time = i / fps if fps > 0 else 0
                is_unsafe = False
                if target_segments:
                    for seg in target_segments:
                        if seg['start'] <= current_time <= seg['end']:
                            is_unsafe = True
                            break
                ground_truth_labels.append(is_unsafe)
            
            # Limit total frames
            if len(frames) >= self.max_frames:
                break
        
        cap.release()
        
        logger.info(f"Sampled {len(frames)} frames from {video_path}")
        if target_segments:
            unsafe_count = sum(ground_truth_labels)
            logger.info(f"  Ground truth: {unsafe_count}/{len(frames)} frames in unsafe segments")
        
        return frames, frame_indices, ground_truth_labels
    
    def test_single_video(self, video_info: Dict[str, Any]) -> Dict[str, Any]:
        """Test Video-Guard on a single video"""
        video_path = video_info['path']
        ground_truth = video_info['label']
        segments = video_info.get('segments', [])
        
        logger.info(f"\nTesting: {video_path}")
        logger.info(f"  Ground truth: {ground_truth}")
        if segments:
            logger.info(f"  Unsafe segments: {segments}")
        
        # Sample frames
        frames, frame_indices, frame_ground_truths = self.sample_frames_from_video(
            video_path, 
            target_segments=segments if ground_truth.startswith('unsafe') else None
        )
        
        if not frames:
            return {
                'video_path': video_path,
                'ground_truth': ground_truth,
                'error': 'Could not load video',
                'prediction': None
            }
        
        # Run Video-Guard analysis
        start_time = time.time()
        try:
            results = self.model.analyze_full_video(frames, frame_indices, video_path=video_path)
            inference_time = time.time() - start_time
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return {
                'video_path': video_path,
                'ground_truth': ground_truth,
                'error': str(e),
                'prediction': None
            }
        
        # Extract predictions
        safety_analysis = results.get('safety_analysis', {})
        is_unsafe_predicted = safety_analysis.get('total_unsafe_frames', 0) > 0
        
        # Extract frame-level predictions
        frame_predictions = []
        for clip in results.get('clips', []):
            for frame_result in clip.get('frame_results', []):
                frame_idx = frame_result.get('frame_idx', -1)
                labels = frame_result.get('labels', [])
                is_unsafe = any('unsafe' in label for label in labels)
                frame_predictions.append({
                    'frame_idx': frame_idx,
                    'predicted_unsafe': is_unsafe,
                    'labels': labels
                })
        
        # Calculate frame-level accuracy if we have ground truth
        frame_accuracy = None
        if frame_ground_truths and frame_predictions:
            correct_frames = 0
            for i, pred in enumerate(frame_predictions[:len(frame_ground_truths)]):
                if pred['predicted_unsafe'] == frame_ground_truths[i]:
                    correct_frames += 1
            frame_accuracy = correct_frames / len(frame_ground_truths)
        
        # Prepare result
        result = {
            'video_path': video_path,
            'video_name': os.path.basename(video_path),
            'ground_truth': ground_truth,
            'ground_truth_is_unsafe': ground_truth.startswith('unsafe'),
            'prediction': 'unsafe' if is_unsafe_predicted else 'safe',
            'prediction_is_unsafe': is_unsafe_predicted,
            'correct': is_unsafe_predicted == ground_truth.startswith('unsafe'),
            'unsafe_frames_predicted': safety_analysis.get('total_unsafe_frames', 0),
            'safe_frames_predicted': safety_analysis.get('total_safe_frames', 0),
            'total_frames': len(frames),
            'unsafe_categories': safety_analysis.get('unsafe_categories_detected', []),
            'final_summary': results.get('final_response', {}).get('content', '') if isinstance(results.get('final_response'), dict) else results.get('final_response', ''),
            'inference_time': inference_time,
            'frame_accuracy': frame_accuracy,
            'frame_predictions': frame_predictions,
            'frame_ground_truths': frame_ground_truths,
            'unsafe_segments': segments
        }
        
        return result
    
    def run_batch_test(
        self, 
        n_unsafe: int = 15, 
        n_safe: int = 5,
        save_intermediate: bool = True
    ) -> Dict[str, Any]:
        """Run batch testing on sampled videos"""
        
        # Sample videos
        unsafe_videos = self.annotations.get_random_unsafe(n_unsafe)
        safe_videos = self.annotations.get_random_safe(n_safe)
        all_videos = unsafe_videos + safe_videos
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting batch test with {len(unsafe_videos)} unsafe and {len(safe_videos)} safe videos")
        logger.info(f"{'='*60}")
        
        # Test each video
        for i, video_info in enumerate(tqdm(all_videos, desc="Testing videos")):
            result = self.test_single_video(video_info)
            self.results.append(result)
            
            # Store for metrics calculation
            if result['prediction'] is not None:
                self.ground_truths.append(1 if result['ground_truth_is_unsafe'] else 0)
                self.frame_predictions.append(1 if result['prediction_is_unsafe'] else 0)
            
            # Save intermediate results
            if save_intermediate and (i + 1) % 5 == 0:
                self.save_intermediate_results(i + 1)
        
        # Calculate final metrics
        metrics = self.calculate_metrics()
        
        # Create final report
        report = self.create_report(metrics)
        
        return report
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics"""
        
        # Basic counts
        total = len(self.results)
        errors = sum(1 for r in self.results if r['prediction'] is None)
        valid = total - errors
        
        if valid == 0:
            return {'error': 'No valid predictions'}
        
        # Separate by ground truth
        unsafe_results = [r for r in self.results if r['ground_truth_is_unsafe'] and r['prediction'] is not None]
        safe_results = [r for r in self.results if not r['ground_truth_is_unsafe'] and r['prediction'] is not None]
        
        # Calculate confusion matrix components
        tp = sum(1 for r in unsafe_results if r['prediction_is_unsafe'])  # True Positives
        fn = len(unsafe_results) - tp  # False Negatives
        fp = sum(1 for r in safe_results if r['prediction_is_unsafe'])  # False Positives
        tn = len(safe_results) - fp  # True Negatives
        
        # Calculate metrics
        accuracy = (tp + tn) / valid if valid > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Frame-level metrics
        frame_accuracies = [r['frame_accuracy'] for r in self.results if r['frame_accuracy'] is not None]
        avg_frame_accuracy = np.mean(frame_accuracies) if frame_accuracies else None
        
        # Inference time statistics
        inference_times = [r['inference_time'] for r in self.results if 'inference_time' in r]
        
        metrics = {
            'total_videos': total,
            'valid_predictions': valid,
            'errors': errors,
            'video_level': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'confusion_matrix': {
                    'true_positives': tp,
                    'false_positives': fp,
                    'true_negatives': tn,
                    'false_negatives': fn
                }
            },
            'frame_level': {
                'average_accuracy': avg_frame_accuracy,
                'num_videos_with_frame_gt': len(frame_accuracies)
            },
            'inference': {
                'average_time': np.mean(inference_times) if inference_times else 0,
                'min_time': np.min(inference_times) if inference_times else 0,
                'max_time': np.max(inference_times) if inference_times else 0,
                'total_time': sum(inference_times) if inference_times else 0
            },
            'per_category_results': self.analyze_per_category()
        }
        
        return metrics
    
    def analyze_per_category(self) -> Dict[str, Any]:
        """Analyze results per unsafe category"""
        category_stats = {}
        
        for result in self.results:
            if result['prediction'] is None:
                continue
            
            for category in result.get('unsafe_categories', []):
                if category not in category_stats:
                    category_stats[category] = {
                        'detected_count': 0,
                        'videos': []
                    }
                category_stats[category]['detected_count'] += 1
                category_stats[category]['videos'].append(result['video_name'])
        
        return category_stats
    
    def create_report(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive test report"""
        
        report = {
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'checkpoint': self.checkpoint_path,
                'device': self.device,
                'sample_fps': self.sample_fps,
                'max_frames': self.max_frames
            },
            'metrics': metrics,
            'detailed_results': self.results,
            'summary': self.create_summary(metrics)
        }
        
        # Save report
        output_dir = Path("/scratch/czr/Video-Guard/datasets/test_results")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"video_guard_test_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\nReport saved to: {report_path}")
        
        # Also save a summary
        self.save_summary(metrics, output_dir / f"summary_{timestamp}.txt")
        
        return report
    
    def create_summary(self, metrics: Dict[str, Any]) -> str:
        """Create text summary of results"""
        
        summary_lines = [
            "VIDEO-GUARD TEST SUMMARY",
            "=" * 60,
            f"Checkpoint: {self.checkpoint_path}",
            f"Total videos tested: {metrics['total_videos']}",
            f"Valid predictions: {metrics['valid_predictions']}",
            f"Errors: {metrics['errors']}",
            "",
            "VIDEO-LEVEL METRICS:",
            f"  Accuracy: {metrics['video_level']['accuracy']:.2%}",
            f"  Precision: {metrics['video_level']['precision']:.2%}",
            f"  Recall: {metrics['video_level']['recall']:.2%}",
            f"  F1-Score: {metrics['video_level']['f1_score']:.2%}",
            "",
            "CONFUSION MATRIX:",
            f"  True Positives: {metrics['video_level']['confusion_matrix']['true_positives']}",
            f"  False Positives: {metrics['video_level']['confusion_matrix']['false_positives']}",
            f"  True Negatives: {metrics['video_level']['confusion_matrix']['true_negatives']}",
            f"  False Negatives: {metrics['video_level']['confusion_matrix']['false_negatives']}",
            ""
        ]
        
        if metrics['frame_level']['average_accuracy'] is not None:
            summary_lines.extend([
                "FRAME-LEVEL METRICS:",
                f"  Average accuracy: {metrics['frame_level']['average_accuracy']:.2%}",
                f"  Videos with frame GT: {metrics['frame_level']['num_videos_with_frame_gt']}",
                ""
            ])
        
        summary_lines.extend([
            "INFERENCE PERFORMANCE:",
            f"  Average time: {metrics['inference']['average_time']:.2f}s",
            f"  Min time: {metrics['inference']['min_time']:.2f}s",
            f"  Max time: {metrics['inference']['max_time']:.2f}s",
            ""
        ])
        
        if metrics['per_category_results']:
            summary_lines.append("DETECTED CATEGORIES:")
            for cat, stats in metrics['per_category_results'].items():
                summary_lines.append(f"  {cat}: {stats['detected_count']} videos")
        
        return "\n".join(summary_lines)
    
    def save_summary(self, metrics: Dict[str, Any], output_path: Path):
        """Save text summary to file"""
        summary = self.create_summary(metrics)
        with open(output_path, 'w') as f:
            f.write(summary)
        print(f"\nSummary saved to: {output_path}")
        print("\n" + summary)
    
    def save_intermediate_results(self, count: int):
        """Save intermediate results during testing"""
        output_dir = Path("/scratch/czr/Video-Guard/datasets/test_results")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        intermediate_path = output_dir / f"intermediate_{count}_videos.json"
        with open(intermediate_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Saved intermediate results ({count} videos) to {intermediate_path}")
    
    def visualize_results(self, metrics: Dict[str, Any], output_dir: Optional[Path] = None):
        """Create visualization of test results"""
        if output_dir is None:
            output_dir = Path("/scratch/czr/Video-Guard/datasets/test_results")
        
        # Create confusion matrix plot
        cm = metrics['video_level']['confusion_matrix']
        cm_array = np.array([
            [cm['true_negatives'], cm['false_positives']],
            [cm['false_negatives'], cm['true_positives']]
        ])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Safe', 'Unsafe'],
                   yticklabels=['Safe', 'Unsafe'])
        plt.title('Confusion Matrix - Video-Guard')
        plt.ylabel('Ground Truth')
        plt.xlabel('Prediction')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(output_dir / f"confusion_matrix_{timestamp}.png")
        plt.close()
        
        logger.info(f"Saved confusion matrix to {output_dir}/confusion_matrix_{timestamp}.png")


def main():
    """Main testing function"""
    
    # Initialize tester
    tester = VideoGuardTester(
        checkpoint_path="/scratch/czr/Video-Guard/training_testing/output_8gpu_full/checkpoint-3500",
        sample_fps=2,
        max_frames=60,
        device="cuda:0"
    )
    
    # Run batch test
    report = tester.run_batch_test(
        n_unsafe=15,  # Test 15 unsafe videos
        n_safe=5,     # Test 5 safe videos
        save_intermediate=True
    )
    
    # Visualize results
    tester.visualize_results(report['metrics'])
    
    print("\n" + "="*60)
    print("TESTING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()