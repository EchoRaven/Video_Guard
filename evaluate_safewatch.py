#!/usr/bin/env python3
"""
Evaluate Video-Guard on SafeWatch-Bench dataset
Records True Positives, True Negatives, False Positives, False Negatives for each category
"""

import os
import sys
import json
import torch
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import random

# Add streaming_model to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'streaming_model'))
from streaming_inference import StreamingModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Category mapping from SafeWatch to our model's categories
CATEGORY_MAPPING = {
    'C1': 'unsafe:C1',  # Sexual content
    'C2': 'unsafe:C2',  # Harassment/bullying
    'C3': 'unsafe:C3',  # Violence/harm  
    'C4': 'unsafe:C4',  # Misinformation
    'C5': 'unsafe:C5',  # Illegal activities
    'C6': 'unsafe:C6',  # Hate speech/extremism
}

class SafeWatchEvaluator:
    def __init__(self, model_checkpoint: str, device: str = "cuda:0"):
        """Initialize evaluator with Video-Guard model"""
        self.model = StreamingModel(
            base_model_path="OpenGVLab/InternVL3-8B",
            checkpoint_path=model_checkpoint,
            device=device
        )
        
        # Initialize metrics storage
        self.metrics = {
            cat: {
                'TP': 0,  # True Positive (correctly predicted unsafe)
                'TN': 0,  # True Negative (correctly predicted safe)
                'FP': 0,  # False Positive (wrongly predicted unsafe)
                'FN': 0,  # False Negative (wrongly predicted safe)
                'videos': []
            }
            for cat in CATEGORY_MAPPING.values()
        }
        
        self.results = []
        
    def load_safewatch_annotations(self) -> List[Dict]:
        """Load SafeWatch-Bench test annotations"""
        annotations = []
        base_path = "/scratch/czr/Video-Guard/datasets/SafeWatch-Bench"
        
        # Process both genai and real categories
        for data_type in ['genai', 'real']:
            type_path = os.path.join(base_path, data_type)
            
            # Process each category C1-C6
            for cat_num in range(1, 7):
                cat_name = f'C{cat_num}'
                cat_path = os.path.join(type_path, cat_name)
                
                # Load benchmark JSON files
                if os.path.exists(cat_path):
                    for json_file in Path(cat_path).glob("*_benchmark.json"):
                        try:
                            with open(json_file, 'r') as f:
                                data = json.load(f)
                                
                            benchmark_name = json_file.stem  # e.g., "evident_benchmark"
                            is_benign = 'benign' in benchmark_name.lower()
                            
                            for item in data:
                                # Extract video file name from video_path
                                video_file = item['video_path'].split('/')[-1]
                                
                                video_path = os.path.join(
                                    base_path, 
                                    data_type, 
                                    'videos',
                                    cat_name,
                                    benchmark_name,
                                    video_file
                                )
                                
                                # Create annotation entry
                                annotation = {
                                    'video_path': video_path,
                                    'data_type': data_type,
                                    'category': cat_name,
                                    'benchmark': benchmark_name,
                                    'video_name': video_file,
                                    'video_content': item.get('video_content', ''),
                                    'violate_reason': item.get('violate_reason', ''),
                                    'labels': item.get('labels', []),
                                    'subcategories': item.get('subcategories', []),
                                    'is_benign': is_benign,
                                    'expected_labels': []
                                }
                                
                                # Map numeric labels to our category format
                                # 1->C1, 2->C2, 3->C3, 4->C4, 5->C5, 6->C6
                                for label_num in item.get('labels', []):
                                    if 1 <= label_num <= 6:
                                        annotation['expected_labels'].append(f'unsafe:C{label_num}')
                                
                                # If no labels but not benign, assume it should have the category label
                                if not annotation['expected_labels'] and not is_benign:
                                    annotation['expected_labels'] = [f'unsafe:C{cat_num}']
                                
                                annotations.append(annotation)
                                
                        except Exception as e:
                            logger.warning(f"Error loading {json_file}: {e}")
        
        logger.info(f"Loaded {len(annotations)} video annotations")
        return annotations
    
    def extract_labels_from_response(self, response: Dict) -> List[str]:
        """Extract safety labels from model response"""
        detected_labels = []
        
        # Check all clips for detected labels
        for clip in response.get('clips', []):
            for frame_result in clip.get('frame_results', []):
                for label in frame_result.get('labels', []):
                    if label.startswith('unsafe:') and label not in detected_labels:
                        detected_labels.append(label)
        
        return detected_labels
    
    def evaluate_single_video(self, annotation: Dict) -> Dict:
        """Evaluate a single video and return results"""
        video_path = annotation['video_path']
        
        # Skip if video doesn't exist
        if not os.path.exists(video_path):
            logger.warning(f"Video not found: {video_path}")
            return None
        
        try:
            # Run inference
            logger.info(f"Processing: {annotation['video_name']} ({annotation['category']}/{annotation['benchmark']})")
            
            # Temporarily reduce logging
            import logging as temp_logging
            original_level = temp_logging.getLogger('streaming_inference').level
            temp_logging.getLogger('streaming_inference').setLevel(temp_logging.WARNING)
            
            response = self.model.process_video_streaming(video_path, fps=0.25)  # Very low fps for faster evaluation
            
            # Restore logging level
            temp_logging.getLogger('streaming_inference').setLevel(original_level)
            
            # Extract detected labels
            detected_labels = self.extract_labels_from_response(response)
            expected_labels = annotation['expected_labels']
            
            # Create result entry
            result = {
                'video_name': annotation['video_name'],
                'category': annotation['category'],
                'benchmark': annotation['benchmark'],
                'is_benign': annotation['is_benign'],
                'expected_labels': expected_labels,
                'detected_labels': detected_labels,
                'data_type': annotation['data_type']
            }
            
            # Update metrics for each category
            for cat_key, cat_label in CATEGORY_MAPPING.items():
                should_detect = cat_label in expected_labels
                did_detect = cat_label in detected_labels
                
                if should_detect and did_detect:
                    self.metrics[cat_label]['TP'] += 1
                elif should_detect and not did_detect:
                    self.metrics[cat_label]['FN'] += 1
                elif not should_detect and did_detect:
                    self.metrics[cat_label]['FP'] += 1
                elif not should_detect and not did_detect:
                    self.metrics[cat_label]['TN'] += 1
                
                # Record video for this category if relevant
                if should_detect or did_detect:
                    self.metrics[cat_label]['videos'].append({
                        'video': annotation['video_name'],
                        'expected': should_detect,
                        'detected': did_detect,
                        'correct': should_detect == did_detect
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            return None
    
    def evaluate_dataset(self, annotations: List[Dict], sample_size: int = None):
        """Evaluate the full dataset or a sample"""
        # Sample if requested
        if sample_size and sample_size < len(annotations):
            annotations = random.sample(annotations, sample_size)
            logger.info(f"Evaluating sample of {sample_size} videos")
        
        # Process each video
        for annotation in tqdm(annotations, desc="Evaluating videos"):
            result = self.evaluate_single_video(annotation)
            if result:
                self.results.append(result)
    
    def calculate_metrics(self) -> Dict:
        """Calculate precision, recall, F1 for each category"""
        metrics_summary = {}
        
        for cat_label, data in self.metrics.items():
            tp = data['TP']
            tn = data['TN']
            fp = data['FP']
            fn = data['FN']
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            
            metrics_summary[cat_label] = {
                'True_Positive': tp,
                'True_Negative': tn,
                'False_Positive': fp,
                'False_Negative': fn,
                'Precision': f"{precision:.3f}",
                'Recall': f"{recall:.3f}",
                'F1_Score': f"{f1:.3f}",
                'Accuracy': f"{accuracy:.3f}",
                'Total_Videos': tp + tn + fp + fn
            }
        
        return metrics_summary
    
    def generate_report(self, output_dir: str = "./evaluation_results"):
        """Generate evaluation report"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate metrics
        metrics_summary = self.calculate_metrics()
        
        # Save metrics to JSON
        metrics_file = os.path.join(output_dir, f"metrics_{timestamp}.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        # Create detailed report
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("VIDEO-GUARD EVALUATION ON SAFEWATCH-BENCH")
        report_lines.append("="*80)
        report_lines.append(f"Timestamp: {timestamp}")
        report_lines.append(f"Total videos evaluated: {len(self.results)}")
        report_lines.append("")
        
        # Print metrics for each category
        for cat_label, metrics in metrics_summary.items():
            cat_name = cat_label.replace('unsafe:', '')
            report_lines.append(f"\n{cat_name} - {self.get_category_description(cat_name)}:")
            report_lines.append("-"*50)
            report_lines.append(f"  True Positives (Correctly detected unsafe):  {metrics['True_Positive']}")
            report_lines.append(f"  True Negatives (Correctly detected safe):    {metrics['True_Negative']}")
            report_lines.append(f"  False Positives (Wrongly detected unsafe):   {metrics['False_Positive']}")
            report_lines.append(f"  False Negatives (Wrongly detected safe):     {metrics['False_Negative']}")
            report_lines.append(f"  Precision: {metrics['Precision']}")
            report_lines.append(f"  Recall:    {metrics['Recall']}")
            report_lines.append(f"  F1 Score:  {metrics['F1_Score']}")
            report_lines.append(f"  Accuracy:  {metrics['Accuracy']}")
        
        # Overall statistics
        total_tp = sum(m['True_Positive'] for m in metrics_summary.values())
        total_tn = sum(m['True_Negative'] for m in metrics_summary.values())
        total_fp = sum(m['False_Positive'] for m in metrics_summary.values())
        total_fn = sum(m['False_Negative'] for m in metrics_summary.values())
        
        report_lines.append("\n" + "="*50)
        report_lines.append("OVERALL STATISTICS:")
        report_lines.append("-"*50)
        report_lines.append(f"  Total True Positives:  {total_tp}")
        report_lines.append(f"  Total True Negatives:  {total_tn}")
        report_lines.append(f"  Total False Positives: {total_fp}")
        report_lines.append(f"  Total False Negatives: {total_fn}")
        
        overall_accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0
        report_lines.append(f"  Overall Accuracy: {overall_accuracy:.3f}")
        
        # Save report
        report_text = "\n".join(report_lines)
        report_file = os.path.join(output_dir, f"evaluation_report_{timestamp}.txt")
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        # Also print to console
        print(report_text)
        
        # Save detailed results to CSV
        if self.results:
            df = pd.DataFrame(self.results)
            csv_file = os.path.join(output_dir, f"detailed_results_{timestamp}.csv")
            df.to_csv(csv_file, index=False)
            
        logger.info(f"\nResults saved to:")
        logger.info(f"  Metrics: {metrics_file}")
        logger.info(f"  Report: {report_file}")
        if self.results:
            logger.info(f"  Details: {csv_file}")
    
    def get_category_description(self, cat: str) -> str:
        """Get description for category"""
        descriptions = {
            'C1': 'Sexual content',
            'C2': 'Harassment/bullying',
            'C3': 'Violence/harm',
            'C4': 'Misinformation',
            'C5': 'Illegal activities',
            'C6': 'Hate speech/extremism'
        }
        return descriptions.get(cat, 'Unknown')


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Video-Guard on SafeWatch-Bench')
    parser.add_argument('--checkpoint', type=str, 
                       default='/scratch/czr/Video-Guard/training_testing/output_4gpu_bs2_16k/checkpoint-8000',
                       help='Path to model checkpoint')
    parser.add_argument('--sample', type=int, default=None,
                       help='Number of videos to sample for evaluation (None for all)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    parser.add_argument('--output', type=str, default='./evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    logger.info(f"Initializing evaluator with checkpoint: {args.checkpoint}")
    evaluator = SafeWatchEvaluator(args.checkpoint, args.device)
    
    # Load annotations
    logger.info("Loading SafeWatch-Bench annotations...")
    annotations = evaluator.load_safewatch_annotations()
    
    if not annotations:
        logger.error("No annotations found!")
        return
    
    # Run evaluation
    logger.info(f"Starting evaluation on {len(annotations) if not args.sample else args.sample} videos...")
    evaluator.evaluate_dataset(annotations, args.sample)
    
    # Generate report
    logger.info("Generating evaluation report...")
    evaluator.generate_report(args.output)
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()