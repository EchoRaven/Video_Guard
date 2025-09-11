#!/usr/bin/env python3
"""
Visualization tool for test results
Creates comprehensive plots and analysis charts
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ResultsVisualizer:
    """Visualize Video-Guard test results"""
    
    def __init__(self, results_path: str = None):
        """Initialize with results JSON file"""
        if results_path:
            with open(results_path, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = None
    
    def create_comprehensive_report(self, results_data: Dict[str, Any] = None):
        """Create comprehensive visual report"""
        
        if results_data:
            data = results_data
        else:
            data = self.data
        
        if not data:
            print("No data to visualize")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Confusion Matrix
        ax1 = plt.subplot(2, 3, 1)
        self.plot_confusion_matrix(data['metrics']['video_level']['confusion_matrix'], ax1)
        
        # 2. Metrics Bar Chart
        ax2 = plt.subplot(2, 3, 2)
        self.plot_metrics_bar(data['metrics']['video_level'], ax2)
        
        # 3. Frame-level Accuracy Distribution
        ax3 = plt.subplot(2, 3, 3)
        self.plot_frame_accuracy_distribution(data['detailed_results'], ax3)
        
        # 4. Inference Time Distribution
        ax4 = plt.subplot(2, 3, 4)
        self.plot_inference_time_distribution(data['detailed_results'], ax4)
        
        # 5. Category Detection Frequency
        ax5 = plt.subplot(2, 3, 5)
        self.plot_category_frequency(data['metrics'].get('per_category_results', {}), ax5)
        
        # 6. Error Analysis
        ax6 = plt.subplot(2, 3, 6)
        self.plot_error_analysis(data['detailed_results'], ax6)
        
        # Add title and adjust layout
        fig.suptitle('Video-Guard Test Results Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"/scratch/czr/Video-Guard/datasets/test_results/visual_report_{timestamp}.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Visual report saved to: {output_path}")
        
        return fig
    
    def plot_confusion_matrix(self, cm_data: Dict[str, int], ax):
        """Plot confusion matrix"""
        cm = np.array([
            [cm_data['true_negatives'], cm_data['false_positives']],
            [cm_data['false_negatives'], cm_data['true_positives']]
        ])
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum() * 100
        
        # Create annotations with both count and percentage
        annotations = np.array([
            [f"{cm[i,j]}\n({cm_percent[i,j]:.1f}%)" for j in range(2)]
            for i in range(2)
        ])
        
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', ax=ax,
                   xticklabels=['Predicted Safe', 'Predicted Unsafe'],
                   yticklabels=['Actual Safe', 'Actual Unsafe'],
                   cbar_kws={'label': 'Count'})
        ax.set_title('Confusion Matrix', fontweight='bold')
        ax.set_xlabel('Prediction')
        ax.set_ylabel('Ground Truth')
    
    def plot_metrics_bar(self, metrics: Dict[str, float], ax):
        """Plot metrics as bar chart"""
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score']
        ]
        
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
        bars = ax.bar(metric_names, metric_values, color=colors)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2%}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score')
        ax.set_title('Video-Level Performance Metrics', fontweight='bold')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    
    def plot_frame_accuracy_distribution(self, results: List[Dict], ax):
        """Plot distribution of frame-level accuracies"""
        frame_accuracies = [r['frame_accuracy'] for r in results 
                           if r.get('frame_accuracy') is not None]
        
        if not frame_accuracies:
            ax.text(0.5, 0.5, 'No frame-level data available',
                   ha='center', va='center', fontsize=12)
            ax.set_title('Frame-Level Accuracy Distribution')
            return
        
        # Create histogram
        ax.hist(frame_accuracies, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(frame_accuracies), color='red', linestyle='--',
                  label=f'Mean: {np.mean(frame_accuracies):.2%}')
        ax.axvline(np.median(frame_accuracies), color='green', linestyle='--',
                  label=f'Median: {np.median(frame_accuracies):.2%}')
        
        ax.set_xlabel('Frame-Level Accuracy')
        ax.set_ylabel('Number of Videos')
        ax.set_title('Frame-Level Accuracy Distribution', fontweight='bold')
        ax.legend()
    
    def plot_inference_time_distribution(self, results: List[Dict], ax):
        """Plot inference time distribution"""
        inference_times = [r['inference_time'] for r in results 
                          if 'inference_time' in r]
        
        if not inference_times:
            ax.text(0.5, 0.5, 'No inference time data',
                   ha='center', va='center', fontsize=12)
            ax.set_title('Inference Time Distribution')
            return
        
        # Create box plot and violin plot
        parts = ax.violinplot([inference_times], positions=[1], widths=0.7,
                             showmeans=True, showmedians=True)
        
        # Color the violin plot
        for pc in parts['bodies']:
            pc.set_facecolor('#9b59b6')
            pc.set_alpha(0.7)
        
        # Add statistics text
        stats_text = f"Mean: {np.mean(inference_times):.2f}s\n"
        stats_text += f"Median: {np.median(inference_times):.2f}s\n"
        stats_text += f"Min: {np.min(inference_times):.2f}s\n"
        stats_text += f"Max: {np.max(inference_times):.2f}s"
        
        ax.text(1.5, np.mean(inference_times), stats_text,
               fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
        
        ax.set_xticks([1])
        ax.set_xticklabels(['Inference Time'])
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Inference Time Distribution', fontweight='bold')
    
    def plot_category_frequency(self, category_data: Dict[str, Any], ax):
        """Plot frequency of detected unsafe categories"""
        if not category_data:
            ax.text(0.5, 0.5, 'No category data available',
                   ha='center', va='center', fontsize=12)
            ax.set_title('Detected Unsafe Categories')
            return
        
        categories = list(category_data.keys())
        counts = [category_data[cat]['detected_count'] for cat in categories]
        
        # Sort by count
        sorted_indices = np.argsort(counts)[::-1]
        categories = [categories[i] for i in sorted_indices]
        counts = [counts[i] for i in sorted_indices]
        
        # Create horizontal bar chart
        bars = ax.barh(categories[:10], counts[:10], color='#e74c3c')  # Top 10
        
        # Add count labels
        for bar, count in zip(bars, counts[:10]):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{count}', ha='left', va='center', fontweight='bold')
        
        ax.set_xlabel('Detection Count')
        ax.set_title('Top Detected Unsafe Categories', fontweight='bold')
    
    def plot_error_analysis(self, results: List[Dict], ax):
        """Analyze and plot error types"""
        # Categorize errors
        false_positives = []
        false_negatives = []
        
        for r in results:
            if r.get('prediction') is None:
                continue
            
            if not r['correct']:
                if r['ground_truth_is_unsafe'] and not r['prediction_is_unsafe']:
                    false_negatives.append(r['video_name'])
                elif not r['ground_truth_is_unsafe'] and r['prediction_is_unsafe']:
                    false_positives.append(r['video_name'])
        
        # Create pie chart of error types
        sizes = [len(false_positives), len(false_negatives)]
        labels = [f'False Positives\n({len(false_positives)})', 
                 f'False Negatives\n({len(false_negatives)})']
        colors = ['#f39c12', '#e74c3c']
        
        if sum(sizes) > 0:
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                              autopct='%1.1f%%', startangle=90)
            
            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_fontweight('bold')
                autotext.set_color('white')
        else:
            ax.text(0.5, 0.5, 'No errors found!', 
                   ha='center', va='center', fontsize=14, fontweight='bold')
        
        ax.set_title('Error Type Distribution', fontweight='bold')
    
    def create_detailed_frame_analysis(self, results: List[Dict]) -> plt.Figure:
        """Create detailed frame-level analysis for videos with ground truth"""
        videos_with_gt = [r for r in results 
                         if r.get('frame_ground_truths') and r.get('frame_predictions')]
        
        if not videos_with_gt:
            print("No videos with frame-level ground truth")
            return None
        
        # Select up to 6 videos for detailed view
        sample_videos = videos_with_gt[:6]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, (video_result, ax) in enumerate(zip(sample_videos, axes)):
            video_name = video_result['video_name']
            gt = video_result['frame_ground_truths']
            pred = [p['predicted_unsafe'] for p in video_result['frame_predictions'][:len(gt)]]
            
            # Create comparison plot
            x = np.arange(len(gt))
            width = 0.35
            
            ax.bar(x - width/2, gt, width, label='Ground Truth', color='#2ecc71', alpha=0.7)
            ax.bar(x + width/2, pred, width, label='Prediction', color='#3498db', alpha=0.7)
            
            # Mark errors
            errors = [i for i in range(len(gt)) if gt[i] != pred[i]]
            if errors:
                ax.scatter(errors, [0.5] * len(errors), color='red', marker='x', s=100, 
                          label='Errors', zorder=5)
            
            ax.set_xlabel('Frame Index')
            ax.set_ylabel('Unsafe (1) / Safe (0)')
            ax.set_title(f'{video_name[:30]}...\nAccuracy: {video_result["frame_accuracy"]:.2%}', 
                        fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.set_ylim(-0.1, 1.1)
        
        # Hide unused subplots
        for idx in range(len(sample_videos), 6):
            axes[idx].axis('off')
        
        fig.suptitle('Frame-Level Predictions vs Ground Truth', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def save_error_report(self, results: List[Dict], output_path: str = None):
        """Save detailed error analysis report"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"/scratch/czr/Video-Guard/datasets/test_results/error_report_{timestamp}.txt"
        
        false_positives = []
        false_negatives = []
        
        for r in results:
            if r.get('prediction') is None:
                continue
            
            if not r['correct']:
                error_info = {
                    'video': r['video_name'],
                    'path': r['video_path'],
                    'ground_truth': r['ground_truth'],
                    'prediction': r['prediction'],
                    'categories_detected': r.get('unsafe_categories', [])
                }
                
                if r['ground_truth_is_unsafe'] and not r['prediction_is_unsafe']:
                    false_negatives.append(error_info)
                elif not r['ground_truth_is_unsafe'] and r['prediction_is_unsafe']:
                    false_positives.append(error_info)
        
        # Write report
        with open(output_path, 'w') as f:
            f.write("ERROR ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"FALSE POSITIVES ({len(false_positives)} videos)\n")
            f.write("-" * 40 + "\n")
            for fp in false_positives:
                f.write(f"Video: {fp['video']}\n")
                f.write(f"  Ground Truth: {fp['ground_truth']}\n")
                f.write(f"  Prediction: {fp['prediction']}\n")
                f.write(f"  Categories: {fp['categories_detected']}\n\n")
            
            f.write(f"\nFALSE NEGATIVES ({len(false_negatives)} videos)\n")
            f.write("-" * 40 + "\n")
            for fn in false_negatives:
                f.write(f"Video: {fn['video']}\n")
                f.write(f"  Ground Truth: {fn['ground_truth']}\n")
                f.write(f"  Prediction: {fn['prediction']}\n")
                f.write(f"  Path: {fn['path']}\n\n")
        
        print(f"Error report saved to: {output_path}")


def visualize_latest_results():
    """Visualize the most recent test results"""
    
    # Find latest results file
    results_dir = Path("/scratch/czr/Video-Guard/datasets/test_results")
    if not results_dir.exists():
        print("No results directory found")
        return
    
    json_files = list(results_dir.glob("video_guard_test_*.json"))
    if not json_files:
        print("No test results found")
        return
    
    latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading results from: {latest_file}")
    
    # Load and visualize
    visualizer = ResultsVisualizer(str(latest_file))
    
    # Create comprehensive report
    visualizer.create_comprehensive_report()
    
    # Create frame-level analysis if available
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    frame_fig = visualizer.create_detailed_frame_analysis(data['detailed_results'])
    if frame_fig:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame_fig.savefig(f"/scratch/czr/Video-Guard/datasets/test_results/frame_analysis_{timestamp}.png")
        print(f"Frame analysis saved")
    
    # Save error report
    visualizer.save_error_report(data['detailed_results'])
    
    plt.show()


if __name__ == "__main__":
    visualize_latest_results()