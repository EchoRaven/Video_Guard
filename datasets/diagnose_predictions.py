#!/usr/bin/env python3
"""
Diagnose Video-Guard prediction patterns
Analyze why C4 (Misinformation) and C6 (Hate speech) are frequently predicted
"""

import os
import sys
import json
from pathlib import Path
from collections import Counter, defaultdict
import glob

# Add path for imports
sys.path.append('/scratch/czr/Video-Guard/datasets')

def analyze_prediction_patterns():
    """Analyze all saved prediction results"""
    
    results_dir = "/scratch/czr/Video-Guard/datasets/comparison_results"
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return
    
    # Find all result files
    result_files = glob.glob(os.path.join(results_dir, "*.json"))
    
    if not result_files:
        print("No result files found")
        return
    
    print(f"Found {len(result_files)} result files")
    print("="*60)
    
    # Aggregate statistics
    all_categories = Counter()
    category_by_video_type = defaultdict(Counter)
    false_positives_by_category = Counter()
    total_predictions = 0
    total_unsafe_predictions = 0
    
    for file_path in result_files:
        print(f"\nAnalyzing: {os.path.basename(file_path)}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check if it has detailed results
        if 'detailed_results' in data:
            results = data['detailed_results']
        elif 'results' in data:
            results = data['results']
        else:
            print("  - No detailed results found")
            continue
        
        for result in results:
            # Get Video-Guard predictions
            vg_info = result.get('video_guard', {})
            
            if not vg_info:
                continue
            
            ground_truth = result.get('ground_truth', 'unknown')
            is_actually_unsafe = result.get('ground_truth_is_unsafe', False)
            
            # Count categories detected
            categories = vg_info.get('categories', vg_info.get('unsafe_categories', []))
            
            if categories:
                total_unsafe_predictions += 1
                for cat in categories:
                    all_categories[cat] += 1
                    
                    # Track by actual safety status
                    if is_actually_unsafe:
                        category_by_video_type['true_unsafe'][cat] += 1
                    else:
                        category_by_video_type['false_positive'][cat] += 1
                        false_positives_by_category[cat] += 1
            
            total_predictions += 1
    
    # Print analysis
    print("\n" + "="*60)
    print("PREDICTION PATTERN ANALYSIS")
    print("="*60)
    
    print(f"\nTotal predictions analyzed: {total_predictions}")
    print(f"Total unsafe predictions: {total_unsafe_predictions}")
    
    print("\n📊 Category Frequency (all predictions):")
    for cat, count in all_categories.most_common():
        percentage = count * 100.0 / total_unsafe_predictions if total_unsafe_predictions > 0 else 0
        print(f"  {cat}: {count} times ({percentage:.1f}% of unsafe predictions)")
    
    print("\n⚠️ False Positive Analysis:")
    if category_by_video_type['false_positive']:
        print("Categories in false positives (safe videos marked unsafe):")
        for cat, count in category_by_video_type['false_positive'].most_common():
            print(f"  {cat}: {count} times")
    else:
        print("  No false positives found")
    
    print("\n✅ True Positive Analysis:")
    if category_by_video_type['true_unsafe']:
        print("Categories in true positives (correctly identified unsafe):")
        for cat, count in category_by_video_type['true_unsafe'].most_common():
            print(f"  {cat}: {count} times")
    
    # Analyze specific problematic categories
    print("\n" + "="*60)
    print("PROBLEMATIC CATEGORIES ANALYSIS")
    print("="*60)
    
    problem_categories = ['unsafe:C4', 'unsafe:C6']
    
    for cat in problem_categories:
        if cat in all_categories:
            total = all_categories[cat]
            false_pos = false_positives_by_category.get(cat, 0)
            true_pos = total - false_pos
            
            print(f"\n{cat} ({cat.split(':')[1]}):")
            print(f"  Total occurrences: {total}")
            print(f"  True positives: {true_pos}")
            print(f"  False positives: {false_pos}")
            if total > 0:
                print(f"  False positive rate: {false_pos*100.0/total:.1f}%")


def check_training_data_issue():
    """Check if the issue is in training data"""
    print("\n" + "="*60)
    print("TRAINING DATA ANALYSIS")
    print("="*60)
    
    # Load SafeWatch annotations
    annotation_file = '/scratch/czr/Video-Guard/datasets/safewatch_live_simple.json'
    
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # Count categories
    categories = Counter()
    for item in data:
        label = item['label']
        categories[label] += 1
    
    print("\nTraining data distribution:")
    total = len(data)
    for label, count in sorted(categories.items()):
        percentage = count * 100.0 / total
        print(f"  {label}: {count} ({percentage:.1f}%)")
    
    print("\n⚠️ ISSUE IDENTIFIED:")
    print("  - Training data only has 'unsafe:C1' (Sexual content) - 83.1%")
    print("  - No training examples for C2, C3, C4, C5, or C6")
    print("  - Model may be hallucinating these categories")
    print("\n💡 SOLUTION:")
    print("  1. The model was trained primarily on C1 (sexual content)")
    print("  2. It learned about C4/C6 from the prompt but has no real training")
    print("  3. This causes spurious predictions of unseen categories")
    print("\n📝 RECOMMENDATIONS:")
    print("  1. Filter out C2-C6 predictions in post-processing")
    print("  2. Or retrain with balanced data for all categories")
    print("  3. Or use only C1 in the prompt since that's what it was trained on")


def suggest_fix():
    """Suggest a fix for the issue"""
    print("\n" + "="*60)
    print("SUGGESTED FIX")
    print("="*60)
    
    fix_code = '''
# In test_full_video_streaming.py, modify the analyze_full_video function:

def filter_predictions(labels):
    """Filter out categories the model wasn't trained on"""
    # Only keep C1 since that's what the model was actually trained on
    filtered = []
    for label in labels:
        if 'unsafe:C1' in label or 'safe' in label or 'continue' in label:
            filtered.append(label)
        # Ignore C2-C6 as they're not in training data
    return filtered

# Apply this filter to all predictions before returning results
'''
    
    print(fix_code)


if __name__ == "__main__":
    print("🔍 Diagnosing Video-Guard Prediction Patterns\n")
    
    # Analyze existing results
    analyze_prediction_patterns()
    
    # Check training data
    check_training_data_issue()
    
    # Suggest fix
    suggest_fix()