#!/usr/bin/env python3
"""
Easy loader for SafeWatch-Bench-Live annotations
"""

import json
from typing import List, Dict, Any, Optional
import random

class SafeWatchLiveAnnotations:
    """Loader for SafeWatch-Bench-Live annotations"""
    
    def __init__(self, json_path: str = '/scratch/czr/Video-Guard/datasets/safewatch_live_simple.json'):
        """Load annotations from JSON file"""
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # Create category index
        self.by_category = {}
        for item in self.data:
            cat = item['label']
            if cat not in self.by_category:
                self.by_category[cat] = []
            self.by_category[cat].append(item)
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all annotations"""
        return self.data
    
    def get_unsafe(self) -> List[Dict[str, Any]]:
        """Get only unsafe videos"""
        return [item for item in self.data if item['label'].startswith('unsafe')]
    
    def get_safe(self) -> List[Dict[str, Any]]:
        """Get only safe videos"""
        return self.by_category.get('safe', [])
    
    def get_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get videos by category (e.g., 'unsafe:C1', 'safe')"""
        return self.by_category.get(category, [])
    
    def get_random_unsafe(self, n: int = 1) -> List[Dict[str, Any]]:
        """Get n random unsafe videos"""
        unsafe = self.get_unsafe()
        return random.sample(unsafe, min(n, len(unsafe)))
    
    def get_random_safe(self, n: int = 1) -> List[Dict[str, Any]]:
        """Get n random safe videos"""
        safe = self.get_safe()
        return random.sample(safe, min(n, len(safe)))
    
    def get_balanced_sample(self, n_per_category: int = 5) -> List[Dict[str, Any]]:
        """Get balanced sample with n videos per category"""
        sample = []
        sample.extend(self.get_random_safe(n_per_category))
        sample.extend(self.get_random_unsafe(n_per_category))
        random.shuffle(sample)
        return sample
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        stats = {
            'total': len(self.data),
            'by_category': {}
        }
        
        for cat, items in self.by_category.items():
            stats['by_category'][cat] = {
                'count': len(items),
                'percentage': len(items) * 100.0 / len(self.data)
            }
        
        # Calculate total unsafe duration
        total_duration = 0
        for item in self.data:
            for seg in item.get('segments', []):
                total_duration += seg['end'] - seg['start']
        
        stats['total_unsafe_duration_seconds'] = total_duration
        stats['average_segments_per_video'] = sum(len(item.get('segments', [])) for item in self.data) / len(self.data)
        
        return stats
    
    def print_examples(self, n: int = 5):
        """Print example entries"""
        print("=== SafeWatch-Bench-Live Annotation Examples ===\n")
        
        # Show unsafe examples
        unsafe = self.get_random_unsafe(n)
        print(f"Unsafe Videos ({len(self.get_unsafe())} total):")
        for item in unsafe:
            print(f"  {item['path'].split('/')[-1]}")
            print(f"    Label: {item['label']}")
            if item['segments']:
                print(f"    Segments: {item['segments']}")
            print()
        
        # Show safe examples
        safe = self.get_random_safe(min(n, len(self.get_safe())))
        print(f"Safe Videos ({len(self.get_safe())} total):")
        for item in safe:
            print(f"  {item['path'].split('/')[-1]}")
            print(f"    Label: {item['label']}")
            print()


# Example usage
if __name__ == '__main__':
    # Load annotations
    loader = SafeWatchLiveAnnotations()
    
    # Print statistics
    stats = loader.get_statistics()
    print("=== Dataset Statistics ===")
    print(f"Total videos: {stats['total']}")
    print("\nCategory distribution:")
    for cat, info in stats['by_category'].items():
        print(f"  {cat}: {info['count']} ({info['percentage']:.1f}%)")
    print(f"\nTotal annotated unsafe duration: {stats['total_unsafe_duration_seconds']:.1f} seconds")
    print(f"Average segments per video: {stats['average_segments_per_video']:.2f}")
    
    print("\n" + "="*50 + "\n")
    
    # Print examples
    loader.print_examples(3)
    
    # Get specific examples for testing
    print("="*50)
    print("\nExamples for testing with test_full_video_streaming.py:\n")
    
    unsafe_samples = loader.get_random_unsafe(3)
    for i, sample in enumerate(unsafe_samples, 1):
        print(f"# Unsafe example {i}")
        print(f"video_path = '{sample['path']}'")
        if sample['segments']:
            print(f"# Unsafe segments: {sample['segments']}")
        print()
    
    safe_samples = loader.get_random_safe(1)
    for sample in safe_samples:
        print(f"# Safe example")
        print(f"video_path = '{sample['path']}'")
        print()