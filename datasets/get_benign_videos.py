#!/usr/bin/env python3
"""
Helper to get benign videos from SafeWatch-Live dataset
"""

import os
import random
from pathlib import Path
from typing import List, Dict, Any

def get_benign_videos(base_path: str = "/scratch/czr/SafeWatch-Bench-Live/benign", 
                      n_videos: int = 10) -> List[Dict[str, Any]]:
    """
    Get random benign videos from SafeWatch-Live
    
    Args:
        base_path: Path to benign folder
        n_videos: Number of videos to sample
    
    Returns:
        List of video info dicts with path and label
    """
    all_videos = []
    
    # Iterate through user folders in benign directory
    for user_folder in os.listdir(base_path):
        user_path = os.path.join(base_path, user_folder)
        if os.path.isdir(user_path):
            # Get all mp4 files in this user folder
            for video_file in os.listdir(user_path):
                if video_file.endswith('.mp4'):
                    video_path = os.path.join(user_path, video_file)
                    all_videos.append({
                        'path': video_path,
                        'label': 'safe',  # All benign videos are safe
                        'source': 'benign',
                        'segments': []  # No unsafe segments
                    })
    
    # Sample n_videos randomly
    if len(all_videos) > n_videos:
        selected = random.sample(all_videos, n_videos)
    else:
        selected = all_videos
    
    return selected


def get_mixed_safe_videos(n_benign: int = 5, n_regular_safe: int = 5) -> List[Dict[str, Any]]:
    """
    Get a mix of benign videos and regular safe videos from annotations
    
    Args:
        n_benign: Number of benign videos to include
        n_regular_safe: Number of regular safe videos from annotations
    
    Returns:
        Combined list of safe videos
    """
    from load_annotations import SafeWatchLiveAnnotations
    
    # Get benign videos
    benign_videos = get_benign_videos(n_videos=n_benign)
    
    # Get regular safe videos from annotations
    annotations = SafeWatchLiveAnnotations()
    regular_safe = annotations.get_random_safe(n_regular_safe)
    
    # Combine and shuffle
    all_safe = benign_videos + regular_safe
    random.shuffle(all_safe)
    
    return all_safe


if __name__ == "__main__":
    # Test the function
    print("Testing benign video loader...")
    
    # Get some benign videos
    benign = get_benign_videos(n_videos=5)
    print(f"\nFound {len(benign)} benign videos:")
    for video in benign:
        print(f"  - {Path(video['path']).name}")
        print(f"    Path: {video['path']}")
        print(f"    Label: {video['label']}")
    
    # Get mixed safe videos
    print("\nTesting mixed safe video loader...")
    mixed = get_mixed_safe_videos(n_benign=3, n_regular_safe=2)
    print(f"\nSelected {len(mixed)} mixed safe videos:")
    for video in mixed:
        source = video.get('source', 'regular')
        print(f"  - {Path(video['path']).name} (source: {source})")