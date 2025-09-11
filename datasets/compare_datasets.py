#!/usr/bin/env python3
"""
Compare SafeBench-Live vs SafeBench-720P datasets
"""

import json
from collections import Counter
from pathlib import Path

print("Dataset Comparison: SafeBench-Live vs SafeBench-720P")
print("="*60)

# Analyze SafeBench-Live
print("\n1. SafeBench-Live (from annotations.json):")
print("-"*40)
annotations_file = '/scratch/czr/Video-Guard/datasets/safewatch_live_annotations.json'
with open(annotations_file, 'r') as f:
    annotations = json.load(f)

live_total = len(annotations)
live_safe = sum(1 for e in annotations if e.get('unsafe_category', 'safe') == 'safe')
live_unsafe = live_total - live_safe
live_categories = Counter()

for entry in annotations:
    unsafe_cat = entry.get('unsafe_category', 'safe')
    if unsafe_cat != 'safe' and ':' in unsafe_cat:
        cat = unsafe_cat.split(':')[1]
        live_categories[cat] += 1

print(f"Total videos: {live_total}")
print(f"Safe videos: {live_safe} ({live_safe*100/live_total:.1f}%)")
print(f"Unsafe videos: {live_unsafe} ({live_unsafe*100/live_total:.1f}%)")
print(f"Main unsafe category: C1 ({live_categories['C1']} videos)")

# Analyze SafeBench-720P
print("\n2. SafeBench-720P (from safewatch_streaming_corrected.jsonl):")
print("-"*40)
jsonl_file = '/scratch/czr/Video-Guard/datasets/safewatch_streaming_corrected.jsonl'

p720_total = 0
p720_safe = 0
p720_unsafe = 0
p720_categories = Counter()

with open(jsonl_file, 'r') as f:
    for line in f:
        p720_total += 1
        data = json.loads(line)
        labels = data['full_video_annotation'].get('labels', [])
        
        if labels:
            p720_unsafe += 1
            for label in labels:
                p720_categories[label] += 1
        else:
            p720_safe += 1

print(f"Total videos: {p720_total}")
print(f"Safe videos: {p720_safe} ({p720_safe*100/p720_total:.1f}%)")
print(f"Unsafe videos: {p720_unsafe} ({p720_unsafe*100/p720_total:.1f}%)")
print("Unsafe category distribution:")
for cat, count in p720_categories.most_common():
    print(f"  {cat}: {count} videos")

print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print("\nSafeBench-Live: Focused on C1 (Sexual Content) - 83% unsafe")
print("SafeBench-720P: Mostly safe content (93%) with some C2/C3 violations")
print("\nFor Video-Guard training:")
print("- If trained on SafeBench-720P: Model sees mostly SAFE videos")
print("- If trained on SafeBench-Live: Model sees mostly C1 UNSAFE videos")
print("\nThis explains why the model might not perform well on C1 detection")
print("if it was trained on SafeBench-720P data!")