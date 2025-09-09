#!/usr/bin/env python3
"""
Comprehensive analysis of SafeWatch descriptions availability
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict

def extract_clip_descriptions_from_explanation(explanation):
    """Extract numbered clip descriptions from EXPLANATION field"""
    descriptions = []
    
    # Multiple patterns to catch different formats
    patterns = [
        r'(\d+)\.\s*(.+?)(?=\d+\.|$)',  # Standard numbered list
        r'Clip (\d+):\s*(.+?)(?=Clip \d+:|$)',  # "Clip N:" format
        r'•\s*(.+?)(?=•|$)',  # Bullet points
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, explanation, re.DOTALL | re.MULTILINE)
        if matches:
            for match in matches:
                if len(match) == 2:
                    desc = match[1].strip()
                else:
                    desc = match[0].strip()
                    
                # Clean up the description
                desc = desc.replace('\n', ' ').strip()
                
                if desc and len(desc) > 15 and not desc.startswith('The video'):
                    descriptions.append(desc)
            
            if descriptions:
                break
    
    return descriptions

def analyze_category(category_path):
    """Analyze one category folder"""
    full_json = category_path / 'full.json'
    
    if not full_json.exists():
        return None
    
    with open(full_json, 'r') as f:
        data = json.load(f)
    
    stats = {
        'category': category_path.name,
        'total_items': len(data),
        'full_videos': 0,
        'clips': 0,
        'full_with_desc': 0,
        'full_with_explanation': 0,
        'clips_with_desc': 0,
        'clips_empty_desc': 0,
        'extracted_from_explanation': 0
    }
    
    # Track video -> clips mapping
    video_clips = defaultdict(list)
    
    for item in data:
        video_path = item['video']
        response = item['conversations'][1]['value']
        
        if '/full/' in video_path:
            stats['full_videos'] += 1
            video_name = video_path.split('/')[-1].replace('.mp4', '')
            
            # Check DESCRIPTION
            if 'DESCRIPTION:' in response:
                desc_start = response.find('DESCRIPTION:')
                desc_end = response.find('GUARDRAIL:', desc_start)
                if desc_end == -1:
                    desc_end = response.find('\n\n', desc_start)
                description = response[desc_start+12:desc_end].strip() if desc_end != -1 else response[desc_start+12:].strip()
                
                if description and len(description) > 20:
                    stats['full_with_desc'] += 1
            
            # Check EXPLANATION for clip descriptions
            if 'EXPLANATION:' in response:
                stats['full_with_explanation'] += 1
                exp_start = response.find('EXPLANATION:')
                explanation = response[exp_start+12:].strip()
                
                clip_descs = extract_clip_descriptions_from_explanation(explanation)
                if clip_descs:
                    stats['extracted_from_explanation'] += len(clip_descs)
                    video_clips[video_name] = clip_descs
                    
        elif '/clip/' in video_path:
            stats['clips'] += 1
            
            # Check clip DESCRIPTION
            if 'DESCRIPTION:' in response:
                desc_start = response.find('DESCRIPTION:')
                desc_end = response.find('GUARDRAIL:', desc_start)
                if desc_end == -1:
                    desc_end = response.find('\n', desc_start)
                description = response[desc_start+12:desc_end].strip() if desc_end != -1 else response[desc_start+12:].strip()
                
                if not description or len(description) < 5:
                    stats['clips_empty_desc'] += 1
                elif len(description) > 20:
                    stats['clips_with_desc'] += 1
            else:
                stats['clips_empty_desc'] += 1
    
    return stats, video_clips

def main():
    base_path = Path('/scratch/czr/Video-Guard/datasets/SafeWatch-Bench-200K-720P/annotation')
    
    # Get all category folders
    categories = sorted([d for d in base_path.iterdir() if d.is_dir()])
    
    print(f"Analyzing {len(categories)} categories...")
    print("="*80)
    
    all_stats = []
    total_extracted_descs = defaultdict(list)
    
    for category_path in categories:
        result = analyze_category(category_path)
        if result:
            stats, video_clips = result
            all_stats.append(stats)
            
            # Aggregate extracted descriptions
            for video, descs in video_clips.items():
                total_extracted_descs[f"{category_path.name}/{video}"] = descs
    
    # Print summary
    total_full = sum(s['full_videos'] for s in all_stats)
    total_clips = sum(s['clips'] for s in all_stats)
    total_full_with_desc = sum(s['full_with_desc'] for s in all_stats)
    total_clips_with_desc = sum(s['clips_with_desc'] for s in all_stats)
    total_clips_empty = sum(s['clips_empty_desc'] for s in all_stats)
    total_extracted = sum(s['extracted_from_explanation'] for s in all_stats)
    
    print("\n📊 OVERALL STATISTICS")
    print("="*80)
    print(f"Total categories analyzed: {len(all_stats)}")
    print(f"Total full videos: {total_full:,}")
    print(f"Total clips: {total_clips:,}")
    print()
    print(f"Full videos with DESCRIPTION: {total_full_with_desc:,} ({total_full_with_desc/total_full*100:.1f}%)")
    print(f"Clips with DESCRIPTION: {total_clips_with_desc:,} ({total_clips_with_desc/total_clips*100:.1f}%)")
    print(f"Clips with empty DESCRIPTION: {total_clips_empty:,} ({total_clips_empty/total_clips*100:.1f}%)")
    print()
    print(f"🔍 Potential clip descriptions from EXPLANATION fields: {total_extracted:,}")
    print(f"   This could provide descriptions for ~{total_extracted/8:.0f} videos (assuming 8 clips/video)")
    
    # Show categories with best/worst coverage
    print("\n📈 TOP 5 CATEGORIES WITH BEST CLIP DESCRIPTION COVERAGE:")
    sorted_stats = sorted(all_stats, key=lambda x: x['clips_with_desc']/x['clips'] if x['clips'] > 0 else 0, reverse=True)
    for s in sorted_stats[:5]:
        if s['clips'] > 0:
            coverage = s['clips_with_desc']/s['clips']*100
            print(f"  {s['category']}: {s['clips_with_desc']}/{s['clips']} clips ({coverage:.1f}%)")
    
    print("\n📉 TOP 5 CATEGORIES WITH WORST CLIP DESCRIPTION COVERAGE:")
    for s in sorted_stats[-5:]:
        if s['clips'] > 0:
            coverage = s['clips_with_desc']/s['clips']*100
            print(f"  {s['category']}: {s['clips_with_desc']}/{s['clips']} clips ({coverage:.1f}%)")
    
    # Recommendation
    print("\n💡 RECOMMENDATIONS:")
    print("="*80)
    
    if total_extracted > total_clips_with_desc:
        print(f"✅ EXPLANATION field extraction can provide {total_extracted:,} additional descriptions")
        print(f"   This would improve coverage from {total_clips_with_desc/total_clips*100:.1f}% to {(total_clips_with_desc+total_extracted)/total_clips*100:.1f}%")
    else:
        print(f"⚠️  EXPLANATION field only provides {total_extracted:,} descriptions")
        print(f"   Current clip DESCRIPTION field already has {total_clips_with_desc:,}")
    
    print("\n🔧 SOLUTION:")
    print("1. Primary source: Use clip DESCRIPTION field when available")
    print("2. Secondary source: Extract from full video EXPLANATION field")
    print("3. Fallback: Generate generic descriptions based on safety labels")

if __name__ == "__main__":
    main()