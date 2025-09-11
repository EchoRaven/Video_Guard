#!/usr/bin/env python3
"""
Check how SafeWatch guardrail info should be used in responses
"""

import json

# Load a few SafeWatch examples
print("Checking SafeWatch data format and guardrail usage...\n")

with open('/scratch/czr/Video-Guard/datasets/safewatch_streaming_corrected.jsonl', 'r') as f:
    for i, line in enumerate(f):
        if i >= 5:  # Check first 5 examples
            break
        
        data = json.loads(line)
        video_path = data['full_video_path'].split('/')[-1]
        
        # Get guardrail info
        guardrail = data['full_video_annotation'].get('guardrail', {})
        description = data['full_video_annotation'].get('description', '')
        
        # Check which categories are flagged
        flagged_categories = []
        for category, is_flagged in guardrail.items():
            if is_flagged:
                # Extract category code (C1, C2, etc.)
                if 'C1' in category:
                    flagged_categories.append('C1')
                elif 'C2' in category:
                    flagged_categories.append('C2')
                elif 'C3' in category:
                    flagged_categories.append('C3')
                elif 'C4' in category:
                    flagged_categories.append('C4')
                elif 'C5' in category:
                    flagged_categories.append('C5')
                elif 'C6' in category:
                    flagged_categories.append('C6')
        
        print(f"Video {i+1}: {video_path}")
        print(f"  Flagged categories: {flagged_categories if flagged_categories else 'None (SAFE)'}")
        print(f"  Description: {description[:100]}...")
        
        # Check clip labels
        clip_labels = data.get('clip_video_labels', [])
        print(f"  Clip labels: {clip_labels}")
        
        # Expected final response format
        if flagged_categories:
            print(f"  Expected response should include:")
            print(f"    - Video description")
            print(f"    - Safety assessment mentioning categories: {flagged_categories}")
        else:
            print(f"  Expected response should include:")
            print(f"    - Video description")
            print(f"    - Confirmation that content is safe")
        
        print()

print("\n" + "="*60)
print("ANALYSIS:")
print("="*60)
print("\nSafeWatch data includes 'guardrail' field with safety categories:")
print("- C1: Sexual Content")
print("- C2: Harassment & Bullying")
print("- C3: Threats, Violence & Harm")
print("- C4: False & Deceptive Information")
print("- C5: Illegal/Regulated Activities")
print("- C6: Hateful Content & Extremism")
print("\nThe final response should:")
print("1. Include the video description")
print("2. Include safety assessment based on guardrail flags")
print("3. For Shot2Story (no guardrail), just use description")
print("\nThis ensures consistent training on safety detection!")