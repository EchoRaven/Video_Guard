#!/usr/bin/env python3
"""
Parse SafeWatch original annotations to understand the description extraction
"""

import json
import re

def parse_explanation(explanation_text):
    """Extract clip descriptions from EXPLANATION text"""
    descriptions = []
    
    # Pattern to match numbered descriptions
    pattern = r'\d+\.\s*(.+?)(?=\d+\.|$)'
    matches = re.findall(pattern, explanation_text, re.DOTALL)
    
    for match in matches:
        desc = match.strip()
        if desc:
            descriptions.append(desc)
    
    return descriptions

# Read the original annotation file
annotation_file = '/scratch/czr/Video-Guard/datasets/SafeWatch-Bench-200K-720P/annotation/abuse_1/full.json'

with open(annotation_file, 'r') as f:
    data = json.load(f)

print("Analyzing SafeWatch annotation structure...")
print("="*60)

# Check first few samples
for i, item in enumerate(data[:3]):
    video_path = item['video']
    video_name = video_path.split('/')[-1]
    
    print(f"\nSample {i+1}: {video_name}")
    print("-"*40)
    
    # Get the GPT response
    gpt_response = item['conversations'][1]['value']
    
    # Parse response parts
    lines = gpt_response.split('\n')
    
    # Find DESCRIPTION
    description = ""
    guardrail = ""
    explanation = ""
    
    for j, line in enumerate(lines):
        if line.startswith("DESCRIPTION:"):
            description = line[12:].strip()
        elif line.startswith("GUARDRAIL:"):
            # Parse JSON guardrail
            guardrail_text = ""
            for k in range(j, len(lines)):
                guardrail_text += lines[k]
                if '}' in lines[k]:
                    break
            guardrail = guardrail_text[10:].strip()
        elif line.startswith("EXPLANATION:"):
            # Get everything after EXPLANATION:
            explanation = '\n'.join(lines[j:])
            explanation = explanation[12:].strip()
    
    print(f"Description: {description[:100]}...")
    
    # Parse guardrail
    if guardrail:
        # Extract the flags
        unsafe_flags = []
        if "\"C1(Sexual Content)\": true" in guardrail:
            unsafe_flags.append("C1")
        if "\"C2(Harassment & Bullying)\": true" in guardrail:
            unsafe_flags.append("C2")
        if "\"C3(Threats, Violence & Harm)\": true" in guardrail:
            unsafe_flags.append("C3")
        if "\"C4(False & Deceptive Information)\": true" in guardrail:
            unsafe_flags.append("C4")
        if "\"C5(Illegal/Regulated Activities)\": true" in guardrail:
            unsafe_flags.append("C5")
        if "\"C6(Hateful Content & Extremism)\": true" in guardrail:
            unsafe_flags.append("C6")
        
        print(f"Unsafe categories: {unsafe_flags if unsafe_flags else 'None (all false)'}")
    
    # Parse clip descriptions from explanation
    if explanation:
        clip_descriptions = parse_explanation(explanation)
        print(f"\nClip descriptions found: {len(clip_descriptions)}")
        for j, desc in enumerate(clip_descriptions[:3]):
            print(f"  Clip {j+1}: {desc[:80]}...")

print("\n" + "="*60)
print("OBSERVATION:")
print("The EXPLANATION field contains numbered clip descriptions.")
print("These should be extracted to create the clip_descriptions field.")
print("\nHowever, in safewatch_streaming_final.jsonl:")
print("- Some videos have clip_descriptions extracted correctly")
print("- Many videos have empty clip_descriptions lists")
print("\nThis suggests the extraction script had issues parsing the EXPLANATION field.")