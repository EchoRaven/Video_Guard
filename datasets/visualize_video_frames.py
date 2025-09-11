#!/usr/bin/env python3
"""
Create a frame samples visualization similar to test_full_video_streaming.py
Shows actual frames from the video with safety labels
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from datetime import datetime
import cv2
from PIL import Image
import os

# Video information
video_path = "/scratch/czr/SafeWatch-Bench-Live/unsafe/aishe8864/20231012_072105_1721197962532212868.mp4"
video_name = "20231012_072105_1721197962532212868"

def extract_frame_samples(video_path, sample_times):
    """Extract frames at specific timestamps"""
    frames = []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video info: {fps:.1f} fps, {total_frames} frames, {duration:.1f}s duration")
    
    for time_sec in sample_times:
        frame_num = int(time_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append({
                'time': time_sec,
                'frame_num': frame_num,
                'image': Image.fromarray(frame)
            })
        else:
            # Create placeholder frame
            placeholder = np.ones((480, 640, 3), dtype=np.uint8) * 128
            frames.append({
                'time': time_sec,
                'frame_num': frame_num,
                'image': Image.fromarray(placeholder)
            })
    
    cap.release()
    return frames

# Sample 12 frames across the video
# Focus on the transition periods
sample_times = [
    0,   # Start - safe
    3,   # Early safe
    6,   # Mid safe
    9,   # Late safe
    12,  # Very late safe
    14,  # Just before unsafe
    15,  # UNSAFE frame
    15.5,  # Mid-UNSAFE (if video has sub-second precision)
    16,  # Just after unsafe
    17,  # Early recovery safe
    18,  # Late safe
    19   # End safe
]

# Extract frames
print(f"Extracting frames from: {video_path}")
frames_data = extract_frame_samples(video_path, sample_times)

if frames_data is None:
    print("Creating simulated frames instead...")
    # Create simulated frames with different colors for visualization
    frames_data = []
    for t in sample_times:
        # Create colored placeholder based on safety
        if 15 <= t < 16:
            # Unsafe - red tinted
            color = np.array([255, 200, 200], dtype=np.uint8)
        else:
            # Safe - green tinted
            color = np.array([200, 255, 200], dtype=np.uint8)
        
        placeholder = np.ones((480, 640, 3), dtype=np.uint8)
        placeholder[:, :] = color
        
        # Add text to show time
        from PIL import Image, ImageDraw, ImageFont
        img = Image.fromarray(placeholder)
        draw = ImageDraw.Draw(img)
        
        # Add time text
        text = f"t={t}s"
        if 15 <= t < 16:
            text += "\nUNSAFE"
            text_color = (139, 0, 0)  # Dark red
        else:
            text += "\nSAFE"
            text_color = (0, 100, 0)  # Dark green
        
        # Draw text (using default font)
        draw.text((320, 240), text, fill=text_color, anchor="mm")
        
        frames_data.append({
            'time': t,
            'frame_num': int(t * 30),  # Assume 30fps
            'image': img
        })

# Create figure with frame grid
fig = plt.figure(figsize=(20, 14))

# Main title
fig.suptitle(f'Video Frame Analysis: {video_name}\nTimeline: 15s Safe → 1s Unsafe → Safe', 
             fontsize=16, fontweight='bold')

# Create figure layout
# Use GridSpec for complex layout: 3 rows of frames + 1 row for timeline
gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.2, height_ratios=[1, 1, 1, 0.5])

# Frame grid (3 rows x 4 columns)
cols = 4
rows = 3
frame_axes = []

for row in range(rows):
    for col in range(cols):
        idx = row * cols + col
        if idx < len(frames_data):
            ax = fig.add_subplot(gs[row, col])
            frame_axes.append(ax)
            
            frame_data = frames_data[idx]
            time = frame_data['time']
            
            # Display frame
            ax.imshow(frame_data['image'])
            ax.axis('off')
            
            # Determine safety and color
            if 15 <= time < 16:
                safety = 'UNSAFE'
                color = '#e74c3c'
                bg_color = '#ffcccc'
            else:
                safety = 'SAFE'
                color = '#27ae60'
                bg_color = '#ccffcc'
            
            # Set title with time and safety
            title = f'Frame @ {time}s\n[{safety}]'
            ax.set_title(title, fontsize=11, color=color, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", 
                                 facecolor=bg_color, alpha=0.7))
            
            # Add colored border
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(4)
                spine.set_visible(True)
            
            # Special annotation for transition frames
            if time == 14:
                ax.text(0.5, -0.15, '⚠️ Last safe before unsafe', 
                       transform=ax.transAxes, ha='center', 
                       fontsize=9, color='orange', fontweight='bold')
            elif time == 15:
                ax.text(0.5, -0.15, '🚨 UNSAFE CONTENT', 
                       transform=ax.transAxes, ha='center',
                       fontsize=9, color='red', fontweight='bold')
            elif time == 16:
                ax.text(0.5, -0.15, '✅ Returns to safe', 
                       transform=ax.transAxes, ha='center',
                       fontsize=9, color='green', fontweight='bold')

# Timeline visualization at bottom (spans all 4 columns)
ax_timeline = fig.add_subplot(gs[3, :])

# Create detailed timeline
timeline_points = np.arange(0, 21, 0.5)
timeline_colors = []
timeline_heights = []

for t in timeline_points:
    if 15 <= t < 16:
        timeline_colors.append('#e74c3c')
        timeline_heights.append(1.0)
    else:
        timeline_colors.append('#27ae60')
        timeline_heights.append(0.5)

# Plot timeline bars
bars = ax_timeline.bar(timeline_points, timeline_heights, width=0.4, 
                       color=timeline_colors, edgecolor='black', linewidth=0.5)

# Highlight the unsafe zone
ax_timeline.axvspan(14.75, 16.25, alpha=0.3, color='red', zorder=0)

# Add markers for sampled frames
for frame_data in frames_data:
    t = frame_data['time']
    ax_timeline.plot(t, 1.2, 'v', markersize=8, color='blue', zorder=5)

ax_timeline.set_xlim(-0.5, 20.5)
ax_timeline.set_ylim(0, 1.5)
ax_timeline.set_xlabel('Time (seconds)', fontsize=12)
ax_timeline.set_ylabel('Safety Status', fontsize=12)
ax_timeline.set_title('Complete Video Timeline with Frame Sample Points (Blue ▼)', fontsize=12)
ax_timeline.grid(True, alpha=0.3, axis='y')

# Add legend
safe_patch = mpatches.Patch(color='#27ae60', label='Safe Period')
unsafe_patch = mpatches.Patch(color='#e74c3c', label='Unsafe Period (1s)')
sample_patch = mpatches.Patch(color='blue', label='Sampled Frames')
ax_timeline.legend(handles=[safe_patch, unsafe_patch, sample_patch], 
                  loc='upper right', fontsize=10)

# Add statistics box
stats_text = f"""Video Statistics:
• Total Duration: ~20s
• Safe Duration: 19s (95%)
• Unsafe Duration: 1s (5%)
• Unsafe Location: 15-16s
• Challenge: Detect 1-second unsafe content without false positives"""

fig.text(0.02, 0.02, stats_text, fontsize=10, 
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
        verticalalignment='bottom')

# Add expected model behavior box
model_text = f"""Expected Model Behavior:
✓ Frames 0-14: Generate 'safe' labels
✓ Frame 15: Detect unsafe:C1, start new clip
✓ Frame 16+: Return to 'safe' labels
✗ Avoid: Falsely marking safe frames as unsafe"""

fig.text(0.98, 0.02, model_text, fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        verticalalignment='bottom', horizontalalignment='right')

# Save the figure
output_path = f"./video_frames_{video_name}.png"
plt.tight_layout()
plt.savefig(output_path, dpi=100, bbox_inches='tight')
print(f"\nFrame visualization saved to: {output_path}")

# Also save individual frame montage
fig2, axes = plt.subplots(1, 3, figsize=(12, 4))
fig2.suptitle('Key Transition Frames', fontsize=14, fontweight='bold')

key_frames = [
    (14, 'Last Safe Frame'),
    (15, 'UNSAFE Frame'),
    (16, 'Back to Safe')
]

for idx, (time, label) in enumerate(key_frames):
    frame_data = frames_data[sample_times.index(time)]
    ax = axes[idx]
    ax.imshow(frame_data['image'])
    ax.axis('off')
    
    if time == 15:
        color = 'red'
        title_color = '#e74c3c'
    else:
        color = 'green'
        title_color = '#27ae60'
    
    ax.set_title(f'{label}\n(t={time}s)', fontsize=12, 
                color=title_color, fontweight='bold')
    
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(5)
        spine.set_visible(True)

plt.tight_layout()
transition_path = f"./video_transition_{video_name}.png"
plt.savefig(transition_path, dpi=100, bbox_inches='tight')
print(f"Transition frames saved to: {transition_path}")

print("\n" + "="*60)
print("FRAME VISUALIZATION COMPLETE")
print("="*60)
print(f"Video: {video_name}")
print(f"Generated 2 visualizations:")
print(f"1. Full frame grid: video_frames_{video_name}.png")
print(f"2. Key transitions: video_transition_{video_name}.png")
print("\nThese show the critical challenge:")
print("- 15 seconds of safe content")
print("- 1 second of unsafe content")
print("- Return to safe content")
print("="*60)