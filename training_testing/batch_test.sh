#!/bin/bash
# Batch testing script for multiple videos

echo "🎬 Batch Testing Video-Guard Model"
echo "===================================="
echo ""

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Configuration
CHECKPOINT_PATH="/scratch/czr/Video-Guard/training_testing/output_8gpu_full/checkpoint-500"
DATASET_DIR="/scratch/czr/Video-Guard/datasets/shot2story-videos/release_134k_videos"
OUTPUT_DIR="./batch_test_results_$(date +%Y%m%d_%H%M%S)"

echo "📍 Configuration:"
echo "  - Checkpoint: $CHECKPOINT_PATH"
echo "  - Dataset dir: $DATASET_DIR"
echo "  - Output dir: $OUTPUT_DIR"
echo ""

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "⚠️  Warning: Checkpoint not found at $CHECKPOINT_PATH"
    echo "   Looking for latest checkpoint..."
    
    LATEST_CHECKPOINT=$(ls -d /scratch/czr/Video-Guard/training_testing/output*/checkpoint-* 2>/dev/null | sort -V | tail -1)
    
    if [ -z "$LATEST_CHECKPOINT" ]; then
        echo "❌ No checkpoint found. Please train the model first."
        exit 1
    fi
    
    CHECKPOINT_PATH=$LATEST_CHECKPOINT
    echo "   Using: $CHECKPOINT_PATH"
fi

echo "🔧 Starting batch test..."
echo ""

# Run batch test on multiple videos
python test_trained_model.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --dataset_dir "$DATASET_DIR" \
    --fps_sample 30 \
    --max_frames 6 \
    --max_videos 5 \
    --output_dir "$OUTPUT_DIR" \
    --device cuda:0

echo ""
echo "✅ Batch test complete!"
echo ""

# Generate summary report
echo "📊 Test Summary:"
python -c "
import json
import glob
import os

summary_file = '$OUTPUT_DIR/test_summary.json'
if os.path.exists(summary_file):
    with open(summary_file) as f:
        summary = json.load(f)
    
    print(f'  Total videos tested: {summary.get(\"total_videos\", 0)}')
    print(f'  Successful: {summary.get(\"successful\", 0)}')
    print(f'  Failed: {summary.get(\"failed\", 0)}')
    
    print('\n  Individual results:')
    for result in summary.get('results', [])[:5]:
        if 'error' not in result:
            video_name = os.path.basename(result.get('video_path', 'Unknown'))
            frames = result.get('total_frames_processed', 0)
            print(f'    - {video_name}: {frames} frames processed')
else:
    print('  Summary file not found')
"

echo ""
echo "📁 All results saved to: $OUTPUT_DIR"
echo "🎨 Visualizations: $OUTPUT_DIR/*_visualization.png"