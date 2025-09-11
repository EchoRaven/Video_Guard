#!/bin/bash
# Script to test the trained Video-Guard model

echo "🎬 Testing Video-Guard Trained Model"
echo "===================================="
echo ""

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Test parameters
CHECKPOINT_PATH="/scratch/czr/Video-Guard/training_testing/output_8gpu_full/checkpoint-500"
OUTPUT_DIR="./test_results_$(date +%Y%m%d_%H%M%S)"

# Sample video for testing
SAMPLE_VIDEO="/scratch/czr/Video-Guard/datasets/shot2story-videos/release_134k_videos/--5gVbAaF_A.9.mp4"

echo "📍 Configuration:"
echo "  - Checkpoint: $CHECKPOINT_PATH"
echo "  - Output dir: $OUTPUT_DIR"
echo "  - Test video: $SAMPLE_VIDEO"
echo ""

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "⚠️  Warning: Checkpoint not found at $CHECKPOINT_PATH"
    echo "   Using latest checkpoint from output directory..."
    
    # Find the latest checkpoint
    LATEST_CHECKPOINT=$(ls -d /scratch/czr/Video-Guard/training_testing/output*/checkpoint-* 2>/dev/null | sort -V | tail -1)
    
    if [ -z "$LATEST_CHECKPOINT" ]; then
        echo "❌ No checkpoint found. Please train the model first."
        exit 1
    fi
    
    CHECKPOINT_PATH=$LATEST_CHECKPOINT
    echo "   Found checkpoint: $CHECKPOINT_PATH"
fi

echo "🔧 Running test..."
echo ""

# Run the test
python test_trained_model.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --video_path "$SAMPLE_VIDEO" \
    --fps_sample 30 \
    --max_frames 8 \
    --output_dir "$OUTPUT_DIR" \
    --device cuda:0

echo ""
echo "✅ Test complete! Results saved to: $OUTPUT_DIR"
echo ""

# Display results if JSON file exists
RESULT_FILE="$OUTPUT_DIR/*_results.json"
if ls $RESULT_FILE 1> /dev/null 2>&1; then
    echo "📊 Quick Summary:"
    python -c "
import json
import glob

result_files = glob.glob('$OUTPUT_DIR/*_results.json')
if result_files:
    with open(result_files[0]) as f:
        data = json.load(f)
    print(f'  - Frames processed: {data.get(\"total_frames_processed\", 0)}')
    print(f'  - Frame labels:')
    for frame in data.get('frame_results', [])[:5]:
        labels = frame.get('labels', ['safe'])
        print(f'    Frame {frame.get(\"frame_number\", 0)}: {labels}')
    if len(data.get('frame_results', [])) > 5:
        print(f'    ... and {len(data.get(\"frame_results\", [])) - 5} more frames')
"
fi

echo ""
echo "🎨 Visualization saved to: $OUTPUT_DIR/*_visualization.png"