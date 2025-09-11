# Video-Guard Model Testing Guide

This guide explains how to test your trained Video-Guard model on real videos.

## 🚀 Quick Start

### 1. Single Video Test
Test the model on a single video with visualization:

```bash
# Quick test with default settings
python quick_model_test.py

# Or use the shell script
./run_test.sh

# Custom test with specific parameters
python test_trained_model.py \
    --checkpoint ./output_8gpu_full/checkpoint-3500 \
    --video_path /path/to/your/video.mp4 \
    --fps_sample 30 \
    --max_frames 10 \
    --output_dir ./my_test_results
```

### 2. Batch Testing
Test multiple videos from a dataset:

```bash
# Test 5 videos from dataset
./batch_test.sh

# Or run directly with custom settings
python test_trained_model.py \
    --checkpoint ./output_8gpu_full/checkpoint-3500 \
    --dataset_dir /path/to/video/dataset \
    --max_videos 10 \
    --fps_sample 30 \
    --max_frames 8 \
    --output_dir ./batch_results
```

## 📋 Test Script Features

The test script (`test_trained_model.py`) completely mimics the training process:

### Key Features:
- ✅ **Exact Training Replication**: Uses identical preprocessing, tokenization, and model inference as training
- ✅ **Dynamic Image Processing**: Supports variable aspect ratios with patch-based processing
- ✅ **Streaming Analysis**: Processes frames sequentially with accumulated context
- ✅ **Multi-label Detection**: Identifies unsafe content categories (C1-C6) and safe content
- ✅ **Shot Summarization**: Generates summaries for video shots
- ✅ **Visualization**: Creates visual analysis reports with frame-by-frame results

### Supported Labels:
- `unsafe:C1`: Sexual content
- `unsafe:C2`: Harassment/bullying
- `unsafe:C3`: Violence/harm
- `unsafe:C4`: Misinformation
- `unsafe:C5`: Illegal activities
- `unsafe:C6`: Hate speech/extremism
- `safe`: Safe content
- `continue`: Shot continuation
- `summary`: Shot summary

## 🔧 Parameters

### Model Parameters:
- `--base_model`: Base model path (default: OpenGVLab/InternVL3-8B)
- `--checkpoint`: Path to LoRA checkpoint
- `--device`: CUDA device to use (default: cuda:0)

### Video Processing:
- `--video_path`: Single video file to test
- `--dataset_dir`: Directory containing videos
- `--fps_sample`: Sample every N frames (default: 30)
- `--max_frames`: Maximum frames per video (default: 10)
- `--max_videos`: Maximum videos to test in batch (default: 5)

### Output:
- `--output_dir`: Directory for results (default: ./test_results)
- `--no_visualization`: Skip visualization generation

## 📊 Output Structure

```
test_results/
├── video_name_results.json       # Detailed analysis results
├── video_name_visualization.png  # Visual analysis report
└── test_summary.json             # Batch test summary
```

### Results JSON Format:
```json
{
  "video_path": "path/to/video.mp4",
  "total_frames_processed": 10,
  "fps_sample": 30,
  "frame_results": [
    {
      "frame_index": 0,
      "frame_number": 1,
      "num_patches": 6,
      "labels": ["safe"],
      "summary": null,
      "raw_response": "..."
    }
  ],
  "final_summary": "Complete video summary...",
  "timestamp": "2024-09-09T10:30:00"
}
```

## 🎨 Visualization

The script generates visualization images showing:
- Frame thumbnails arranged in grid
- Color-coded labels for each frame
- Frame summaries when available
- Legend with label definitions
- Final video summary

## 🔍 Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**:
   - Reduce `--max_frames` parameter
   - Use smaller batch size
   - Try different GPU with `--device cuda:X`

2. **Video Loading Error**:
   - Ensure video file exists and is readable
   - Install decord for better performance: `pip install decord`
   - Check video codec compatibility

3. **Checkpoint Not Found**:
   - Verify checkpoint path exists
   - Use absolute paths for checkpoints
   - Check training output directory

4. **Slow Processing**:
   - Reduce `--max_frames` for faster testing
   - Increase `--fps_sample` to skip more frames
   - Use GPU with more memory

## 📈 Performance Tips

1. **Optimal Settings for Quick Testing**:
   ```bash
   --fps_sample 60  # Sample every 2 seconds
   --max_frames 5   # Process 5 frames total
   ```

2. **Detailed Analysis**:
   ```bash
   --fps_sample 15  # Sample every 0.5 seconds
   --max_frames 20  # Process 20 frames
   ```

3. **Batch Processing**:
   - Process videos in parallel using multiple GPUs
   - Adjust `--max_videos` based on available memory

## 🧪 Example Commands

### Test Latest Checkpoint:
```bash
# Find latest checkpoint
LATEST=$(ls -d output*/checkpoint-* | sort -V | tail -1)

# Run test
python test_trained_model.py \
    --checkpoint $LATEST \
    --video_path sample_video.mp4
```

### Test Specific Videos:
```bash
python test_trained_model.py \
    --checkpoint ./output_8gpu_full/checkpoint-3500 \
    --video_list video1.mp4 video2.mp4 video3.mp4 \
    --output_dir ./specific_test_results
```

### High-Resolution Analysis:
```bash
python test_trained_model.py \
    --checkpoint ./output_8gpu_full/checkpoint-3500 \
    --video_path important_video.mp4 \
    --fps_sample 10 \
    --max_frames 30 \
    --output_dir ./detailed_analysis
```

## 📝 Notes

- The test script uses the exact same preprocessing pipeline as training
- LoRA weights are automatically merged for inference
- All outputs are saved in both JSON and visualization formats
- The script supports both single video and batch testing modes
- Frame indices are 0-based in the output

## 🤝 Support

For issues or questions:
1. Check the training logs for model configuration
2. Verify checkpoint compatibility with base model
3. Ensure all dependencies are installed
4. Review the error messages in console output