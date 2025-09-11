#!/bin/bash

echo "Running Enhanced Video-Guard vs GPT-4 Comparison Test"
echo "====================================================="
echo ""
echo "Configuration:"
echo "  - 10 unsafe videos (2 fps sampling)"
echo "  - 10 safe videos: 5 benign (1 fps) + 5 regular (2 fps)"
echo "  - Early stopping enabled for Video-Guard"
echo "  - GPT-4 comparison included"
echo ""

cd /scratch/czr/Video-Guard/datasets

# Set GPU
export CUDA_VISIBLE_DEVICES=7

# Run the enhanced comparison
python3 run_model_comparison_v2.py

echo ""
echo "Test completed! Check the comparison_results folder for detailed output."