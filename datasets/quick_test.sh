#!/bin/bash

# Quick test script for Video-Guard vs GPT-4 comparison on SafeWatch-Live dataset

echo "============================================"
echo "Video-Guard vs GPT-4 Comparison Test"
echo "Dataset: SafeWatch-Bench-Live"
echo "============================================"

# Set GPU
export CUDA_VISIBLE_DEVICES=7

# Navigate to the correct directory
cd /scratch/czr/Video-Guard/datasets

# Run the comparison test
echo ""
echo "Starting test with 5 unsafe and 2 safe videos..."
echo ""

python3 run_model_comparison.py

echo ""
echo "Test completed! Check results in ./comparison_results/"
echo ""

# Show latest results
echo "Latest results:"
ls -la comparison_results/*.json 2>/dev/null | tail -1