#!/usr/bin/env bash
# Build, generate test data, and run the batch image filter.
set -e

echo "=== Step 0: Install Python dependencies ==="
pip3 install numpy opencv-python 2>/dev/null || pip3 install numpy 2>/dev/null || true

echo ""
echo "=== Step 1: Generate test images ==="
bash generate_test_images.sh

echo ""
echo "=== Step 2: Build ==="
make clean build

echo ""
echo "=== Step 3: Run with 5x5 box filter ==="
mkdir -p output
./batch_image_filter.exe -i sample_images -o output -w 5 -h 5 -l processing_log_5x5.csv

echo ""
echo "=== Step 4: Run with 11x11 box filter ==="
rm -rf output/*
./batch_image_filter.exe -i sample_images -o output -w 11 -h 11 -l processing_log_11x11.csv

echo ""
echo "=== Step 5: Run with 21x21 box filter ==="
rm -rf output/*
./batch_image_filter.exe -i sample_images -o output -w 21 -h 21 -l processing_log_21x21.csv

echo ""
echo "=== All runs complete ==="
echo "Log files: processing_log_5x5.csv, processing_log_11x11.csv, processing_log_21x21.csv"
echo "Sample output images in: output/"
