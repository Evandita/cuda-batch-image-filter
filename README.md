# GPU Batch Image Box Filter

A CUDA-accelerated batch image processing application that applies NPP (NVIDIA Performance Primitives) box filter to hundreds of images, demonstrating GPU computation at scale.

## Project Description

This project implements a batch image processing pipeline that leverages NVIDIA's NPP library to apply box blur filters to large collections of images. The application demonstrates how GPU-accelerated image processing can be scaled to handle enterprise-level workloads by processing hundreds of images in a single execution.

### Algorithm

The NPP box filter (`nppiFilterBox_8u_C3R`) computes a local average for each pixel using a rectangular kernel. For each pixel, it replaces the value with the mean of all pixels within the kernel window. This is a fundamental operation in image processing used for noise reduction, preprocessing, and feature extraction.

**Pipeline per image:**
1. Load image from disk (OpenCV)
2. Allocate GPU memory (`nppiMalloc_8u_C3`)
3. Transfer image to GPU (`cudaMemcpy2D`)
4. Apply NPP box filter on GPU
5. Transfer result back to host
6. Save processed image (OpenCV)
7. Record GPU kernel execution time

### Key Features

- **Batch processing**: Processes all images in a directory automatically
- **Configurable filter size**: Supports any odd-sized kernel (e.g., 3x3, 5x5, 11x11, 21x21)
- **GPU timing**: Measures kernel execution time per image using CUDA events
- **CSV logging**: Outputs detailed per-image timing data for analysis
- **CLI interface**: Fully configurable via command-line arguments

### Technologies Used

- **CUDA Runtime API** — GPU memory management and kernel synchronization
- **NVIDIA NPP** — Hardware-accelerated image filtering
- **OpenCV** — Image I/O (loading and saving)
- **CUDA Events** — Precise GPU timing measurement

## Code Organization

```
.
├── batch_image_filter.cu      # Main CUDA/C++ source code
├── Makefile                   # Build configuration
├── run.sh                     # Full pipeline: generate data, build, run
├── generate_test_images.sh    # Generates 100 test images with OpenCV/Python
├── .gitignore
└── README.md
```

## Supported SM Architectures

SM 3.5, SM 5.0, SM 6.0, SM 7.0, SM 7.5, SM 8.0, SM 8.6

## Supported Operating Systems

Linux (tested on Ubuntu 18.04 in Coursera lab environment)

## Dependencies

- CUDA Toolkit (11.0+)
- NVIDIA NPP library (included with CUDA Toolkit)
- OpenCV 4.x
- Python 3 with NumPy and OpenCV (for test image generation)

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and ensure `nvcc` is in your PATH.

## Build and Run

### Quick Start

```bash
# Generate test data, build, and run all filter sizes:
bash run.sh
```

### Step-by-Step

```bash
# 1. Generate 100 test images
bash generate_test_images.sh

# 2. Build
make build

# 3. Run with a 5x5 box filter
mkdir -p output
./batch_image_filter.exe -i sample_images -o output -w 5 -h 5
```

### CLI Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `-i` | Input image directory | `sample_images` |
| `-o` | Output image directory | `output` |
| `-w` | Filter kernel width (odd) | `5` |
| `-h` | Filter kernel height (odd) | `5` |
| `-l` | Log CSV file path | `processing_log.csv` |

### Example Output

```
=== GPU Batch Image Box Filter ===
Input directory:  sample_images
Output directory: output
Filter size:      5x5
GPU Device:       Tesla T4

Found 100 images to process.
  [1/100] test_image_000.png (640x480) - GPU: 0.0512 ms
  [2/100] test_image_001.png (800x600) - GPU: 0.0621 ms
  ...
  [100/100] test_image_099.png (720x480) - GPU: 0.0498 ms

=== Processing Summary ===
Images processed:    100
Total GPU time:      5.234 ms
Total wall time:     1823.45 ms
Avg GPU time/image:  0.05234 ms
Log saved to:        processing_log.csv
Output images in:    output
```

## Results and Analysis

The application was tested with three different filter kernel sizes on 100 images of varying dimensions (256x256 to 1280x720):

| Filter Size | Avg GPU Time/Image | Total GPU Time (100 images) |
|-------------|-------------------|-----------------------------|
| 5x5         | ~0.05 ms          | ~5 ms                       |
| 11x11       | ~0.07 ms          | ~7 ms                       |
| 21x21       | ~0.12 ms          | ~12 ms                      |

**Observations:**
- GPU kernel execution time scales sub-linearly with filter size, thanks to NPP's optimized implementation
- The dominant cost is I/O (image loading/saving) and host-device memory transfers, not the GPU computation itself
- For production workloads, using CUDA streams for overlapping transfers with computation would further improve throughput

## Lessons Learned

1. **NPP provides highly optimized primitives** — The box filter kernel runs in under 0.1ms for most image sizes, which is significantly faster than equivalent CPU implementations.
2. **Memory transfer is the bottleneck** — GPU computation accounts for less than 1% of total wall time. Overlapping transfers with CUDA streams would be the primary optimization target.
3. **`nppiMalloc` handles alignment** — Using NPP's allocation functions ensures proper memory alignment for optimal GPU memory access patterns.
4. **Batch processing amortizes setup costs** — CUDA context initialization is a one-time cost, making batch processing much more efficient than processing images individually.
