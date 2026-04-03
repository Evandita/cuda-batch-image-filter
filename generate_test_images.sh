#!/usr/bin/env bash
# Generate test images for batch processing using Python + OpenCV.
# Creates 100 random-colored images of varying sizes.

set -e

OUTPUT_DIR="sample_images"
mkdir -p "$OUTPUT_DIR"

echo "Generating 100 test images in $OUTPUT_DIR ..."

python3 << 'EOF'
import numpy as np
import cv2
import os

output_dir = "sample_images"
os.makedirs(output_dir, exist_ok=True)

np.random.seed(42)

# Generate 100 images with random patterns and varying sizes
sizes = [
    (640, 480), (800, 600), (1024, 768), (320, 240), (512, 512),
    (1280, 720), (256, 256), (400, 300), (600, 400), (720, 480)
]

for i in range(100):
    w, h = sizes[i % len(sizes)]
    # Create image with random geometric patterns
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Random background color
    bg_color = np.random.randint(0, 256, 3).tolist()
    img[:] = bg_color

    # Add random rectangles
    for _ in range(np.random.randint(5, 20)):
        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
        x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
        color = np.random.randint(0, 256, 3).tolist()
        thickness = np.random.choice([-1, 1, 2, 3])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # Add random circles
    for _ in range(np.random.randint(3, 10)):
        cx, cy = np.random.randint(0, w), np.random.randint(0, h)
        radius = np.random.randint(10, min(w, h) // 4)
        color = np.random.randint(0, 256, 3).tolist()
        thickness = np.random.choice([-1, 1, 2])
        cv2.circle(img, (cx, cy), radius, color, thickness)

    # Add some noise
    noise = np.random.randint(0, 30, (h, w, 3), dtype=np.uint8)
    img = cv2.add(img, noise)

    filename = os.path.join(output_dir, f"test_image_{i:03d}.png")
    cv2.imwrite(filename, img)

print(f"Generated 100 test images in {output_dir}/")
EOF

echo "Done. $(ls -1 $OUTPUT_DIR/*.png 2>/dev/null | wc -l) images created."
