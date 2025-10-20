#!/usr/bin/env python3
"""
Performance benchmark for optimized JPEG compression
"""
import time
import numpy as np
from PIL import Image

# Create a test image if one doesn't exist
def create_test_image(filename="test_image.png", size=(512, 512)):
    """Create a test image with various patterns for benchmarking."""
    # Create a colorful test pattern
    width, height = size
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create gradient pattern
    for y in range(height):
        for x in range(width):
            img_array[y, x, 0] = (x * 255) // width          # Red gradient
            img_array[y, x, 1] = (y * 255) // height         # Green gradient  
            img_array[y, x, 2] = ((x + y) * 255) // (width + height)  # Blue diagonal
    
    # Add some noise and patterns
    noise = np.random.randint(0, 50, (height, width, 3))
    img_array = np.clip(img_array.astype(int) + noise, 0, 255).astype(np.uint8)
    
    # Create image and save
    img = Image.fromarray(img_array)
    img.save(filename)
    return img

def benchmark_compression(img_path="test_image.png", runs=3):
    """Benchmark the optimized JPEG compression."""
    from main import myImageClass
    
    print("=== JPEG Compression Benchmark ===")
    print(f"Running {runs} iterations for accuracy...")
    print()
    
    # Load image
    img = Image.open(img_path).convert("RGB")
    print(f"Test image: {img.size[0]}x{img.size[1]} pixels")
    
    times = []
    
    for run in range(runs):
        print(f"\n--- Run {run + 1}/{runs} ---")
        start_time = time.time()
        
        # Run compression pipeline
        my_img = myImageClass(img)
        my_img.do_YCbCr()
        my_img.pad_image()
        my_img.ycbcr_to_dct()
        my_img.count_rle()
        my_img.build_huffman_tables()
        bitstreams = my_img.encode_all()
        
        end_time = time.time()
        elapsed = end_time - start_time
        times.append(elapsed)
        
        print(f"Compression time: {elapsed:.3f} seconds")
        
        if run == 0:  # Show results for first run
            total_bits = sum(len(bitstreams[ch]) for ch in bitstreams)
            original_bits = my_img.height * my_img.width * 24
            ratio = original_bits / total_bits
            print(f"Compression ratio: {ratio:.2f}:1")
            print(f"Output size: {total_bits//8:,} bytes")
    
    print(f"\n=== Benchmark Results ===")
    print(f"Average time: {np.mean(times):.3f} ± {np.std(times):.3f} seconds")
    print(f"Best time: {min(times):.3f} seconds")
    print(f"Blocks per second: {(my_img.height * my_img.width // 64) / np.mean(times):,.0f}")

if __name__ == "__main__":
    # Create test image if it doesn't exist
    try:
        img = Image.open("test_image.png")
    except FileNotFoundError:
        print("Creating test image...")
        img = create_test_image()
        print("✓ Test image created")
    
    benchmark_compression()