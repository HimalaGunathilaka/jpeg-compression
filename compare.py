#!/usr/bin/env python3
"""
Image comparison utility for JPEG compression results
"""
import os
from PIL import Image
import numpy as np

def compare_images(original_path="image.png", compressed_path="compressed_output.jpg"):
    """Compare original and compressed images."""
    print("=== Image Compression Comparison ===")
    
    # Check if files exist
    if not os.path.exists(original_path):
        print(f"❌ Original image not found: {original_path}")
        return
    if not os.path.exists(compressed_path):
        print(f"❌ Compressed image not found: {compressed_path}")
        return
    
    # Load images
    orig_img = Image.open(original_path)
    comp_img = Image.open(compressed_path)
    
    # Get file sizes
    orig_size = os.path.getsize(original_path)
    comp_size = os.path.getsize(compressed_path)
    
    print(f"📁 File Sizes:")
    print(f"   Original:   {orig_size:,} bytes ({orig_size/1024/1024:.2f} MB)")
    print(f"   Compressed: {comp_size:,} bytes ({comp_size/1024/1024:.2f} MB)")
    print(f"   Reduction:  {((orig_size - comp_size) / orig_size * 100):.1f}%")
    print(f"   Ratio:      {orig_size / comp_size:.2f}:1")
    print()
    
    print(f"🖼️  Image Properties:")
    print(f"   Original size:   {orig_img.size[0]}x{orig_img.size[1]} pixels")
    print(f"   Compressed size: {comp_img.size[0]}x{comp_img.size[1]} pixels")
    print(f"   Original format: {orig_img.format}")
    print(f"   Compressed format: {comp_img.format}")
    print()
    
    # Calculate quality metrics if same size
    if orig_img.size == comp_img.size:
        # Convert to numpy arrays
        orig_array = np.array(orig_img.convert('RGB'))
        comp_array = np.array(comp_img.convert('RGB'))
        
        # Calculate MSE and PSNR
        mse = np.mean((orig_array.astype(float) - comp_array.astype(float)) ** 2)
        
        if mse > 0:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
            print(f"📊 Quality Metrics:")
            print(f"   MSE:  {mse:.2f}")
            print(f"   PSNR: {psnr:.2f} dB")
            
            if psnr > 40:
                quality = "Excellent"
            elif psnr > 35:
                quality = "Very Good"
            elif psnr > 30:
                quality = "Good"
            elif psnr > 25:
                quality = "Fair"
            else:
                quality = "Poor"
            
            print(f"   Quality: {quality}")
        else:
            print(f"📊 Images are identical (MSE = 0)")
    else:
        print(f"⚠️  Images have different dimensions - cannot calculate quality metrics")
    
    print()
    print(f"✅ Compression successful!")
    print(f"   Original: {original_path}")
    print(f"   Compressed: {compressed_path}")

def show_compression_artifacts():
    """Display information about JPEG compression artifacts."""
    print("\n=== JPEG Compression Effects ===")
    print("🔍 What to look for when comparing images:")
    print("   • Blocking artifacts: 8x8 pixel block boundaries may be visible")
    print("   • Color bleeding: Colors may appear less sharp around edges")
    print("   • Loss of fine detail: Small textures and patterns may be simplified")
    print("   • Smooth gradients: May show banding instead of smooth transitions")
    print()
    print("💡 The PSNR value indicates quality:")
    print("   • >40 dB: Excellent quality (barely visible artifacts)")
    print("   • 35-40 dB: Very good quality (minor artifacts)")
    print("   • 30-35 dB: Good quality (noticeable but acceptable)")
    print("   • 25-30 dB: Fair quality (visible artifacts)")
    print("   • <25 dB: Poor quality (significant artifacts)")

if __name__ == "__main__":
    compare_images()
    show_compression_artifacts()