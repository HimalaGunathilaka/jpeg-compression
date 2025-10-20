import numpy as np
from PIL import Image
import math
import heapq
import itertools
import os

# ---------- helper functions ----------
def huf_symbol(run, size):
    """JPEG symbol: high 4 bits = run, low 4 bits = size"""
    return (run << 4) | size

def value_size(val):
    """Number of amplitude bits needed for 'val' (0 -> 0)."""
    v = abs(int(val))
    if v == 0:
        return 0
    return v.bit_length()  # floor(log2(v)) + 1

def amplitude_bits(val):
    """
    Return (size, bits_string) for the amplitude following JPEG rules.
    For positive: normal binary of val in 'size' bits.
    For negative: encoded as val + (2^size - 1) (i.e. inverted magnitude).
    """
    size = value_size(val)
    if size == 0:
        return size, ""
    if val > 0:
        bits = format(val, 'b').rjust(size, '0')
    else:
        # For negative v, JPEG stores (v + (2^size - 1)) in size bits
        magnitude = (1 << size) + val - 1  # since val is negative
        bits = format(magnitude, 'b').rjust(size, '0')
    return size, bits

# ---------- Huffman building ----------
def build_huffman_codes(freq_dict):
    """
    freq_dict: {symbol: frequency}
    returns codes: {symbol: '0101...'}
    """
    # Edge-case: if only one symbol, give it code '0'
    if len(freq_dict) == 0:
        return {}
    if len(freq_dict) == 1:
        only_symbol = next(iter(freq_dict))
        return {only_symbol: '0'}

    counter = itertools.count()
    heap = []
    for sym, freq in freq_dict.items():
        heapq.heappush(heap, (freq, next(counter), {'sym': sym}))

    # build tree
    while len(heap) > 1:
        f1, _, node1 = heapq.heappop(heap)
        f2, _, node2 = heapq.heappop(heap)
        merged = {'left': node1, 'right': node2}
        heapq.heappush(heap, (f1 + f2, next(counter), merged))

    # now heap[0] is root
    _, _, root = heap[0]

    codes = {}
    def walk(node, prefix):
        if 'sym' in node:
            codes[node['sym']] = prefix or '0'  # avoid empty string
            return
        walk(node['left'], prefix + '0')
        walk(node['right'], prefix + '1')

    walk(root, "")
    return codes


# ---------- Encoder integration with your class ----------
class myImageClass():
    def __init__(self, img):
        img_np = np.array(img, dtype=np.uint8)  # Ensure uint8 for memory efficiency
        self.img_np = img_np
        self.height, self.width = img_np.shape[:2]
        
        # Use int16 instead of int32 for RGB values (0-255 range)
        self.R = img_np[:,:,0].astype(np.int16)
        self.G = img_np[:,:,1].astype(np.int16)  
        self.B = img_np[:,:,2].astype(np.int16)
        
        # Quantization table as uint8 (all values < 256)
        self.quat_table = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68,109,103, 77],
            [24, 35, 55, 64, 81,104,113, 92],
            [49, 64, 78, 87,103,121,120,101],
            [72, 92, 95, 98,112,100,103, 99]
        ], dtype=np.uint8)
        
        # Pre-allocate RLE lists with reasonable initial capacity
        estimated_size = (self.height * self.width) // 64 * 10  # Rough estimate
        self.Y_rle = []
        self.Cr_rle = []
        self.Cb_rle = []
        
        # Use defaultdict for frequency counters (faster than dict.get())
        from collections import defaultdict
        self.Y_rle_count = defaultdict(int)
        self.Cb_rle_count = defaultdict(int)
        self.Cr_rle_count = defaultdict(int)
        
        # will hold codes (after building)
        self.Y_codes = {}
        self.Cb_codes = {}
        self.Cr_codes = {}
        
    # ---------- color conversion ----------
    def do_YCbCr(self):
        print("Stage 1/5: Converting RGB to YCbCr color space...")
        # Optimized vectorized conversion using matrix multiplication
        # Convert to float32 for faster computation, then back to int32
        rgb_stack = np.stack([self.R, self.G, self.B], axis=-1).astype(np.float32)
        
        # YCbCr transformation matrix (ITU-R BT.601)
        # Optimized coefficients for integer arithmetic approximation
        transform_matrix = np.array([
            [0.299,    0.587,    0.114   ],  # Y
            [-0.16874, -0.33126,  0.5     ],  # Cb  
            [0.5,      -0.41869, -0.08131 ]   # Cr
        ], dtype=np.float32)
        
        # Apply transformation: [H, W, 3] @ [3, 3] -> [H, W, 3]
        ycbcr = rgb_stack @ transform_matrix.T
        
        # Add offsets and convert to int32
        self.Y = (ycbcr[:, :, 0] + 16).astype(np.int32)
        self.Cb = (ycbcr[:, :, 1] + 128).astype(np.int32)
        self.Cr = (ycbcr[:, :, 2] + 128).astype(np.int32)
        
        print("✓ YCbCr conversion completed (20%)")
        print()
        
    # ---------- padding ----------
    def pad_image(self):
        print("Stage 2/5: Padding image to 8x8 block boundaries...")
        # Store original dimensions before padding
        self.original_height = self.height
        self.original_width = self.width
        
        original_size = f"{self.height}x{self.width}"
        pad_h = (8 - self.height % 8) % 8
        pad_w = (8 - self.width % 8) % 8
        
        self.height += pad_h
        self.width += pad_w

        self.Y  = np.pad(self.Y,  ((0, pad_h), (0, pad_w)), mode='edge')
        self.Cb = np.pad(self.Cb, ((0, pad_h), (0, pad_w)), mode='edge')
        self.Cr = np.pad(self.Cr, ((0, pad_h), (0, pad_w)), mode='edge')
        print(f"✓ Image padded from {original_size} to {self.height}x{self.width} (40%)")
        print()
        
    def mask(self,i=0, j=0):
        row_start = i*8
        col_start = j*8
        kernel_Y = self.Y[row_start:row_start+8, col_start:col_start+8]
        kernel_Cb = self.Cb[row_start:row_start+8, col_start:col_start+8]
        kernel_Cr = self.Cr[row_start:row_start+8, col_start:col_start+8]
        return (kernel_Y, kernel_Cb, kernel_Cr)
    
     # ---------- DCT, quantize, zigzag, RLE ----------
    def ycbcr_to_dct(self):
        print("Stage 3/5: Applying DCT, quantization, and RLE encoding...")
        h_blocks = self.height // 8
        w_blocks = self.width // 8
        total_blocks = h_blocks * w_blocks
        
        # Precompute quantization table as float32 for faster division
        quat_table_float = self.quat_table.astype(np.float32)
        
        # Progress reporting optimization - update less frequently for large images
        progress_step = max(1, total_blocks // 20)  # Update every 5% instead of 10%
        
        processed_blocks = 0
        for row in range(h_blocks):
            # Process entire row of blocks for better cache locality
            row_start = row * 8
            row_end = row_start + 8
            
            for col in range(w_blocks):
                col_start = col * 8
                col_end = col_start + 8
                
                # Direct array slicing instead of mask function
                k_y = self.Y[row_start:row_end, col_start:col_end]
                k_cb = self.Cb[row_start:row_end, col_start:col_end]
                k_cr = self.Cr[row_start:row_end, col_start:col_end]
                
                # DCT and quantization in one step
                k_y = np.floor(dctTransform(k_y) / quat_table_float).astype(np.int16)
                k_cb = np.floor(dctTransform(k_cb) / quat_table_float).astype(np.int16)
                k_cr = np.floor(dctTransform(k_cr) / quat_table_float).astype(np.int16)
                
                # Zigzag and RLE
                self.Y_rle.extend(jpeg_rle(zig_zag_traversal_optimized(k_y)))
                self.Cb_rle.extend(jpeg_rle(zig_zag_traversal_optimized(k_cb)))
                self.Cr_rle.extend(jpeg_rle(zig_zag_traversal_optimized(k_cr)))
                
                processed_blocks += 1
                if processed_blocks % progress_step == 0 or processed_blocks == total_blocks:
                    progress = (processed_blocks / total_blocks) * 100
                    print(f"  Processing blocks: {processed_blocks}/{total_blocks} ({progress:.1f}%)")
        
        print(f"✓ DCT and RLE completed for {total_blocks} blocks (60%)")
        print()

    # ---------- count RLE -> frequencies (use the JPEG symbol = run<<4 | size) ----------
    def count_rle(self):
        print("Counting symbol frequencies for Huffman encoding...")
        def add_counts(rle_list, counter_dict):
            for run, val in rle_list:
                size = value_size(val)
                sym = huf_symbol(run, size)
                counter_dict[sym] += 1  # defaultdict handles missing keys
        add_counts(self.Y_rle, self.Y_rle_count)
        add_counts(self.Cb_rle, self.Cb_rle_count)
        add_counts(self.Cr_rle, self.Cr_rle_count)
        print(f"  Y channel: {len(self.Y_rle_count)} unique symbols")
        print(f"  Cb channel: {len(self.Cb_rle_count)} unique symbols")
        print(f"  Cr channel: {len(self.Cr_rle_count)} unique symbols")
          
     # ---------- build per-channel Huffman codes ----------
    def build_huffman_tables(self):
        print("Stage 4/5: Building Huffman tables...")
        self.Y_codes = build_huffman_codes(self.Y_rle_count)
        self.Cb_codes = build_huffman_codes(self.Cb_rle_count)
        self.Cr_codes = build_huffman_codes(self.Cr_rle_count)
        print(f"✓ Huffman tables built (80%)")
        print()
        
     # ---------- encode RLE lists to final bitstrings ----------
    def encode_channel(self, rle_list, codes_dict):
        """
        returns string of '0'/'1' bits for the channel
        RLE list contains tuples (run, val)
        codes_dict maps symbol->huffman string
        """
        out_bits = []
        for run, val in rle_list:
            size = value_size(val)
            sym = huf_symbol(run, size)
            # fetch huffman code for symbol
            code = codes_dict.get(sym)
            if code is None:
                # if symbol unseen during table build (rare), handle gracefully:
                # fallback: raw fixed-length encoding (not JPEG-standard) OR raise
                raise KeyError(f"Symbol {sym:#x} not found in Huffman table.")
            out_bits.append(code)
            # append amplitude bits (if any)
            _, amp = amplitude_bits(val)
            if amp:
                out_bits.append(amp)
        return ''.join(out_bits)
    
    def encode_all(self):
        """Return dict with bit-strings for each channel."""
        print("Stage 5/5: Encoding channels with Huffman codes...")
        print("  Encoding Y channel...")
        y_bits = self.encode_channel(self.Y_rle, self.Y_codes)
        print("  Encoding Cb channel...")
        cb_bits = self.encode_channel(self.Cb_rle, self.Cb_codes)
        print("  Encoding Cr channel...")
        cr_bits = self.encode_channel(self.Cr_rle, self.Cr_codes)
        print("✓ All channels encoded (100%)")
        print()
        return {'Y': y_bits, 'Cb': cb_bits, 'Cr': cr_bits}
    
    # ---------- Decoding functions ----------
    def decode_channel(self, bits_string, codes_dict):
        """Decode a bit string back to RLE tuples using Huffman codes."""
        # Create reverse lookup table
        reverse_codes = {code: symbol for symbol, code in codes_dict.items()}
        
        rle_list = []
        i = 0
        
        while i < len(bits_string):
            # Find the longest matching Huffman code
            found = False
            for length in range(1, min(17, len(bits_string) - i + 1)):  # Max Huffman code length is 16
                code = bits_string[i:i+length]
                if code in reverse_codes:
                    symbol = reverse_codes[code]
                    run = (symbol >> 4) & 0xF
                    size = symbol & 0xF
                    
                    i += length
                    
                    if size == 0:
                        if run == 0:  # EOB
                            rle_list.append((0, 0))
                        else:  # ZRL (15 zeros)
                            rle_list.append((15, 0))
                    else:
                        # Read amplitude bits
                        if i + size <= len(bits_string):
                            amp_bits = bits_string[i:i+size]
                            val = decode_amplitude(amp_bits, size)
                            rle_list.append((run, val))
                            i += size
                        else:
                            break
                    found = True
                    break
            
            if not found:
                # If we can't decode anymore, break (end of valid data)
                break
        
        return rle_list
    

    
    def save_lossy_reconstruction(self, output_path="compressed_output.jpg"):
        """Create a lossy reconstruction to demonstrate compression effects."""
        print("=== Creating Lossy Reconstruction ===")
        print("Simulating JPEG compression artifacts...")
        
        # We'll reconstruct directly from the stored quantized DCT coefficients
        # This simulates the lossy compression without needing full decoding
        
        # Reconstruct image by going through the compression pipeline in reverse
        h_blocks = self.height // 8
        w_blocks = self.width // 8
        
        # Create output arrays
        Y_out = np.zeros((self.height, self.width), dtype=np.float32)
        Cb_out = np.zeros((self.height, self.width), dtype=np.float32) 
        Cr_out = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Process each 8x8 block
        for row in range(h_blocks):
            for col in range(w_blocks):
                # Get original block
                row_start, row_end = row * 8, (row + 1) * 8
                col_start, col_end = col * 8, (col + 1) * 8
                
                orig_y = self.Y[row_start:row_end, col_start:col_end]
                orig_cb = self.Cb[row_start:row_end, col_start:col_end]  
                orig_cr = self.Cr[row_start:row_end, col_start:col_end]
                
                # Apply DCT, quantize, then dequantize and inverse DCT
                # This introduces the lossy compression artifacts
                
                # Forward DCT and quantization
                dct_y = dctTransform(orig_y)
                dct_cb = dctTransform(orig_cb)
                dct_cr = dctTransform(orig_cr)
                
                quant_y = np.round(dct_y / self.quat_table.astype(np.float32))
                quant_cb = np.round(dct_cb / self.quat_table.astype(np.float32))
                quant_cr = np.round(dct_cr / self.quat_table.astype(np.float32))
                
                # Dequantization and inverse DCT
                dequant_y = quant_y * self.quat_table.astype(np.float32)
                dequant_cb = quant_cb * self.quat_table.astype(np.float32)
                dequant_cr = quant_cr * self.quat_table.astype(np.float32)
                
                recon_y = idctTransform(dequant_y)
                recon_cb = idctTransform(dequant_cb)
                recon_cr = idctTransform(dequant_cr)
                
                # Store reconstructed block
                Y_out[row_start:row_end, col_start:col_end] = recon_y
                Cb_out[row_start:row_end, col_start:col_end] = recon_cb
                Cr_out[row_start:row_end, col_start:col_end] = recon_cr
        
        # Convert YCbCr back to RGB
        print("Converting YCbCr back to RGB...")
        rgb_image = self.ycbcr_to_rgb(Y_out, Cb_out, Cr_out)
        
        # Crop to original size and save
        rgb_cropped = rgb_image[:self.original_height, :self.original_width]
        rgb_uint8 = np.clip(rgb_cropped, 0, 255).astype(np.uint8)
        
        # Save as JPEG to show compression artifacts  
        img_pil = Image.fromarray(rgb_uint8, 'RGB')
        img_pil.save(output_path, 'JPEG', quality=85)
        
        print(f"✓ Lossy reconstruction saved as: {output_path}")
        
        # Calculate and display quality metrics
        original_rgb = self.img_np[:self.original_height, :self.original_width]
        mse = np.mean((original_rgb.astype(float) - rgb_uint8.astype(float)) ** 2)
        
        if mse > 0:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
            print(f"  PSNR: {psnr:.2f} dB (higher is better)")
        
        print(f"  File size reduction: Original vs Compressed")
        
        return img_pil
    
    def save_compression_data(self, filename="compression_data.npz"):
        """Save the compression data for later reconstruction or analysis."""
        print(f"Saving compression data to {filename}...")
        
        # Convert defaultdicts to regular dicts for saving
        y_counts = dict(self.Y_rle_count)
        cb_counts = dict(self.Cb_rle_count)
        cr_counts = dict(self.Cr_rle_count)
        
        np.savez_compressed(filename,
            # Image metadata
            original_height=self.original_height,
            original_width=self.original_width,
            padded_height=self.height,
            padded_width=self.width,
            
            # Quantization table
            quat_table=self.quat_table,
            
            # RLE data (convert to arrays for saving)
            Y_rle=np.array(self.Y_rle, dtype=object),
            Cb_rle=np.array(self.Cb_rle, dtype=object),
            Cr_rle=np.array(self.Cr_rle, dtype=object),
            
            # Frequency counts
            Y_counts=np.array(list(y_counts.items())),
            Cb_counts=np.array(list(cb_counts.items())),
            Cr_counts=np.array(list(cr_counts.items())),
            
            # Huffman codes (save as string arrays)
            Y_codes=np.array(list(self.Y_codes.items()), dtype=object),
            Cb_codes=np.array(list(self.Cb_codes.items()), dtype=object),
            Cr_codes=np.array(list(self.Cr_codes.items()), dtype=object)
        )
        
        print(f"✓ Compression data saved to {filename}")
        
        # Calculate and display sizes
        file_size = os.path.getsize(filename)
        original_size = self.original_height * self.original_width * 3  # RGB bytes
        print(f"  Compressed data size: {file_size:,} bytes")
        print(f"  Original image size: {original_size:,} bytes")
        print(f"  Compression ratio: {original_size / file_size:.2f}:1")
        
        return filename
    
    def reconstruct_blocks_simple(self):
        """Simplified reconstruction using stored RLE data."""
        h_blocks = self.height // 8
        w_blocks = self.width // 8
        
        Y_blocks = np.zeros((h_blocks, w_blocks, 8, 8), dtype=np.float32)
        Cb_blocks = np.zeros((h_blocks, w_blocks, 8, 8), dtype=np.float32)
        Cr_blocks = np.zeros((h_blocks, w_blocks, 8, 8), dtype=np.float32)
        
        # We need to track which RLE entries belong to which block
        # Since we stored them sequentially during encoding
        rle_y_idx = 0
        rle_cb_idx = 0  
        rle_cr_idx = 0
        
        for row in range(h_blocks):
            for col in range(w_blocks):
                # Reconstruct Y block
                block_y, rle_y_idx = self.rle_to_block_from_list(self.Y_rle, rle_y_idx)
                Y_blocks[row, col] = block_y
                
                # Reconstruct Cb block  
                block_cb, rle_cb_idx = self.rle_to_block_from_list(self.Cb_rle, rle_cb_idx)
                Cb_blocks[row, col] = block_cb
                
                # Reconstruct Cr block
                block_cr, rle_cr_idx = self.rle_to_block_from_list(self.Cr_rle, rle_cr_idx)
                Cr_blocks[row, col] = block_cr
        
        return Y_blocks, Cb_blocks, Cr_blocks
    
    def rle_to_block_from_list(self, rle_list, start_idx):
        """Convert RLE entries back to an 8x8 block."""
        block = np.zeros(64, dtype=np.float32)
        
        # First coefficient is DC (stored separately in real JPEG, but we'll approximate)
        ac_idx = 1  # Start from AC coefficients
        rle_idx = start_idx
        
        while rle_idx < len(rle_list) and ac_idx < 64:
            run, val = rle_list[rle_idx]
            
            if run == 0 and val == 0:  # EOB
                rle_idx += 1
                break
            elif run == 15 and val == 0:  # ZRL (15 zeros)
                ac_idx += 15
                rle_idx += 1
            else:
                ac_idx += run  # Skip 'run' zeros
                if ac_idx < 64:
                    block[ac_idx] = val
                    ac_idx += 1
                rle_idx += 1
        
        # Reshape to 8x8 and apply inverse zigzag
        return self.inverse_zigzag(block).reshape(8, 8)
    
    def inverse_zigzag(self, flat_block):
        """Convert flat array back to 8x8 using inverse zigzag pattern."""
        indices = get_zigzag_indices()
        block_2d = np.zeros((8, 8), dtype=np.float32)
        
        for i, (row, col) in enumerate(indices):
            if i < len(flat_block):
                block_2d[row, col] = flat_block[i]
        
        return block_2d
    
    def ycbcr_to_rgb(self, Y, Cb, Cr):
        """Convert YCbCr back to RGB."""
        # Inverse transformation matrix
        inv_transform = np.array([
            [1.0,  0.0,      1.402   ],
            [1.0, -0.34414, -0.71414 ],
            [1.0,  1.772,    0.0     ]
        ], dtype=np.float32)
        
        # Remove offsets
        Y_norm = Y - 16
        Cb_norm = Cb - 128
        Cr_norm = Cr - 128
        
        # Stack and apply inverse transformation
        ycbcr_stack = np.stack([Y_norm, Cb_norm, Cr_norm], axis=-1)
        rgb = ycbcr_stack @ inv_transform.T
        
        return rgb

                
# Precomputed zigzag indices for 8x8 blocks
_zigzag_indices = None

def get_zigzag_indices():
    """Precompute zigzag traversal indices for 8x8 blocks."""
    global _zigzag_indices
    if _zigzag_indices is None:
        indices = []
        n, m = 8, 8
        for s in range(n + m - 1):
            if s % 2 == 0:
                # Even sum: move up-right
                x = min(s, n - 1)
                y = s - x
                while x >= 0 and y < m:
                    indices.append((x, y))
                    x -= 1
                    y += 1
            else:
                # Odd sum: move down-left
                y = min(s, m - 1)
                x = s - y
                while y >= 0 and x < n:
                    indices.append((x, y))
                    x += 1
                    y -= 1
        _zigzag_indices = np.array(indices)
    return _zigzag_indices

def zig_zag_traversal_optimized(kernel):
    """Optimized zigzag traversal using precomputed indices."""
    indices = get_zigzag_indices()
    kernel_np = np.array(kernel)
    return kernel_np[indices[:, 0], indices[:, 1]].tolist()

def zig_zag_traversal(kernel):
    """Original zigzag traversal for compatibility."""
    n = len(kernel)
    m = len(kernel[0])
    result = []
    for s in range(n + m - 1):
        if s % 2 == 0:
            # Even sum: move up-right
            x = min(s, n - 1)
            y = s - x
            while x >= 0 and y < m:
                result.append(kernel[x][y])
                x -= 1
                y += 1
        else:
            # Odd sum: move down-left
            y = min(s, m - 1)
            x = s - y
            while y >= 0 and x < n:
                result.append(kernel[x][y])
                x += 1
                y -= 1
    return result

def jpeg_rle(block):
    """Optimized RLE for AC coefficients of a zigzagged 8x8 block (skip DC)."""
    ac = block[1:]  # skip DC
    if not ac:  # Empty block
        return [(0, 0)]
    
    # Pre-allocate output list with reasonable size to reduce memory allocations
    output = []
    zero_count = 0

    for val in ac:
        if val == 0:
            zero_count += 1
        else:
            # if zero_count > 15, emit multiple 15,0 symbols
            while zero_count > 15:
                output.append((15, 0))
                zero_count -= 16
            output.append((zero_count, int(val)))  # Ensure int type
            zero_count = 0

    # trailing zeros → EOB
    if zero_count > 0:
        output.append((0, 0))
    
    return output

        
# Optimized DCT using separable transform and precomputed cosine matrix
_dct_matrix = None

def get_dct_matrix():
    """Precompute the 8x8 DCT transformation matrix for reuse."""
    global _dct_matrix
    if _dct_matrix is None:
        _dct_matrix = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                if i == 0:
                    ci = 1 / math.sqrt(8)
                else:
                    ci = math.sqrt(2 / 8)
                _dct_matrix[i, j] = ci * math.cos((2 * j + 1) * i * math.pi / 16)
    return _dct_matrix

def dctTransform(mat, m=8, n=8):
    """Optimized DCT using matrix multiplication (separable transform)."""
    # Center the values around -128 to 127 for DCT
    mat = np.array(mat, dtype=np.float32) - 128
    
    # Get precomputed DCT matrix
    T = get_dct_matrix()
    
    # Apply separable 2D DCT: T * mat * T^T
    return T @ mat @ T.T

def idctTransform(mat, m=8, n=8):
    """Inverse DCT using matrix multiplication."""
    # Get precomputed DCT matrix
    T = get_dct_matrix()
    
    # Apply separable 2D IDCT: T^T * mat * T
    result = T.T @ mat @ T
    
    # Add back the DC offset and clip to valid range
    result = result + 128
    return np.clip(result, 0, 255)

def decode_amplitude(bits_str, size):
    """Decode amplitude bits back to the original value."""
    if size == 0:
        return 0
    
    # Convert binary string to integer
    val = int(bits_str, 2)
    
    # Check if it's positive or negative encoding
    # If MSB is 1, it's positive
    # If MSB is 0, it's negative (stored as complement)
    if bits_str[0] == '1':
        return val
    else:
        # Negative value: subtract (2^size - 1)
        return val - (1 << size) + 1

# ---------- minimal main to run pipeline ----------
def main():
    print("=== JPEG Compression Pipeline ===")
    print()
    
    img = Image.open("image.png").convert("RGB")
    print(f"Input image size: {img.size[0]}x{img.size[1]}")
    print()
    
    my_img = myImageClass(img)
    my_img.do_YCbCr()
    my_img.pad_image()
    my_img.ycbcr_to_dct()
    my_img.count_rle()
    my_img.build_huffman_tables()
    bitstreams = my_img.encode_all()
    
    print("=== Compression Results ===")
    print(f"Y channel:  {len(bitstreams['Y']):,} bits")
    print(f"Cb channel: {len(bitstreams['Cb']):,} bits")
    print(f"Cr channel: {len(bitstreams['Cr']):,} bits")
    total_bits = len(bitstreams['Y']) + len(bitstreams['Cb']) + len(bitstreams['Cr'])
    print(f"Total:      {total_bits:,} bits ({total_bits/8:,.0f} bytes)")
    
    # Calculate compression ratio (assuming 24-bit RGB input)
    original_bits = my_img.original_height * my_img.original_width * 24
    compression_ratio = original_bits / total_bits
    print(f"Compression ratio: {compression_ratio:.2f}:1")
    print("=== Compression Complete! ===")
    print()
    
    # Save the compressed image
    try:
        compressed_img = my_img.save_lossy_reconstruction("compressed_output.jpg")
        print("  You can compare 'image.png' with 'compressed_output.jpg'")
    except Exception as e:
        print(f"✗ Error saving compressed image: {e}")

if __name__ == "__main__":
    main()


