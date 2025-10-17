from PIL import Image
import numpy as np

img = Image.open("image.png").convert("RGB")
img_np = np.array(img)

print(img_np.shape)

height = img_np.shape[0]
width = img_np.shape[1]

# print(height, width)

R = img_np[:,:, 0]
G = img_np[:,:, 1]
B = img_np[:,:, 2]

Y = 16 + (((R << 6) + (R<<1) + (G << 7) + G + (B << 3) + B)>>8)
Cb = 128 + ((-((R << 5) + (R << 2) + (R << 1)) - ((G << 6) + (G << 3) + (G<<1)) + (B << 7)- (B << 4)) >> 8)
Cr = 128 + (((R << 7) - (R << 4) - ((G << 6)+(G << 5) - (G << 1)) - ((B << 4) + (B << 1))) >> 8)

# print(Y.shape)
# print(len(Y[2])) = 7435

# height_mod = height % 8

# if height_mod:
#     for row in range(8 - height_mod):
#         Y = np.pad(Y, (Y[height-1]), mode='constant')
    
# width += height_mod
# width_mod = width % 8
