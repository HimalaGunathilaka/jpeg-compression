import numpy as np
from PIL import Image
import math




# YCbCr conversion
# print(Y.shape, Cb.shape, Cr.shape)


class myImageClass():
    def __init__(self, img):
        img_np = np.array(img)
        self.height, self.width = self.img_np.shape[:2]
        self.R = img_np[:,:,0]
        self.G = img_np[:,:,1]
        self.B = img_np[:,:,2]
        self.quat_table = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68,109,103, 77],
            [24, 35, 55, 64, 81,104,113, 92],
            [49, 64, 78, 87,103,121,120,101],
            [72, 92, 95, 98,112,100,103, 99]
        ])
        
    def do_YCbCr(self):
        self.Y  = 16 + (((self.R << 6) + (self.R << 1) + (self.G << 7) + self.G + (self.B << 3) + self.B) >> 8)
        self.Cb = 128 + ((-((self.R << 5) + (self.R << 2) + (self.R << 1)) - ((self.G << 6) + (self.G << 3) + (self.G << 1)) + (self.B << 7) - (self.B << 4)) >> 8)
        self.Cr = 128 + (((self.R << 7) - (self.R << 4) - ((self.G << 6) + (self.G << 5) - (self.G << 1)) - ((self.B << 4) + (self.B << 1))) >> 8)
        
    def pad_image(self):
        """
        Here image is padded to have a multiple of 8 width and height
        """
        # Compute padding sizes
        pad_h = (8 - self.height % 8) % 8
        pad_w = (8 - self.width % 8) % 8
        
        self.height += pad_h
        self.width += pad_w

        # Pad using edge values
        self.Y  = np.pad(self.Y,  ((0, pad_h), (0, pad_w)), mode='edge')
        self.Cb = np.pad(self.Cb, ((0, pad_h), (0, pad_w)), mode='edge')
        self.Cr = np.pad(self.Cr, ((0, pad_h), (0, pad_w)), mode='edge')
        
    def mask(self,i=0, j=0):
        # arr[row_start:row_end, col_start:col_end]
        row_start = i*8
        col_start = j*8
        kernel_Y = self.Y[row_start:row_start+8, col_start:col_start+8]
        kernel_Cb = self.Cb[row_start:row_start+8, col_start:col_start+8]
        kernel_Cr = self.Cr[row_start:row_start+8, col_start:col_start+8]
        return (kernel_Y, kernel_Cb, kernel_Cr)
    
    def ycbcr_to_dct(self):
        for col in range(self.width//8):
            for row in range(self.height//8):
                k_y,k_cb,k_cr = self.mask(i=row, j=col)
                
                k_y = np.floor(dctTransform(k_y) / self.quat_table)
                k_cb = np.floor(dctTransform(k_cb) / self.quat_table)
                k_cr = np.floor(dctTransform(k_cr) / self.quat_table)
                
    # def zig_zag_Serialization(self, kernel):
    #     n = len(kernel)
    #     m = len(kernel[0])
        
    #     row = 0
    #     col = 0
        
    #     row_inc = False
        
    #     mn = min(m,n)
        
    #     # First half
    #     for length in range(1, mn+1):
    #         for i in range(length):
                
                
                    
            
        
        
# Discrete cosine tansform
def dctTransform(mat, m=8, n=8):
    # store the cosine information
    dct = [[0.0 for _ in range(n)] for _ in range(m)]
    
    for i in range(m):
        for j in range(n):
            if i == 0:
                ci = 1 / math.sqrt(m)
            else:
                ci = math.sqrt(2 / m)
            if j == 0:
                cj = 1 / math.sqrt(n)
            else:
                cj = math.sqrt(2 / n)
            
            sum_val = 0.0
            for k in range(m):
                for l in range(n):
                    sum_val += mat[k][l] * \
                        math.cos((2 * k + 1) * i * math.pi / (2 * m)) * \
                        math.cos((2 * l + 1) * j * math.pi / (2 * n))
            dct[i][j] = ci * cj * sum_val
    return dct

def main():
    img = Image.open("image.png").convert("RGB")
    my_img = myImageClass(img)
    my_img.do_YCbCr()
    my_img.pad_image()
    my_img.ycbcr_to_dct()

if __name__ == "__main__":
    main()


