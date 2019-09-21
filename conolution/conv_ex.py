import numpy as np

class Conv3x3:
    def __init__(self,num_filt):
        self.num_filt = num_filt
        self.filter = np.random.randn(self.num_filt,3,3)/9 # random filter
        # self.filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).reshape(1, 3, 3)  # Sobel filter horizontal
        # self.filter = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, -1]]).reshape(1, 3, 3) # Sobel filter vertical

    def iter_regions(self,img):
        h,w = img.shape
        for y in range(h - 2):
            for x in range(w - 2):
                img_region = img[y:(y + 3), x:(x + 3)]
                yield img_region, y, x

    def forward(self,input):
        h,w = input.shape
        output = np.zeros((h-2,w-2,self.num_filt))

        for img_region, y,x in self.iter_regions(input):
            output[y,x] = np.sum(img_region * self.filter, axis=(1,2))

        return output

