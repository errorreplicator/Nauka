import numpy as np

class Conv3x3:
    def __init__(self,num_filt):
        self.num_filt = num_filt
        self.filters = np.random.randn(self.num_filt,3,3)/9

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
            output[y,x] = np.sum(img_region * self.filters, axis=(1,2))

        return output

