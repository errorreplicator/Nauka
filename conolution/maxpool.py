import numpy as np

class MaxPool2D:

    def IterRegions(self,img):
        h,w,_ = img.shape
        new_h = h // 2
        new_w = w // 2
        for y in range(new_h):
            for x in range(new_w):
                img_reg = img[(y*2):(y*2+2),(x*2):(x*2+2)]
                yield img_reg, y, x

    def forward(self, img):

        y,x, num_filters = img.shape
        output = np.zeros((y//2, x//2, num_filters))

        for img_reg, y, x in self.IterRegions(img):
            output[y, x] = np.amax(img_reg, axis=(0, 1))

        return output

