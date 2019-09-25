import ex2
import numpy as np

img_d1 = ex2.r_imd2D()
f0 = ex2.r_filter0()

print(img_d1)
print(f0)


def iter_regions2D(image):

    h, w = image.shape # change to 3D
    for y in range(h-2):
        for x in range(w-2):
            yield image[y:(y+3), x:(x+3)], y, x


def forward(image):

    h, w = image.shape
    output = np.zeros((h, w))
    for region, y, x in iter_regions2D(image):
        output[y, x] = np.sum(region * f0)

    return output

zm1 = forward(img_d1)