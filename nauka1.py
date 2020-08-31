from PIL import Image
from PIL import ImageFilter
from matplotlib import pyplot as plt
import cv2
import numpy as np
file = 'c:/1/barbara.jpg'
file2 = 'c:/1/1.jpg'

img = cv2.imread(file)
#768,448
def auto_canny(image, sigma=0.33):
    v = np.median(image)
    # apply automatic Canny edge detection using the computed media
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image,lower,upper)
    return edged

# img_edge = auto_canny(img)
blure = cv2.blur(img,(20,20))

plt.imshow(blure)

plt.show()