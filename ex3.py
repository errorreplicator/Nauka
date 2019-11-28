import numpy as np
import pandas as pd
from Embedd import dataproc,modeler
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array
from PIL import Image
dataproc.fix_seeds(1)
# ar = np.array([[6,7,3,4],[6,3,7,1]])
# args = np.zeros((2,5))
# tmp = np.concatenate((ar,args),1)
# resh = tmp.reshape(-1,9,2,4)
# dim3 = np.concatenate((tmp,tmp,tmp))
#
# test = dim3.reshape(-1,3,3,3)
# l1 = ['3','5']
# l2 = ['4','2']
# l3 = l1 + l2
# print(l3)

img = Image.open('/home/piotr/Pictures/1.png')
# vgg16.preprocess_input(img, mode='tf')
# print(img)
img = img_to_array(img)
print(img.shape)
print(type(img))
print(img)
# img_prep = preprocess_input(img)
# print(img_prep)

import numpy as np











# print()
# print(tmp)
# print(resh)
# print(dim3)
# print(dim3.shape)
# print(test)