import numpy as np
import pandas as pd
from Embedd import dataproc,modeler
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array
from PIL import Image
dataproc.fix_seeds(1)
ar = np.array([[[6],[7],[3]],[[6],[3],[7]]])
# args = np.zeros((2,5))
# tmp = np.concatenate((ar,args),1)
# resh = tmp.reshape(-1,9,2,4)
# dim3 = np.concatenate((tmp,tmp,tmp))
# print(np.tile(A=ar,reps=[1,3]))

print(np.full((2, 2), 255))



# img = Image.open('/home/piotr/Pictures/1.png').convert('LA')
# img = img.resize((10,10))
# num_img = np.asarray(img)
# # vgg16.preprocess_input(img, mode='tf')
# # print(img)
# # img = img_to_array(img)
# print(type(num_img))
# print(num_img.shape)
# print(num_img)
# # img_prep = preprocess_input(img)
# # print(img_prep)
