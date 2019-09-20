import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

img = Image.open('c:/Dataset/44.jpg').convert('L')
img = np.asarray(img)
print(img.shape)
zm1 = np.random.randn(4,4)/9
zm2 = [[1,2,1],[0,0,0],[-1,-2,-1]]

zm1 = np.pad(zm1,1,mode='constant')

print(zm1)


