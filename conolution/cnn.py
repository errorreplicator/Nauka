from conolution.conv_ex import Conv3x3
from conolution.maxpool import MaxPool2D
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


img = Image.open('/home/piotr/data/22.jpg').convert('L')

print(np.asarray(img).shape)
# samp_img = np.random.randint(0,255,size=(5,5))

np_img = np.pad(np.asarray(img),1,mode='constant')
conv = Conv3x3(8)
output = conv.forward(np_img)
print(str(output.shape) + 'after conv')
pool = MaxPool2D()
output = pool.forward(output)

# output = output.reshape(output.shape[0],output.shape[1])
print(str(output.shape) + 'after max pool')


# plt.imshow(output, cmap='gray')
# plt.show()
