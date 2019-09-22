from covolution.conv_ex import Conv3x3
from covolution.maxpool import MaxPool2D
from covolution.softmax import Softmax
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

np.set_printoptions(linewidth=np.inf)

# img = Image.open('/home/piotr/data/22.jpg').convert('L')
img = np.random.randint(0,255,size=(5,5))

print(np.asarray(img).shape)

np_img = np.pad(np.asarray(img),1,mode='constant')
conv = Conv3x3(1)
output = conv.forward(np_img)
print(str(output.shape) + 'after conv')
pool = MaxPool2D()
output = pool.forward(output)

# output = output.reshape(output.shape[0],output.shape[1])
print(str(output.shape) + 'after max pool')

soft = Softmax(output.shape[0]*output.shape[1]*output.shape[2],10)
final = soft.forward(output)
print(final)

# plt.imshow(output, cmap='gray')
# plt.show()
