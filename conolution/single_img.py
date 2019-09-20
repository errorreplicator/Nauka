from conolution.conv_ex import Conv3x3
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

img = Image.open('c:/Dataset/44.jpg').convert('L')

# plt.imshow(np.asarray(img))
# plt.show()
# # # display(img)

print(np.asarray(img).shape)

np_img = np.pad(np.asarray(img),1,mode='constant')
conv = Conv3x3(1)
output = conv.forward(np_img)
print(output.shape)
plt.imshow(output, cmap='gray')
plt.show()
