from conolution.conv_ex import Conv3x3
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

np.random.seed(30)
samp_img = np.random.randint(0,255,size=(3,3))
pad_img = np.pad(np.asarray(samp_img), 1, mode='constant')
print(pad_img)

conv = Conv3x3(3)
output = conv.forward(pad_img)
# # output = output.reshape(output.shape[0],output.shape[1])
# print(output.shape)
# plt.imshow(output, cmap='gray')
# plt.show()
# # print(output.shape)
# # print()
# # print(output)