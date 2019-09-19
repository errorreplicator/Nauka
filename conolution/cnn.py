from conolution.conv_ex import Conv3x3
import numpy as np


np.random.seed(100)
samp_img = np.random.randint(0,255,size=(5,5))

# print(type(samp_img))

print(samp_img)
print(samp_img.shape)
conv = Conv3x3(1)
output = conv.forward(samp_img)
print(output.shape)
print()
print(output)