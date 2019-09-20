import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# img = Image.open('c:/Dataset/44.jpg').convert('L')
# img = np.asarray(img)
# print(img.shape)
zm1 = np.random.randn(4,4)/9
zm2 = [[1,2,1],[0,0,0],[-1,-2,-1]]

zm2 = np.pad(zm2,1,mode='constant')

print(zm2)
print(zm2.shape)

h,w = zm2.shape

for y in range(h-1):
    for x in range(w-1):
        region = zm2[y:(y+2),x:(x+2)]
        print(region,y,x)


new_h = h // 2
new_w = w // 2

for i in range(new_h):
  for j in range(new_w):
    im_region = zm2[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
    print(im_region, i, j)