import numpy as np
import mnist
from matplotlib import pyplot as plt
np.set_printoptions(linewidth=np.inf)

HOW EXP WIRKS AT THE END

imgs = mnist.test_images()
print(len(imgs))
print(type(mnist))
print(imgs[0])

plt.imshow(imgs[0])
plt.show()


np.random.seed(20)

biases = np.zeros(4)

img = np.random.randint(0, 20, size=(2,2))/7
print(img, '->random image after maxpool', end='\n\n')

img = img.flatten()
print(img, '->image flatter', end='\n\n')

weights = np.random.randint(0, 25, size=(4, 4))
print(weights, '-> weights', end='\n\n')

total = np.dot(img,weights) + biases
print(total, '->dot total + bias',end='\n\n')

exp = np.exp(total) # check exp how it works
print(exp,end='\n\n')

retu = exp / np.sum(exp,axis=0)
print(retu)