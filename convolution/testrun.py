import numpy as np

# [1.00000000e+00 7.33897182e-25 2.50111192e-12 2.41812824e-10]
lista = [4,3,1,1]
print(np.exp(lista))
print(np.exp(lista)/np.sum(np.exp(lista),axis=0))