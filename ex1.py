import numpy as np

# Working on 2D array
array = np.arange(12).reshape(3, 4)
print("INPUT ARRAY : \n", array)

array2 = np.array([0.423097,0.664932,0.844051,0.006873,0.043565]).reshape(5,1) #reshape x,n

# No axis mentioned, so works on entire array
print("\nMax element : ", np.argmax(array))

# returning Indices of the max element
# as per the indices
print("\nIndices of Max element : ", np.argmax(array, axis=0))
print("\nIndices of Max element : ", np.argmax(array, axis=-1))

print(array2)
print(array2.shape)
print(np.argmax(array2,axis=-1))