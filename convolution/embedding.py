import numpy as np
from scipy import spatial
np.random.seed(123)
covab = ['number_1', "number_2", "number_3"]


X = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])
X_week = np.array([
    [1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1]
])

y = np.array([[.9, .7, .5, .3, .4, .6, .8]]).T # what is the difference / practice dimension inputs / outputs on forward and backward pass


n_categories = 7
emd_dim = 3

weights0 = np.random.random((n_categories, emd_dim))
weights1 = np.random.random((emd_dim, 1))

learning_rate = 0.01

def sigmoid(x,deriv=False):
    if deriv == True:
        return x*(1-x)
    return 1/(1+np.exp(-x))

for x in range(30000):
    output0 = np.dot(X_week, weights0)
    output1 = np.dot(output0, weights1)
    output_end = sigmoid(output1)
    # print(output_end.shape)
    # print(y.shape)
    error = y - output_end
    # print('error shape',str(error.shape))
    # print(error)
    if x % 10000 == 0:
        print('error rate: ', str(np.mean(np.abs(error))))

    error_delta = error * sigmoid(output_end,True)
    l1_error = error_delta.dot(weights1.T)
    l1_delta = l1_error

    weights1 += learning_rate*output0.T.dot(error_delta)
    weights0 += learning_rate*X_week.T.dot(l1_delta)


print(output_end)
print(weights0)

print(spatial.distance.cosine(weights0[0], weights0[1]))
print(spatial.distance.cosine(weights0[0], weights0[2]))
print(spatial.distance.cosine(weights0[0], weights0[3]))
print(spatial.distance.cosine(weights0[0], weights0[4]))
print(spatial.distance.cosine(weights0[0], weights0[5]))
print(spatial.distance.cosine(weights0[0], weights0[6]))
print()
print(spatial.distance.cosine(weights0[1], weights0[2]))
print(spatial.distance.cosine(weights0[0], weights0[4]))
print(spatial.distance.cosine(weights0[0], weights0[6]))


