import numpy as np
n_categories = 3  # number of possible categories
emb_dim = 5       # dimension of the emdedding vectors


X = np.array([
    [1,0,0],  # Category 1
    [0,1,0],  # Category 2
    [0,0,1],  # Category 3
    [0,0,1]   # Category 3
])

y = np.array([[1,1,0,0]]).T

# initialize the weights
weights0 = np.random.random((n_categories, emb_dim))
weights1 = np.random.random((emb_dim, 1))



def sigmoid(x, deriv=False):
    if(deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))



learning_rate = 0.1


for j in range(60000):
    # forward pass
    output1 = np.dot(X, weights0)  # A linear hidden layer
    output2 = sigmoid(np.dot(output1, weights1))  # Output layer with sigmoid activation
    # computing error
    l2_error = y - output2
    if (j% 10000) == 0:
        print("Error:" + str(np.mean(np.abs(l2_error))))
    # backward pass
    l2_delta = l2_error * sigmoid(output2, deriv=True)
    l1_error = l2_delta.dot(weights1.T)
    l1_delta = l1_error
    # update the weigths
    weights1 += learning_rate * output1.T.dot(l2_delta)
    weights0 += learning_rate * X.T.dot(l1_delta)
