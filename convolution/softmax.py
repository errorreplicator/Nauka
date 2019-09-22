import numpy as np

class Softmax:

    def __init__(self,input_len,nodes):
        self.weights = np.random.randn(input_len,nodes) / input_len
        self.bias = np.zeros(nodes)

    def forward(self,input):
        input = input.flatten()

        input_len, nodes = self.weights.shape

        total = np.dot(input, self.weights) + self.bias
        exp = np.exp(total)
        return exp / np.sum(exp, axis=0)