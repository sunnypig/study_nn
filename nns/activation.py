import numpy as np


class Activation(object):

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, y, dy):
        raise NotImplementedError()


class Sigmoid(Activation):

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, y, dy):
        return y * (1 - y) * dy


class Tanh(Activation):

    def forward(self, x):
        e = np.exp(x)
        e_minus = np.exp(-x)
        return (e - e_minus) / (e + e_minus)

    def backward(self, y, dy):
        return dy * (1 - (y * y))


class Relu(Activation):

    def forward(self, x):
        return x * np.int64(x > 0)

    def backward(self, y, dy):
        return dy * np.int64(y > 0)

