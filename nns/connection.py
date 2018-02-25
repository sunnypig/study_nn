import numpy as np


class Connection(object):

    matrix_w = None
    matrix_b = None
    matrix_dw = None
    matrix_db = None

    def __init__(self, number_input_neurons, number_output_neurons, activation):
        self.activation = activation
        self.initialize(number_input_neurons, number_output_neurons)
        assert (self.matrix_w is not None)
        assert (self.matrix_b is not None)

    def initialize(self, number_input_neurons, number_output_neurons):
        self.matrix_w = np.random.randn(
            number_output_neurons, number_input_neurons)
        self.matrix_b = np.zeros((number_output_neurons, 1))

    def forward(self, x):
        return self.activation.forward(
            np.dot(self.matrix_w, x) + self.matrix_b)

    def backward(self, x, y, dy):
        dz = self.activation.backward(y, dy)
        self.matrix_dw = np.dot(dz, x.T)
        self.matrix_db = np.sum(dz, axis=1, keepdims=True)
        return np.dot(self.matrix_w.T, dz)

    def update(self, learning_rate):
        assert (self.matrix_dw is not None)
        assert (self.matrix_db is not None)
        self.matrix_w = self.matrix_w - learning_rate * self.matrix_dw
        self.matrix_b = self.matrix_b - learning_rate * self.matrix_db

