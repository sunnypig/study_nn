import numpy as np


class Loss(object):

    def calculate(self, y, y_predict):
        raise NotImplementedError()

    def calculate_derivative(self, y, y_predict):
        raise NotImplementedError()


class CrossEntropy(Loss):

    def calculate(self, y, y_predict):
        return (-np.log(y_predict) * y) + (-np.log(1 - y_predict) * (1 - y))

    def calculate_derivative(self, y, y_predict):
        return (y_predict - y) / (y_predict * (1 - y_predict))
