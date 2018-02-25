from .connection import Connection


class Layer(object):

    def __init__(self, number_neurons, activation, connection_class=Connection):
        self.activation = activation
        self.number_neurons = number_neurons
        self.connection_class = connection_class

    def generate_connection(self, input_layer):
        return self.connection_class(
            input_layer.number_neurons, self.number_neurons, self.activation)

