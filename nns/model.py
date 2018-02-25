

class Model(object):

    def __init__(self, loss, learning_rate=0.1):
        self.connections = []
        self.layers = []
        self.forward_cache = []
        self.learning_rate = learning_rate
        self.loss = loss

    def append(self, layer):
        self.layers.append(layer)

    def compile(self):
        self.connections.clear()
        input_layer = None
        for layer in self.layers:
            if input_layer:
                self.connections.append(
                    layer.generate_connection(
                        input_layer))
            input_layer = layer

    def forward_propagation(self, x):
        self.forward_cache.clear()
        self.forward_cache.append(x)
        for conn in self.connections:
            x = conn.forward(x)
            self.forward_cache.append(x)
        return x

    def backward_propagation(self, dy):
        num_connections = len(self.connections)
        assert (len(self.forward_cache) == num_connections + 1)
        for i, conn in enumerate(reversed(self.connections)):
            dy = conn.backward(
                self.forward_cache[num_connections - i - 1],
                self.forward_cache[num_connections - i], dy)
            conn.update(self.learning_rate)

    def predict(self, x):
        return self.forward_propagation(x)

    def train(self, x, y, number_iterations, trace=None):
        for i in range(0, number_iterations):
            y_predict = self.forward_propagation(x)
            dy = self.loss.calculate_derivative(y, y_predict)
            self.backward_propagation(dy / x.shape[1])

            if trace:
                trace.add(i, self.loss.calculate(y, y_predict).mean())
