class NeuralNetwork:
    def __init__(self, layers=[]):
        self.layers = layers
    
    def forward(self, data):
        """
        forward propagation
        """
        for layer in self.layers:
            data = layer.forward(data)
        return data
    
    def backward(self, data, loss, learning_rate=0.01, epochs=100):
        """
        backward propagation
        """
        print(self.layers)
        for x in range(0, epochs):
            for layer in self.layers:
                data = layer.backward(data[x], loss[x], learning_rate)
        return data