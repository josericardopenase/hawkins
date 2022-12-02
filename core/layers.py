from abc import abstractmethod
import numpy as np
from core.helpers import NeuralNetworkHelper as nnh

class Layer:
    def __init__(self, activation_function=nnh.sigmoid, derivated_activation_function=nnh.sigmoid_derivate, input_size=3, number_of_neurons=1, learning_rate=0.01):
        self.activation_function = activation_function
        self.input_size = input_size
        self.number_of_neurons = number_of_neurons
        self.weights = np.random.rand(number_of_neurons, input_size + 1)
        self.derivated_activation_fun = derivated_activation_function
        self.learning_rate=learning_rate
    
    def loss_function(self, loss, result):
        """
        loss function
        """
        return loss - result

    @abstractmethod
    def forward(self, data):
        """
        forward propagation
        """
        pass

    @abstractmethod
    def backward(self, data, loss, learning_rate):
        """
        backward propagation
        """
        pass


class Linear(Layer):
    """
    class that represents a layer of a neural network.
    """

    def forward(self, data):
        """
        forward propagation
        """
        result = np.array([])
        for x in range(0, self.number_of_neurons):
            result[x] = self.activation_function(np.sum(np.dot(self.weights[x], data)))
        return result
    
    def backward(self, data, loss, learning_rate=0.01):
        """
        backward propagation
        """
        weights_mod = np.zeros((self.weights.shape))
        for x in range(0, self.number_of_neurons):
            result=self.activation_function(np.sum(np.dot(self.weights[x], np.insert(data, 0, 1))))
            error=self.loss_function(loss, result)*self.derivated_activation_fun(result)
            weights_mod[x]=error*data[x]*learning_rate
        self.weights += weights_mod
            
    
class Conv2D(Layer):
    """
    class that represents a convulutional layer of a neural network.
    """
    pass

class MaxPool2D(Layer):
    """
    class that represents a maxpool layer of a neural network.
    """
    pass

class Flatten(Layer):
    """
    class that represents a flatten layer of a neural network.
    """
    pass

