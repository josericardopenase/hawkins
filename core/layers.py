from abc import abstractmethod
import numpy as np
from core.helpers import NeuralNetworkHelper as nnh

class Layer:
    def __init__(self, activation_function=nnh.sigmoid, derivated_activation_function=nnh.sigmoid_derivate, input_size=3, number_of_neurons=1, learning_rate=0.01, final_layer=False):
        self.activation_function = activation_function
        self.input_size = input_size
        self.number_of_neurons = number_of_neurons
        self.weights = np.random.rand(number_of_neurons, input_size + 1)
        self.derivated_activation_fun = derivated_activation_function
        self.learning_rate=learning_rate
        self.final_layer=final_layer
    
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

    def forward_one_line(self, data):
        """
        forward propagation
        """
        return self.activation_function(np.sum(np.dot(self.weights, np.insert(data, 0, 1))))

    def forward(self, data):
        """
        forward propagation
        """
        result = np.zeros((self.number_of_neurons))
        for x in range(0, self.number_of_neurons):
            result[x]=self.forward_one_line(data)

        if self.final_layer:
            return result[0]
        return result

    def backward_one_line(self, data, loss, learning_rate):
        """
        backward propagation
        """
        result=self.activation_function(np.sum(np.dot(self.weights, np.insert(data, 0, 1))))
        error=self.loss_function(loss, result)*self.derivated_activation_fun(result)
        return error*learning_rate

    def backward(self, curr_grad, curr_bias, curr_out, prev_act):
        """
        backward propagation
        """
        d_curr_out = self.derivated_activation_fun(curr_out, init_grad=curr_grad)
        d_curr_weight = np.dot(d_curr_out, prev_act.T)
        d_curr_bias = d_curr_out
        d_prev_act = np.dot(self.weights.T, d_curr_out)
        return d_prev_act, d_curr_weight, d_curr_bias


    
    
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
