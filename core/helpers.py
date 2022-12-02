import numpy as np

class NeuralNetworkHelper:
    """
    Class that contains all the helper functions for the neural network
    """
    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function
        """
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def  sigmoid_derivate(val):
        """
        Derivate of the sigmoid activation function
        """
        return val * (1 - val)
    
    @staticmethod
    def relu(val):
        """
        Relu activation function
        """
        return max(0, val)

    @staticmethod
    def relu_derivate(val):
        """
        Derivate of the relu activation function
        """
        return 1 if val > 0 else 0

    @staticmethod
    def loss_function(loss, result):
        """
        loss function
        """
        return loss - result