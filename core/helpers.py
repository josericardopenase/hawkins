import numpy as np

class Value:
    value = 0

    def __init__(self, value, children=(), op='') -> None:
        self.value=value
    
    def __repr__(self) -> str:
        return "Value: " + self.value
    
    def __str__(self) -> str:
        return "Value: " + self.value

    def __add__(self, o: object) -> object:
        return Value(self.value + o.value, op='+', children=(self, o))
    
    def __sub__(self, o: object) -> object:
        return Value(self.value - o.value, op='-', children=(self, o))

    def __mul__(self, o: object) -> object:
        return Value(self.value * o.value, op='*', children=(self, o))
    
    def __truediv__(self, o: object) -> object:
        return Value(self.value / o.value, op='/', children=(self, o))
    
    def __pow__(self, o: object) -> object:
        return Value(self.value ** o.value, op='**', children=(self, o))

    def __eq__(self, o: object) -> bool:
        return self.value == o.value
    
    def __ne__(self, o: object) -> bool:
        return self.value != o.value

    def __lt__(self, o: object) -> bool:
        return self.value < o.value
    
    def __le__(self, o: object) -> bool:
        return self.value <= o.value
    
    def __gt__(self, o: object) -> bool:
        return self.value > o.value
    
    def __ge__(self, o: object) -> bool:
        return self.value >= o.value
    


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
    def  sigmoid_derivate(val, init_grad=1):
        """
        Derivate of the sigmoid activation function
        """
        return val * (1 - val) * init_grad
    
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
    
    @staticmethod
    def error_medio_cuadratico(data, solutions):
        error=0
        for x in range(0, len(data)):
            error+=(data[x]-solutions[x])**2
        return error/len(data)
    
    @staticmethod
    def tanh(val):
        """
        Tanh activation function
        """
        return np.tanh(val)
    
    @staticmethod
    def tanh_derivate(val):
        """
        Derivate of the tanh activation function
        """
        return 1 - val**2

    
