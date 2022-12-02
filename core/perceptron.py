import numpy as np
from core.helpers import NeuralNetworkHelper
import matplotlib as plt
import matplotlib.pyplot as plt


class Perceptron:
    """
    Class that represents a perceptron.
    """

    input_number=0
    weights=np.array([])
    learning_rate=0
    activation_function="sigmoid"


    def __init__(self, input_number=2, activation_function="sigmoid", learning_rate=0.01) :
        """
        constructor of the class
        """
        self.weights=np.random.rand(input_number+1)
        self.input_number=2
        self.learning_rate=learning_rate
        self.activation_function=activation_function

    def loss_fun(self, val, result):
        """
        loss function
        """
        return val - result
    
    def activation_fun(self, val):
        """
        activation function
        """
        if self.activation_function == "sigmoid":
            return NeuralNetworkHelper.sigmoid(val)
        elif self.activation_function == "relu":
            return NeuralNetworkHelper.relu(val)
        
    def derivated_activation_fun(self, val):
        """
        derivated activation function
        """
        if self.activation_function == "sigmoid":
            return NeuralNetworkHelper.sigmoid_derivate(val)
        elif self.activation_function == "relu":
            return NeuralNetworkHelper.relu_derivate(val)

    def run(self, values):
        """
        run the perceptron
        """
        values = np.insert(values, 0, 1)
        return self.activation_fun(np.sum(np.dot(self.weights, values)))

    def train(self, data, solutions, number_of_iterations=1000, epochs=100):
        """
        train the perceptron
        """
        weights_mod=np.zeros((self.weights.shape))
        for x in range(0, epochs):
            for i in range(0, number_of_iterations):
                result=self.run(data[i])
                loss=self.loss_fun(solutions[i], result)
                sol = np.insert(data[i], 0, 1)
                weights_mod += self.learning_rate * loss * self.derivated_activation_fun(result) * sol
            self.weights += weights_mod
        
    def predict(self, data):
        """
        predict the output of the perceptron
        """
        return self.run(data)
    
    def print_chart(self, data, solutions):
        """
        print the chart
        """
        x=np.linspace(-10, 10, 100)
        plt.plot(x, -(self.weights[1] * x + self.weights[0]) / self.weights[2])
        plt.scatter(data[:,0], data[:,1], c=solutions)
        plt.show()