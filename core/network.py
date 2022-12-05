import numpy as np

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
    
    def backward(self, data, loss, learning_rate=0.01, epochs=100, batch_size=1000):
        """
        backward propagation
        """
        accuracy_history=[]
        accuracy=0
        result=None
        for index in range(0, batch_size):
            curr_data=data[index], 0, 1
            curr_loss=loss[index]
            predicted = 1 if self.forward(data[index]) > 0.5 else 0
            accuracy=accuracy+(loss[index]==predicted)
            accuracy_history.append(accuracy/(index+1))
            print("learning... predicted: ", predicted, "solution: ",  loss[index], "accuracy: " , accuracy/(index+1))
            for layer in reversed(self.layers):
                result_data, result_loss = layer.backward(curr_data, curr_loss, learning_rate)
                curr_data=np.insert(result_data, 0, 1)
                curr_loss=result_loss

        self.print_accuracy_graph(accuracy_history) 
        return result
    
    def print_accuracy_graph(self, accuracy_history):
        """
        print the accuracy graph
        """
        import matplotlib.pyplot as plt

        plt.plot(accuracy_history)
        plt.ylabel('accuracy')
        plt.xlabel('epochs')
        plt.show()
        

    def test(self, data, solution):
        """
        test the layer
        """
        error_medio_cuadratico=0
        accuracy=0
        for x in range(0, len(data)):
            result=self.forward(data[x])
            result=1 if result>0.5 else 0
            error_medio_cuadratico=error_medio_cuadratico+(solution[x]-result)**2
            accuracy=accuracy+(solution[x]==result)
        print("error promedio", error_medio_cuadratico/len(data), " accuracy", accuracy/len(data))