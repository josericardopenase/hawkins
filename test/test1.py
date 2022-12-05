
import numpy as np
from core.network import NeuralNetwork
from core.layers import Linear
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

model = NeuralNetwork([
    Linear(input_size=2, number_of_neurons=1)
]
)

data=np.array([])
solutions=np.array([])


def accuracy(y, y_pred):
    return np.sum(y == y_pred) / len(y)

x, y = datasets.make_blobs(n_samples=1000, centers=2, n_features=2, random_state=0)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123) 

model.backward(x_train, y_train, epochs=100, learning_rate=0.01)

for x in range(0, len(x_test)):
    print("prediccion:", model.forward(x_test[x]), "real:", y_test[x])

print("accuracy:", accuracy(y_test, [model.forward(x) for x in x_test]))