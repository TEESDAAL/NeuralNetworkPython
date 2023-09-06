from typing import Any
import PIL
import numpy as np
import csv
import string
import matplotlib.pyplot as plt
import random
from dataclasses import dataclass
from abc import abstractmethod, abstractproperty

letters = string.ascii_lowercase

with open('AB Dataset.csv', 'r') as file:
    line_reader = csv.reader(file)
    images = [[letters[int(image[0])], np.array(image[1:]).astype(int).reshape((28,28))] for image in line_reader]
    file.close()

count = len(images)
n_training_instances = int(count * 0.9)
random.shuffle(images)
training, test = images[:n_training_instances], images[n_training_instances:]
plt.imshow(training[0][1], cmap='Greys')
plt.show()


@dataclass
class BaseLayer:
    size: int
    num_inputs: int

    @abstractproperty
    def get_size(self) -> int:
        pass

    @abstractproperty
    def get_num_inputs(self) -> int:
        pass

    @abstractmethod
    def feedforward(self, input: np.ndarray[float]) -> np.ndarray[float]:
        pass
    
    @abstractmethod
    def backpropogate(self, input: np.ndarray[float]) -> np.ndarray[float]:
        pass

@dataclass
class NeuralNetwork:
    layers: list[BaseLayer]

    def __init__(self, layer_sizes: list[int]) -> None:
        self.layers = []
        self.layers.append(InputLayer(layer_sizes[0]))
        self.layers.append(ActivationLayer(layer_sizes[0], layer_sizes[0]))
        for i in range(1, len(layer_sizes)):
            self.layers.append(FeedForwardLayer(layer_sizes[i], layer_sizes[i-1]))
            self.layers.append(ActivationLayer(layer_sizes[i], layer_sizes[i]))

    def calculate_inputs(self, input: np.ndarray[float]):
        for layer in self.layers:
            print(input.shape)
            input = layer.feedforward(input)
        
        return input


@dataclass
class ActivationLayer(BaseLayer):
    def __init__(self, size: int, num_inputs: int) -> None:
        self.size = size
        self.num_inputs = num_inputs

    def get_size(self) -> int:
        return self.size
    
    def get_num_inputs(self) -> int:
        return self.num_inputs

    def activation(self, x: float) -> float:
        return np.tanh(x)
    
    def activation_prime(self, x: float) -> float:
        cosh = np.cosh(x)
        return 1 / (cosh * cosh)
    
    def feedforward(self, input: np.ndarray[float]) -> np.ndarray[float]:
        return self.activation(input)
        
    def backpropogate(self, input: np.ndarray[float]) -> np.ndarray[float]:
        pass

@dataclass
class FeedForwardLayer(BaseLayer):
    ninputs: int
    size: int
    weights: np.ndarray
    biases: np.ndarray
    alpha: float

    def __init__(self, size: int, num_inputs: int) -> None:
        self.size = size
        self.num_inputs = num_inputs
        self.weights = np.random.rand(size, num_inputs)
        self.biases = np.random.rand(size)
        self.alpha = 0.01

    def get_size(self) -> int:
        return self.size
    
    def get_num_inputs(self) -> int:
        return self.num_inputs
    
    def feedforward(self, input: np.ndarray[float]) -> np.ndarray[float]:
        return self.weights.dot(input) + self.biases
    
    def backpropogate(self, input: np.ndarray[float]) -> np.ndarray[float]:
        pass
    
class InputLayer(BaseLayer):
    def __init__(self, size: int):
        self.size = size
        self.num_inputs = 0
    
    def feedforward(self, input: np.ndarray[float, Any]) -> np.ndarray[float, Any]:
        #print(input.shape)
        return input
    
    def backpropogate(self, input: np.ndarray[float, Any]) -> np.ndarray[float, Any]:
        pass
        

nn = NeuralNetwork([2,2])

'''
for layer in nn.layers:
    if isinstance(layer, FeedForwardLayer):
        print(layer.weights, layer.biases)
    else:
        print(layer.size)
'''

print(nn.calculate_inputs(np.array([1, 1])))