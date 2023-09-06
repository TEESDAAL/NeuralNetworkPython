import PIL
import numpy as np
import csv
import string
import matplotlib.pyplot as plt
import random
from dataclasses import dataclass
from abc import abstractmethod, abstractproperty

'''
letters = string.ascii_lowercase

with open('A_Z Handwritten Data.csv', 'r') as file:
    line_reader = csv.reader(file)
    images = [[letters[image[0]], np.array(image[1:]).astype(int).reshape((28,28))] for image in line_reader if image[0] in ['0', '1']]
    file.close()

count = len(images)
n_training_instances = int(count * 0.9)
random.shuffle(images)
training, test = images[:n_training_instances], images[n_training_instances:]
'''


@dataclass
class BaseLayer:
    size: int
    num_inputs: int

    @abstractmethod
    def __init__(self, size: int, num_inputs: int) -> None:
        pass

    @abstractproperty
    def get_size() -> int:
        pass

    @abstractproperty
    def get_num_inputs() -> int:
        pass

    @abstractmethod
    def feedforward(input: list[float]) -> list[float]:
        pass
    
    @abstractmethod
    def backpropogate(input: list[float]) -> list[float]:
        pass

@dataclass
class NeuralNetwork:
    layers: list[BaseLayer]

    def __init__(self, layer_size: list[int]) -> None:
        self.layers = []
        for i in range(layer_size):
            self.layers.append(FeedForwardLayer(layer_size[i], layer_size[i-1]))

    def feedforward(self, input: list[float]):
        for layer in self.layers:
            pass


@dataclass
class ActivationLayer(BaseLayer):
    def __init__(self, size: int, num_inputs: int) -> None:
        pass

    def activation(self, x: float) -> float:
        return np.tanh(x)
    
    def activation_prime(self, x: float) -> float:
        cosh = np.cosh(x)
        return 1 / (cosh * cosh)
    
    def feedforward(self, input: list[float]) -> list[float]:
        return [self.activation(x) for x in input]
        
    def backpropogate(self, input: list[float]) -> list[float]:
        pass


@dataclass
class FeedForwardLayer(BaseLayer):
    ninputs: int
    size: int

    def __init__(self, size: int, num_inputs: int) -> None:
        self.size = size
        self.num_inputs = num_inputs