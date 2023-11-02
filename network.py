from random import shuffle
import numpy as np

print("importing data")
from letternn import training, test
print("imported Data")


def sigmoid(x: float):
    """Return the sigmod(x)

    Args:
        z float

    Returns:
        float: A float between 0 and 1
    """
    return 1/(1 + np.exp(-x))


def sigmoid_prime(x):
    """The derivative of the sigmoid function"""
    return sigmoid(x)*(1 - sigmoid(x))


class NeuralNetwork:
    """A Class to represent a Neural Network"""
    def __init__(self, layer_sizes: list[int]):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        # Represents the pairs of layers layer_sizes = [1,2,3,4], layer_connections = [(1,2), (2,3), (3,4)]
        layer_connections: list[tuple[int, int]] = list(zip(layer_sizes[:-1], layer_sizes[1:]))

        # The first layer is considered to be the input layer so we don't set any biases for it
        self.biases = [np.random.randn(x, 1) for x in layer_sizes[1:]]
        self.weights = [np.random.randn(j, k) for k, j in layer_connections]

    def feedforward(self, a):
        """ Return the output of the network if 'a' is input ."""
        # for biases, weights in zip(self.biases, self.weights):
        #     a = sigmoid(np.dot(weights, a)+biases)
        for i in range(self.num_layers - 1):
            a = sigmoid(np.dot(self.weights[i], a)+self.biases[i])
        return a

    def stochastic_gradient_descent(self, training_data: list[tuple], num_epochs: int,
                                    mini_batch_size: int, learning_rate: float, test_data: list[tuple] | None=None):
        """Train the neural network using mini-batch stochastic gradientdescent

        Args:
            training_data (list[tuple]): A list of tuples (training input, desired output)
            num_epochs (int): The number of times to run through the training set
            mini_batch_size (int): the size of each batch to learn from
            learning_rate (float): a number that represents how fast the AI learns
                (higher number means faster learning rate, but chance to skip local minima)
            test_data (list[tuple], optional): The data to test the AI on. Defaults to None.
        """
        for epoch in range(num_epochs):
            shuffle(training_data)

            mini_batches = [training_data[i: i+mini_batch_size] for i in range(0, len(training_data), mini_batch_size)]

            for batch in mini_batches:
                self.update_from_mini_batch(batch, learning_rate)

                if test_data:
                    num_successes = self.evaluate(test_data)
                    print(f"Epoch {epoch}: {num_successes} / {len(test_data)}\
                        = {num_successes / len(test_data)}%")
                else:
                    print(f"Epoch {epoch}: complete")

    def update_from_mini_batch(self, mini_batch: list[tuple], learning_rate: float):
        """Update the weights and biases by applying gradient descent using back propagation to a single mini batch .

        Args:
            mini_batch (list[tuple]): a subset of the training data to train on
            learning_rate (float): an float used to determine how much accuracy to train the NN with
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for image, label in mini_batch:
            delta_nabla_b, delta_nabla_w = self.back_propagation(image, label)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

            needs_good_name = learning_rate/len(mini_batch)
            self.weights = [weight - needs_good_name*nw for weight, nw in zip(self.weights, nabla_w)]
            self.biases = [bias - needs_good_name*nb for bias, nb in zip(self.biases, nabla_b)]

    def back_propagation(self, x, y) -> tuple:
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        #for b, w in zip(self.biases, self.weights):
        for l in range(self.num_layers - 1):
            z = np.dot(self.weights[l], activation)+self.biases[l]
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def home_made_backprop(self, x, y):
        """return the suggested changes to the weights and the baises
        Args:
            a: the data
            y: the label for the data

        Returns:

        """
        #
        L = len(self.layer_sizes)
        zs = []
        for j in reversed(self.layer_sizes):
            activation_input = sum(self.weights[L][j][k] * activations[L-1][k] for k in range(self.layer_sizes[L-1])) + self.biases[j]
            activation_inputs.append(zj)

            activations = [sigmoid(z) for z in activation_inputs]
            
            dCost_dActivation = 2 * (activations[j] - y[j])

            dCost_dWeight = dActivationInput_dWeight * dActivation_dActivationInput * dCost_dActivation

    def cost_derivative(self, output_activations, y):
        r""" Return the vector of partial derivatives (dCx/da) for the output activations."""
        return output_activations - y

    def evaluate(self, test_data):
        """ Return the number of test inputs for which the neural network outputs the correct result."""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data ]
        return sum(int(x == y) for (x, y) in test_results)


net = NeuralNetwork([784, 30, 2])

net.stochastic_gradient_descent(training, 30, 10, 3.0, test_data=test)
print("Done :)")
