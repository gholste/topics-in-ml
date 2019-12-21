import numpy as np
import random
import tqdm
from time import sleep


def softmax(f):
  denom = np.sum(np.exp(f))

  return np.exp(f) / denom

def sigmoid(z):
    return 1. / (1. + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1. - sigmoid(z))

def one_hot(y, C):
    res = np.zeros((C, 1))
    res[y] = 1.

    return res

def normalize(data):
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))


class NeuralNet:
    '''Basic implementation of a feedforward neural network.

    Args:
        sizes (list)    : list of all layer sizes (number of neurons):
                            [n_features, n_hidden1, ..., n_hiddenk, n_out]
        y_type (string) : string denoting whether to perform regression or classification
                            must be "discrete" or "continuous"

    Attributes:
        sizes (list)    : list of all layer sizes (number of neurons):
                           [n_features, n_hidden1, ..., n_hiddenk, n_out]
        y_type (string) : string denoting whether to perform regression or classification
                            must be "discrete" or "continuous"
        n_layers (int)  : number of layers in network (determined by "sizes" parameter)
        weights (list)  : list of weights at each layer, where each element
                          is an ndarray of shape (n_out, n_in)
        biases (list)   : list of biases at each layer, where each element
                          is an ndarray of shape (n_out, 1)
    '''
    def __init__(self, sizes, y_type):
        assert (y_type in ['discrete', 'continuous']), "y_type must be 'discrete' or 'continuous'"
        self.y_type = y_type
        self.n_layers = len(sizes)
        np.random.seed(0)
        self.weights = [np.random.randn(i, j) / np.sqrt(j)
                        for i, j in zip(sizes[1:], sizes[:-1])]
        self.biases = [np.random.randn(i, 1) for i in sizes[1:]]


    def forward(self, x):
        if self.y_type == "discrete":
            for w, b in zip(self.weights, self.biases):
                x = sigmoid(np.dot(w, x) + b)
        else:
            for w, b in zip(self.weights[:-1], self.biases[:-1]):
                x = sigmoid(np.dot(w, x) + b)

            x = np.dot(self.weights[-1], x) + self.biases[-1]  # linear activation

        return x

    def backprop(self, x, y):
        Z = []
        A = [x]
        a = A[0]

        # Forward pass: store z and a values for each layer
        if self.y_type == "discrete":
            for w, b in zip(self.weights, self.biases):
                z = np.dot(w, a) + b
                Z.append(z)
                a = sigmoid(z)
                A.append(a)
        else:
            for w, b in zip(self.weights[:-1], self.biases[:-1]):
                z = np.dot(w, a) + b
                Z.append(z)
                a = sigmoid(z)
                A.append(a)

            z = a = np.dot(self.weights[-1], a) + self.biases[-1]
            Z.append(z)
            A.append(a)

        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]

        # Compute gradient at output layer 
        if self.y_type == "discrete":
            delta = self.C_prime(A[-1], y) * sigmoid_prime(Z[-1])
        else:
            delta = self.C_prime(A[-1], y)
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, A[-2].T)

        # Backward pass: propagate errors backward
        for l in range(2, self.n_layers):
            delta = np.dot(self.weights[-l + 1].T, delta) * sigmoid_prime(Z[-l])
            grad_b[-l] = delta
            grad_w[-l] = np.dot(delta, A[-l - 1].T)

        return grad_w, grad_b

    def C_prime(self, output, y):
        # Derivative of MSE: L(x) = 0.5*(y - f(x))^2
        return output - y

    def train(self, data, batch_size, epochs=10, lr=0.1):
        """Train neural network with stochastic gradient descent.

        Args:
            data (list)      : list of (x, y) tuples where each x is an ndarray of shape (p, 1)
                               and each y is of shape (1, 1) if continuous and of shape (C, 1)
                               if discrete, where p = # features and C = # output classes
            batch_size (int) : number of samples in each mini-batch
            epochs (int)     : number of epochs (full passes through the training set) to train
            lr (float)       : learning rate ("step size") for weight updates in backprop
        """
        n = len(data)
        bar = tqdm.trange(epochs, desc="", leave=True)
        for e in bar:
            random.shuffle(data)
            mini_batches = [data[k:k + batch_size] for k in range(0, n, batch_size)]

            for mini_batch in mini_batches:
                delta_w = [np.zeros(w.shape) for w in self.weights]
                delta_b = [np.zeros(b.shape) for b in self.biases]

                for x, y in mini_batch:
                    grad_w, grad_b = self.backprop(x, y)
                    delta_w = [dw + gw for dw, gw in zip(delta_w, grad_w)]
                    delta_b = [db + gb for db, gb in zip(delta_b, grad_b)]

                self.weights = [w - (lr / len(mini_batch)) * dw
                                for w, dw in zip(self.weights, delta_w)]
                self.biases = [b - (lr / len(mini_batch)) * db
                               for b, db in zip(self.biases, delta_b)]

            if self.y_type == "discrete":
                bar.set_description(f"Epoch {e + 1} | Accuracy: {round(self.evaluate(data), 3)}")
                bar.refresh()
                sleep(0.01)
            else:
                bar.set_description(f"Epoch {e + 1} | MSE: {round(self.evaluate(data), 3)}")
                bar.refresh()
                sleep(0.01)


    def predict(self, X):
        if self.y_type == "discrete":
            y_pred = np.array([np.argmax(self.forward(x)) for x in X])
        else:
            y_pred = np.array([self.forward(x).item(0) for x in X])

        return y_pred

    def predict_proba(self, X):
        assert (y_type == "discrete"), "y_type must be 'discrete' to call predict_proba"
        if self.y_type == "discrete":
            y_pred = np.array([softmax(self.forward(x)) for x in X])

        return y_pred    

    def evaluate(self, data):
        if self.y_type == "discrete":
            y_pred = np.array([np.argmax(self.forward(x)) for x, _ in data])
            y_true = np.array([np.argmax(y) for _, y in data])

            return np.sum(y_pred == y_true) / len(data)
        else:
            y_pred = np.array([self.forward(x).item(0) for x, _ in data])
            y_true = np.array([y.item(0) for _, y in data])

            return np.mean((y_pred - y_true)**2) / 2
