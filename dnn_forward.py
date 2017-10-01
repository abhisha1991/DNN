import matplotlib.pyplot as plt
from DNN.dnn_utils_v2 import *
from DNN.dnn_utils_v2 import Activations
import numpy as np

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)


class DNNForwardModel(object):

    def initialize_parameters(self, n_x, n_h, n_y):
        """
        Argument:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer

        Returns:
        parameters -- python dictionary containing your parameters:
                        W1 -- weight matrix of shape (n_h, n_x)
                        b1 -- bias vector of shape (n_h, 1)
                        W2 -- weight matrix of shape (n_y, n_h)
                        b2 -- bias vector of shape (n_y, 1)
        """

        np.random.seed(1)

        W1 = np.random.randn(n_h, n_x) * 0.01
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_y, n_h) * 0.01
        b2 = np.zeros((n_y, 1))

        assert (W1.shape == (n_h, n_x))
        assert (b1.shape == (n_h, 1))
        assert (W2.shape == (n_y, n_h))
        assert (b2.shape == (n_y, 1))

        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}

        return parameters

    def initialize_parameters_deep(self, layer_dims):
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network

        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """

        np.random.seed(3)
        parameters = {}
        L = len(layer_dims)  # number of layers in the network

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

            assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
            assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

        return parameters

    def linear_forward(self, A, W, b):
        """
        Implement the linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter
        cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """

        Z = np.dot(W, A) + b  # in a 1 layer NN, A = X (n_x, m), W = W1 (n_h, n_x), Z = (n_h, m)

        assert (Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)

        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function, also called the post-activation value
        cache -- a python dictionary containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
        """
        act = Activations()
        if activation == "sigmoid":
            Z, linear_cache = self.linear_forward(A_prev, W, b)  # because W = (n_h, n_x) and A_prev = (n_x, m), Z = (n_h, m)
            A, activation_cache = act.sigmoid(Z)  # A = (n_h, m)

        elif activation == "relu":
            Z, linear_cache = self.linear_forward(A_prev, W, b)  # Z = (n_h, m)
            A, activation_cache = act.relu(Z)  # A = (n_h, m)

        assert (A.shape == (W.shape[0], A_prev.shape[1]))  # LHS = (n_h, m), RHS = (W.shape[0], X.shape[1]) => (n_h, m)
        cache = (linear_cache, activation_cache)

        return A, cache

    def L_model_forward(self, X, parameters):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()

        Returns:
        AL -- last post-activation value or y_hat value
        caches -- list of caches containing:
                    every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                    the cache of linear_sigmoid_forward() (there is one, indexed L-1)
        """

        caches = []
        A = X
        L = len(parameters) // 2  # number of layers in the neural network

        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A
            print("Shape of the input: " + str(A_prev.shape))
            W = parameters["W" + str(l)]
            print("Shape of the weights: " + str(W.shape))
            b = parameters["b" + str(l)]
            print("Shape of the bias: " + str(b.shape))
            A, cache = self.linear_activation_forward(A_prev, W, b, "relu")
            caches.append(cache)

        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        print("Final layer")
        A_prev = A
        print("Shape of the input: " + str(A_prev.shape))
        W = parameters["W" + str(L)]
        print("Shape of the weights: " + str(W.shape))
        b = parameters["b" + str(L)]
        print("Shape of the bias: " + str(b.shape))
        AL, cache = self.linear_activation_forward(A_prev, W, b, "sigmoid")
        caches.append(cache)

        assert (AL.shape == (1, X.shape[1]))

        return AL, caches



