import numpy as np
import h5py
import matplotlib.pyplot as plt
from DNN.testCases_v3 import *
from DNN.dnn_utils_v2 import *


class DNNBackProp(object):

    def linear_backward(self, dZ, cache):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev = cache[0]
        W = cache[1]
        b = cache[2]

        print("Shape of A_prev: " + str(A_prev.shape))
        print("Shape of W: " + str(W.shape))
        print("Shape of b: " + str(b.shape))
        print("Shape of dZ: " + str(dZ.shape))
        m = A_prev.shape[1]

        dW = 1 / m * np.dot(dZ, A_prev.T)  # dZ wrt W is going to be A, hence, dL/dW = (dL/dZ) * (dZ/dW) = dZ * A
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)  # add dZ for all m examples
        dA_prev = np.dot(W.T, dZ)  # dZ wrt A is going to be W, hence, dL/dA = (dL/dZ) * (dZ/dA) = dZ * W

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.

        Arguments:
        dA -- post-activation gradient for current layer l
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        linear_cache, activation_cache = cache
        # linear cache has values A_prev, W, b - from which we calculate dA_prev, dW, db
        # activation cache has value Z from which we calculate dZ

        if activation == "relu":
            dZ = relu_backward(dA, activation_cache) # dL/dZ = (dL/dA) * (dA/dZ). The second term depends on what type of activation
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        elif activation == "sigmoid":
            dZ = sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    def L_model_backward(self, AL, Y, caches):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

        Returns:
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ...
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ...
        """
        grads = {}
        L = len(caches)  # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

        # Initializing the backpropagation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))  # This assumes a logistic loss [generally good for classifcation problems]

        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
        current_cache = caches[L - 1]
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(AL, current_cache, activation="sigmoid")

        for l in reversed(range(L - 1)):  # will contain a reversed list starting from l-2 going up to 0
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
            current_cache = caches[l]
            # Note that now the input will be gradient of the right most layer, not the last layer
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation="relu")
            grads["dA" + str(l + 1)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    # GRADED FUNCTION: update_parameters

    def update_parameters(self, parameters, grads, learning_rate):
        """
        Update parameters using gradient descent

        Arguments:
        parameters -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients, output of L_model_backward

        Returns:
        parameters -- python dictionary containing your updated parameters
                      parameters["W" + str(l)] = ...
                      parameters["b" + str(l)] = ...
        """

        L = len(parameters) // 2  # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        for l in range(1, L + 1):
            parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
            parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
        return parameters