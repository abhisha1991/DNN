from DNN.dnn_backprop import *
from DNN.dnn_forward import *
from DNN.cost import *
from DNN.dnn_utils_v2 import *
from DNN.testCases_v3 import *


class DNNTest(object):

    def perform_tests(self):
        fm = DNNForwardModel()
        bk = DNNBackProp()

        parameters = fm.initialize_parameters(3, 2, 1)
        print("W1 = " + str(parameters["W1"]))
        print("b1 = " + str(parameters["b1"]))
        print("W2 = " + str(parameters["W2"]))
        print("b2 = " + str(parameters["b2"]))

        parameters = fm.initialize_parameters_deep([5, 4, 3])
        print("W1 = " + str(parameters["W1"]))
        print("b1 = " + str(parameters["b1"]))
        print("W2 = " + str(parameters["W2"]))
        print("b2 = " + str(parameters["b2"]))

        A, W, b = linear_forward_test_case()

        Z, linear_cache = fm.linear_forward(A, W, b)
        print("Z = " + str(Z))

        A_prev, W, b = linear_activation_forward_test_case()

        A, linear_activation_cache = fm.linear_activation_forward(A_prev, W, b, activation="sigmoid")
        print("With sigmoid: A = " + str(A))

        A, linear_activation_cache = fm.linear_activation_forward(A_prev, W, b, activation="relu")
        print("With ReLU: A = " + str(A))

        X, parameters = L_model_forward_test_case_2hidden()
        AL, caches = fm.L_model_forward(X, parameters)
        # This was for a 2 layer NN with input layer = 5 nodes, h1 = 4 nodes, h2 = 3 nodes, output layer = 1 node
        print("AL = " + str(AL))
        print("Note that sum of all items in AL = 1 because last layer is sigmoid")
        print(sum(np.squeeze(AL)))
        print("Length of caches list = " + str(len(caches)))

        Y, AL = compute_cost_test_case()

        print("cost = " + str(compute_cost(AL, Y)))

        # Set up some test inputs
        dZ, linear_cache = linear_backward_test_case()

        dA_prev, dW, db = bk.linear_backward(dZ, linear_cache)
        print("dA_prev = " + str(dA_prev))
        print("dW = " + str(dW))
        print("db = " + str(db))

        AL, linear_activation_cache = linear_activation_backward_test_case()

        dA_prev, dW, db = bk.linear_activation_backward(AL, linear_activation_cache, activation="sigmoid")
        print("sigmoid:")
        print("dA_prev = " + str(dA_prev))
        print("dW = " + str(dW))
        print("db = " + str(db) + "\n")

        dA_prev, dW, db = bk.linear_activation_backward(AL, linear_activation_cache, activation="relu")
        print("relu:")
        print("dA_prev = " + str(dA_prev))
        print("dW = " + str(dW))
        print("db = " + str(db))

        AL, Y_assess, caches = L_model_backward_test_case()
        grads = bk.L_model_backward(AL, Y_assess, caches)
        print_grads(grads)

        parameters, grads = update_parameters_test_case()
        parameters = bk.update_parameters(parameters, grads, 0.1)

        print("W1 = " + str(parameters["W1"]))
        print("b1 = " + str(parameters["b1"]))
        print("W2 = " + str(parameters["W2"]))
        print("b2 = " + str(parameters["b2"]))

