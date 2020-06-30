import numpy as np
from sklearn.metrics import accuracy_score
from nn_utils import *

def initialize_parameters(n_x, n_y, n_h):

    W1 = np.random.randn(n_h, n_x)*1e-4
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*1e-4
    b2 = np.zeros((n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def forward_propagation(X, parameters):
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]

    Z1 = np.dot(W1, X.T) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)

    cache = {"Z1": Z1, "A1": A1,
             "Z2": Z2, "A2": A2}

    return A2, cache

def compute_cost(Y, A, parameters, lamda):
    m = Y.shape[0]

    cost = -1 / m * np.sum(
                     np.sum( Y.T * np.log(A) + (1 - Y.T) * np.log(1 - A), axis=0 )
                     )

    return cost + np.sum([lamda * np.sum(np.power(parameters["W"+str(i)], 2)) for i in [1,2]])

def backward_propagation(X, Y, cache, parameters, lamda):

    Z1, A1 = cache["Z1"], cache["A1"]
    Z2, A2 = cache["Z2"], cache["A2"]

    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]

    m = Y.shape[0]

    dZ2 = A2-Y.T
    dW2 = 1 / m * np.dot(dZ2, A1.T) + 2*lamda*W2
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * sigmoid_backward(Z1) # sigmoid_backward(Z2)
    dW1 = 1 / m * np.dot(dZ1, X) + 2*lamda*W1
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)


    grads = {"dW1": dW1, "db1": db1,
             "dW2": dW2, "db2": db2}

    return grads

def update_parameters(parameters, grads, learning_rate):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    W1 = W1-learning_rate*dW1
    b1 = b1-learning_rate*db1
    W2 = W2-learning_rate*dW2
    b2 = b2-learning_rate*db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def predictions(X, parameters):
    A, cache = forward_propagation(X, parameters)
    predictions = np.argmax(A, axis=0)

    return predictions
