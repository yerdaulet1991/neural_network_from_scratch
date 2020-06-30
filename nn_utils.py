from math import floor
import numpy as np

def random_mini_batches(X, Y, mini_batch_size = 256):

    m = X.shape[0]
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation,:]

    num_complete_minibatches = floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def sigmoid_backward(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))

def softmax(Z):
    """numerically stable version of softmax"""
    shift_Z = Z - np.max(Z)
    exps     = np.exp(shift_Z)
    return exps / np.sum(exps, axis=0)

# def relu(Z):
#     return np.maximum(0, Z)
#
# def relu_backward(Z):
#     Z[Z <= 0] = 0
#     Z[Z > 0]  = 1
#     return Z
