from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
from nn_utils import *
from nn_functions import *

mndata = MNIST('mnist')

X_train, Y_train = mndata.load_training()
X_test, Y_test = mndata.load_testing()

X_train = np.asarray(X_train)
Y_train = np.eye(10)[np.asarray(Y_train)]
X_test  = np.asarray(X_test)
Y_test  = np.eye(10)[np.asarray(Y_test)]

print("Size of X_train = {}".format(X_train.shape))
print("Size of Y_train = {}".format(Y_train.shape))
print("Size of X_test = {}".format(X_test.shape))
print("Size of Y_test = {}".format(Y_test.shape))


def nn_model(X_train, Y_train, X_test, Y_test,
             learning_rate, num_epochs, mini_batch_size, lamda, num_neurons):

    n_x = X_train.shape[1]
    n_y = Y_train.shape[1]
    n_h = num_neurons

    costs = []

    parameters = initialize_parameters(n_x, n_y, n_h)

    for epoch in range(1, num_epochs+1):
        mini_batch_cost = 0.
        num_mini_batches = int(X_train.shape[0] / mini_batch_size)
        mini_batches = random_mini_batches(X_train, Y_train, mini_batch_size)
        for mini_batch in mini_batches:
            (X_mini_batch, Y_mini_batch) = mini_batch

            Y_hat, cache   = forward_propagation(X_mini_batch, parameters)
            temp_cost      = compute_cost(Y_mini_batch, Y_hat, parameters, lamda=lamda)
            grads          = backward_propagation(X_mini_batch, Y_mini_batch, cache, parameters, lamda=lamda)
            parameters     = update_parameters(parameters, grads, learning_rate=learning_rate)

            mini_batch_cost += temp_cost / num_mini_batches

        if epoch % 5 == 0 or epoch == 1:
            costs.append(mini_batch_cost)
            print("Mini-batch cost, epoch {} = {}".format(epoch, mini_batch_cost))


    print("Making predictions on test set...")
    # print(parameters)
    preds = predictions(X_test, parameters)
    trues = np.argmax(Y_test, axis=1)
    print("Test set accuracy = {}".format(accuracy_score(trues, preds)))
    print("Plotting cost over epochs...")
    plt.plot(costs)
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.show();


if __name__ == "__main__":
    nn_model(X_train, Y_train, X_test, Y_test,
             learning_rate=0.05, num_epochs=50, mini_batch_size=1024,
             lamda=0.01, num_neurons=32)
