# IMPLEMENTATION OF PERCEPTRON ALGORITHM

import numpy as np
import csv


def Perceptron(X, Y, output_file):
    '''
        n - Number of rows in the training data
        X - Training data with a column of 1's added to it
        Y - Labels of the training data
        f_X - Affine function which is evaluated
        w - Weight vector at each iteration
        iter - Keeps track of the number of iterations
        weights - List containing the entire sequence of weight vectors for every iteration

    '''

    # shape[0] is row and shape[1] is column
    # let's get number of rows or number of training data sets in n
    n = X.shape[0]
    X = np.insert(X, obj=X.shape[1], values=1, axis=1)
    w = np.zeros(X.shape[1])     # returns an array of 0 of length equal to the column length which we will consider as weight vector for each column or say for each feature / dimension
    iter = 0
    weights = []

    # Updating weights until convergence
    while True:
        w_initial = w
        for i in range(n):                                 # n is the number of training data set
            f_X = np.dot(w, X[i])                          # dot product of weight and first training data set

            #now, let's implement sigmoid activation function in an easy way

            if f_X > 0:
                f_X = 1
            else:
                f_X = -1

            #adjust the weight here
            if Y[i] * f_X <= 0:
                w = w + Y[i] * X[i]

        #check if there is convergence in weight

        deviation = np.linalg.norm(w - w_initial, ord=1)
        iter = iter + 1
        # record the weight vector here
        weights.append(w)
        if deviation == 0:
            break

    # Visualizes the final graph
    # visualize(X, Y, w)

    # Printing the weights in the output_file
    for w in weights:
        with open(output_file, 'a', newline='') as csvfile:
            fieldnames = ['w_A', 'w_B', 'w_intercept']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'w_A': w[0], 'w_B': w[1], 'w_intercept': w[2]})

    csvfile.close()

    return "SUCCESS"


# VISUALIZING THE TRAINING DATA AND THE PREDICTED LINE
def visualize(X, Y, w):
    import matplotlib
    import matplotlib.pyplot as plt

    label1 = [i for i, label in enumerate(Y) if label == 1]
    label2 = [i for i, label in enumerate(Y) if label == -1]

    X_pos = X[label1]
    X_neg = X[label2]

    fig = plt.gcf()
    fig.canvas.set_window_title('Assignment-3.1: Perceptron')

    plt.plot(X_pos[:, 0], X_pos[:, 1], 'o', color='blue')
    plt.plot(X_neg[:, 0], X_neg[:, 1], 'o', color='red')
    plt.xlabel('A')
    plt.xlim(0, 16)
    plt.ylabel('B')
    plt.ylim(-30, 30)

    intercept = -w[2] / w[1]
    slope = -w[0] / w[1]
    plt.plot(X, X * slope + intercept, color='black')
    plt.grid(True)
    plt.show()