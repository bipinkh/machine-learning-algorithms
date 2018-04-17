
import csv
import numpy as np

def pla(trainDataSet, labelSet, op_file):

    loopCount = 0
    weightsList = []

    # shape[0] is row and shape[1] is column
    # let's get number of rows or number of training data sets in trainNumber

    trainDataNum = trainDataSet.shape[0]
    trainDataSet = np.insert(trainDataSet, obj=trainDataSet.shape[1], values=1, axis=1)

    weightVector = np.zeros(trainDataSet.shape[1])
    # returns an array of 0 of length equal to the column length which we will consider as weight vector for each column or say for each feature / dimension



    while True:
        w_initial = weightVector
        for i in range(trainDataNum):                                 # n is the number of training data set
            f_X = np.dot(weightVector, trainDataSet[i])               # dot product of weight and first training data set

            #now, let's implement sigmoid activation function in an easy way

            if f_X > 0:
                f_X = 1
            else:
                f_X = -1

            #update the weight
            if labelSet[i] * f_X <= 0:
                weightVector = weightVector + labelSet[i] * trainDataSet[i]

        #check if there is convergence in weight
        diff = np.linalg.norm(weightVector - w_initial, ord=1)
        loopCount = loopCount + 1

        # record the weight vector here
        weightsList.append(weightVector)
        if diff == 0:
            break

    # Visualize the final graph
    visualize(trainDataSet, labelSet, weightVector)

    # Printing the weights in the output_file
    for w in weightsList:
        with open(op_file, 'a', newline='') as csvfile:
            fieldnames = ['weight_X', 'weight_B', 'weight_intercept']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'weight_X': w[0], 'weight_B': w[1], 'weight_intercept': w[2]})
    csvfile.close()

    return "SUCCESS"


# this function is gracefully copied from https://github.com/aabs/edx-ai-week7-project

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