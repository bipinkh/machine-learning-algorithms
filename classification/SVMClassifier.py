
import csv

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV

from .ClassifierProperties import classifier_list


def SVM(X, Y, output_file):

    names = []          #list to hold all the names
    best_scores =[]     #list to hold all the train scores
    test_scores = []    #list to hold all the test scores

    # split into train and test data
    train_index, test_index = stratified_split(X, Y)
    X_train = X[train_index]
    Y_train = Y[train_index]
    X_test = X[test_index]
    Y_test = Y[test_index]

    # scale the data
    X_train = scale(X_train)
    X_test = scale(X_test)

    # implement all the classifier available
    for classifier in classifier_list:

        # get kernels and the parameters
        svr = classifier["svr"]
        parameters = classifier["parameters"]

        # train the classifier
        model = GridSearchCV(svr, parameters, cv=5)
        model.fit(X_train, Y_train)

        #predict the result
        trainDataPrediction = model.predict(X_train)
        testDataPrediction = model.predict(X_test)

        #calculate the score
        test_score = accuracy_score(testDataPrediction, Y_test)
        best_score = accuracy_score(trainDataPrediction, Y_train)

        #record the result
        names.append(classifier["name"])
        best_scores.append(best_score)
        test_scores.append(test_score)


    #write output to the file
    with open(output_file, 'a', newline='') as csvfile:
        fieldnames = ['name', 'best_score', 'test_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        for i in range (0, len(names)):
            writer.writerow({
                    'name': names[i],
                    'best_score': best_scores[i],
                    'test_score': test_scores[i]
                 })

    return "SUCCESS"


# function for stratified sampling
def stratified_split(X, Y):
    stratified_sampler = StratifiedShuffleSplit(n_splits=1, test_size=0.4, train_size=0.6, random_state=0)
    train_index = []
    test_index = []

    for t1_index, t2_index in stratified_sampler.split(X, Y):
        train_index = t1_index
        test_index = t2_index

    return train_index, test_index


# scale the data
def scale(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std
    return X