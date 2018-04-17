 ### file is supposed to be launched this way
 #  python3 problem1_3.py input1.csv output1.csv

import sys
import numpy as np

from perceptron.PerceptronLearning import pla

if __name__ == "__main__":

    if len(sys.argv)==3:

        #read the argument and load input data
        input_csv = sys.argv[1]
        input = np.genfromtxt(input_csv, delimiter=',')

        #let's get the number of dimensions
        dimension = input.shape[1]

        X = input[:, 0:(dimension-1)]                    #training data sets
        Y = input[:, -1]                                #label of training data

        #get the output file
        output_file = sys.argv[2]

        #call the function for logic implementation
        pla(X, Y, output_file)

    else:
        print ("Illegal number of arguments."
               "file is supposed to be launched this way ::  "
               "python3 problem1_3.py input1.csv output1.csv")