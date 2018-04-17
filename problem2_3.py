### file is supposed to be launched this way
#  python3 problem2_3.py input2.csv output2.csv

import sys
import numpy as np
import os

from linearRegression.LinearRegression import lr


if __name__ == "__main__":

    if len(sys.argv) == 3:

        #read the argument and load input data
        input_csv = sys.argv[1]
        input = np.genfromtxt(input_csv, delimiter=',')

        # let's get the number of dimensions
        dimension = input.shape[1]

        #let's get some input data set and output label
        X = input[:, 0:(dimension - 1)]     #training data sets
        Y = input[:, -1]                    #label of training data

        # get the output file
        output_csv = sys.argv[2]

        #prevent overwrite to the existing file. so, delete if there is any
        try:
            os.remove(output_csv)
        except OSError:
            pass

        # call the function for logic implementation
        lr(X, Y, output_csv)

    else:
        print("Illegal number of arguments."
              "file is supposed to be launched this way ::  "
              "python3 problem2_3.py input2.csv output2.csv")