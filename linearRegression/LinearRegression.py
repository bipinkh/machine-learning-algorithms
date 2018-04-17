
import numpy as np
import csv

minThreshold = 0.001
learn_rate = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 0.9]

def lr(dataItems, actual_op, output_file):

    #let's scale these all columns
    dataItems = scaling(dataItems)

    #add extra column of 1 in the begining of the data matrix
    dataItems = np.insert(dataItems, obj=0, values=1, axis=1)

    trainDataNum = dataItems.shape[0]
    dimension = dataItems.shape[1]
    actual_op = actual_op.reshape(trainDataNum, 1)


    #let's run the algorithm for each given learning rate

    for alpha in learn_rate:

        #run each interation for 100 times with different alpha
        loopCount = 0

        # number of weights is same as the number of dimension with one extra constant term for the polynomial
        beta = np.zeros([dimension, 1])

        # Update beta until convergence
        while loopCount < 100:
            beta_prev = beta
            f_X = np.dot(dataItems, beta)

            # calculation of risk derivative obtained from the formula
            dR = 1 / trainDataNum * np.dot(np.transpose(dataItems), (f_X - actual_op))

            #calculate new weights
            beta = beta - alpha * dR

            #check how much variation is there
            deviation = np.linalg.norm(beta - beta_prev, ord=1)

            # if there is not so much of variation, we thus obtained the converged value of the weights
            # or, stop if the loop count has reached maximum
            if deviation < minThreshold or loopCount > 100:
                break

            #d loop counter
            loopCount = loopCount + 1

        with open(output_file, 'a', newline='') as csvfile:
            fieldnames = ['alpha', 'iterations', 'b_0', 'b_age', 'b_weight']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(
            {'alpha': alpha, 'iterations': loopCount, 'b_0': float(beta[0]), 'b_age': float(beta[1]),
                 'b_weight': float(beta[2])}
            )

                # {'alpha': alpha,
                #  'iterations': loopCount,
                #  'b_0': ("%.5f" % round(float(beta[0]),5)),
                #  'b_age': ("%.5f" % round(float(beta[1]),5)),
                #  'b_weight': ("%.5f" % round(float(beta[2]),5))
                #  }
            # )



    return "SUCCESS"


#   we scale each data dimension so that no any dimension could dominate in the graph
def scaling(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data = (data - mean) / std
    return data


# this function is gracefully copied from https://github.com/aabs/edx-ai-week7-project
# thanks to the owner of github usename @aabs

def visualize(X, Y, beta):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    age = X[:, 0]
    weight = X[:, 1]
    height = Y
    age_grid, weight_grid = np.meshgrid(age, weight)
    # ax.plot_surface(age, weight, )
    ax.plot_surface(age_grid, weight_grid, beta[1] * age + beta[2] * weight + beta[0], rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)
    ax.scatter(age, weight, height, c='red')
    ax.set_xlabel('Age(Years)')
    ax.set_ylabel('Weight(Kilograms)')
    ax.set_zlabel('Height(Meters)')

    plt.show()
