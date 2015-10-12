"""
@author: agrim
"""

import numpy as np
from pset2_agrim_part1 import data_generator1, data_generator2, gen_coefficients

#Question5
def train_ridgeregression(x, y, lmda): #5b (lmda is lambda here)
    prod1 = np.dot(x.T, x) + lmda * np.identity(len(x.T))
    prod2 = np.dot(np.linalg.inv(prod1), x.T)
    w = np.dot(prod2.T, y)
    return w

#Question6
print "Question 6"

def calculate_MSE(trainX, trainY, testX, testY, lmda):
    w = train_ridgeregression(trainX, trainY, lmda)
    predictY = np.dot(testX.T, w)
    length = len(testY)
    total = 0
    for i in xrange(length):
        total += np.square(predictY[i][0] - testY[i][0])
    return total / length

def show_average_error(lambdas, training_samples_count, testing_samples_count):#6a
    print "Using", training_samples_count, "training samples"
    errors = np.zeros(len(lambdas))
    iteration_count = 10 #6b
    for i in xrange(iteration_count):
        trainX, trainY = data_generator1(training_samples_count)
        testX, testY = data_generator1(testing_samples_count)
        for index, lmda in enumerate(lambdas):
            mse = calculate_MSE(trainX, trainY, testX, testY, lmda)
            errors[index] += mse / iteration_count
    for lmda, error in zip(lambdas, errors):
        print "MSE with lambda =", lmda, "is", error

lambdas = [np.exp(-30), 5] #6c
show_average_error(lambdas, 100, 1000)
show_average_error(lambdas, 500, 1000) #6e

"""
6(d) We achieved better results with lambda = 5

6(f) The MSE is lower with more training samples for lambda = 5
Conversely, we have a higher MSE with more training samples for lambda = 1e -30

6(g) As more training data become available, stronger regularization i.e. a higher lambda value should be used.
"""

#Question7
print "Question 7"

def holdout():
    errors = []
    iteration_count = 10 #7a
    for i in xrange(iteration_count):
        samples_count = 500
        testing_samples_count = 100
        training_samples_count = samples_count - testing_samples_count

        lmda = np.exp(-2)
        x, y, v = data_generator2(samples_count, gen_coefficients())
        permuted = np.random.permutation(zip(x.T, y))
        testXtrans, testY = zip(*permuted[0:testing_samples_count])
        testX = np.array(testXtrans).T
        testY = np.array(testY)

        trainXtrans, trainY = zip(*permuted[testing_samples_count:])
        trainX = np.array(trainXtrans).T
        trainY = np.array(trainY)

        mse = calculate_MSE(trainX, trainY, testX, testY, lmda)
        errors.append(mse)

    print "Holdout average MSE: ", sum(errors)/len(errors) #7b
    print "Holdout variance: ", np.var(errors)

holdout()

def five_fold(): #7c
    errors = []
    iteration_count = 10
    sub_iteration_count = 5

    for i in xrange(iteration_count):
        samples_count = 500
        testing_samples_count = 100
        training_samples_count = samples_count - testing_samples_count

        lmda = np.exp(-2)
        x, y, v = data_generator2(samples_count, gen_coefficients())
        permuted = np.random.permutation(zip(x.T, y))

        mse = 0
        for j in xrange(sub_iteration_count):
            testXtrans, testY = zip(*permuted[j*testing_samples_count:(j+1)*testing_samples_count])
            testX = np.array(testXtrans).T
            testY = np.array(testY)
            permutedTrain = np.append(permuted[0:j*testing_samples_count], permuted[(j+1)*testing_samples_count:], axis=0)
            trainXtrans, trainY = zip(*permutedTrain)
            trainX = np.array(trainXtrans).T
            trainY = np.array(trainY)

        mse += calculate_MSE(trainX, trainY, testX, testY, lmda) / sub_iteration_count #7d

    errors.append(mse)

    print "Five fold average MSE: ", sum(errors)/len(errors) #7e
    print "Five fold variance: ", np.var(errors)

five_fold()

"""
7(f) 5-fold cross-validation protocol results in both smaller variance and better estimate i.e. lower average MSE.
The difference in average MSE, however, is insignificant.
"""
