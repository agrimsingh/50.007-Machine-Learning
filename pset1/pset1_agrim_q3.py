import matplotlib.pyplot as plt
#import scipy as sp
import numpy as np
import copy

"""
X normal error 26.00%
X scaled by 1000 error 49.00%
Standard deviation:  [0.57246035382147897, 0.58166186606711379]
X scaled by standard deviation error 26.00%

data for the photos provided.
"""

def plotData(data, label, name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data[0],data[1], c=label)
    plt.show
    plt.savefig(name + ".png")

def computeError(originalData, predictedData):
    error = 0
    for original, predicted in zip(originalData, predictedData):
        error += (1 if original != predicted else 0)
    return (error * 100) / len(originalData)

#PART B
def generateYfromX(X):
    return [np.random.choice(a = [-1, 1], p = [0.25, 0.75] if x >= 0 else [0.75, 0.25]) for x in X[1]]

X2 = sp.random.uniform(low=-1, high=1, size=(5000,2)).T
y2 = generateYfromX(X2)
X3 = sp.random.uniform(low=-1, high=1, size=(1000,2)).T
y3 = generateYfromX(X3)
plotData(X2, y2, '3-training')

#PART C
def getDistanceSquared(x0, x1, xTest0, xTest1):
    return ((x0 - xTest0) ** 2 + (x1 - xTest1) ** 2)

def knn(xtrain, ytrain, xtest, k):
    ytest=[]
    for xTest0, xTest1 in zip(xtest[0], xtest[1]):
        distances = [(getDistanceSquared(x0, x1, xTest0, xTest1), y) for x0, x1, y in zip(xtrain[0], xtrain[1], ytrain)]
        distances.sort(key=lambda distance: distance[0])
        ySum = sum([distance[1] for distance in distances[:k]])
        ytest.append(1 if ySum >= 0 else -1)
    return ytest

#PART D
'''
Although the core of the algorithm calls for a square root, we don't require it since the points of minimum euclidian metrics
will have minimum distance after square rooting as well.
'''

#PART E
yPredict = knn(X2, y2, X3, 100)
print ("X normal error %02.2f%%" % computeError(y3, yPredict))
plotData(X3, yPredict, '3-normal')
'''Average error is 25%'''

#part f
def getScaledDimensionalData (dimensionalData, scale):
    return [point * scale for point in dimensionalData]
def getScaledData(data, scales, dimensions):
    dataCopy = copy.deepcopy(data)
    for dimension, scale in zip(dimensions, scales):
        dataCopy[dimension] = getScaledDimensionalData(dataCopy[dimension], scale)
    return dataCopy

X2scaled = getScaledData(X2, [1000], [0])
X3scaled = getScaledData(X3, [1000], [0])
yPredictScaled = knn(X2scaled, y2, X3scaled, 100)
print ("X scaled by 1000 error %02.2f%%" % computeError(y3, yPredictScaled))
plotData(X3scaled, yPredictScaled, '3-scaled')
'''Average error is 50%'''

#PART G
'''
The scaled error is about twice as high as the normal error.
'''

#PART H
X2std = [np.std(dimensionalData) for dimensionalData in X2]
print "Standard deviation: ", X2std

#PART I
X2scaledStd = getScaledData (X2, X2std, [0, 1])

X3std = [np.std(dimensionalData) for dimensionalData in X3]
X3scaledStd = getScaledData (X3, X3std, [0, 1])

yPredictScaledStd = knn(X2scaledStd, y2, X3scaledStd, 100)
print ("X scaled by standard deviation error %02.2f%%" % computeError(y3, yPredictScaledStd))
plotData(X3scaledStd, yPredictScaledStd,'3-stdScaled')
'''Average error 25%'''

#PART K
'''
With standard deviation of x0 as s0 and standard deviation of x1 as s1
d = sqrt ((x0 - xTest0) ** 2 / (s0 ** 2) + (x1 - xTest1) ** 2 / (s1 ** 2))
'''
