'''
@author: agrim
'''
import numpy as np

#Question2
def data_generator1(N): #2b
    mean = np.zeros(48) #2c - mean is the zero vector
    diagonalVector = [1.0/(i+1.0) for i in xrange(48)]
    covar = np.identity(48)*diagonalVector #2c - covariance matrix diagonal entry is 1/(1+i) for dimension i
    x = np.array([np.random.multivariate_normal(mean, covar) for k in xrange(N)]).T

    v = np.ones((48, 1)) #2e
    epsilon = np.random.normal(0, 2, (1, N))
    y = np.reshape(np.dot(v.T, x) + epsilon, (N,1))

    return x, y

"""
2(d) Yes it is possible to create this feature from 48 1-D draw
since the covariance matrix is diagonal, therefore we only need to generate 48
univariate normal distributions with mean 0, variance (1/(i+1)), i = xrange(48)
"""

#Question3
def gen_coefficients(): #3b
    coeff = np.append(
      np.random.uniform(0.6, 1, 12),
      np.random.uniform(0, 0.2, 36)
    )
    np.random.shuffle(coeff)
    v = np.reshape(coeff, (48,1))
    return v

#Question4
def data_generator2(N, v): #4b
    covar = np.identity(48) #4c
    mean = np.zeros(48)
    x = np.array([np.random.multivariate_normal(mean, covar) for k in xrange(N)]).T
    epsilon = np.random.normal(0, 2, (1, N))
    y = np.reshape(np.dot(v.T, x) + epsilon, (N, 1))
    return x, y, v

"""
4(d) It is possible to create the feature by taking a vector consists of
48 univariate normal distributions with mean 0, variance 1.
"""
