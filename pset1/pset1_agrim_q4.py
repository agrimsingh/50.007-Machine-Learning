import numpy as np
import matplotlib.pyplot as plt

"""
Linear transforms in 2 variables
"""
#4A
def generate_gaussian_data(n):
    x1 = np.random.normal(1,1,n)
    x2 = np.random.normal(3,1,n)
    return np.array([x1,x2])

def question4():
    #Plot the 100 points
    x = generate_gaussian_data(100)
    plt.plot(x[0,:],x[1,:],'r.')
    plt.axhline(0)
    plt.axvline(0)
    plt.show()


    #Linear mapping A1 - 2x2 matrix to mirror data across Y axis

    a1 = [[-1, 0], [0, 1]]
    a1x = np.dot(a1, x)
    plt.plot(a1x[0,:],a1x[1,:],'g.')
    plt.axhline(0)
    plt.axvline(0)
    plt.show()

    #Linear mapping A2 - scale X axis by 0.5
    """
    scaling is [[x, 0] based on whichever axis is being scaled
                [0, y]]
    """

    a2 = [[0.5, 0], [0, 1]]

    a2x = np.dot(a2, x)

    plt.plot(a2x[0,:],a2x[1,:],'b.')
    plt.axhline(0)
    plt.axvline(0)
    plt.show()

    #Linear mapping A3 - rotate data by 45 degrees

    """
    rotation is done by matrix [[cos T, sin T]
                                [-sin T, cos T]]
    for 45 degrees it comes with dilation of sqrt(2). new matrix is [[1, 1], [-1, 1]]
    """

    a3 = [[1, 1], [-1, 1]]
    a3x = np.dot(a3, x)
    plt.plot(a3x[0,:],a3x[1,:],'y.')
    plt.axhline(0)
    plt.axvline(0)
    plt.show()

    #Linear mapping A4 - mirror data along X axis

    a4 = [[1, 0], [0, -1]]
    a4x = np.dot(a4, x)

    #Linear mapping A5 - A2A1A4 = ((A2A1)A4)

    a5 = np.dot(np.dot(a2, a1), a4)
    a5x = np.dot(a5, x)
    plt.plot(a5x[0,:],a5x[1,:],'k.')
    plt.axhline(0)
    plt.axvline(0)
    plt.show()

question4()
