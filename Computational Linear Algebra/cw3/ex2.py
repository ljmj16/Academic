import numpy as np
import cla_utils
import matplotlib.pyplot


def matrixconstruction(n):
    '''Construct a matrix A 2nx2n, like the one requested in the exercise 2 of the coursework'''
    A = np.zeros((2*n,2*n))
    A += np.diag(np.ones(2*n-1),k=1)
    A += np.diag(-np.ones(2*n-1),k=-1)
    return A

A = matrixconstruction(10)
#print(np.linalg.eig(A))
A = cla_utils.pure_QR(A,100,1.0e-6)
# to plot :
# matplotlib.pyplot.matshow(A.real)
# matplotlib.pyplot.show()
# print(A)
