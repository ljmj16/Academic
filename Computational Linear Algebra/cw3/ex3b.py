import numpy as np
import cla_utils
from cla_utils.exercises8 import hessenberg
import scipy
from cla_utils.exercises9 import pure_QR
import matplotlib.pyplot


def creationofA(m):
    '''Creates the matrix A of exercisse 3c'''
    A = np.zeros((m,m))
    for i in range (m):
        for j in range (m):
            A[i,j] = 1/(i+j+1)
    return A

A = creationofA(5)
B = cla_utils.hessenberg(A)
#If we want to plot them:
#matplotlib.pyplot.matshow(B)
#matplotlib.pyplot.show()
# print(B)
C = cla_utils.pure_QR(B,1000,0,case=True)
#If we want to plot:
#matplotlib.pyplot.matshow(C.real)
#matplotlib.pyplot.show()
#print(C)

eigenvalues2 = np.diagonal(C)
print(np.linalg.det(A-eigenvalues2[3]*np.ones((5,5))))
#print (eigenvalues2)

