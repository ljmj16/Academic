import numpy as np
import cla_utils
from numpy import random

def creation_matrix(n):
    '''This function creates a random matrix with dimension (4n+1)x(4n+1), with respect of the property (2) given in the exercise 4'''
    random.seed(500)
    A = np.eye(4*n+1)
    A[0:5,0:5] += 0.1*random.uniform(low=0,high=1,size=(5,5))
    for i in range (2,n+1):
        #I generate random 5x5 blocks to add to A
        A[4*(i-1):4*i+1,4*(i-1):4*i+1] += 0.1*random.uniform(low=0,high=1,size=(5,5))
    return A


n = 3
A = creation_matrix(n)

A = cla_utils.LU_inplace(A)
U = np.triu(A) 
L = np.eye(4*n+1) + np.tril(A, k=-1) 


print (L,U)