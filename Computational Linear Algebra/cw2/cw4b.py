import numpy as np
import cla_utils
from numpy import random
from numpy import copy
from copy import deepcopy

def creation_matrix(n):
    '''This function creates a random matrix with dimension (4n+1)x(4n+1), with respect of the property (2) given in the exercise 4'''
    random.seed(500)
    A = np.eye(4*n+1)
    A[0:5,0:5] += 0.1*random.uniform(low=0,high=1,size=(5,5))
    #I generate random 5x5 blocks to add to A
    for i in range (2,n+1):
        A[4*(i-1):4*i+1,4*(i-1):4*i+1] += 0.1*random.uniform(low=0,high=1,size=(5,5))
    return A


def bandedalgorithm(A):
    '''This function is the implementation of a modification of of the banded matrix algorithm'''
    
    n,_ = A.shape
    #The function is implemented in place
    for k in range (n-1):
        #Let's try to identify in which block we are 'working'
        block_number = 1+k//4
        #Let's define the max index
        max= 4*block_number+1
        A[k+1:max,k] = A[k+1:max,k]/A[k,k]
        A[k+1:max,k+1:max] = A[k+1:max,k+1:max] - np.outer( A[k+1:max,k],  A[k,k+1:max])
    return A





