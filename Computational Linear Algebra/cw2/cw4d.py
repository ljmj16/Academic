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
    for i in range (2,n+1):
        A[4*(i-1):4*i+1,4*(i-1):4*i+1] += 0.1*random.uniform(low=0,high=1,size=(5,5))
    return A

def oddLUbandedalgorithm(A,b):
    '''This function is the implementation of a modification of of the banded matrix algorithm that runs only on odd blocks'''
    
    n,_ = A.shape
    #The function is implemented in place
    for k in range (n-1):
        #Let's try to identify in which block we are 'working'
        block_number = 1+k//4
        #Let's define the max index
        max= 4*block_number+1
        if block_number%2!=0:
           A[k+1:max,k] = A[k+1:max,k]/A[k,k]
           A[k+1:max,k+1:max] = A[k+1:max,k+1:max] - np.outer( A[k+1:max,k],  A[k,k+1:max])
           B = 1.0*A[k:max,k:max]
           A[k:max,k:max] = np.triu(A[k:max,k:max])
           I = np.eye(max-k)
           L = I + np.tril(B,i=-1)
           b[k:max] = np.dot(L,b[k:max])
    return A, b


def ULalgorithm(A):
    '''Given a matrix A transforms it in lower triangulars '''
    n,m = A.shape
    L = 1.0*A
    U = np.eye(m)
    
    for l in range(1,m):
        U[:l,l] = L[:l,l]/L[l,l]
        L[:l,l:] = L[:l,l:] - np.outer(U[:l,l,], L[l,l:])
    return L,U  

def evenULalgorithm(A,b):
    '''Given a 4n+1x4n+1 matrix as in our case perform lower triangular transformation to the even blocks'''
    
    n,_ = A.shape
    for k in range (n-1):
        #Let's try to identify in which block we are 'working'
        block_number = 1+k//4
        #Let's define the max index
        max= 4*block_number+1
        if block_number%2==0:
            A[k:max,k:max],U = ULalgorithm(A[k:max,k:max])
            b[k:max] = np.dot(U,b[k:max])
    return A,b

A = creation_matrix(2)
b = np.random.randn(9)
A,b = evenULalgorithm(A,b)

