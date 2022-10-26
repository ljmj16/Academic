import numpy as np
from numpy import random


def flattenerslow(u):
    '''takes a 2D numpy array and returns the equivalent serialised 1D vector v'''
    m,n = u.shape
    v = np.zeros(m*n)
    for k in range(m*n):
        v[k*n:(k+1)*n] = u[k,:]
    return v

def fastserialiser(A):
    '''The same as the previous one but with the .flatten'''
    return (np.matrix.flatten(A.T))

def reformer(u,m,n):
    '''does the inverse trasformation, taking a 1D array and creating the 2D version'''
    return (u.reshape(n,m)).T

