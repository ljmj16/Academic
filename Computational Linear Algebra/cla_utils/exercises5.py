import numpy as np
from numpy import random

from cla_utils.exercises3 import householder_solve


def randomQ(m):
    """
    Produce a random orthogonal mxm matrix.

    :param m: the matrix dimension parameter.
    
    :return Q: the mxm numpy array containing the orthogonal matrix.
    """
    Q, R = np.linalg.qr(random.randn(m, m))
    return Q


def randomR(m):
    """
    Produce a random upper triangular mxm matrix.

    :param m: the matrix dimension parameter.
    
    :return R: the mxm numpy array containing the upper triangular matrix.
    """
    
    A = random.randn(m, m)
    return np.triu(A)


def backward_stability_householder(m):
    """
    Verify backward stability for QR factorisation using Householder for
    real mxm matrices.

    :param m: the matrix dimension parameter.
    """
    # repeat the experiment a few times to capture typical behaviour
    for k in range(20):
        Q1 = randomQ(m)
        R1 = randomR(m)
        A = np.dot(Q1,R1)
        #I find Q2 and R2
        Q2,R2 = np.linalg.qr(A)
        #Now I calculate the three norms which I call k1,k2,k3
        k1 = np.linalg.norm(Q2-Q1)
        k2 = np.linalg.norm(R2-R1)
        k3 = np.linalg.norm(A-np.dot(Q2,R2))
        #Now I can print the results
        print(k1,k2,k3) 

def solve_R(R, b):
    """
    Solve the system Rx=b where R is an mxm upper triangular matrix 
    and b is an m dimensional vector.

    :param A: an mxm-dimensional numpy array
    :param b: an m-dimensional numpy array

    :param x: an m-dimensional numpy array
    """

    m = len(b)
    #I initialize a vector of zeros
    x = np.empty(m) 
    #first step
    x[m-1] = b[m-1] / R[m-1,m-1] 
    #loop backwards
    for i in range(m-2,-1,-1): 
        alpha = 1 / R[i,i]
        x[i] = alpha*(b[i] - np.dot(R[i,i+1:],x[i+1:]))            
    return x


def back_stab_solve_R(m):
    """
    Verify backward stability for back substitution for
    real mxm matrices.

    :param m: the matrix dimension parameter.
    """
    # repeat the experiment a few times to capture typical behaviour
    for k in range(20):
        A = random.randn(m, m)
        R = np.triu(A)
        #I proceed as for the stability check done before
        x = random.randn(m)
        #I compute the vector b
        b = np.dot(A,x) 
        #Then I calculare x again starting from the previous function
        x1 = solve_R(R,b)
        #Then I calculate the norm of x1-x
        k = np.linalg.norm(x1-x)
        print(k)


def back_stab_householder_solve(m):
    """
    Verify backward stability for the householder algorithm
    for solving Ax=b for an m dimensional square system.

    :param m: the matrix dimension parameter.
    """
    for k in range(20):
        #random generation
        A = random.randn(m, m)
        x = random.randn(m)
        #creation of b
        b = np.dot(A,x)
        #I calculate x using householder
        x1 = householder_solve(A,b)
        #norm
        k = np.linalg.norm(x1-x)
        print(k)


