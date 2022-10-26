import numpy as np
from scipy import linalg as scipylinalg



def householder(A, kmax=None):
    """
    Given a real mxn matrix A, find the reduction to upper triangular matrix R
    using Householder transformations.

    :param A: an mxn-dimensional numpy array
    :param kmax: an integer, the number of columns of A to reduce \
    to upper triangular. If not present, will default to n.

    :return R: an mxn-dimensional numpy array containing the upper \
    triangular matrix
    """
    
    m, n = A.shape
    if kmax is None:
        kmax = n
    #I use a loop 
    for k in range(0,kmax):
        x = A[k:m,k].copy()
        # Implementation of e_1
        e_1 = np.zeros((m-k),dtype=float)
        e_1[0] = 1
        # sign of x_1
        if x[0]  == 0:
            l = 1,
        else:
            l = np.sign(x[0])
        # v and A
        v = l*np.linalg.norm(x)*e_1 + x
        v = v / np.linalg.norm(v)
        A[k:m,k:n] = A[k:m,k:n] - 2*np.outer( v, np.dot( v.T, A[k:m,k:n]))
    # I copy A in R
    R= A[:,:]

    return R


def householder_solve(A, b):
    """
    Given a real mxm matrix A, use the Householder transformation to solve
    Ax_i=b_i, i=1,2,...,k.

    :param A: an mxm-dimensional numpy array
    :param b: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors b_1,b_2,...,b_k.

    :return x: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors x_1,x_2,...,x_k.
    """
    m = A.shape[0]
    #I am computing an extended array Ahat
    Ahat = np.append(A,b,axis=1)
    #I use the householder function that I defined above
    Ahat = householder(Ahat,m)
    #I write R and bhat using the numpy slice notation
    R = Ahat[:,:m]
    bhat = Ahat[:,m:]
    #Then I use the built-in triangular solve algorithm 
    x = scipylinalg.solve_triangular(R,bhat)

    return x


def householder_qr(A):
    """
    Given a real mxn matrix A, use the Householder transformation to find
    the full QR factorisation of A.

    :param A: an mxn-dimensional numpy array

    :return Q: an mxm-dimensional numpy array
    :return R: an mxn-dimensional numpy array
    """

    #we are finding the dimensions first
    m, n = A.shape
    
    I = np.eye(m)
    #I have just to adapt what I have done before
    Ahat = np.append(A,I,axis=1)
    Rhat = householder(Ahat, n)
    #I use the slice notation to write R
    R = Rhat[:,:n]
    Q_star = Rhat[:,n:]
    Q = Q_star.conj().T

    return Q, R


def householder_ls(A, b):
    """
    Given a real mxn matrix A and an m dimensional vector b, find the
    least squares solution to Ax = b.

    :param A: an mxn-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an n-dimensional numpy array
    """
    
    m,n = A.shape
    b = b.reshape(m,1)
    Ahat = np.append(A, b, axis=1)
    Ahat = householder(Ahat,n)
    #Using the slice notation I extract R and Q*b
    R = Ahat[:,:n]
    Q_star_b = Ahat[:,n]
    #I obtain R_hat and Q_hat*b so from the matrices that I have above using the slice notation and ellipsis
    Rhat = R[:n,...]
    Q_hat_star_b = Q_star_b[:n,...]
    #now to solve the triangular system R_hat x = Q_hat* b I use the scipy builtin function solve_triangular
    x = scipylinalg.solve_triangular(Rhat,Q_hat_star_b)

    return x
