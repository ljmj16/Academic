import numpy as np


def operator_2_norm(A):
    """
    Given a real mxn matrix A, return the operator 2-norm.

    :param A: an mxn-dimensional numpy array

    :return o2norm: the norm
    """
    #I create A^TA and I look for its eigenvalue
    B = np.dot(A.transpose(),A) 
    #A vector containing all the eigenvalues of B
    l = np.linalg.eig(B) 
    #I find the max of l
    maxeig = np.amax(l[0])
    #Is just the square root of the max eigenvalue
    o2norm = np.sqrt(maxeig) 
    return o2norm

def verify1(m,n):
    """
    Given m,n generates a random matrix A mxn and a random array x and verifies that the inequality ||Ax||<=||A||||X|| holds
    """
    np.random.seed(3168*m+2765*n)
    A = np.random.randn(m,n) #generates random matrix
    x = np.random.randn(n) #generates random array
    #I check the inequality
    if operator_2_norm(A) <= np.linalg.norm(A)*np.linalg.norm(x): 
        print('True')
    else:
        print('False')

def verify2(l,m,n):
    """
    Given m,n,l generates random matrices A and B and verifies that the inequality ||AB||(l,n) <= ||A||(l,m)||X||(m,n) holds
    """
    np.random.seed(3168*m+2765*n)
    A = np.random.randn(m,n) #generates random matrix
    B = np.random.randn(m,n) #generates random matrix
    C = np.dot(A,B)
    if operator_2_norm(C) <= operator_2_norm(A)*operator_2_norm(B): #checks the inequality
        print('True')
    else:
        print('False')


def cond(A):
    """
    Given a real mxn matrix A, return the condition number in the 2-norm.

    :return A: an mxn-dimensional numpy array

    :param ncond: the condition number
    """

    #We can see the condition number as the ratio of maximum stretching to maximum shrinking in a unit vector when A is multiplied with it
    #we calculate the eigenvalues of A^TA so that we can look fo max  strecht and min shrink
    l = np.linalg.eig(np.dot(A.T,A)) 
    l1 = np.amin(l[0]) #min eigenvalue
    l2 = np.amax(l[0]) #max eigenvalue
    k1 = np.sqrt(l1)
    k2 = np.sqrt(l2)
    ncond = k2 / k1

    return ncond
