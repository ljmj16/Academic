import numpy as np
from numpy.lib.function_base import append
import numpy.random as random
import cla_utils
from cla_utils.exercises3 import householder_ls


def arnoldi(A, b, k):
    """
    For a matrix A, apply k iterations of the Arnoldi algorithm,
    using b as the first basis vector.

    :param A: an mxm numpy array
    :param b: m dimensional numpy array, the starting vector
    :param k: integer, the number of iterations

    :return Q: an mx(k+1) dimensional numpy array containing the orthonormal basis
    :return H: a (k+1)xk dimensional numpy array containing the upper \
    Hessenberg matrix
    """
 
    m , _ = A.shape
    #initialising Q and H
    Q = np.zeros((m,k+1),dtype=complex)
    H = np.zeros((k+1,k),dtype=complex)
    #qi are the columns of Q, so we can work directly on Q and use slice notation to indicate the vectors qi
    #now we follow the algorithm
    Q[:,0] = b/np.linalg.norm(b)
    #we use only one loop
    for i in range(k):
        v = np.dot(A,Q[:,i])
        H[0:i+1,i] = np.dot(np.conjugate(Q[:,0:i+1]).T, v)
        v = v - np.dot( Q[:,0:i+1],H[0:i+1,i])
        H[i+1,i] = np.linalg.norm(v)
        Q[:,i+1] = v/np.linalg.norm(v)
    return Q,H    


1

def GMRES(A, b, maxit, tol, x0=None, return_residual_norms=False,
          return_residuals=False, situation = None, preconditioning = None):
    """
    For a matrix A, solve Ax=b using the basic GMRES algorithm.

    :param A: an mxm numpy array
    :param b: m dimensional numpy array
    :param maxit: integer, the maximum number of iterations
    :param tol: floating point number, the tolerance for termination
    :param x0: the initial guess (if not present, use b)
    :param return_residual_norms: logical
    :param return_residuals: logical

    :return x: an m dimensional numpy array, the solution
    :return nits: if converged, the number of iterations required, otherwise \
    equal to -1
    :return rnorms: nits dimensional numpy array containing the norms of \
    the residuals at each iteration
    :return r: mxnits dimensional numpy array, column k contains residual \
    at iteration k
    """

    if x0 is None:
        x0 = b
    m = b.shape[0]

    if preconditioning != None:
        b = preconditioning(b)


    #I write Q and H as vectors and I will stack to them other vectors during the process in order to obtain the final matrices
    Q = np.array(b.reshape(m,1)/np.linalg.norm(b),dtype=complex)
    H = np.zeros((2,1),dtype=complex)
    #rnorms introduction (as befor we will stack element to this array later)
    rnorms = np.array([],dtype=complex)
    #r introduction
    r = np.zeros((m,1),dtype=complex)
    #now i introduce nits
    nits = -1
    #now I can start the algorithm, we will compute everything and only at the end we will use if statements to decide what to return
    for i in range(maxit):
        #we apply arnoldi
        if (i>0):
            H = np.column_stack((H,np.zeros(H.shape[0])))
            H = np.vstack((H,np.zeros(H.shape[1])))
        if situation == None and preconditioning == None:
            v = np.dot(A,Q[:,i])
        elif situation == None and preconditioning != None:
            v = preconditioning(np.dot(A,Q[:,i]))
        elif situation != None and preconditioning != None:
            v = preconditioning(situation(Q[:,i]))
        else:
            v = situation(Q[:,i])
        H[0:i+1,i] = np.dot(np.conjugate(Q[:,0:i+1]).T, v)
        v = v - np.dot( Q[:,0:i+1],H[0:i+1,i])
        H[i+1,i] = np.linalg.norm(v)

        #minimezer y
        e1 = np.zeros(i+2)
        e1[0] = 1
        y = cla_utils.householder_ls(H,np.linalg.norm(b)*e1)
        #xn=Qhatn*y and residuals
        x = np.dot(Q,y)
        if situation == None:
            r[:,i] = np.dot(A,x)-b
        else:
            r[:,i] = situation(x)-b
        #adding elements to rnorm
        t = np.linalg.norm(r[:,i])
        rnorms = append(rnorms,t)
        #tolerance
        if (t<tol):
            nits = i+1
            break
        Q = np.column_stack((Q, v/H[i+1,i]))
        if (i<maxit-1):
            r = np.column_stack((r,np.zeros(m)))
    if (return_residual_norms == True):
        return x, nits, rnorms
    if (return_residuals == True):
        return x, nits, r
    if (return_residual_norms == True and return_residuals == True):
        return x, nits, rnorms, r
    else:
        return x, nits  


def get_AA100():
    """
    Get the AA100 matrix.

    :return A: a 100x100 numpy array used in exercises 10.
    """
    AA100 = np.fromfile('AA100.dat', sep=' ')
    AA100 = AA100.reshape((100, 100))
    return AA100


def get_BB100():
    """
    Get the BB100 matrix.

    :return B: a 100x100 numpy array used in exercises 10.
    """
    BB100 = np.fromfile('BB100.dat', sep=' ')
    BB100 = BB100.reshape((100, 100))
    return BB100


def get_CC100():
    """
    Get the CC100 matrix.

    :return C: a 100x100 numpy array used in exercises 10.
    """
    CC100 = np.fromfile('CC100.dat', sep=' ')
    CC100 = CC100.reshape((100, 100))
    return CC100
