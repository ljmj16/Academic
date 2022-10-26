import numpy as np
import numpy.random as random
import scipy

from cla_utils.exercises1 import column_matvec

def get_A100():
    """
    Return A100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    return A


def get_B100():
    """
    Return B100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A[np.tril_indices(m, -2)] = 0
    return A


def get_C100():
    """
    Return C100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A = 0.5*(A + np.conj(A).T)
    return A


def get_D100():
    """
    Return D100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A = 0.5*(A + np.conj(A).T)
    A[np.tril_indices(m, -2)] = 0
    A[np.triu_indices(m, 2)] = 0
    return A


def get_A3():
    """
    Return A3 matrix for investigating power iteration.
    
    :return A3: a 3x3 numpy array.
    """

    return np.array([[ 0.68557183+0.46550108j,  0.12934765-0.1622676j ,
                    0.24409518+0.25335939j],
                  [ 0.1531015 +0.66678983j,  0.45112492+0.18206976j,
                    -0.02633966+0.43477693j],
                  [-0.10817164-1.16879196j, -0.18446849+0.03755672j,
                   0.06430325-0.44757084j]])


def get_B3():
    """
    Return B3 matrix for investigating power iteration.

    :return B3: a 3x3 numpy array.
    """
    return np.array([[ 0.46870499+0.37541453j,  0.19115959-0.39233203j,
                    0.12830659+0.12102382j],
                  [ 0.90249603-0.09446345j,  0.51584055+0.84326503j,
                    -0.02582305+0.23259079j],
                  [ 0.75419973-0.52470311j, -0.59173739+0.48075322j,
                    0.51545446-0.21867957j]])


def pow_it(A, x0, tol, maxit, store_iterations = False):
    """
    For a matrix A, apply the power iteration algorithm with initial
    guess x0, until either 

    ||r|| < tol where

    r = Ax - lambda*x,

    or the number of iterations exceeds maxit.

    :param A: an mxm numpy array
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of power iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing all \
    the iterates.
    :return lambda0: the final eigenvalue.
    """
    m = A.shape[0]
    k=0
    lambda0 = np.dot(x0.conj().T, np.dot(A,x0))
    v = np.zeros((m,maxit), dtype = complex)
    #first case (without iterations)
    if  store_iterations == False:
        #we put the two conditions for the loop
       while np.linalg.norm(np.dot(A,x0)-lambda0*x0) > tol and k <= maxit:
         x0 = np.dot(A,x0)
         x0 = x0/np.linalg.norm(x0)
         lambda0 = np.dot(x0.conj().T, np.dot(A,x0))
         k += 1        
       return x0, lambda0
    #second case (with iterations)
    if store_iterations == True:
        #we put the two conditions for the loop
       while np.linalg.norm(np.dot(A,x0)-lambda0*x0) > tol and k <= maxit:
         x0 = np.dot(A,x0)
         x0 = x0/np.linalg.norm(x0)
         lambda0 = np.dot(x0.conj().T, np.dot(A,x0))
         v[:,k] = x0
         k += 1        
       return v, lambda0


def inverse_it(A, x0, mu, tol, maxit, store_iterations = False):
    """
    For a Hermitian matrix A, apply the inverse iteration algorithm
    with initial guess x0, using the same termination criteria as
    for pow_it.

    :param A: an mxm numpy array
    :param mu: a floating point number, the shift parameter
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of inverse iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing \
    all the iterates.
    :return l: a floating point number containing the final eigenvalue \
    estimate, or if store_iterations, an m dimensional numpy array containing \
    all the iterates.
    """
    
    m = A.shape[0]
    v = np.zeros((m,maxit),dtype=complex)
    L = np.zeros(m,dtype=complex)
    #first case (without iterations)
    if  store_iterations == False:
        #we put the two conditions for the loop
       for k in range(maxit):
         x0 = np.linalg.solve(A - mu*np.eye(m),x0)
         x0 = x0/np.linalg.norm(x0)
         l = np.dot(x0.conj().T,np.dot(A,x0))
         if (np.linalg.norm(A@x0-l*x0)<tol):
             break
       return x0, l
    #second case (with iterations)
    if store_iterations == True:
        #we put the two conditions for the loop
       for k in range(maxit):
         x0 = np.linalg.solve(A - mu*np.eye(m),x0)
         x0 = x0/np.linalg.norm(x0)
         l = np.dot(x0.conj().T,np.dot(A,x0))
         v[:,k] = x0
         L[k] = l
         if (np.linalg.norm(A@x0-l*x0)<tol):
             break
       return v, l
        

    


def rq_it(A, x0, tol, maxit, store_iterations = False):
    """
    For a Hermitian matrix A, apply the Rayleigh quotient algorithm
    with initial guess x0, using the same termination criteria as
    for pow_it.

    :param A: an mxm numpy array
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of inverse iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing \
    all the iterates.
    :return l: a floating point number containing the final eigenvalue \
    estimate, or if store_iterations, an m dimensional numpy array containing \
    all the iterates.
    """
    #we can just adapt the code of the previous function
    m = A.shape[0]
    v = np.zeros((m,maxit),dtype=complex)
    L = np.zeros(m,dtype=complex)
    l = np.dot(x0.conj().T,np.dot(A,x0))
    #first case (without iterations)
    if  store_iterations == False:
        #we put the two conditions for the loop
       for k in range(maxit):
         x0 = np.linalg.solve(A - l*np.eye(m),x0)
         x0 = x0/np.linalg.norm(x0)
         l = np.dot(x0.conj().T,np.dot(A,x0))
         if (np.linalg.norm(A@x0-l*x0)<tol):
             break
       return x0, l
    #second case (with iterations)
    if store_iterations == True:
        #we put the two conditions for the loop
       for k in range(maxit):
         x0 = np.linalg.solve(A - l*np.eye(m),x0)
         x0 = x0/np.linalg.norm(x0)
         l = np.dot(x0.conj().T,np.dot(A,x0))
         v[:,k] = x0
         L[k] = l
         if (np.linalg.norm(A@x0-l*x0)<tol):
             break
       return v, l

def householdercomplex(A, kmax=None):
    """
    we write householder for complex matrices mxm
    """
    
    m,n = A.shape
    if kmax is None:
        kmax = n
    #I use a loop 
    for k in range(0,m):
        x = A[k:m,k]
        # Implementation of e_1 
        e_1 = np.zeros((m-k),dtype=complex)
        e_1[0] = 1
        #we adapt to A being complex taking x0/|x0| instead of sgn(x0)
        l = x[0]/np.linalg.norm(x[0]) if np.linalg.norm(x[0])>0 else 1
        #v and A
        v = l*np.linalg.norm(x)*e_1 + x
        v = v / np.linalg.norm(v)
        A[k:m,k:n] -= 2*np.outer( v, np.dot( v.conj().T, A[k:m,k:n]))
    return A 

def householder_qrcomplex(A):
    """
    Given a complex mxm matrix A, use the Householder transformation to find
    the full QR factorisation of A.

    :param A: an mxm-dimensional numpy array

    :return Q: an mxm-dimensional numpy array
    :return R: an mxm-dimensional numpy array
    """

    #we are finding the dimensions first
    m,n = A.shape
    I = np.eye(m,dtype=complex)
    #I have just to adapt what I have done before
    Ahat = np.append(A,I,axis=1)
    Rhat = householdercomplex(Ahat)
    #I use the slice notation to write R
    R = Rhat[:,:n]
    Q_star = Rhat[:,n:]
    Q = Q_star.conj().T

    return Q, R    

def pure_QR(A, maxit, tol,case = False, case2 = False, case3 = False):
    """
    For matrix A, apply the QR algorithm and return the result.

    :param A: an mxm numpy array
    :param maxit: the maximum number of iterations
    :param tol: termination tolerance

    :return Ak: the result
    """
    m = A.shape[0]
    
    if case == False and case3==False:
      for k in range(maxit):
        Q,R = householder_qrcomplex(A)
        A = R@Q
        if (np.linalg.norm(A[np.tril_indices(m,-1)])/m**2<tol):
            break
        
    if case == True and case3==False:
       v = []
       for k in range(maxit):
        Q,R = householder_qrcomplex(A)
        A = R@Q
        v.append(abs(A[m-1,m-2]))
        if (abs(A[m-1,m-2]<1.0e-12)):
            break    


    if case3 == True:
       v = []
       for k in range(maxit):
        delta = 0.5*(A[m-2,m-2]-A[m-1,m-1])
        b = A[m-1,m-2]
        a = A[m-1,m-1]
        sign = delta/np.linalg.norm(delta) if delta else 1
        mu = a-sign*b**2/(np.abs(delta)+np.sqrt(delta**2+b**2))
        Q,R = scipy.linalg.qr(A-mu*np.eye(m))
        A = R@Q+mu*np.eye(m)
        v.append(abs(A[m-1,m-2]))
        if (abs(A[m-1,m-2]<1.0e-12)):
            break 
        




    if case2 == False:
        return A
    if case2 == True:
        return A, v
