import numpy as np
from numpy.random.mtrand import rand
import cla_utils
import time
from numpy import nested_iters, random
import pytest
import matplotlib.pyplot
import timeit


from cla_utils.exercises10 import GMRES
import timeit

from cla_utils.exercises7 import solve_LUP


def LUsolve(A,b):
    m = A.shape[0]
    A = cla_utils.LU_inplace(A)
    U = np.triu(A) 
    I = np.eye(m) 
    L = I + np.tril(A, k=-1) 
    y = cla_utils.solve_L(L,b) 
    x = cla_utils.solve_U(U,y) 
    return x
#fist we define the HSS code implementation


def HSS_LU(A,b,n,alpha,er):
    '''This function implements the HSS method for a system Ax=b, where A has the condition given in the paper
    :param A: mxm array
    :param b: m-dim array
    :param n: max number of iteration 
    :param alpha: parameter 
    :param er: tollerance parameter we use to stop the loop

    We use householder to solve the two half steps
    
    :return x: solution of the system obtained with the method
    :return i: number of iterations'''

    m,_ = A.shape
    I = np.identity(m)
    x = np.zeros(m,dtype=complex)
    #I generate the matrices H and S
    H = (A+np.conj(A).T)/2
    S = (A-np.conj(A).T)/2
    # Now I write (M_1,N_1)=(alphaI+H,alphaI-S) and (M_2,N_2)=(alphaI+S,alphaI-H)
    M_1 = alpha*I+H
    N_1 = alpha*I-S
    M_2 = alpha*I+S
    N_2 = alpha*I-H
    #steps initialization
    i = 0
    #I start a loop 
    e = er + 0.1
    while i < n and e>er:
        #I count the steps
        i += 1
        #first half step
        x_1 = LUsolve(M_1,N_1@x+b)
        #second half step
        x = LUsolve(M_2,N_2@x_1+b)
        #I define the distance from two consecutive approximate solution
        e = np.linalg.norm(x-x_1)
    return x, i

def HSS_numpy(A,b,n,alpha,er):
    '''This function implements the HSS method for a system Ax=b, where A has the condition given in the paper
    :param A: mxm array
    :param b: m-dim array
    :param n: max number of iteration 
    :param alpha: parameter 
    :param er: tollerance parameter we use to stop the loop
    
    We use the numpy function solve to solve the two half steps systems (numpy.linalg.solve is based on LU)

    :return x: solution of the system obtained with the method
    :return i: number of iterations'''

    m,_ = A.shape
    I = np.identity(m)
    x = np.zeros(m,dtype=complex)
    #I generate the matrices H and S
    H = (A+np.conj(A).T)/2
    S = (A-np.conj(A).T)/2
    # Now I write (M_1,N_1)=(alphaI+H,alphaI-S) and (M_2,N_2)=(alphaI+S,alphaI-H)
    M_1 = alpha*I+H
    N_1 = alpha*I-S
    M_2 = alpha*I+S
    N_2 = alpha*I-H
    #steps initialization
    i = 0
    #I start a loop 
    e = er + 0.1
    while i < n and e>er:
        #I count the steps
        i += 1
        #first half step
        x_1 = np.linalg.solve(M_1,N_1@x+b)
        #second half step
        x = np.linalg.solve(M_2,N_2@x_1+b)
        #I define the distance from two consecutive approximate solution
        e = np.linalg.norm(x-x_1)
    return x, i

def HSS_inv(A,b,n,alpha,er):
    '''This function implements the HSS method for a system Ax=b, where A has the condition given in the paper
    :param A: mxm array
    :param b: m-dim array
    :param n: max number of iteration 
    :param alpha: parameter 
    :param er: tollerance parameter we use to stop the loop
    
    We use the numpy function linalg.inv

    :return x: solution of the system obtained with the method
    :return i: number of iterations'''

    m,_ = A.shape
    I = np.identity(m)
    x = np.zeros(m,dtype=complex)
    #I generate the matrices H and S
    H = (A+np.conj(A).T)/2
    S = (A-np.conj(A).T)/2
    # Now I write (M_1,N_1)=(alphaI+H,alphaI-S) and (M_2,N_2)=(alphaI+S,alphaI-H)
    M_1 = alpha*I+H
    N_1 = alpha*I-S
    M_2 = alpha*I+S
    N_2 = alpha*I-H
    #invertions
    P_1 = np.linalg.inv(M_1)
    P_2 = np.linalg.inv(M_2)
    #steps initialization
    i = 0
    #I start a loop 
    e = er + 0.1
    while i < n and e>er:
        #I count the steps
        i += 1
        #first half step
        x_1 = P_1@N_1@x+P_1@b
        #second half step
        x = P_2@N_2@x_1+P_2@b
        #I define the distance from two consecutive approximate solution
        e = np.linalg.norm(x-x_1)
    return x, i



#now we define the IHSS implementation

def IHSS(A,b,n,alpha,er):
    '''This function implements the IHSS method 
    :param A: mxm matrix
    :param b: m-dim array
    :n: max number of iteration
    :alpha: parameter
    :param er: tollerance parameter we use to stop the loop
    
    :return: approximate solution x of the system'''

    m,_ = A.shape
    I = np.identity(m)
    x = np.zeros(m,dtype=float)
    #I generate the matrices H and S
    H = (A+np.conj(A).T)/2
    S = (A-np.conj(A).T)/2
    # Now I write (M_1,N_1)=(alphaI+H,alphaI-S) and (M_2,N_2)=(alphaI+S,alphaI-H)
    M_1 = alpha*I+H
    N_1 = alpha*I-S
    M_2 = alpha*I+S
    N_2 = alpha*I-H

    #steps initialization
    i = 0
    e = er + 0.1
    #I start a loop 
    while i < n and e>er:
        #I count the steps
        i += 1
        #first half step
        x_1,nits = GMRES(M_1,N_1@x+b, m, er )
        #second half step
        x,nits = GMRES(M_2,N_2@x_1+b,m,er)
        #I define the distance from two consecutive approximate solution
        e = np.linalg.norm(x-x_1)
    return x, i

#The following code is to develop the case of the two dimensional convection-diffusion process
def convdiff(M,q):

    I = np.identity(M)
    #I generate the Reynolds number 
    h = 1/(M+1)
    Re = q*h/2
    #I generate T
    T = np.zeros((M,M))
    T += np.diag(2*np.ones(M),k=0)
    T += np.diag((-1-Re)*np.ones(M-1),k=-1)
    T += np.diag((-1+Re)*np.ones(M-1),k=1)
    #I write A
    A = np.kron(T,I)+np.kron(I,T)
    return A

def aux1():
    A = convdiff(14,1)
    random.seed(8473)
    b = random.randn(14**2)
    HSS_LU(A,b,100,1,1.0e-4)

def aux2():
    A = convdiff(14,1)
    random.seed(8473)
    b = random.randn(14**2)
    IHSS(A,b,100,1,1.0e-4)

def timing():
    print(timeit.Timer(aux1).timeit(number=1), timeit.Timer(aux2).timeit(number=1))







'''NUMERICAL EXPERIMENTS'''
#NUMERICAL EXPERIMENTS


def timingexperiments():
    #we check the timings for different dimension
    list = [] 
    list2 = []
    list3 = []
    list4 = []
    for n in range (4,18):
        def auxiliardimensions1():
        #alpha=1
            A = convdiff(n,10)
            b = random.randn(n**2)
            x = IHSS(A,b,1000,1,1.0e-5)
        def auxiliardimensions2():
        #alpha=12
            A = convdiff(n,10)
            b = random.randn(n**2)
            x = IHSS(A,b,1000,12,1.0e-5)
        def auxiliardimensions3():
        #alpha=4
            A = convdiff(n,10)
            b = random.randn(n**2)
            x = IHSS(A,b,1000,4,1.0e-5)
        def auxiliardimensions4():
        #alpha=4
            A = convdiff(n,10)
            H = (A+np.conj(A).T)/2
            eigs = np.linalg.eig(H)[0]
            alph_pseudoptimal = np.sqrt(eigs[0]*eigs[(n-1)**2])
            b = random.randn(n**2)
            x = IHSS(A,b,1000,alph_pseudoptimal,1.0e-5)
        #we generate bamded matrices
        t = timeit.Timer(auxiliardimensions1).timeit(number=1)
        m = timeit.Timer(auxiliardimensions2).timeit(number=1)
        p = timeit.Timer(auxiliardimensions3).timeit(number=1)
        pseudalp = timeit.Timer(auxiliardimensions4).timeit(number=1)
        list.append(t)
        list2.append(m)
        list3.append(p)
        list4.append(pseudalp)
    #Now we plot it
    matplotlib.pyplot.plot([i for i in range (4,18)],list)
    matplotlib.pyplot.plot([i for i in range (4,18)],list2)
    matplotlib.pyplot.plot([i for i in range (4,18)],list3)
    matplotlib.pyplot.plot([i for i in range (4,18)],list4)

    matplotlib.pyplot.show()


'''AUTOMATIC TESTING'''
#Now I test the HSS and the IHSS for the convection-diffusion equation


@pytest.mark.parametrize('m,q',[(4,1),(5,10)])
def test_definitepositive(m,q):
    A = convdiff(m,q)
    eig = np.linalg.eig(A)[0]
    for i in eig:
        assert(i>0)

@pytest.mark.parametrize('m,q',[(4,1),(5,10)])
def test_IHSS(m,q):
    A = convdiff(m,q)
    random.seed(1235*m)
    b = random.randn(m**2)
    x, i = IHSS(A,b,1000,10,1.0-6)
    assert(np.linalg.norm(A@x-b)<1.0e-5)

@pytest.mark.parametrize('m,q',[(4,1),(5,10)])
def test_HSS(m,q):
    A = convdiff(m,q)
    random.seed(1235*m)
    b = random.randn(m**2)
    x, i = HSS_inv(A,b,10000,10,1.0-6)
    assert(np.linalg.norm(A@x-b)<1.0e-5)


'''
A = convdiff(24,100)
random.seed(8473)
b = random.randn(24**2)
x1,h1 = HSS_numpy(A,b,1000,1,1.0e-4)

print(x1,h1)'''

#TO RUN THE TIMINGS
#to time chose one of the followings :


#timing()
timingexperiments()



#to print plots of the matrices
A = convdiff(10,10)
#matplotlib.pyplot.matshow(A)
#matplotlib.pyplot.show()