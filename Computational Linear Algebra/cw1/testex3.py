#This is the test for excercise 4 (before called as 3), which is called in the cw1 directory 'ex3.py'
from numpy import random
import numpy as np
import copy
import pytest
import cla_utils
import cw1
from cw1.ex3 import multiplicationQstarb, solvels
import ex1
from ex3 import householderVR

@pytest.mark.parametrize('m', [20, 40, 87])
def test_householderVR(m):
    random.seed(1878*m)
    A = random.randn(m, m)
    #I am making a copy of A
    A0 = copy.deepcopy(A) 
    Rv = householderVR(A0, None)
    #Now I get rid of the v column vectors using the builtin numpy function
    R = np.triu(Rv) 
    #Now the tests
    assert(np.allclose(R, np.triu(R))) #1
    assert(np.linalg.norm(np.dot(R.T, R) - np.dot(A.T, A)) < 1.0e-6) #2
    
    #Now I have to recreate Q from Rv
    m = Rv.shape[0]
    Q = np.identity(m)
    for i in range(m) :
        v = Rv[i:,i] / np.linalg.norm(Rv[i:,i])
        Q[i:,:]=Q[i:,:]-2 * np.outer(v,np.dot(v,Q[i:,:]))
    
    # check orthonormality
    assert(np.linalg.norm(np.dot(Q.transpose(), Q) - np.identity(m)) < 1.0e-6) #3
    assert(np.linalg.norm(np.dot(Q.transpose(), R) - A) < 1.0e-6)


@pytest.mark.parametrize('m, n', [(20, 7), (40, 13), (87, 9)])
def test_Qb(m,n): 
    random.seed(1878*m)
    A = random.randn(m, n)
    A0 = copy.deepcopy(A)  
    R_v = householderVR(A0)
    # I generate a random b
    b = random.randn(m)
    b_0 = copy.deepcopy(b)
    #Now I compute the multiplication Q*b
    Qb = multiplicationQstarb(R_v,b)
    #I use the builtin numpy factorisation
    Q = np.linalg.qr(A,mode='complete')[0]
    #Now I implement Q*b starting from the Q obtained with the numpy function
    realQ= np.dot(Q.transpose().conjugate(),b_0) 
    assert(np.linalg.norm(realQ-Qb)<1.0e-5)

@pytest.mark.parametrize('m, n', [(20, 7), (40, 13), (87, 9)])
def test_solvels(m,n) :
    random.seed(1878*m)
    A = random.randn(m, n)
    A_0 = copy.deepcopy(A)  
    R_v = householderVR(A_0)  
    #I generate a random vector
    b = random.randn(m)
    b_0 = copy.deepcopy(b)   
    #real x with a numpy builtin function
    realx = np.linalg.lstsq(A, b,rcond=-1)
    x = solvels(R_v,b_0)
    #now the test
    assert(np.linalg.norm(x-realx[0])<1.0e-6) 
    
    
    
    
