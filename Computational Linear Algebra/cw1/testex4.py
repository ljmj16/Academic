#This is the test for excercise 4 (before called as 3), which is called in the cw1 directory 'ex3.py'
import numpy as np
from numpy import random
import pytest
import cla_utils
import cw1
from cw1.ex4 import findlambda, findlambdaloop, solvewoodbury
import ex4

@pytest.mark.parametrize('m, n', [(20, 7), (40, 13), (87, 9)])
def test_solvewoodbury(m,n) :
    random.seed(1878*m)
    #I create random A and b 
    A = random.randn(m, n)
    b = random.randn(m,1)
    #I create a random lambda (we call it k)
    k= random.uniform(low=-9999,high=9999)
    #I run the function for finding the solution x
    x = solvewoodbury(A,b,k)
    #We write the derivate of Phi in x
    half_derivate =  k * x +np.dot(np.dot(A.T,A), x) - np.dot(A.T, b) 
    #we are checking that the point is stationary
    assert(np.linalg.norm(half_derivate)<1.0e-5) 

@pytest.mark.parametrize('m, c, d', [(20, 0.00001, 100), (4, 0.0000001, 150), (8, 0.000001, 200)])
def test_findlooplambda(m,c,d) :   
    random.seed(1878*m)
    #I generate A and b
    A = random.randn(m, m)
    b = random.randn(m,1)
    #I look for the lambda (which we call v)
    v= findlambda(A,b,c,d)
    #I compute x so that I can check
    x = solvewoodbury(A,b,v)
    #we check that the norm of x is 1
    assert(abs(np.linalg.norm(x)-1)<1.0e-2 )
    

