import numpy as np
from cla_utils.exercises7 import solve_LUP

from cw3.ex4a import fastserialiser, reformer


def M_solve(u,lamb = 1,mu = 1):
    '''M solver on ex 4d'''
    n = u.shape[0]
    m = int(np.sqrt(n))
    v = reformer(u,m,m)
    #construction of matrix
    M = np.zeros((m,m))
    M += 2*np.identity(m)
    M += np.diag(-1*np.ones(m-1),k=-1)
    M += np.diag(-1*np.ones(m-1),k=1)
    #multiplying by lambda
    M *= lamb
    M += mu*np.eye(m)
    t = np.zeros((m,m))
    y = np.zeros((m,m))
    #loop
    for i in range(m):
        t[i,:] = solve_LUP(M,v[i,:])
    #another loop
    for i in range(m):
        y[:,i] = solve_LUP(M,v[:,i]-(M-mu*np.identity(m))@t[:,i])
    return fastserialiser(y)

