import numpy as np

from cw3.ex4a import fastserialiser, reformer

#we can substitute n, mu, lamb as we prefer (do it also in the tests if needed)

def H_apply(v,n = 5,mu = 1,lamb = 1,case=False):
    '''function for ex 4b'''
    u = reformer(v, n, n)
    #I compute the laplacian
    laplacian = np.zeros((n,n),dtype=complex) #Initialize the vector
    devx1 = u[:-1,:]-u[1:,:]
    devx2 = u[1:,:]-u[:-1,:]
    devy1 = u[:,:-1]-u[:,1:]
    devy2 = u[:,1:]-u[:,:-1]
    laplacian[:-1,:] += devx1
    laplacian[1:,:] += devx2
    laplacian[:,:-1] += devy1
    laplacian[:,1:] += devy2

    #I compute the sum (I will use it in the test)
    sum = np.sum(laplacian)
    #Now I do the rescaling
    laplacian = lamb*laplacian
    laplacian += mu*u
    if case == False:
        return fastserialiser(laplacian)
    if case == True:
        return fastserialiser(laplacian),sum
