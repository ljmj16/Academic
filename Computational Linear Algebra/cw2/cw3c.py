import numpy as np

def quadratic_solve(A,dtype=complex):
    '''Given a 2x2 matrix A computes the eigenvalues'''
    
    
    lambdaone = 0.5*(A[0,0]+A[1,1]+np.sqrt((A[0,0]+A[1,1])**2-4*(A[0,0]*A[1,1]-A[0,1]*A[1,0])))
    lambdatwo = 0.5*(A[0,0]+A[1,1]-np.sqrt((A[0,0]+A[1,1])**2-4*(A[0,0]*A[1,1]-A[0,1]*A[1,0])))
    return lambdaone, lambdatwo


#Now let's define a matrix A1
A = np.zeros((2,2),dtype=complex)
A[0,0]=1
A[1,1]=1
#Now let's write the exact solutions
a = np.zeros(2,dtype=complex)
a[0] = 1
a[1] = 1
#let's find the solutions through the algorithm
l1,l2=quadratic_solve(A)
#I am storing them in a vector to compute the error easily
b=np.zeros(2,dtype=complex)
b[0] = l1
b[1] = l2
#error
error1 = np.linalg.norm(a-b)
#I print the results obtained with the algo and the error
print("The eigenvalues of A1 through the algorithm are {}".format(b))
print ("The error for the first case is {}".format(error1))



#Now let's define a matrix A2
B = np.zeros((2,2),dtype=complex)
B[0,0]=1+1e-14
B[1,1]=1
#Now let's write the exact solutions
c = np.zeros(2,dtype=complex)
c[0] = 1+1e-14
c[1] = 1
#let's find the solutions through the algorithm
m1,m2=quadratic_solve(B)
#I am storing them in a vector to compute the error easily
d=np.zeros(2,dtype=complex)
d[0] = m1
d[1] = m2
#error
error2 = np.linalg.norm(c-d)
#I print the results obtained with the algo and the error
print("The eigenvalues of A2 through the algorithm are {}".format(d))
print ("The error for the first case is {}".format(error2))