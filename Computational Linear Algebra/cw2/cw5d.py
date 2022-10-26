import numpy as np
import cla_utils
from numpy import random
from scipy.sparse import diags
import matplotlib.pyplot as plt



def LUfactorisationbanded(A,p,q):
    '''Given a banded matrix with bandwidths p and q performs the LU factorisation for banded matrices'''
    m = A.shape[0]
    #now I can start the loop
    for k in range(m-1):
        A[k+1:min(k+p,m),k] = A[k+1:min(k+p,m),k] / A[k,k]
        A[k+1:min(k+p,m),k+1:min(k+q,m)] = A[k+1:min(k+p,m),k+1:min(k+q,m)] - np.outer( A[k+1:min(k+p,m),k],  A[k,k+1:min(k+q,m)])
    U = np.triu(A) 
    I = np.eye(m) 
    L = I + np.tril(A, k=-1) 
    return L,U

def forwardsubbanded(L,b):
    '''Given a matrix L with lower bandwidht p, solves Lx=b using forward sub'''
    n,_ = L.shape
    p = n-np.count_nonzero(L[:,0]==0)-1
    v = np.zeros(n)
    #now we go with the loop
    for k in range(n):
        l = max(0,k-p)
        v[k] = (b[k] - np.dot(L[k,l:k], v[l:k]))/L[k,k]
    return v

def backwardsubbanded(U,b,q):
    '''Given a matrix U with upper bandwidht q, solves Ux=b using backward sub'''
    n,_ = U.shape
    v = np.zeros(n)
    #now we go with the loop
    for k in reversed(range(n)):
        l = min(n,k+q+1)
        v[k] = (b[k] - np.dot(U[k,k+1:l], v[k+1:l]))/U[k,k]
    return v   

def solvebanded(A,b):
    '''Given a banded matrix, calculates the bandwidhts, performs LU and solves Ax=b using backward and forward substitution'''
    n,_ =A.shape
    v = np.zeros(n)
    l = np.zeros(n)
    #bandwidhts and L and U
    p = n-np.count_nonzero(A[:,0]==0)-1
    q = n-np.count_nonzero(A[0,:]==0)-1
    L,U = LUfactorisationbanded(A,q,p)
    #Notice that could also be written recalling the previous functions
    #Lv=b
    for k in range(n):
        ma = max(0,k-p)
        v[k] = (b[k] - np.dot(L[k,ma:k],v[ma:k]))/L[k,k]
    #Ul=v
    for k in reversed(range(n)):
        mi = min(n,k+q+1)
        l[k] = (v[k]-np.dot(U[k,k+1:mi],v[k+1:mi]))/U[k,k]
    return l    
         

def create_Sandb(s0,r0,n,alpha):
    '''Creates a matrix S and b given s0,r0, alpha and a dimension n'''
    S = np.zeros((n-1,n-1))
    b = np.zeros(((n-1)**2,2))
    for i in range(n-1):
        for j in range(n-1):
            S[i,j] = s0*np.exp(-((i-0.25)**2)/r0**2-((j-0.25)**2)/r0**2)
            b[(n-1)*i+j] = [alpha*(-np.sin((np.pi)*i)*np.cos((np.pi)*j)), alpha*(np.cos((np.pi)*i)*np.sin((np.pi)*j))]                      
    return S, b

def LHS_matrix(b,mu,c,n,S):
    '''Creates the matrix on the LHS of the matrix vector interpretaion of (6) and (7) and flattens S'''
    #Matrix on the LHS for the half step
    #Note : If I muliply both the LHS and RHS for (delta x)^2 I get better results
    A0 = np.zeros(((n-1)**2,(n-1)**2))
    A0 += np.diag(0.5*b[0:(n-1)**2-(n-1),0]/n-mu*np.ones((n-1)**2-(n-1)),n-1)
    A0 += np.diag(-0.5*b[n-1:(n-1)**2,0]/n-mu*np.ones((n-1)**2-(n-1)),-n+1)
    A0 += np.diag((c/(n**2)+4*mu)*np.ones((n-1)**2),0)
    #RHS matrix for the half step
    B0 = np.zeros(((n-1)**2,(n-1)**2))
    B0 += np.diag(0.5*b[0:(n-1)**2-(n-1),0]/n-mu*np.ones((n-1)**2-(n-1)),n-1)
    B0 += np.diag(-0.5*b[n-1:(n-1)**2,0]/n-mu*np.ones((n-1)**2-(n-1)),-n+1)
    #Matrix on the LHS for the full step
    A1 = np.zeros(((n-1)**2,(n-1)**2))
    A1 += np.diag(0.5*b[0:(n-1)**2-1,1]/n-mu*np.ones((n-1)**2-1),1)
    A1 += np.diag(-0.5*b[1:(n-1)**2,1]/n-mu*np.ones((n-1)**2-1),-1)
    A1 += np.diag((c/(n**2)+4*mu)*np.ones((n-1)**2),0)
    #Matrix on the RHS for the full step
    B1 = np.zeros(((n-1)**2,(n-1)**2))
    B1 += np.diag(0.5*b[0:(n-1)**2-1,1]/n-mu*np.ones((n-1)**2-1),1)
    B1 += np.diag(-0.5*b[1:(n-1)**2,1]/n-mu*np.ones((n-1)**2-1),-1)
    #Full matrix on the LHS for the expression (4)
    B = 1.0*A0
    C = 1.0*A1
    A = np.zeros(((n-1)**2,(n-1)**2))
    A[:,:] = B+C-np.diag((c/(n**2)+4*mu)*np.ones((n-1)**2),0)
    s = S.flatten()
    return A0,A1,B0,B1,A,s

def solutionofPDE(n,mu,c,s0,r0,alpha):
    #create S and b
    S,b = create_Sandb(s0,r0,n,alpha)
    A0,A1,B0,B1,A,s = LHS_matrix(b,mu,c,n,S)
    #I will use as a starting vector for the iteration a vector of zeros
    v_0 = np.zeros((n-1)**2)
    k = 0
    #Now I use a loop while, that keeps going until the error is small
    while np.linalg.norm(s-np.dot(A,v_0))>1e-5 and k<55:
        #I compute the RHS calculations for the half step
        b0 = s - np.dot(B1,v_0)
        #I could use backward and forward substitution to solve the system (using the function I defined above)
        #I will use the built in function to optimize the result
        x = np.linalg.solve(A0,b0)
        #I compute the RHS calculations for the full step
        b1 = s - np.dot(B0,x)
        #I solve
        v_0 = np.linalg.solve(A1,b1)
        k += 1
    err = np.linalg.norm(s-np.dot(A,v_0))
    return v_0, err, k
   
v_0, err, k= solutionofPDE(5,1,0.5,1,2,0.5)


print(err, k)

def sol_analysis1():
    list1 = []
    for i in range(10):
        v_0,err,k = solutionofPDE(5+i,1,0.5,1,2,0.5)
        list1.append(err)
    plt.plot([i+1 for i in range (0,10)],list1)
    plt.xlabel('n')
    plt.ylabel('err')
    plt.show()

def sol_analysis2():
    list2 = []
    for i in range(10):
        v_0,err,k = solutionofPDE(5+i,1,0.5,1,2,0.5)
        list2.append(k)

    plt.plot([i+1 for i in range (0,10)],list2)
    plt.xlabel('n')
    plt.ylabel('iterations')
    plt.show()

def sol_analysis3():
    list3 = []
    for i in range(10):
        v_0,err,k = solutionofPDE(5,1,0.1+0.1*i,1,2,0.5)
        list3.append(err)
    plt.plot([i+1 for i in range (0,10)],list3)
    plt.xlabel('n')
    plt.ylabel('err')
    plt.show()

def sol_analysis4():
    list4 = []
    for i in range(10):
        v_0,err,k = solutionofPDE(5,1,0.1+0.1*i,1,2,0.5)
        list4.append(k)
    plt.plot([i+1 for i in range (0,10)],list4)
    plt.xlabel('n')
    plt.ylabel('iterations')
    plt.show()

sol_analysis1()
sol_analysis2()