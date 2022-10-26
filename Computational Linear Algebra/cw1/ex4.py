import numpy as np
from numpy import random
import cla_utils
from cla_utils.exercises3 import householder_solve
 
 #we write k instead of lambda
def solvewoodbury(A,b,k):
  m,n=A.shape
  Q,R=np.linalg.qr(A,mode='complete')
  I=np.eye(m)
  #we write c as in the report
  c = np.dot(A.transpose(), b)
  #we use householder to find y as in the report
  y=cla_utils.householder_solve(I+ (1/k)*np.dot(R,R.transpose()), np.dot(R,c))
  #now I can write x
  x = (1/k)*c-(1/k**2)*np.dot(R.transpose(),y)
  return x

def findlambda(A,b,k1,k2):
  #This function is just an idea (better look at the version with the loop)
  #k1<k2 are the starting points. We suppose to have a solution in [k1,k2]
  #I write the medium point and I find the x using the function I wrote above
  v=(k1+k2)/2
  x=solvewoodbury(A,b,v)
  #and then I take its norm
  y=np.linalg.norm(x)
  #now we use a statement if to decide if our solution is enough and to decide in case how to change our starting points and we will use recursion
  #1.0e-4 indicates how much near to 1 we want the approximated solution of our problem to be. We can change it to get worse or better solutions
  if abs(1-y) < 1.0e-4:
      return(v)
  #now I am checking what happens if y is not near enough to 1
  if y<1:
    return findlambda(A,b,k1,v)
  else:
    return findlambda(A,b,v,k2)

def findlambdaloop(A,b,k1,k2):
    #We have to make sure that we have a solution in[k1,k2]. We insert two loops that change k1 and k2 when needed in order to have a solution in [k1,k2]
    #now we proceed with the algorithm
    v=(k1+k2)/2
    x=solvewoodbury(A,b,v)
    y=np.linalg.norm(x)
    if abs(1-y) < 1.0e-2:
      return(v)
    else:
        while abs(1-y)>1.0e-2:
          if y>1:
              k2=(k1+k2)/2
          else:
              k1=(k1+k2)/2
          v=(k1+k2)/2
          x=solvewoodbury(A,b,v)
          y=np.linalg.norm(x)
    return v


    

