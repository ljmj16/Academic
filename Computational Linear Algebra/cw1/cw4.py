import numpy as np
import cla_utils
from cla_utils.exercises3 import householder_solve
 
 #we write k instead of lambda
def solve_woodbury(A,b,k):
  m,n=A.shape
  Q,R=np.linalg.qr(A)
  I=np.eye(m)
  #we write c as in the report
  c=[(R.T).dot(Q.transpose())]*b
  #we use householder to find y as in the report
  y=cla_utils.householder_solve(I+ (1/k)*R.dot(R.transpose()),R*c)
  #now I can write x
  x = (1/k)*c-(1/k)^2*R.transpose()*y
  return x

def find_lambda(A,b,k1,k2):
  #k1<k2 are the starting points. We suppose to have a solution in [k1,k2]
  #I write the medium point and I find the x using the function I wrote above
  v=(k1+k2)/2
  x=solve_woodbury(A,b,v)
  #and then I take its norm
  y=np.linalg.norm(x)
  #now we use a statement if to decide if our solution is enough and to decide in case how to change our starting points and we will use recursion
  #1.0e-4 indicates how much near to 1 we want the approximated solution of our problem to be. We can change it to get worse or better solutions
  if abs(1-y) < 1.0e-4:
      return(v)
  #now I am checking what happens if y is not near enough to 1
  if y<1:
    return find_lambda(A,b,k1,v)
  else:
    return find_lambda(A,b,v,k2)

def find_lambda_loop(A,b,k1,k2):
    #this function does the same thing as the other one but through a loop instead of using recursion. We are again supposing to have a solution in [k1,k2]
    v=(k1+k2)/2
    x=solve_woodbury(A,b,v)
    y=np.linalg.norm(x)
    if abs(1-y) < 1.0e-4:
      return(v)
    else:
        while abs(1-y)<1.0e-4:
            if y<1: 
              k2=(k1+k2)/2
            else:
              k1=(k1+k2)/2
            v=(k1+k2)/29
            x=solve_woodbury(A,b,v)
            y=np.linalg.norm(x)
    return v
    

