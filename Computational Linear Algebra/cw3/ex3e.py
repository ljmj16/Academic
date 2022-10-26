import numpy as np
import matplotlib.pyplot as plt

from cw3.ex3c import scriptcmodified
from cw3.ex3d import scriptdmodified3d


A = np.zeros((15,15))
v = np.array([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
np.fill_diagonal(A,v)
A += np.ones((15,15))


B,v,list = scriptcmodified(A)
C,t,nist = scriptdmodified3d(A)

n = len(list)

print(list)
plt.plot([i for i in range (n)],list)
plt.yscale('log')
plt.xlabel('')
plt.ylabel('')
plt.show()

m = len(nist)

print(nist)
plt.plot([i for i in range (m)],nist)
plt.yscale('log')
plt.xlabel('')
plt.ylabel('')
plt.show()