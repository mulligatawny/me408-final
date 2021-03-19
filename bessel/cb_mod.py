import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from subroutines import coll_der_mat

n = 7
N = 257
t = np.linspace(0, np.pi, N+1)
x = np.cos(t) # flipped
D = coll_der_mat.cheby_coll_der_mat(N, x)

A = (x**2)*D@D +x*D - (n**2)*np.eye(N+1) 
B = -(x**2)*np.eye(N+1)
#B[0,:] = 0
#print(B)

u, _ = scipy.linalg.eig(A, B)

print(np.sort(abs(np.real(u))))
