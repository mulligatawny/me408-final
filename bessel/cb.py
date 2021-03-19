import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from subroutines import coll_der_mat

n = 7
N = 129
#N = np.array([9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31])
#N = np.array([13, 19, 29, 39, 59, 69, 99])

t = np.linspace(0, np.pi, N+1)
x = np.cos(t) # flipped
D = coll_der_mat.cheby_coll_der_mat(N, x)

A = D@D +(1/x)*D - (n**2/x**2)*np.eye(N+1) 
B = -np.eye(N+1)
B[0,:] = 0
#print(B)

u, _ = scipy.linalg.eig(A, B)


print(np.sort(abs(np.real(u)))[:10])

