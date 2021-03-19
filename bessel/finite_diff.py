import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

n = 7
N = 4
L = 2
j = np.arange(0, N)
A = np.zeros((N+1, N+1))
dx = L/(N-1)

# populate finite difference matrix
np.fill_diagonal(A, -2*j**2 -n**2)
np.fill_diagonal(A[1:], j**2 -j/2)
np.fill_diagonal(A[:,1:], j**2 +j/2)
A[-1,-1] = -2*N**2 -n**2

B = np.zeros((N+1, N+1))

# populate RHS matrix
np.fill_diagonal(B, (-j**2)*(dx**2))
B[-1,-1] = (-N**2)*(dx**2)

# make first row the BC since it's zero anyway
print(A)
print(B)
u, v = scipy.linalg.eig(A, B)
