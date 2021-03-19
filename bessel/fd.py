import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

n = 7
#N = int(2**9)
N = np.array([8, 16, 32, 64, 128, 256, 512, 1024])
def lmin(N):
    x = np.linspace(-1, 1, N+1)
    x[int(len(x)/2)] = 1e-32
    dx = 2/(N-1)

    # fill LHS matrix 
    A = np.zeros((N+1, N+1))
    np.fill_diagonal(A, -2*(x**2)/dx**2 -n**2)
    np.fill_diagonal(A[1:], (x[1:]**2)/dx**2 - x[1:]/(2*dx))
    np.fill_diagonal(A[:,1:], (x[:-1]**2/(dx**2) + x[:-1]/(2*dx)))
    A[-1,-1] = 0
    A[-2,-1] = 0

    # fill RHS matrix
    B = np.zeros((N+1, N+1))
    np.fill_diagonal(B, -x**2)

    # compute eigenvalues
    u, _ = scipy.linalg.eig(A, B)
    # smallest eigenvalues (ignoring spurious values)
    ls = np.sort(np.real(u))[2]
    return abs((ls - 122.907600204)/122.907600204)

e = np.zeros(len(N))
for i in range(len(N)):
    e[i] = lmin(N[i])

plt.loglog(N, e, 'ro-', label='finite diff.')
plt.loglog(N, 1/N, 'k-', label='slope-1')
plt.title('Using finite differences')
plt.xlabel('$N$')
plt.ylabel('error')
plt.grid(which='both')
plt.legend()
plt.show()
