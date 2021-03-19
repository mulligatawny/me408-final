###############################################################################
# Bessel Eigenvalue Solver Using Chebyshev Expansions (Singular Form)         #
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from subroutines import coll_der_mat

def find_nearest(array, value):
    """
    https://stackoverflow.com/a/2566508/13434335
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

n = 7
N = np.array([16, 32, 64, 128, 256, 512, 1024])
N = N+1
l = 122.907600204

def lmin(N):
    t = np.linspace(0, np.pi, N+1)
    x = np.cos(t) 
    # compute chebyshev collocation derivative matrix
    D = coll_der_mat.cheby_coll_der_mat(N, x)
    A = D@D +(1/x)*D - (n**2/x**2)*np.eye(N+1) 
    B = -np.eye(N+1)
    B[0,:] = 0
    # solve eigenvalue problem
    u, _ = scipy.linalg.eig(A, B)
    # find nearest eigenvalue
    lc = find_nearest(abs(np.real(u)), l)
    return abs((lc - l)/l)

e1 = np.zeros(len(N))
e2 = np.zeros(len(N))

for i in range(len(N)):
    e1[i] = lmin(N[i])
    e2[i] = lmin(N[i]+1)

plt.loglog(N, e1, '^-', color='green', label='$N$ odd')
plt.loglog(N, e2, 'o-', color='brown', label='$N$ even')
plt.title('Using Chebyshev expansions (singular equation)')
plt.xlabel('$N$')
plt.ylabel('error')
plt.grid(which='both')
plt.ylim([1e-3, 1e3])
plt.legend()
plt.show()
