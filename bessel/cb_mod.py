###############################################################################
# Bessel Eigenvalue Solver Using Chebyshev Expansions (Non-singular Form)     #
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
N = np.array([7, 11, 15, 19, 23, 27, 31, 33])
l = 122.907600204

def lmin(N):
    t = np.linspace(0, np.pi, N+1)
    x = np.cos(t) 
    D = coll_der_mat.cheby_coll_der_mat(N, x)
    # compute chebyshev collocation derivative matrix
    A = (x**2)*D@D +x*D - (n**2)*np.eye(N+1) 
    B = -(x**2)*np.eye(N+1)
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
plt.title('Using Chebyshev expansions (non-singular equation)')
plt.xlabel('$N$')
plt.ylabel('error')
plt.grid(which='both')
plt.legend()
plt.show()
