import numpy as np
import matplotlib.pyplot as plt
from subroutines import compute_fk

def fu(uk):
    a = np.zeros_like(uk)
    d = np.zeros_like(uk)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if n1[i] == n2[j] == 0:
                    a[i,j,k] = 1
                else:
                    a[i,j,k] = (n1[i]*fku[i,j,k] + n2[j]*fkv[i,j,k]\
                                )*n1[i]/(n1[i]**2 +\
                                n2[j]**2)
                d[i,j,k] = nu*(n1[i]**2 + n2[j]**2)*uk[i,j,k]
    return fku -a -d

def fv(vk):
    a = np.zeros_like(vk)
    d = np.zeros_like(vk)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if n1[i] == n2[j] == n3[k] == 0:
                    a[i,j,k] = 1
                else:
                    a[i,j,k] = (n1[i]*fku[i,j,k] + n2[j]*fkv[i,j,k]\
                                )*n2[j]/(n1[i]**2 +\
                                n2[j]**2)
                d[i,j,k] = nu*(n1[i]**2 + n2[j]**2)*vk[i,j,k]
    return fkv -a -d


N = 8
L = 2
nu = 10
rho = 1
V0 = 1e-2

# mesh
x = np.linspace(-1, 1, N+1)[:-1]
y = np.linspace(-1, 1, N+1)[:-1]
X, Y = np.meshgrid(x, y)

# wavenumbers
n1 = np.arange(-N/2, N/2)*(2*np.pi/L)
n2 = np.arange(-N/2, N/2)*(2*np.pi/L)

# initialize
u0 = V0*np.sin(X/L)*np.cos(Y/L)*np.cos(Z/L)
v0 = -V0*np.cos(X/L)*np.sin(Y/L)*np.cos(Z/L)
t = 0.0
tf = 0.03
dt = 2e-4

# transform I.C.
uk = np.fft.fftshift(np.fft.fftn(u0))/N**3
vk = np.fft.fftshift(np.fft.fftn(v0))/N**3

# allocate storage for (n+1) and (n-1)th timestep
uknp1 = np.zeros_like(uk)
vknp1 = np.zeros_like(vk)
uknm1 = np.zeros_like(uk)
vknm1 = np.zeros_like(vk)

# first timestep with forward Euler
# time integrate
fku = compute_fk.compute_fk(N, n1, n2, n3, uk, vk, wk, dim=0)
fkv = compute_fk.compute_fk(N, n1, n2, n3, uk, vk, wk, dim=1)

#uknp1 = uk + dt*fu(uk)
#vknp1 = vk + dt*fv(vk)
#wknp1 = wk + dt*fw(wk)
#
#uknm1 = uk
#vknm1 = vk
#wknm1 = wk
#
#uk = uknp1
#vk = vknp1
#wk = wknp1

while t < tf:

    uknp1 = uk + dt*fu(uk)
    vknp1 = vk + dt*fv(vk)

    uk = uknp1
    vk = vknp1

    fku = compute_fk.compute_fk(N, n1, n2, n3, uk, vk, wk, dim=0)
    fkv = compute_fk.compute_fk(N, n1, n2, n3, uk, vk, wk, dim=1)

    t = t + dt

solu = np.fft.ifftn(np.fft.ifftshift(uknp1))*(N**3)
solv = np.fft.ifftn(np.fft.ifftshift(vknp1))*(N**3)
#plt.contourf(X[:,:,5], Y[:,:,5], np.real(u0[:,:,5]))
plt.contourf(X, Y, np.real(solu))
#plt.quiver(X[:,:,4], Y[:,:,4], np.real(solv[:,:,4]))
plt.colorbar()
plt.show()
