###############################################################################
# 2D Incompressible Navier-Stokes Solver with Periodic Boundaries, Galerkins' #
# Method and Adams-Bashforth Time Integration                                 #
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
from subroutines import compute_fk
from subroutines import compute_vorticity
from subroutines import compute_pressure
from matplotlib import cm

def fu(uk, vk):
    fku = compute_fk.compute_fk(N, n1, n2, uk, vk, dim=0)
    fkv = compute_fk.compute_fk(N, n1, n2, uk, vk, dim=1)
    a = np.zeros_like(uk)
    d = np.zeros_like(uk)
    for i in range(N):
        for j in range(N):
            if n1[i] == n2[j] == 0:
                a[i,j] = 1
            else:
                a[i,j] = (n1[i]*fku[i,j] + n2[j]*fkv[i,j]\
                            )*n1[i]/(n1[i]**2 +\
                            n2[j]**2)
            d[i,j] = nu*(n1[i]**2 + n2[j]**2)*uk[i,j]
    return fku -a -d

def fv(uk, vk):
    fku = compute_fk.compute_fk(N, n1, n2, uk, vk, dim=0)
    fkv = compute_fk.compute_fk(N, n1, n2, uk, vk, dim=1)
    a = np.zeros_like(vk)
    d = np.zeros_like(vk)
    for i in range(N):
        for j in range(N):
            if n1[i] == n2[j] == 0:
                a[i,j] = 1
            else:
                a[i,j] = (n1[i]*fku[i,j] + n2[j]*fkv[i,j]\
                            )*n2[j]/(n1[i]**2 +\
                            n2[j]**2)
            d[i,j] = nu*(n1[i]**2 + n2[j]**2)*vk[i,j]
    return fkv -a -d

N = 32
L = 2*np.pi
nu = 1

# mesh
x = np.linspace(-L/2, L/2, N+1)[:-1]
y = np.linspace(-L/2, L/2, N+1)[:-1]
X, Y = np.meshgrid(x, y)

# wavenumbers
n1 = np.arange(-N/2, N/2)*(2*np.pi/L)
n2 = np.arange(-N/2, N/2)*(2*np.pi/L)

# initialize
u0 = 0.5*(np.sin(Y+X) +np.sin(Y-X))
v0 = -0.5*(np.sin(X+Y) +np.sin(X-Y))
t = 0.0
tf = 10.0
dt = 5e-4
nt = int(tf/dt+1)
e = np.zeros(nt)
count = 0

# transform I.C.
uk = np.fft.fftshift(np.fft.fftn(u0))/N**2
vk = np.fft.fftshift(np.fft.fftn(v0))/N**2

# allocate storage for (n+1) and (n-1)th timestep
uknp1 = np.zeros_like(uk)
vknp1 = np.zeros_like(vk)
uknm1 = np.zeros_like(uk)
vknm1 = np.zeros_like(vk)

# first timestep with forward Euler
fku = compute_fk.compute_fk(N, n1, n2, uk, vk, dim=0)
fkv = compute_fk.compute_fk(N, n1, n2, uk, vk, dim=1)
uknp1 = uk + dt*fu(uk, vk)
vknp1 = vk + dt*fv(uk, vk)
uknm1 = uk
vknm1 = vk
uk = uknp1
vk = vknp1

# time integrate using Adams-Bashforth
while t < tf:
    uknp1 = uk + (dt/2)*(3*fu(uk, vk) - fu(uknm1, vknm1))
    vknp1 = vk + (dt/2)*(3*fv(uk, vk) - fv(uknm1, vknm1))
    uknm1 = uk
    vknm1 = vk
    uk = uknp1
    vk = vknp1
    u = np.real(np.fft.ifftn(np.fft.ifftshift(uk))*(N**2))
    v = np.real(np.fft.ifftn(np.fft.ifftshift(vk))*(N**2))
    e[count] = (np.mean(u**2) + np.mean(v**2))/2
    count = count + 1
    t = t + dt

w = compute_vorticity.compute_vorticity(N, n1, n2, uk, vk)
p = compute_pressure.compute_pressure(N, n1, n2, uk, vk)
u = np.real(np.fft.ifftn(np.fft.ifftshift(uk))*(N**2))
v = np.real(np.fft.ifftn(np.fft.ifftshift(vk))*(N**2))

#np.save('./data/32x32/t2/t.npy', t)
#np.save('./data/32x32/t2/x.npy', X)
#np.save('./data/32x32/t2/y.npy', Y)
#np.save('./data/32x32/t2/u.npy', u)
#np.save('./data/32x32/t2/v.npy', v)
#np.save('./data/32x32/t2/w.npy', w)
#np.save('./data/32x32/t2/p.npy', p)
np.save('./data/32x32/t2/e.npy', e)

#plt.contourf(X, Y, p, cmap=cm.bone)
#plt.colorbar()
#plt.quiver(X, Y, solu, solv, cmap=cm.gray)
#plt.show()