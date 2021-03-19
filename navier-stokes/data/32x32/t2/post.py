import numpy as np
import matplotlib.pyplot as plt
from subroutines import compute_fk
from matplotlib import cm

u = np.load('./u.npy')
v = np.load('./v.npy')
x = np.load('./x.npy')
y = np.load('./y.npy')
t = np.load('./t.npy')
w = np.load('./w.npy')
p = np.load('./p.npy')
e = np.load('./e.npy')

fig1 = plt.figure(1)
plt.contourf(x, y, u, cmap=cm.inferno, vmin=-.05, vmax=.05)
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Contours of $u$ at t = {:.2f}'.format(t))
plt.quiver(x, y, u, v, cmap=cm.gray)
plt.show()

fig2 = plt.figure(2)
plt.contourf(x, y, v, cmap=cm.inferno, vmin=-.05, vmax=.05)
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Contours of $v$ at t = {:.2f}'.format(t))
plt.quiver(x, y, u, v, cmap=cm.gray)
plt.show()

fig3 = plt.figure(3)
plt.contourf(x, y, w, cmap=cm.GnBu, vmin=-.05, vmax=.05)
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Contours of $\omega$ at t = {:.2f}'.format(t))
plt.quiver(x, y, u, v, cmap=cm.gray)
plt.show()

fig4 = plt.figure(4)
plt.contourf(x, y, p, cmap=cm.OrRd, vmin=.9998, vmax=1.0002)
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Contours of $p$ at t = {:.2f}'.format(t))
plt.quiver(x, y, u, v, cmap=cm.gray)
plt.show()
