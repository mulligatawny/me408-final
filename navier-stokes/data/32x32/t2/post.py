import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

u = np.load('./u.npy')
v = np.load('./v.npy')
x = np.load('./x.npy')
y = np.load('./y.npy')
t = np.load('./t.npy')
w = np.load('./w.npy')
p = np.load('./p.npy')
e = np.load('./e.npy')

plt.contourf(x, y, u, cmap=cm.inferno, vmin=-.1, vmax=.1)
#plt.contourf(x, y, w, cmap=cm.binary)
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Contours of $\omega$ at t = {:.2f}'.format(t))
plt.quiver(x, y, u, v, cmap=cm.gray)

#plt.semilogy((e[:-1]))
plt.show()
