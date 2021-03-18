import numpy as np
import matplotlib.pyplot as plt
from subroutines import compute_fk
from matplotlib import cm

u = np.load('./data/32x32/t2/u.npy')
v = np.load('./data/32x32/t2/v.npy')
x = np.load('./data/32x32/t2/x.npy')
y = np.load('./data/32x32/t2/y.npy')
t = np.load('./data/32x32/t2/t.npy')

plt.contourf(x, y, u, cmap=cm.inferno)
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Contours of $u$ at t = {:.2f}'.format(t))
plt.quiver(x, y, u, v, cmap=cm.gray)
plt.show()
