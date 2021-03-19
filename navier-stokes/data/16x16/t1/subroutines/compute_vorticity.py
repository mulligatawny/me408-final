import numpy as np

def compute_vorticity(N, n1, n2, uk, vk):
    dvdx = np.fft.ifftn(np.fft.ifftshift(1j*n1*vk))*N**2
    dudy = np.fft.ifftn(np.fft.ifftshift(1j*n2*uk))*N**2
    return np.real(dvdx - dudy)
