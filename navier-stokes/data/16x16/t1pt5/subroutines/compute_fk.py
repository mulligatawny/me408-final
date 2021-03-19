import numpy as np

def compute_fk(N, n1, n2, uk, vk, dim=0):
    """ Compute the advection term in physical space 
        and return its Fourier transform.

        Parameters:
        uk (2d_array) : FT of u in x, y
        vk (2d_array) : FT of v in x, y
        dim (int)     : flag indicating which momentum equation
                        0 for x, 1 for y
        Returns:
        fk (2d_array) : Fourier transform of the advection term in x, y
    """
    u = np.fft.ifft2(np.fft.ifftshift(uk))*N**2
    v = np.fft.ifft2(np.fft.ifftshift(vk))*N**2
    if dim == 0:
        dudx = np.fft.ifft2(np.fft.ifftshift(1j*n1*uk))*N**2
        dudy = np.fft.ifft2(np.fft.ifftshift(1j*n2*uk))*N**2 
        f = -u*dudx -v*dudy
    elif dim == 1:
        dvdx = np.fft.ifft2(np.fft.ifftshift(1j*n1*vk))*N**2
        dvdy = np.fft.ifft2(np.fft.ifftshift(1j*n2*vk))*N**2 
        f = -u*dvdx -v*dvdy

    return np.fft.fftshift(np.fft.fft2(f))/N**2
