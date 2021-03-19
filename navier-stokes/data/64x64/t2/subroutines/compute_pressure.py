import numpy as np
from subroutines import compute_fk

def compute_pressure(N, n1, n2, uk, vk):
    pk = np.zeros_like(uk)
    fku = compute_fk.compute_fk(N, n1, n2, uk, vk, dim=0)
    fkv = compute_fk.compute_fk(N, n1, n2, uk, vk, dim=1)
    for i in range(N):
        for j in range(N):
            if n1[i] == n2[j] == 0:
                pk[i,j] = 1
            else:
                pk[i,j] = 1j*(n1[i]*fku[i,j] + n2[j]*fkv[i,j]\
                            )/(n1[i]**2 + n2[j]**2)

    return np.real(np.fft.ifftn(np.fft.ifftshift(pk))*N**2)
