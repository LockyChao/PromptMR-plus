import numpy as np
import torch
from mri_utils.fftc import fft2c_new, ifft2c_new

def fft2(x):
    """ FFT with shifting DC to the center of the image"""
    return torch.fft.fftshift(torch.fft.fft2(x), dim=[-1, -2])


def ifft2(x):
    """ IFFT with shifting DC to the corner of the image prior to transform"""
    return torch.fft.ifft2(torch.fft.ifftshift(x, dim=[-1, -2]))


def fft2_m(x):
    """ FFT for multi-coil """
    return torch.view_as_complex(fft2c_new(torch.view_as_real(x)))


def ifft2_m(x):
    """ IFFT for multi-coil """
    return torch.view_as_complex(ifft2c_new(torch.view_as_real(x)))

class SinglecoilMRI_real:
    def __init__(self, image_size, mask):
        self.image_size = image_size
        self.mask = mask

    def A(self, x):
        return fft2(x) * self.mask

    def A_dagger(self, x):
        return torch.real(ifft2(x))

    def AT(self, x):
        return self.A_dagger(x)
    
    
class SinglecoilMRI_comp:
    def __init__(self, image_size, mask):
        self.image_size = image_size
        self.mask = mask

    def A(self, x):
        return fft2_m(x) * self.mask

    def A_dagger(self, x):
        return ifft2_m(x)

    def A_T(self, x):
        return self.A_dagger(x)


class MulticoilMRI:
    def __init__(self, image_size, mask, sens):
        self.image_size = image_size
        self.mask = mask
        self.sens = sens

    def A(self, x):
        return fft2_m(self.sens * x) * self.mask

    def AT(self, x):
        return torch.sum(torch.conj(self.sens) * ifft2_m(x * self.mask), dim=1).unsqueeze(dim=1)
    
    def A_dagger(self, x):
        return self.AT(x)
    
    
def CG(A_fn, b_cg, x, n_inner=10, eps=1e-8):
    r = b_cg - A_fn(x)
    p = r.clone()
    rs_old = torch.matmul(r.view(1, -1), r.view(1, -1).T)
    for _ in range(n_inner):
        Ap = A_fn(p)
        a = rs_old / torch.matmul(p.view(1, -1), Ap.view(1, -1).T)

        x += a * p
        r -= a * Ap

        rs_new = torch.matmul(r.view(1, -1), r.view(1, -1).T)
        
        if torch.sqrt(rs_new) < eps:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x