import numpy as np
import torch
from ..util.backend_functions import backend as bd
from ..util.scaled_FT import scaled_fourier_transform

"""
MPL 2.0 License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.
"""

def angular_spectrum_method(simulation, E, z, λ, scale_factor = 1):
    """
    Compute the field in distance equal to z with the angular spectrum method. 
    By default (scale_factor = 1), the ouplut plane coordinates is the same than the input.
    Otherwise, it's recommended to use the two_steps_fresnel_method as it's computationally cheaper.
    To arbitrarily choose and zoom in a region of interest, use bluestein method instead.

    Reference: https://rafael-fuente.github.io/simulating-diffraction-patterns-with-the-angular-spectrum-method-and-python.html
    """
    global bd
    from ..util.backend_functions import backend as bd

    # compute angular spectrum
    fft_c = bd.fft.fft2(E)
    c = bd.fft.fftshift(fft_c)

    fx = bd.fft.fftshift(bd.fft.fftfreq(simulation.Nx, d = simulation.dx))
    fy = bd.fft.fftshift(bd.fft.fftfreq(simulation.Ny, d = simulation.dy))
    fxx, fyy = bd.meshgrid(fx, fy)

    argument = (2 * bd.pi)**2 * ((1. / λ) ** 2 - fxx ** 2 - fyy ** 2)

    #Calculate the propagating and the evanescent (complex) modes
    tmp = bd.sqrt(bd.abs(argument))
    kz = bd.where(argument >= 0, tmp, 1j*tmp)


    if scale_factor == 1:

        # propagate the angular spectrum a distance z
        E = bd.fft.ifft2(bd.fft.ifftshift(c * bd.exp(1j * kz * z)))

    else:
        nn_, mm_ = bd.meshgrid(bd.arange(simulation.Nx)-simulation.Nx//2, bd.arange(simulation.Ny)-simulation.Ny//2)
        factor = ((simulation.dx *simulation.dy)* bd.exp(bd.pi*1j * (nn_ + mm_)))


        simulation.x = simulation.x*scale_factor
        simulation.y = simulation.y*scale_factor

        simulation.dx = simulation.dx*scale_factor
        simulation.dy = simulation.dy*scale_factor

        extent_fx = (fx[1]-fx[0])*simulation.Nx
        simulation.xx, simulation.yy, E = scaled_fourier_transform(fxx, fyy, factor*c * bd.exp(1j * kz * z),  λ = -1, scale_factor = simulation.extent_x/extent_fx * scale_factor, mesh = True)
        simulation.extent_x = simulation.extent_x*scale_factor
        simulation.extent_y = simulation.extent_y*scale_factor

    return E


def angular_spectrum_method_torch(simulation, E, z, λ, scale_factor=1):
    """
    Angular spectrum method rewritten in PyTorch for GPU execution.
    E should be a complex64 or complex128 tensor on the desired device (usually CUDA).
    """
    device = E.device

    # 2D FFT of the input field
    fft_c = torch.fft.fft2(E)
    c = torch.fft.fftshift(fft_c)

    # Frequency coordinates
    fx = torch.fft.fftshift(torch.fft.fftfreq(simulation.Nx, d=simulation.dx, device=device))
    fy = torch.fft.fftshift(torch.fft.fftfreq(simulation.Ny, d=simulation.dy, device=device))
    fxx, fyy = torch.meshgrid(fx, fy, indexing='xy')

    # Argument inside the square root for kz
    argument = (2 * torch.pi) ** 2 * ((1. / λ) ** 2 - fxx ** 2 - fyy ** 2)

    tmp = torch.sqrt(torch.abs(argument))
    kz = torch.where(argument >= 0, tmp, 1j * tmp)

    if scale_factor == 1:

        # Propagate the angular spectrum a distance z
        propagated = c * torch.exp(1j * kz * z)
        E_out = torch.fft.ifft2(torch.fft.ifftshift(propagated))

    else:
        # Scaling — not fully implemented yet
        raise NotImplementedError("Scale factor != 1 not implemented in this PyTorch version.")

    return E_out
