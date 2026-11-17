"""Mao et al., Phys. Rev. B 104, 155145 (2021), Eq. (13).

The published coefficient is fixed at J_eff=0.35 (J=1); no refitting.
Periodic bonds are counted once, with rho=<sigma_x> in the Pauli convention.
"""
import numpy as np
from .uniform_ising import homogeneous

J_EFF = 0.35


class GGA:
    def __init__(self, L, g0, uniform):
        self.n = L
        self.g0 = g0
        self.uniform = uniform

    def value_grad(self, x):
        x = np.asarray(x)
        local, gradient = self.uniform.value_grad(x)
        u = x*x
        bonds = u - np.roll(u, -1)
        energy = float(np.sum(local) + J_EFF*np.dot(bonds, bonds))
        gradient = gradient + 4*J_EFF*x*(2*u-np.roll(u, 1)-np.roll(u, -1))
        return energy, gradient

    def value(self, x):
        return self.value_grad(x)[0]

    def hessian(self):
        _, rho, chi = homogeneous(self.g0)
        q = 2*np.pi*np.arange(self.n)/self.n
        return 1/chi + 32*J_EFF*rho*rho*np.sin(q/2)**2

    def background_valid(self, x):
        return bool(np.max(np.abs(x)) < self.uniform.bound)
