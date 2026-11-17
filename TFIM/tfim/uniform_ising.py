import numpy as np
from scipy.special import ellipe, ellipkm1
from scipy.interpolate import CubicHermiteSpline


def homogeneous(g):
    """Thermodynamic (energy/site, transverse density, susceptibility)."""
    a = np.abs(np.asarray(g, dtype=float))
    b = np.maximum(a, 1e-4)
    d = ((b - 1) / (b + 1)) ** 2
    ek = ellipkm1(d)
    ee = ellipe(1 - d)
    with np.errstate(invalid='ignore'):
        e = -2 * (1 + b) * ee / np.pi
        rho = ((1 + b) * ee + (b - 1) * ek) / (np.pi * b)
        chi = ((1 + b*b) * ek - (1 + b)**2 * ee) / (np.pi*b*b*(1+b))
    # Series removes cancellation at the origin.
    e = np.where(a < 1e-4, -1-a*a/4-a**4/64, e)
    rho = np.where(a < 1e-4, a/2+a**3/16, rho)
    chi = np.where(a < 1e-4, .5+3*a*a/16, chi)
    rho = np.where(a == 1, 2/np.pi, rho)
    return e, rho*np.sign(g), chi


def field_grid(dense=1):
    return np.unique(np.r_[np.linspace(0, .9, 901*dense),
        1-np.geomspace(1e-12,.1,500*dense), 1.,
        1+np.geomspace(1e-12,.1,500*dense),
        np.linspace(1.1,3,1901*dense), np.geomspace(3,1e4,1000*dense)])


class UniformFunctional:
    """Hermite g(rho) with exact 1/chi slopes; integrate the SAME interpolant."""
    def __init__(self, dense=1):
        g = field_grid(dense)
        _,rho,chi = homogeneous(g)
        keep = np.r_[True, np.diff(rho)>0]
        self.inverse = CubicHermiteSpline(rho[keep], g[keep], 1/chi[keep], extrapolate=False)
        self.primitive = self.inverse.antiderivative()
        self.bound = float(rho[-1])

    def field(self, rho):
        return self.inverse(np.minimum(np.abs(rho),self.bound))*np.sign(rho)

    def value_grad(self, rho):
        a = np.minimum(np.abs(rho),self.bound)
        return self.primitive(a)-self.primitive(0)-1, self.field(rho)
