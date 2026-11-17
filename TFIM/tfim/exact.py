"""Independent thermodynamic Bloch-JW integration, no finite-size floor."""
import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import svd
def solve(cell,eps=2e-11):
    g=np.asarray(cell);L=len(g)
    if min(g)<=0:raise ValueError('This Bloch validation protocol uses positive fields')
    def integrand(k):
        t=np.diag(g).astype(complex);t[np.arange(L-1),np.arange(1,L)]=-1;t[-1,0]=-np.exp(1j*k)
        u,s,vh=svd(t,check_finite=False);return np.r_[-sum(s)/L,(u*vh.T).real.sum(axis=1)]/np.pi
    effective_log_mass=abs(np.log(g).sum());points=np.unique(np.clip(effective_log_mass*np.array([.001,.01,.1,1,10,100]),0,np.pi))
    y,err,info=quad_vec(integrand,0,np.pi,epsabs=eps,epsrel=eps,points=points,limit=1000,norm='max',full_output=True)
    return dict(energy_per_site=float(y[0]),rho=y[1:].tolist(),quadrature_error=float(err),success=bool(info.success),evaluations=info.neval,log_geometric_mean=float(np.log(g).mean()))
