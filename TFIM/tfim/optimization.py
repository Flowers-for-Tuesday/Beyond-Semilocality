import time
import numpy as np
from scipy.optimize import minimize
from .common import CONFIG
def solve(f,g,start):
    tick=time.perf_counter();g=np.asarray(g);x=np.asarray(start).copy();bound=f.uniform.bound
    def obj(x):e,gr=f.value_grad(x);return e-g@x,gr-g
    # Prefer local residual continuation; it remains accurate when energy differences cancel.
    iteration=0
    for iteration in range(300):
        e,gr=obj(x);err=max(abs(gr))
        if not np.isfinite(err) or err<CONFIG['solver_tolerance']:break
        step=np.fft.ifft(np.fft.fft(gr)/f.k).real;norm=np.linalg.norm(gr);accepted=False
        for a in [1,.5,.25,.125,.0625,.03125,.015625,.0078125]:
            y=x-a*step
            if max(abs(y))>=bound:continue
            ey,gy=obj(y)
            if np.all(np.isfinite(gy)) and np.linalg.norm(gy)<norm and ey<=e+1e-11:x=y;accepted=True;break
        if not accepted:break
    # A generic optimizer is a fallback; every outcome is checked by its actual residual.
    fallback=False
    if max(abs(obj(x)[1]))>CONFIG['solver_tolerance']:
        fallback=True;s=minimize(obj,x,jac=True,method='L-BFGS-B',bounds=[(-bound,bound)]*len(x),options=dict(maxiter=600,ftol=1e-15,gtol=1e-11,maxls=40));x=s.x
        for _ in range(100):
            e,gr=obj(x)
            if max(abs(gr))<CONFIG['solver_tolerance']:break
            step=np.fft.ifft(np.fft.fft(gr)/f.k).real;accepted=False
            for a in [1,.5,.25,.125,.0625,.015625]:
                y=x-a*step
                if max(abs(y))>=bound:continue
                ey,gy=obj(y)
                if np.all(np.isfinite(gy)) and np.linalg.norm(gy)<np.linalg.norm(gr) and ey<=e+1e-11:x=y;accepted=True;break
            if not accepted:break
    e,gr=obj(x);mass=f.background_mass(x)
    return dict(rho=x.tolist(),energy_per_site=float(e/len(g)),gradient=float(max(abs(gr))),iterations=iteration+1,fallback=fallback,background_mass=mass,background_resolved=abs(mass)>CONFIG['critical_background_resolution'],seconds=time.perf_counter()-tick,boundary_active=bool(max(abs(x))>bound-1e-7))
