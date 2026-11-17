"""The existing bound-constrained / uniform-Hessian preconditioned protocol."""
import time
import numpy as np
from scipy.optimize import minimize
def optimize(f,g,start,method='L-BFGS-B',polish=True):
    tick=time.perf_counter();g=np.asarray(g);bound=f.uniform.bound
    def obj(x):e,gr=f.value_grad(x);return e-g@x,gr-g
    opts=dict(maxiter=2000,ftol=1e-15)
    if method=='L-BFGS-B':opts.update(gtol=1e-12,maxls=60,maxcor=30)
    sol=minimize(obj,start,jac=True,method=method,bounds=[(-bound,bound)]*len(g),options=opts);x=sol.x;before=float(max(abs(obj(x)[1])));iters=0
    if polish:
        K=f.hessian()
        for iters in range(150):
            e,gr=obj(x)
            if max(abs(gr))<1e-14:break
            step=np.fft.ifft(np.fft.fft(gr)/K).real;norm=np.linalg.norm(gr);accepted=False
            for a in [1.,.5,.25,.125,.0625]:
                y=x-a*step
                if max(abs(y))>=bound or not f.background_valid(y):continue
                ey,gy=obj(y)
                if np.linalg.norm(gy)<norm and ey<=e+2e-12:x=y;accepted=True;break
            if not accepted:break
    e,gr=obj(x);pg=gr.copy();pg[(x<=-bound+1e-8)&(gr>0)]=0;pg[(x>=bound-1e-8)&(gr<0)]=0
    return dict(rho=x.tolist(),energy=float(e),projected_gradient=float(max(abs(pg))),pre_polish_gradient=before,polish_iterations=iters,success=bool(sol.success),message=str(sol.message),iterations=int(sol.nit),seconds=time.perf_counter()-tick,optimizer=method,boundary_active=bool(max(abs(x))>=bound-1e-8),background_patch_valid=f.background_valid(x))
