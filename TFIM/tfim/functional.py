"""Same NRC-LDA expression as core_study, with a broad critical background table."""
import numpy as np
from scipy.interpolate import CubicHermiteSpline,PchipInterpolator
from .uniform_ising import UniformFunctional,homogeneous,field_grid
from .susceptibility import response_table
from .common import ROOT,NEW,CONFIG,init,masses

def grid(L,refine=1):
    local=np.concatenate([1+side*m*(1+np.linspace(-.2,.2,(CONFIG['kernel_local_nodes']-1)*refine+1)) for m in masses(L) for side in [-1,1]])
    geom=np.geomspace(1e-9,.8,CONFIG['kernel_log_nodes']*refine)
    return np.unique(np.r_[.05,np.linspace(.1,.9,81*refine),1-geom,1.,1+geom,np.linspace(1.1,3,101*refine),local])

class Uniform(UniformFunctional):
    def __init__(self,L,refine=1):
        refs=grid(L,refine);base=field_grid(2);base=base[np.min(abs(base[:,None]-refs),axis=1)>2e-13];g=np.sort(np.r_[base,refs]);_,r,c=homogeneous(g);keep=np.r_[True,np.diff(r)>0];self.inverse=CubicHermiteSpline(r[keep],g[keep],1/c[keep],extrapolate=False);self.primitive=self.inverse.antiderivative();self.bound=float(r[-1])

TABLES={}
class Functional:
    def __init__(self,L,g0,kind='NRC-LDA',refine=1):
        init();self.L=L;self.g0=g0;self.kind=kind;key=(L,refine)
        if key not in TABLES:
            uni=Uniform(L,refine);gs=grid(L,refine);rho=homogeneous(gs)[1];path=NEW/'cache'/f'kernel_L{L}_refine{refine}.npz'
            version_ok=False
            if path.exists():
                z=np.load(path);assert np.array_equal(z['fields'],gs);K=z['K'];version_ok='quadrature_version' in z and int(z['quadrature_version'])==2
            if not path.exists() or not version_ok or not np.all(np.isfinite(K)):
                print('Building uniform kernel',L,refine,len(gs),flush=True);q=2*np.pi*np.arange(L//2+1)/L;small=np.array([1/response_table(x,gs,CONFIG['kernel_order']) for x in q]).T;ix=np.minimum(np.arange(L),L-np.arange(L));K=small[:,ix];np.savez_compressed(path,fields=gs,K=K,quadrature_version=2)
            keep=np.r_[True,np.diff(rho)>0]
            TABLES[key]=(uni,PchipInterpolator(rho[keep],K[keep],axis=0,extrapolate=False))
        self.uniform,self.interp=TABLES[key];r=homogeneous(g0)[1];self.k=np.full(L,1/homogeneous(g0)[2]) if kind=='LDA' else np.asarray(self.interp(r));self.k[0]=1/homogeneous(g0)[2]
    def delta(self,r):
        d=np.asarray(self.interp(r))-float(self.uniform.inverse.derivative()(r));dp=np.asarray(self.interp(r,1))-float(self.uniform.inverse.derivative(2)(r));d[0]=dp[0]=0;return d,dp
    def value_grad(self,x):
        x=np.asarray(x);loc,gr=self.uniform.value_grad(x);e=float(sum(loc))
        if self.kind=='LDA':return e,gr
        r2=float(np.mean(x*x));r=np.sqrt(r2)
        if not self.interp.x[0]<r<self.interp.x[-1]:return np.inf,np.full(len(x),np.nan)
        z=np.fft.fft(x*x/r2-1);d,dp=self.delta(r);S=d@abs(z)**2;Sp=dp@abs(z)**2
        return e+r2*S/(8*len(x)),gr+x*np.fft.ifft(d*z).real/2+x/len(x)*(r*Sp/(8*len(x))-S/(4*len(x)))
    def background_mass(self,x):return float(self.uniform.inverse(np.sqrt(np.mean(np.asarray(x)**2)))-1)

