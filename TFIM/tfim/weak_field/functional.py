"""NRC-LDA, one local baseline, and a pure tail-removal control."""
import numpy as np
from scipy.interpolate import CubicHermiteSpline,PchipInterpolator
from ..uniform_ising import UniformFunctional,homogeneous,field_grid
from ..susceptibility import response_table
from .common import ROOT,RESULTS,CONFIG,init
def fields(g0,nodes=None):return g0+.12*abs(g0-1)*np.linspace(-1,1,nodes or CONFIG['background_nodes'])
class Uniform(UniformFunctional):
    def __init__(self):
        refs=np.unique(np.concatenate([fields(1+s*2*np.pi/L) for L in CONFIG['periods'] for s in [-1,1]]));base=field_grid();base=base[np.min(abs(base[:,None]-refs),axis=1)>1e-12];g=np.sort(np.r_[base,refs]);_,r,chi=homogeneous(g);keep=np.r_[True,np.diff(r)>0]
        self.inverse=CubicHermiteSpline(r[keep],g[keep],1/chi[keep],extrapolate=False);self.primitive=self.inverse.antiderivative();self.bound=float(r[-1])
UNIFORM=Uniform();TABLES={}
def kernel(L,side,nfft,nodes=None):
    nodes=nodes or CONFIG['background_nodes'];key=(L,side,nfft,nodes)
    if key not in TABLES:
        init();gs=fields(1+side*2*np.pi/L,nodes);rho=homogeneous(gs)[1];name=f'patch_L{L}_side{side}_fft{nfft}.npz';new=RESULTS/'cache'/f'K_L{L}_side{side}_fft{nfft}_nodes{nodes}.npz'
        archived=ROOT/'03_Error-Cancellation_Mechanism/cache'/new.name
        if new.exists() or archived.exists():
            z=np.load(new if new.exists() else archived);K=z['K'];assert np.array_equal(z['fields'],gs)
        else:
            q=2*np.pi*np.arange(nfft//2+1)/nfft;small=np.array([1/response_table(x,gs,CONFIG['response_quadrature_order']) for x in q]).T;idx=np.minimum(np.arange(nfft),nfft-np.arange(nfft));K=small[:,idx];np.savez_compressed(new,fields=gs,rho=rho,K=K)
        TABLES[key]=PchipInterpolator(rho,K,axis=0)
    return TABLES[key]
class Functional:
    def __init__(self,L,side,kind,nodes=None):
        self.n=L;self.side=side;self.kind=kind;self.g0=1+side*2*np.pi/L;self.uniform=UNIFORM;self.nfft=8*L if kind=='NRC-LDA-R8' else L;self.interp=kernel(L,side,self.nfft,nodes) if kind.startswith('NRC-LDA') else None
    def delta(self,r):
        d=np.asarray(self.interp(r))-float(UNIFORM.inverse.derivative()(r));dp=np.asarray(self.interp(r,nu=1))-float(UNIFORM.inverse.derivative(2)(r));d[0]=dp[0]=0
        if self.kind=='NRC-LDA-R8':
            dist=np.minimum(np.arange(self.nfft),self.nfft-np.arange(self.nfft))
            for a in [d,dp]:
                real=np.fft.ifft(a).real;real[dist>CONFIG['tail_range']]=0;real[0]-=real.sum();a[:]=np.fft.fft(real).real;a[0]=0
            d=d[::8];dp=dp[::8]
        return d,dp
    def value_grad(self,x):
        x=np.asarray(x);loc,gr=UNIFORM.value_grad(x);e=float(loc.sum());n=len(x)
        if self.kind=='LDA':return e,gr
        r2=float(np.mean(x*x))
        if r2==0:return e,gr
        r=np.sqrt(r2);z=np.fft.fft(x*x/r2-1);d,dp=self.delta(r);power=abs(z)**2;ss=d@power;sp=dp@power
        return e+r2*ss/(8*n),gr+x*np.fft.ifft(d*z).real/2+x/n*(r*sp/(8*n)-ss/(4*n))
    def value(self,x):return self.value_grad(x)[0]
    def hessian(self):
        r=float(homogeneous(self.g0)[1]);k0=float(1/homogeneous(self.g0)[2]);q=2*np.pi*np.arange(self.n)/self.n
        if self.kind=='LDA':return np.full(self.n,k0)
        return k0+self.delta(r)[0]
    def background_valid(self,x):
        if self.interp is None:return True
        r=np.sqrt(np.mean(np.asarray(x)**2));return bool(self.interp.x[0]<r<self.interp.x[-1])
