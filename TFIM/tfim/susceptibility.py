import numpy as np
from scipy.integrate import quad
from scipy.special import roots_legendre
from .uniform_ising import homogeneous


def integrand(k,q,g):
    # Pair-creation coherence factor of the transverse-density vertex tau_z.
    a=g-np.cos(k); b=g-np.cos(k+q)
    x=np.sin(k); y=np.sin(k+q)
    d=np.hypot(a,x); e=np.hypot(b,y)
    # At g=1, rounded quadrature nodes can coincide with a single band minimum.
    # The symmetric endpoint value is finite for q != 0; changing a measure-zero
    # value does not change the integral. q=0 is handled analytically by callers.
    coherence=np.divide(a*b-x*y,d*e,out=np.zeros_like(d*e),where=d*e>0)
    return np.divide(1-coherence,d+e,out=np.zeros_like(d+e),where=d+e>0)


def finite_response(q,g,n):
    k=2*np.pi*(np.arange(n)+.5)/n
    return float(np.mean(integrand(k,q,g)))


def exact_response(q,g):
    q=float(abs((q+np.pi)%(2*np.pi)-np.pi))
    if q<1e-15:
        return float(homogeneous(g)[2])
    m=abs(g-1)
    # Include the periodic image of the second minimum near +pi. Without it,
    # adaptive quadrature can miss a narrow endpoint contribution at q=pi.
    points=[-np.pi,0.,-q,2*np.pi-q,np.pi]
    for center in [0.,-q,2*np.pi-q]:
        for scale in [m,q]:
            if scale>0:
                for f in [.1,1,10,100]:
                    points += [center-scale*f,center+scale*f]
    points=np.unique(np.clip(points,-np.pi,np.pi))
    return sum(quad(lambda k:integrand(k,q,g),a,b,epsabs=2e-10,
                    epsrel=2e-10,limit=200)[0] for a,b in zip(points[:-1],points[1:]))/(2*np.pi)


def response_table(q,gs,order=16):
    """Thermodynamic quadrature, geometrically resolved at both band minima."""
    q=float(abs((q+np.pi)%(2*np.pi)-np.pi))
    gs=np.asarray(gs)
    if q<1e-15:
        return homogeneous(gs)[2]
    offsets=np.r_[0.,np.geomspace(1e-13,8,100)]
    points=np.unique(np.clip(np.r_[-offsets,offsets,-q-offsets,-q+offsets,2*np.pi-q-offsets,2*np.pi-q+offsets],-np.pi,np.pi))
    x,w=roots_legendre(order)
    a=points[:-1,None]; b=points[1:,None]
    nodes=((a+b)/2+(b-a)*x/2).ravel()
    weights=((b-a)*w/2).ravel()/(2*np.pi)
    out=[]
    for chunk in np.array_split(gs,max(1,int(np.ceil(len(gs)/64)))):
        out.extend(integrand(nodes[None,:],q,chunk[:,None])@weights)
    return np.asarray(out)
