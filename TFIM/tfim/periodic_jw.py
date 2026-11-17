"""Exact finite-ring Bloch factorization of the existing JW Majorana matrix.

The input is a repeated unit cell. Both spin parity sectors and the lowest
singular-value parity correction are retained; no thermodynamic approximation.
"""
import numpy as np
from scipy.linalg import svd

def solve_periodic(cell,n):
    g=np.asarray(cell,float);L=len(g)
    if n%L or L<3:raise ValueError('A period >=3 dividing N is required')
    M=n//L;states=[]
    for parity in [1,-1]:
        ks=(2*np.pi*np.arange(M)+(np.pi if parity==1 else 0))/M
        density=np.zeros(L);ss=[];pieces=[]
        for k in ks:
            t=np.diag(g).astype(complex);t[np.arange(L-1),np.arange(1,L)]=-1;t[-1,0]=-np.exp(1j*k)
            u,s,vh=svd(t,check_finite=False);d=(u*vh.T).real
            density+=d.sum(axis=1)/M;ss.extend(s.tolist());pieces.append((s,d))
        all_s=np.sort(ss);e=-sum(ss)
        # det(T) for real positive g: even sector always positive;
        # odd sector sign is sign(prod(g)^M - 1).
        vacuum=1 if parity==1 or M*np.log(g).sum()>0 else -1
        if vacuum!=parity:
            block,index=min(((j,int(np.argmin(s))) for j,(s,d) in enumerate(pieces)),key=lambda p:pieces[p[0]][0][p[1]])
            density-=2*pieces[block][1][:,index]/M;e+=2*all_s[0];levels=[e,-sum(ss)+2*all_s[1]]
        else:levels=[e,e+2*(all_s[0]+all_s[1])]
        states.append(dict(energy=float(e),rho=density,levels=levels,parity=parity,quasiparticle_gap=float(2*all_s[0])))
    emin=min(s['energy'] for s in states);best=min(states,key=lambda s:s['energy']).copy()
    deg=[s for s in states if abs(s['energy']-emin)<1e-10]
    best['rho']=np.mean([s['rho'] for s in deg],axis=0);levels=sorted(x for s in states for x in s['levels'])
    best['gap']=max(0.,float(levels[1]-levels[0]));best['rho_block']=best.pop('rho');best['N']=n;best['period']=L
    return best
