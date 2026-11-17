import json,time
import numpy as np
from .common import ROOT,NEW,PARTS,CONFIG,init,fields,masses,rms,save,write,sha
from .uniform_ising import homogeneous
from .functional import Functional
from .exact import solve as exact_solve
from .optimization import solve

def exact(g,tag,eps=None):
    path=NEW/'cache'/f'exact_{tag}.json'
    if path.exists():
        cached=json.loads(path.read_text())
        if 'cell' in cached and not np.array_equal(cached['cell'],np.asarray(g)):raise ValueError('Cached external field mismatch')
        return cached
    result=exact_solve(g,eps or CONFIG['exact_epsabs']);result['cell']=np.asarray(g).tolist();save(path,result);return result

def amplitudes(L,m,peak):
    rows=[dict(lam=a*m/peak,alpha=a,kind='scaled') for a in CONFIG['peak_amplitudes_over_mass']]
    if L==CONFIG['fixed_period']:rows += [dict(lam=a,alpha=a*peak/m,kind='absolute') for a in CONFIG['fixed_absolute_amplitudes']]
    return sorted(rows,key=lambda r:r['lam'])

def measure(L,m,side,shape,amp,f,start=None):
    g0=1+side*m;v=np.array(shape['v']);lam=amp['lam'];g=g0+lam*v;r0=float(homogeneous(g0)[1]);linear=np.fft.ifft(np.fft.fft(v)/f.k).real
    tag=f'L{L}_m{m:.10g}_s{side}_{shape["family"]}_l{lam:.12g}';ex=exact(g,tag);xe=np.array(ex['rho']);st=r0+lam*linear if start is None else start;sol=solve(f,g,st);xa=np.array(sol['rho']);lda=homogeneous(g)[1];signal=rms(xe-r0);err=rms(xa-xe);minK=min(f.k);numeric=3*ex['quadrature_error']+sol['gradient']/minK+2e-13
    ok=bool(ex['success'] and sol['gradient']<1e-9 and sol['background_resolved'] and not sol['boundary_active'] and np.isfinite(err))
    row=dict(id=tag,protocol='fixed_period' if L==CONFIG['fixed_period'] else 'qxi_one',L=L,m=m,side=side,g0=g0,family=shape['family'],amplitude_kind=amp['kind'],lambda_value=lam,alpha=amp['alpha'],q_over_m=2*np.pi/(L*m),peak=shape['peak'],signal_rms=signal,error_rms=err,relative_error=err/signal,relative_error_upper=(err+numeric)/max(signal-3*ex['quadrature_error'],1e-30),relative_LDA_error=rms(lda-xe)/signal,exact_nonlinearity=rms(xe-r0-lam*linear)/signal,numerical_error_bound=numeric,gradient=sol['gradient'],background_mass=sol['background_mass'],background_resolved=sol['background_resolved'],same_background_side=bool(side*sol['background_mass']>0),local_field_crosses_critical=bool(min(g)<=1<=max(g)),minimum_local_critical_distance=float(min(abs(g-1))),log_geometric_mean=ex['log_geometric_mean'],quadrature_error=ex['quadrature_error'],quadrature_success=ex['success'],solver_valid=ok,solver_seconds=sol['seconds'],fallback=sol['fallback'])
    row['lambda']=row.pop('lambda_value')
    save(NEW/'profiles'/f'{tag}_{amp["kind"]}.json',dict(**row,rho_SARC=xa.tolist(),rho_exact=xe.tolist(),rho_LDA=lda.tolist(),linear_response=linear.tolist(),v=v.tolist(),solver=sol))
    return row,xa

def run():
    init();Ls=[CONFIG['fixed_period']]+CONFIG['crossover_periods'];shapes=[s for L in Ls for s in fields(L)];save(NEW/'config/frozen_shapes.json',shapes);rows=[]
    for L in Ls:
        for m in masses(L):
            for side in CONFIG['sides']:
                f=Functional(L,1+side*m)
                for shape in [s for s in shapes if s['L']==L]:
                    previous=None;oldlam=0
                    for amp in amplitudes(L,m,shape['peak']):
                        # Continue the uniform-connected branch as amplitude increases.
                        if previous is not None:
                            linear=np.fft.ifft(np.fft.fft(shape['v'])/f.k).real;start=previous+(amp['lam']-oldlam)*linear
                        else:start=None
                        row,xa=measure(L,m,side,shape,amp,f,start);rows.append(row)
                        if row['solver_valid'] and row['same_background_side']:previous=xa;oldlam=amp['lam']
                        else:previous=None;oldlam=0
                    write(NEW/'data/window_scan.csv',rows);print('SCAN',L,m,side,shape['family'],'rows',len(rows),'max relative',max(r['relative_error'] for r in rows[-len(amplitudes(L,m,shape['peak'])):]),flush=True)
    return rows

def tests():
    from .periodic_jw import solve_periodic
    out=[]
    for m in [.1,.001,.0001]:
        for side in [-1,1]:
            g0=1+side*m;ex=exact_solve(np.full(8,g0));h=homogeneous(g0);err=rms(np.array(ex['rho'])-h[1]);assert err<1e-10;out.append(dict(test='thermodynamic_uniform',m=m,side=side,error=err,energy_error=abs(ex['energy_per_site']-h[0]),quad_error=ex['quadrature_error']))
    L=32;m=.03
    for side in [-1,1]:
        g0=1+side*m;v=np.array(fields(L)[0]['v']);g=g0+.001*v;ex=exact_solve(g);fin=solve_periodic(g,2048);err=rms(np.array(ex['rho'])-fin['rho_block']);assert err<1e-10;out.append(dict(test='finite_ring_to_thermodynamic',m=m,side=side,error=err,energy_error=abs(ex['energy_per_site']-fin['energy']/2048)))
        f=Functional(L,g0);r0=float(homogeneous(g0)[1]);x=r0+.0001*v;d=np.random.default_rng(31).normal(size=L);d/=np.linalg.norm(d);h=2e-6;fd=(f.value_grad(x+h*d)[0]-f.value_grad(x-h*d)[0])/(2*h);err=abs(fd-f.value_grad(x)[1]@d);assert err<1e-7;out.append(dict(test='full_gradient',m=m,side=side,error=err))
        h=1e-7;mode=np.cos(2*np.pi*np.arange(L)/L);hmode=(f.value_grad(r0+h*mode)[1]-f.value_grad(r0-h*mode)[1])/(2*h);err=rms(hmode-f.k[1]*mode);assert err<1e-7;out.append(dict(test='uniform_Hessian',m=m,side=side,error=err))
    write(ROOT/PARTS[1]/'data/structural_checks.csv',out);print('New thermodynamic solver and broad-kernel functional tests passed',flush=True)
