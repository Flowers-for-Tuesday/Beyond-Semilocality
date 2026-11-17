import json
import numpy as np
from numpy.polynomial.legendre import leggauss
from ..uniform_ising import homogeneous
from ..susceptibility import finite_response
from ..periodic_jw import solve_periodic
from .common import CONFIG,RESULTS,METHODS,write,jsave,jload,rms
from .functional import Functional
from .optimization import optimize

def field_shapes(L):
    theta=2*np.pi*np.arange(L)/L;rng=np.random.default_rng(CONFIG['seed']+L);out=[]
    for family in CONFIG['field_families']:
        if family=='localized_packet':v=np.exp(2*np.cos(theta-.73))*np.cos(3*theta+.41);metadata=dict(center=.73,concentration=2,carrier=3,phase=.41)
        else:
            modes=np.array([1,2,3,4] if family=='four_low_modes' else [1,3,L//8,L//4] if family=='mixed_scales' else list(range(1,13)))
            weights=rng.uniform(.5,1.5,len(modes))/np.sqrt(np.arange(1,len(modes)+1));phases=rng.uniform(-np.pi,np.pi,len(modes));v=np.sum(weights[:,None]*np.cos(modes[:,None]*theta+phases[:,None]),axis=0);metadata=dict(modes=modes.tolist(),weights=weights.tolist(),phases=phases.tolist())
        v-=v.mean();v/=rms(v);out.append(dict(L=L,family=family,v=v.tolist(),metadata=metadata,mean=float(v.mean()),rms=rms(v),maximum=float(max(abs(v)))))
    return out
def exact_case(L,side,family,ratio,v,nfactor=8):
    path=RESULTS/'cache'/f'JW_L{L}_s{side}_{family}_a{ratio:g}_M{nfactor}.json'
    if path.exists():return json.loads(path.read_text())
    m=2*np.pi/L;g0=1+side*m;out=[]
    for sign in [1,-1]:
        a=solve_periodic(g0+sign*ratio*m*np.asarray(v),nfactor*L);a['rho_block']=a['rho_block'].tolist();out.append(a)
    path.write_text(json.dumps(out),encoding='utf8');return out
def relaxation_penalty(f,g,xopt,xexact,order):
    delta=np.asarray(xexact)-xopt;nodes,weights=leggauss(order);values=[np.dot(f.value_grad(xopt+(t+1)*delta/2)[1]-g,delta)/len(g) for t in nodes]
    return float(np.dot(weights,values)/2)
def fit(rows):
    fits=[];eff=[]
    metrics=['density_rms','euler_defect_rms','relaxation_penalty']
    for L,side,family,method in sorted(set((r['L'],r['side'],r['family'],r['method']) for r in rows)):
        rr=sorted([r for r in rows if (r['L'],r['side'],r['family'],r['method'])==(L,side,family,method)],key=lambda r:r['ratio'])
        for metric in metrics:
            for label,used in [('primary',[r for r in rr if r['primary_window'] and r['valid_density'] and r['valid_euler'] and r['valid_relaxation']]),('all_four',rr)]:
                rec=dict(L=L,side=side,family=family,method=method,metric=metric,window=label,count=len(used))
                if len(used)>=3 and all(r[metric]>0 for r in used):
                    xx=np.log([r['lambda'] for r in used]);yy=np.log([r[metric] for r in used]);slope,c=np.polyfit(xx,yy,1);rec.update(slope=float(slope),coefficient=float(np.exp(c)),max_log_residual=float(max(abs(yy-slope*xx-c))))
                fits.append(rec)
            for a,b in zip(rr[:-1],rr[1:]):
                eff.append(dict(L=L,side=side,family=family,method=method,metric=metric,ratio_mid=np.sqrt(a['ratio']*b['ratio']),exponent=np.log(b[metric]/a[metric])/np.log(b['lambda']/a['lambda']) if a[metric]>0 and b[metric]>0 else None))
    write('new_error_orders',fits);write('new_effective_exponents',eff)
def run():
    shapes=[r for L in CONFIG['periods'] for r in field_shapes(L)];jsave('frozen_field_shapes',shapes);rows=[];profiles=[];controls=[];response=[]
    for L in CONFIG['periods']:
        for side in [-1,1]:
            m=2*np.pi/L;g0=1+side*m;r=float(homogeneous(g0)[1]);n=8*L;models={k:Functional(L,side,k) for k in METHODS};ke=models['NRC-LDA'].hessian();q=2*np.pi*np.arange(L//2+1)/L;chismall=np.array([finite_response(qq,g0,n) for qq in q]);ix=np.minimum(np.arange(L),L-np.arange(L));chiN=chismall[ix];uniform=solve_periodic(np.full(4,g0),n);bias=abs(np.mean(uniform['rho_block'])-r)
            for shape in [s for s in shapes if s['L']==L]:
                family=shape['family'];v=np.array(shape['v']);vz=np.fft.fft(v);num_linear=rms(np.fft.ifft((1/ke-chiN)*vz).real)
                for method,f in models.items():
                    k=f.hessian();a=np.fft.ifft(vz/k).real;de=np.fft.ifft((1/k-1/ke)*vz).real;force=np.fft.ifft((k/ke-1)*vz).real
                    response.append(dict(L=L,side=side,family=family,method=method,linear_density_coefficient=rms(de),linear_euler_coefficient=rms(force),linear_relaxation_coefficient=float(np.mean(de*np.fft.ifft(k*np.fft.fft(de)).real)/2),finite_vs_thermo_response=num_linear,minimum_Hessian=float(min(k))))
                    for ratio in CONFIG['lambda_over_mass']:
                        lam=ratio*m;g=g0+lam*v;exact=exact_case(L,side,family,ratio,v);sols=[optimize(f,g0+s*lam*v,r+s*lam*a) for s in [1,-1]];xa,xb=[np.array(s['rho']) for s in sols];ea,eb=[np.array(e['rho_block']) for e in exact];delta=xa-ea;odd=(xa-xb-ea+eb)/2;even=(xa+xb-ea-eb)/2;defect=f.value_grad(ea)[1]-g
                        penalty8=relaxation_penalty(f,g,xa,ea,8);penalty4=relaxation_penalty(f,g,xa,ea,4);direct=(f.value(ea)-g@ea-sols[0]['energy'])/L
                        quadratic=float(np.mean(delta*np.fft.ifft(k*np.fft.fft(delta)).real)/2);identity=delta+np.fft.ifft(np.fft.fft(defect)/k).real;grad=max(s['projected_gradient'] for s in sols);density_floor=bias+grad/min(k)+lam*num_linear;force_floor=max(k)*bias+lam*max(k)*num_linear+1e-14;energy_floor=abs(penalty8-penalty4)+rms(delta)*max(grad,1e-14)
                        row=dict(L=L,N=n,side=side,g0=g0,mass=m,xi=1/m,family=family,method=method,ratio=ratio,lambda_value=lam,density_rms=rms(delta),euler_defect_rms=rms(defect),relaxation_penalty=penalty8,penalty_direct_subtraction=direct,penalty_GL4=penalty4,line_integral_difference=abs(penalty8-penalty4),quadratic_relaxation_prediction=quadratic,penalty_over_quadratic=penalty8/quadratic if quadratic>0 else None,
                          response_identity_residual=rms(identity),odd_density_error=rms(odd),even_density_error=rms(even),measured_odd_density_coefficient=rms(odd)/lam,predicted_linear_density_coefficient=rms(de),predicted_linear_euler_coefficient=rms(force),predicted_linear_relaxation_coefficient=float(np.mean(de*np.fft.ifft(k*np.fft.fft(de)).real)/2),
                          projected_gradient=grad,uniform_finite_size_bias=bias,finite_linear_mismatch=num_linear,density_numerical_floor=density_floor,euler_numerical_floor=force_floor,relaxation_numerical_floor=energy_floor,valid_density=bool(rms(delta)>20*max(density_floor,5e-14)),valid_euler=bool(rms(defect)>20*force_floor),valid_relaxation=bool(penalty8>20*energy_floor),primary_window=ratio<=CONFIG['primary_max_lambda_over_mass'],boundary_active=any(s['boundary_active'] for s in sols),background_valid=all(s['background_patch_valid'] for s in sols),gap=exact[0]['gap'],qp_gap=exact[0]['quasiparticle_gap'])
                        row['lambda']=row.pop('lambda_value');rows.append(row);profiles.append(dict(**row,rho_plus=xa.tolist(),rho_minus=xb.tolist(),exact_plus=ea.tolist(),exact_minus=eb.tolist(),euler_defect=defect.tolist(),linear_density_direction=de.tolist(),solvers=sols))
                        if (L==64 and ratio==.01 or L==128 and ratio==.001) and method in ['NRC-LDA-R8','NRC-LDA']:
                            if L==64:other=optimize(f,g,np.full(L,r),method='SLSQP',polish=False)
                            else:other=optimize(f,g,r+lam*np.random.default_rng(CONFIG['seed']).normal(size=L))
                            controls.append(dict(L=L,side=side,family=family,method=method,ratio=ratio,density_difference=rms(np.array(other['rho'])-xa),primary_gradient=sols[0]['projected_gradient'],other=other))
                write('new_experiment',rows);jsave('new_profiles',profiles,'profiles');write('linear_predictions',response);jsave('optimizer_controls',controls);print('New field family complete',L,side,family,flush=True)
    fit(rows)
