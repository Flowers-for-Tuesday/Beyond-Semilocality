import json
import numpy as np
from .common import NEW,CONFIG,read,write,save,rms
from .analysis import accepted
from .functional import Functional
from .optimization import solve
from .uniform_ising import homogeneous
from .exact import solve as exact_solve
from .susceptibility import exact_response,response_table

def load_profile(r):return json.loads((NEW/'profiles'/f'{r["id"]}_{r["amplitude_kind"]}.json').read_text())
def run():
    rows=read(NEW/'data/window_scan.csv');windows=read(NEW/'data/applicability_windows.csv');boundary=[];excontrols=[];spectral=[]
    chosen=set(w[k] for w in windows for k in ['limit_id','failure_id'] if w[k])
    for r in [r for r in rows if r['id'] in chosen]:
        L=int(r['L']);g0=r['g0'];p=load_profile(r);v=np.array(p['v']);g=g0+r['lambda']*v;x=np.array(p['rho_NRC_LDA']);xe=np.array(p['rho_exact']);f=Functional(L,g0);fine=Functional(L,g0,refine=2);s=solve(fine,g,x);shift=rms(np.array(s['rho'])-x);errfine=rms(np.array(s['rho'])-xe)/r['signal_rms']
        r0=float(homogeneous(g0)[1]);alt=solve(f,g,r0+r['lambda']*np.array(p['linear_response']));difference=rms(np.array(alt['rho'])-x)
        # Full Hessian at the reported branch; central gradient differences, not
        # an energy-difference test that would lose significant digits.
        h=1e-7;H=np.empty((L,L))
        for j in range(L):
            d=np.zeros(L);d[j]=h;H[:,j]=(f.value_grad(x+d)[1]-f.value_grad(x-d)[1])/(2*h)
        eig=np.linalg.eigvalsh((H+H.T)/2);asym=float(np.max(abs(H-H.T)))
        rec=dict(id=r['id'],protocol=r['protocol'],L=L,m=r['m'],side=r['side'],family=r['family'],lambda_value=r['lambda'],relative_error_original=r['relative_error'],relative_error_refined=errfine,refinement_density_shift=shift,refinement_shift_over_signal=shift/r['signal_rms'],refined_gradient=s['gradient'],alternative_start_density_difference=difference,alternative_start_gradient=alt['gradient'],minimum_Hessian=float(eig[0]),Hessian_asymmetry=asym,classification_1pct_agrees=(errfine<=.01)==(r['relative_error']<=.01),classification_5pct_agrees=(errfine<=.05)==(r['relative_error']<=.05))
        boundary.append(rec);write(NEW/'data/boundary_controls.csv',boundary)
        if len(boundary)%20==0:print('Checked boundary points',len(boundary),flush=True)
    # Tight quadrature at small/large mass and the largest cell, around both thresholds.
    candidates=[r for r in rows if r['id'] in chosen and (r['L']==256 or r['L']==32 and r['m'] in [.1,.0001])]
    for r in candidates:
        p=load_profile(r);g=r['g0']+r['lambda']*np.array(p['v']);e=exact_solve(g,eps=2e-13);shift=rms(np.array(e['rho'])-p['rho_exact']);excontrols.append(dict(id=r['id'],L=r['L'],m=r['m'],density_shift=shift,shift_over_signal=shift/r['signal_rms'],quadrature_error=e['quadrature_error'],success=e['success']))
    for q in [2*np.pi/256,2*np.pi/32,np.pi]:
        for g in [1-.0001,1,1+.0001]:
            a=exact_response(q,g);b=response_table(q,np.array([g]),24)[0];spectral.append(dict(q=q,g=g,relative_difference=abs(a-b)/a))
    write(NEW/'data/exact_quadrature_controls.csv',excontrols);write(NEW/'data/uniform_quadrature_controls.csv',spectral)
    summary=dict(boundary_points=len(boundary),exact_controls=len(excontrols),max_refinement_shift_over_signal=max(r['refinement_shift_over_signal'] for r in boundary),max_quadrature_shift_over_signal=max(r['shift_over_signal'] for r in excontrols),minimum_boundary_Hessian=min(r['minimum_Hessian'] for r in boundary),max_alternative_start_difference=max(r['alternative_start_density_difference'] for r in boundary),classification_disagreements=sum(not r['classification_1pct_agrees'] or not r['classification_5pct_agrees'] for r in boundary),max_uniform_quadrature_difference=max(r['relative_difference'] for r in spectral))
    save(NEW/'data/control_summary.json',summary);print('Boundary controls',summary,flush=True)
