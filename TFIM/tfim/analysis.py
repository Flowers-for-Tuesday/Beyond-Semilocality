import json
import numpy as np
from .common import NEW,CONFIG,read,write,save
GROUP=['protocol','L','m','side','family']
def groupkey(r):return tuple(r[k] for k in GROUP)
def accepted(r,t):return r['solver_valid'] and r['same_background_side'] and r['relative_error_upper']<=t
def run():
    rows=read(NEW/'data/window_scan.csv');windows=[];weak=[]
    for key in sorted(set(groupkey(r) for r in rows)):
        rr=sorted([r for r in rows if groupkey(r)==key],key=lambda r:r['lambda']);base=dict(zip(GROUP,key))
        for threshold in CONFIG['relative_error_thresholds']:
            passed=[];failed=None
            for r in rr:
                if accepted(r,threshold):passed.append(r)
                else:failed=r;break
            last=passed[-1] if passed else None
            windows.append(dict(**base,tolerance=threshold,lambda_verified=last['lambda'] if last else 0,alpha_verified=last['alpha'] if last else 0,relative_error_at_limit=last['relative_error'] if last else None,lambda_first_failure=failed['lambda'] if failed else None,alpha_first_failure=failed['alpha'] if failed else None,boundary_status=('error_threshold' if failed['solver_valid'] and failed['same_background_side'] else 'numerically_or_branch_unresolved') if failed else 'right_censored_by_scan',failure_id=failed['id'] if failed else None,limit_id=last['id'] if last else None,passing_prefix=len(passed),total_points=len(rr),LDA_error_at_limit=last['relative_LDA_error'] if last else None,local_critical_crossing_at_limit=last['local_field_crosses_critical'] if last else None))
        # Fit only resolved weak points, and never use this extrapolation to define windows.
        used=[r for r in rr if r['amplitude_kind']=='scaled' and r['alpha']<=.1 and r['solver_valid'] and r['error_rms']>20*r['numerical_error_bound']]
        w=dict(**base,points=len(used))
        if len(used)>=3:
            x=np.log([r['lambda'] for r in used]);y=np.log([r['error_rms'] for r in used]);slope,c=np.polyfit(x,y,1);w.update(slope=float(slope),C2=float(np.median([r['error_rms']/r['lambda']**2 for r in used])),linear_signal_coefficient=float(np.median([r['signal_rms']/r['lambda'] for r in used])))
            w['predicted_lambda_1pct']=.01*w['linear_signal_coefficient']/w['C2']
        weak.append(w)
    write(NEW/'data/applicability_windows.csv',windows);write(NEW/'data/weak_nonlinear_coefficients.csv',weak)
    envelope=[]
    for protocol,L,m,t in sorted(set((w['protocol'],w['L'],w['m'],w['tolerance']) for w in windows)):
        subset=[w for w in windows if (w['protocol'],w['L'],w['m'],w['tolerance'])==(protocol,L,m,t)]
        # alpha limits can be compared across shapes; lambda bounds use the smallest
        # verified lambda and are likewise conservative for all four sampled cases.
        envelope.append(dict(protocol=protocol,L=L,m=m,tolerance=t,lambda_verified_all=min(w['lambda_verified'] for w in subset),alpha_verified_all=min(w['alpha_verified'] for w in subset),lambda_first_failure_any=min((w['lambda_first_failure'] for w in subset if w['lambda_first_failure']),default=None),alpha_first_failure_any=min((w['alpha_first_failure'] for w in subset if w['alpha_first_failure']),default=None),cases=len(subset),right_censored=sum(w['boundary_status']=='right_censored_by_scan' for w in subset)))
    write(NEW/'data/conservative_windows.csv',envelope)
    summary=dict(rows=len(rows),coarse_scan_rows=sum(r['amplitude_kind']in ['scaled','absolute'] for r in rows),boundary_refinement_rows=sum(r['amplitude_kind'].startswith('boundary') for r in rows),groups=len(set(groupkey(r) for r in rows)),solver_invalid=sum(not r['solver_valid'] for r in rows),background_side_changes=sum(not r['same_background_side'] for r in rows),max_exact_quadrature_error=max(r['quadrature_error'] for r in rows),windows=len(windows),weak_order_count=sum('slope' in w for w in weak),weak_order_range=[min(w['slope'] for w in weak if 'slope'in w),max(w['slope'] for w in weak if 'slope'in w)],resolved_window_boundaries=sum(w['boundary_status']=='error_threshold' for w in windows),unresolved_window_boundaries=sum(w['boundary_status']=='numerically_or_branch_unresolved' for w in windows))
    save(NEW/'data/summary.json',summary);print('Window analysis',summary,flush=True)
