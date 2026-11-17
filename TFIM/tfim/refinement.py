"""Resolve coarse pass/fail brackets; no functional parameters are changed."""
import json
import numpy as np
from .common import NEW,read,write,save,fields
from .analysis import GROUP,groupkey,accepted,run as analyze
from .functional import Functional
from .study import measure

def run():
    protocol=dict(purpose='Bracket each 1%/5% applicability threshold more tightly than the initial grid',maximum_failure_over_pass_amplitude=1.12,maximum_steps=8,selection='Geometric bisection of first pass/fail bracket only',no_functional_change=True)
    save(NEW/'config/boundary_refinement_protocol.json',protocol);rows=read(NEW/'data/window_scan.csv');windows=read(NEW/'data/applicability_windows.csv');extra=[r for r in rows if r['amplitude_kind'].startswith('boundary')]
    for w in windows:
        if w['boundary_status']!='error_threshold' or not w['lambda_verified']:continue
        L=int(w['L']);f=Functional(L,w['g0'] if 'g0'in w else 1+w['side']*w['m']);shape=next(s for s in fields(L) if s['family']==w['family']);t=w['tolerance']
        for step in range(protocol['maximum_steps']):
            rr=sorted([r for r in rows if groupkey(r)==groupkey(w)],key=lambda r:r['lambda']);passed=[];failed=None
            for r in rr:
                if accepted(r,t):passed.append(r)
                else:failed=r;break
            if not passed or failed is None or failed['lambda']/passed[-1]['lambda']<=1.12:break
            last=passed[-1];lam=np.sqrt(last['lambda']*failed['lambda']);p=json.loads((NEW/'profiles'/f'{last["id"]}_{last["amplitude_kind"]}.json').read_text());start=np.array(p['rho_NRC_LDA'])+(lam-last['lambda'])*np.array(p['linear_response']);amp=dict(lam=lam,alpha=lam*shape['peak']/w['m'],kind=f'boundary_{int(t*100)}pct');row,_=measure(L,w['m'],int(w['side']),shape,amp,f,start);rows.append(row);extra.append(row)
            write(NEW/'data/window_scan.csv',rows);write(NEW/'data/boundary_refinement.csv',extra)
        print('BOUNDARY',L,w['m'],w['side'],w['family'],t,'extra',len(extra),flush=True)
    analyze()
