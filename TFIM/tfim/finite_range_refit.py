"""Reconstructed finite-range refit control; not recovered fifth-round data."""
import json
from datetime import datetime, timezone
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits
from .common import ROOT, PARTS, read, write, save, sha, rms
from .weak_field.functional import Functional, UNIFORM
from .weak_field.optimization import optimize

BASE = ROOT / PARTS[1]
CORE = ROOT / PARTS[2]
CONFIG = BASE / 'config/finite_range_refit.json'


def basis(q, R):
    return 4 * np.sin(np.asarray(q)[:, None] * np.arange(1, R + 1) / 2) ** 2


def fit_kernel(k, R):
    n = len(k)
    q = 2 * np.pi * np.arange(n) / n
    A = basis(q, R)
    train = np.arange(0, n // 2 + 1, 2)
    valid = np.arange(1, n // 2, 2)
    B = A[train] / k[train, None]
    y = (k[train] - k[0]) / k[train]
    aub = np.r_[np.c_[B, -np.ones(len(B))], np.c_[-B, -np.ones(len(B))]]
    bub = np.r_[y, -y]
    cost = np.r_[np.zeros(R), 1.]
    fit = linprog(cost, A_ub=aub, b_ub=bub, bounds=[(None, None)] * R + [(0, None)],
                  method='highs', options={'dual_feasibility_tolerance':1e-9, 'primal_feasibility_tolerance':1e-9})
    if not fit.success: raise RuntimeError(fit.message)
    b = fit.x[:R]
    kfit = k[0] + A @ b
    error = abs(kfit / k - 1)
    dual_gap = abs(fit.fun - bub @ fit.ineqlin.marginals)
    violation = max(0., float(np.max(aub @ fit.x - bub)))
    assert dual_gap < 1e-7 and violation < 1e-7 and np.min(kfit) > 0
    return b, kfit, dict(train_error=float(max(error[train])), validation_error=float(max(error[valid])),
                         lp_lower_bound=float(bub @ fit.ineqlin.marginals), dual_gap=float(dual_gap),
                         constraint_violation=violation, minimum_kernel=float(min(kfit)))


def truncated_kernel(k, R):
    n = len(k)
    real = np.fft.ifft(k - k[0]).real
    real[np.minimum(np.arange(n), n - np.arange(n)) > R] = 0
    real[0] -= real.sum()
    return k[0] + np.fft.fft(real).real


class RefittedFunctional(Functional):
    def __init__(self, L, side, rho, coefficients):
        self.n = L
        self.side = side
        self.g0 = 1 + side * 2 * np.pi / L
        self.kind = 'Refit-R8'
        self.uniform = UNIFORM
        self.interp = PchipInterpolator(rho, coefficients, axis=0, extrapolate=False)
        self.A = basis(2 * np.pi * np.arange(L) / L, coefficients.shape[1])

    def delta(self, r):
        return self.A @ self.interp(r), self.A @ self.interp(r, 1)


def run():
    config = json.loads(CONFIG.read_text(encoding='utf8'))
    shapes = json.loads((CORE / 'config/frozen_field_shapes.json').read_text(encoding='utf8'))
    reference = json.loads((CORE / 'profiles/weak_field_profiles.json').read_text(encoding='utf8'))
    old_orders = read(CORE / 'data/weak_field_orders.csv')
    inputs = [CORE / 'config/frozen_field_shapes.json', CORE / 'profiles/weak_field_profiles.json',
              CORE / 'data/weak_field_orders.csv', CORE / 'data/weak_field_results.csv']
    inputs += [CORE / 'cache' / f'K_L{L}_side{side}_fft{8*L}_nodes25.npz'
               for L in config['periods'] for side in config['sides']]
    hashes = {str(p.relative_to(ROOT)).replace('\\', '/'):sha(p) for p in inputs}
    kernels, nodes, results, profiles, orders, summaries, checks = [], [], [], [], [], [], []
    for L in config['periods']:
        for side in config['sides']:
            source = CORE / 'cache' / f'K_L{L}_side{side}_fft{8*L}_nodes25.npz'
            z = np.load(source)
            center = len(z['fields']) // 2
            k = z['K'][center]
            mass = 2 * np.pi / L
            nfft = len(k)
            train = np.arange(0, nfft//2+1, 2)
            for R in config['ranges']:
                b, ka, stats = fit_kernel(k, R)
                cut = truncated_kernel(k, R)
                kernels.append(dict(L=L,side=side,m=mass,R=R,**stats,
                                    truncated_train_error=float(max(abs(cut[train]/k[train]-1))),
                                    coefficient_l1=float(sum(abs(b))),coefficients=json.dumps(b.tolist())))
            coefficients = []
            for node, (field, rho, kk) in enumerate(zip(z['fields'], z['rho'], z['K'])):
                b, ka, stats = fit_kernel(kk, config['density_range'])
                coefficients.append(b)
                nodes.append(dict(L=L,side=side,node=node,g0=float(field),rho=float(rho),**stats,
                                  **{f'b{i+1}':float(x) for i,x in enumerate(b)}))
            coefficients = np.asarray(coefficients)
            np.savez_compressed(BASE/'cache'/f'refit_L{L}_side{side}.npz',fields=z['fields'],rho=z['rho'],coefficients=coefficients)
            f = RefittedFunctional(L,side,z['rho'],coefficients)
            kfit = f.hessian()
            kexact = k[::8]
            r0 = float(z['rho'][center])
            vtest = np.asarray(next(s['v'] for s in shapes if s['L']==L))
            point = r0 + .003 * mass * vtest
            direction = np.random.default_rng(137).normal(size=L)
            direction /= np.linalg.norm(direction)
            h = 2e-6
            fd = (f.value(point+h*direction)-f.value(point-h*direction))/(2*h)
            gradient_error = abs(fd - f.value_grad(point)[1]@direction)
            assert gradient_error < 1e-7
            mode = np.cos(2*np.pi*np.arange(L)/L)
            hm = (f.value_grad(r0+1e-7*mode)[1]-f.value_grad(r0-1e-7*mode)[1])/2e-7
            hessian_error = rms(hm-kfit[1]*mode)
            assert hessian_error < 1e-7
            checks.append(dict(L=L,side=side,gradient_error=float(gradient_error),hessian_error=hessian_error))
            for shape in [s for s in shapes if s['L']==L]:
                family = shape['family']
                v = np.asarray(shape['v'])
                vz = np.fft.fft(v)
                linear = np.fft.ifft(vz/kfit).real
                predicted = rms(np.fft.ifft((1/kfit-1/kexact)*vz).real)
                group = []
                for ratio in config['lambda_over_mass']:
                    index = next(i for i,p in enumerate(reference) if p['L']==L and p['side']==side and p['family']==family and p['method']=='SARC' and p['ratio']==ratio)
                    exact = reference[index]
                    lam = ratio*mass
                    sols = [optimize(f,f.g0+sign*lam*v,r0+sign*lam*linear) for sign in [1,-1]]
                    xa,xb = [np.asarray(s['rho']) for s in sols]
                    ea,eb = np.asarray(exact['exact_plus']),np.asarray(exact['exact_minus'])
                    odd = (xa-xb-ea+eb)/2
                    defect = f.value_grad(ea)[1]-(f.g0+lam*v)
                    gradient = max(s['projected_gradient'] for s in sols)
                    row = dict(L=L,side=side,m=mass,family=family,ratio=ratio,lambda_value=lam,
                               density_rms=rms(xa-ea),euler_rms=rms(defect),predicted_linear_coefficient=predicted,
                               measured_odd_coefficient=rms(odd)/lam,gradient=gradient,
                               background_valid=all(s['background_patch_valid'] for s in sols),
                               boundary_active=any(s['boundary_active'] for s in sols),exact_profile_index=index)
                    assert gradient < 1e-9 and row['background_valid'] and not row['boundary_active']
                    group.append(row);results.append(row)
                    profiles.append(dict(**row,rho_plus=xa.tolist(),rho_minus=xb.tolist(),euler_defect=defect.tolist()))
                primary = group[:3]
                pd = np.polyfit(np.log([r['lambda_value'] for r in primary]),np.log([r['density_rms'] for r in primary]),1)[0]
                pe = np.polyfit(np.log([r['lambda_value'] for r in primary]),np.log([r['euler_rms'] for r in primary]),1)[0]
                orders.append(dict(L=L,side=side,family=family,p_density=float(pd),p_euler=float(pe),
                                   predicted_linear_coefficient=predicted,measured_odd_coefficient=group[0]['measured_odd_coefficient'],
                                   coefficient_relative_difference=abs(group[0]['measured_odd_coefficient']/predicted-1)))
            fitted = [r for r in orders if r['L']==L and r['side']==side]
            sarc = [r['slope'] for r in old_orders if r['L']==L and r['side']==side and r['method']=='SARC' and r['metric']=='density_rms' and r['window']=='primary']
            summaries.append(dict(L=L,side=side,m=mass,
                                  predicted_C1_max=max(r['predicted_linear_coefficient'] for r in fitted),
                                  measured_C1_max=max(r['measured_odd_coefficient'] for r in fitted),
                                  refit_p_min=min(r['p_density'] for r in fitted),refit_p_max=max(r['p_density'] for r in fitted),
                                  sarc_p_min=min(sarc),sarc_p_max=max(sarc)))
            print('REFIT',L,side,'groups',len(orders),'density rows',len(results),flush=True)
    for name,rows in [('kernels',kernels),('nodes',nodes),('results',results),('orders',orders),('summary',summaries),('checks',checks)]:
        write(BASE/'data'/f'finite_range_refit_{name}.csv',rows)
    save(BASE/'profiles/finite_range_refit_profiles.json',profiles)
    assert all(sha(ROOT/p)==value for p,value in hashes.items())
    save(BASE/'logs/finite_range_refit.json',dict(status='passed',created_utc=datetime.now(timezone.utc).isoformat(),
         provenance='reconstructed experiment; original fifth-round files unavailable',config_sha256=sha(CONFIG),
         input_sha256=hashes,kernel_fits=len(kernels),background_node_fits=len(nodes),density_cases=len(results),
         density_groups=len(orders),max_dual_gap=max(r['dual_gap'] for r in nodes+kernels),
         density_order_range=[min(r['p_density'] for r in orders),max(r['p_density'] for r in orders)],
         max_coefficient_relative_difference=max(r['coefficient_relative_difference'] for r in orders),
         max_gradient=max(r['gradient'] for r in results)))
    print('Finite-range refit complete; original inputs unchanged.',flush=True)


def verify():
    log = json.loads((BASE/'logs/finite_range_refit.json').read_text(encoding='utf8'))
    assert log['status']=='passed' and sha(CONFIG)==log['config_sha256']
    assert all(sha(ROOT/p)==value for p,value in log['input_sha256'].items())
    kernels=read(BASE/'data/finite_range_refit_kernels.csv')
    nodes=read(BASE/'data/finite_range_refit_nodes.csv')
    rows=read(BASE/'data/finite_range_refit_results.csv')
    orders=read(BASE/'data/finite_range_refit_orders.csv')
    summaries=read(BASE/'data/finite_range_refit_summary.csv')
    profiles=json.loads((BASE/'profiles/finite_range_refit_profiles.json').read_text(encoding='utf8'))
    reference=json.loads((CORE/'profiles/weak_field_profiles.json').read_text(encoding='utf8'))
    shapes=json.loads((CORE/'config/frozen_field_shapes.json').read_text(encoding='utf8'))
    assert (len(kernels),len(nodes),len(rows),len(orders),len(profiles),len(summaries))==(24,150,96,24,96,6)
    models={}
    for L in [64,128,256]:
        for side in [-1,1]:
            z=np.load(CORE/'cache'/f'K_L{L}_side{side}_fft{8*L}_nodes25.npz')
            cache=np.load(BASE/'cache'/f'refit_L{L}_side{side}.npz')
            f=RefittedFunctional(L,side,cache['rho'],cache['coefficients'])
            models[L,side]=f
            for r in [r for r in kernels if r['L']==L and r['side']==side]:
                R=int(r['R']);k=z['K'][12];q=2*np.pi*np.arange(len(k))/len(k)
                b=np.array(json.loads(r['coefficients']));ka=k[0]+basis(q,R)@b
                train=np.arange(0,len(k)//2+1,2);valid=np.arange(1,len(k)//2,2)
                assert abs(max(abs(ka[train]/k[train]-1))-r['train_error'])<1e-12
                assert abs(max(abs(ka[valid]/k[valid]-1))-r['validation_error'])<1e-12
                assert r['train_error']<=r['truncated_train_error']+1e-9
                assert abs(fit_kernel(k,R)[2]['lp_lower_bound']-r['lp_lower_bound'])<1e-9
            for r in [r for r in nodes if r['L']==L and r['side']==side]:
                i=int(r['node']);assert np.array_equal(cache['coefficients'][i],[r[f'b{j}'] for j in range(1,9)])
    for r,p in zip(rows,profiles):
        f=models[int(r['L']),int(r['side'])]
        exact=reference[int(r['exact_profile_index'])]
        xa,xb=np.asarray(p['rho_plus']),np.asarray(p['rho_minus'])
        ea,eb=np.asarray(exact['exact_plus']),np.asarray(exact['exact_minus'])
        v=np.asarray(next(s['v'] for s in shapes if s['L']==r['L'] and s['family']==r['family']))
        lam=r['lambda_value'];g=f.g0+lam*v
        assert abs(rms(xa-ea)-r['density_rms'])<1e-20
        assert abs(rms((xa-xb-ea+eb)/2)/lam-r['measured_odd_coefficient'])<1e-14
        assert rms(f.value_grad(ea)[1]-g-np.asarray(p['euler_defect']))<1e-13
        assert max(abs(f.value_grad(xa)[1]-g))<1e-9
    for o in orders:
        group=[r for r in rows if (r['L'],r['side'],r['family'])==(o['L'],o['side'],o['family'])]
        primary=sorted(group,key=lambda r:r['ratio'])[:3]
        slope=np.polyfit(np.log([r['lambda_value'] for r in primary]),np.log([r['density_rms'] for r in primary]),1)[0]
        assert abs(slope-o['p_density'])<1e-12
        assert o['coefficient_relative_difference']<1e-4
    for s in summaries:
        group=[o for o in orders if (o['L'],o['side'])==(s['L'],s['side'])]
        assert s['refit_p_min']==min(o['p_density'] for o in group) and s['refit_p_max']==max(o['p_density'] for o in group)
        assert s['predicted_C1_max']==max(o['predicted_linear_coefficient'] for o in group)
        assert s['measured_C1_max']==max(o['measured_odd_coefficient'] for o in group)
    assert max(r['dual_gap'] for r in nodes+kernels)<1e-7
    checks=read(BASE/'data/finite_range_refit_checks.csv')
    assert len(checks)==6 and max(max(r['gradient_error'],r['hessian_error']) for r in checks)<1e-7
    return dict(status='passed',kernel_fits=24,density_cases=96,density_groups=24,original_round_recovered=False)


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
