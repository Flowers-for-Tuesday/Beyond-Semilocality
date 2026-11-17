"""Published GGA control on retained fields; exact evidence is read-only."""
import json
import numpy as np
from .common import ROOT, PARTS, read, write, save, sha, rms
from .gga import GGA, J_EFF
from .uniform_ising import homogeneous
from .weak_field.functional import UNIFORM, Functional
from .weak_field.optimization import optimize
from .weak_field.experiment import relaxation_penalty
from .functional import Uniform

MINIMAL, WEAK, CRITICAL = [ROOT/PARTS[i] for i in [1, 2, 3]]
CONFIG_PATH = WEAK/'config/gga_control.json'


def load(path):
    return json.loads(path.read_text(encoding='utf8'))


def structural_checks():
    """Independent finite differences, symmetry, and uniform-limit checks."""
    rows = []
    for side in [-1, 1]:
        L = 64
        g0 = 1+side*2*np.pi/L
        rho = float(homogeneous(g0)[1])
        f = GGA(L, g0, UNIFORM)
        theta = 2*np.pi*np.arange(L)/L
        x = rho+.001*np.cos(theta)+.0007*np.sin(7*theta)
        direction = np.random.default_rng(20261117).normal(size=L)
        direction /= np.linalg.norm(direction)
        h = 2e-5
        e, grad = f.value_grad(x)
        fd = (-f.value(x+2*h*direction)+8*f.value(x+h*direction)
              -8*f.value(x-h*direction)+f.value(x-2*h*direction))/(12*h)
        values = {'gradient_difference': abs(fd-grad@direction)}
        eta = np.cos(3*theta)+.4*np.sin(9*theta)
        h = 1e-6
        finite = (f.value_grad(rho+h*eta)[1]-f.value_grad(rho-h*eta)[1])/(2*h)
        expected = np.fft.ifft(f.hessian()*np.fft.fft(eta)).real
        values['uniform_Hessian'] = rms(finite-expected)
        signs = np.random.default_rng(73).choice([-1, 1], L)
        es, gs = f.value_grad(x*signs)
        values['local_sign_symmetry'] = max(abs(e-es), rms(gs-signs*grad))
        et, gt = f.value_grad(np.roll(x, 7))
        values['translation_symmetry'] = max(abs(e-et), rms(gt-np.roll(grad, 7)))
        eu, gu = f.value_grad(np.full(L, rho))
        values['uniform_limit'] = max(abs(eu-np.sum(UNIFORM.value_grad(np.full(L, rho))[0])), rms(gu-g0))
        # Direct bond-by-bond loop independently checks periodic normalization.
        correction = sum(J_EFF*(x[i]**2-x[(i+1) % L]**2)**2 for i in range(L))
        values['periodic_bond_energy'] = abs(e-np.sum(UNIFORM.value_grad(x)[0])-correction)
        rows.extend(dict(test=k, side=side, error=float(v), tolerance=1e-7, passed=bool(v<1e-7)) for k,v in values.items())
    write(MINIMAL/'data/gga_structural_checks.csv', rows)
    assert all(r['passed'] for r in rows)
    return rows


def weak_field():
    config = load(CONFIG_PATH)['weak_field']
    shapes = {(int(s['L']), s['family']): np.array(s['v']) for s in load(WEAK/'config/frozen_field_shapes.json')}
    sources = [p for p in load(WEAK/'profiles/weak_field_profiles.json') if p['method']=='NRC-LDA']
    rows, profiles, orders = [], [], []
    for L in config['periods']:
        for side in config['sides']:
            g0 = 1+side*2*np.pi/L
            rho = float(homogeneous(g0)[1])
            f = GGA(L, g0, UNIFORM)
            k = f.hessian()
            ke = Functional(L, side, 'NRC-LDA').hessian()
            for family in config['families']:
                v = shapes[L, family]
                vf = np.fft.fft(v)
                direction = np.fft.ifft(vf/k).real
                linear_error = np.fft.ifft((1/k-1/ke)*vf).real
                for p in sorted([p for p in sources if (p['L'],p['side'],p['family'])==(L,side,family)], key=lambda p:p['ratio']):
                    lam = p['lambda']
                    g = g0+lam*v
                    sols = [optimize(f, g0+s*lam*v, rho+s*lam*direction) for s in [1, -1]]
                    xa, xb = [np.array(s['rho']) for s in sols]
                    ea, eb = np.array(p['exact_plus']), np.array(p['exact_minus'])
                    defect = f.value_grad(ea)[1]-g
                    grad = max(s['projected_gradient'] for s in sols)
                    error = rms(xa-ea)
                    odd = rms((xa-xb-ea+eb)/2)/lam
                    penalty = relaxation_penalty(f, g, xa, ea, 8)
                    gl4 = relaxation_penalty(f, g, xa, ea, 4)
                    row = dict(L=L, N=p['N'], side=side, g0=g0, mass=p['mass'], family=family,
                               method='GGA', J_eff=J_EFF, ratio=p['ratio'], **{'lambda':lam},
                               density_rms=error, euler_defect_rms=rms(defect), relaxation_penalty=penalty,
                               penalty_GL4=gl4, line_integral_difference=abs(penalty-gl4),
                               projected_gradient=grad, predicted_linear_density_coefficient=rms(linear_error),
                               measured_odd_density_coefficient=odd,
                               linear_coefficient_relative_difference=abs(odd/rms(linear_error)-1),
                               primary_window=p['ratio'] in config['primary_ratios'],
                               density_numerical_floor=p['uniform_finite_size_bias']+lam*p['finite_linear_mismatch']+grad/min(k),
                               solver_valid=grad<1e-10 and not any(s['boundary_active'] for s in sols))
                    assert row['solver_valid'] and error>20*max(row['density_numerical_floor'],5e-14)
                    rows.append(row)
                    profiles.append(dict(**row, rho_plus=xa.tolist(),rho_minus=xb.tolist(),exact_plus=ea.tolist(),exact_minus=eb.tolist(),euler_defect=defect.tolist(),v=v.tolist(),solvers=sols))
                group = [r for r in rows if (r['L'],r['side'],r['family'])==(L,side,family) and r['primary_window']]
                for metric in ['density_rms','euler_defect_rms','relaxation_penalty']:
                    xx, yy = np.log([r['lambda'] for r in group]), np.log([r[metric] for r in group])
                    slope, intercept = np.polyfit(xx, yy, 1)
                    orders.append(dict(L=L,side=side,family=family,method='GGA',metric=metric,window='primary',count=len(group),slope=float(slope),coefficient=float(np.exp(intercept)),max_log_residual=float(max(abs(yy-slope*xx-intercept)))))
            print('GGA weak field complete',L,side,flush=True)
    write(WEAK/'data/gga_weak_field.csv',rows)
    write(WEAK/'data/gga_orders.csv',orders)
    save(WEAK/'profiles/gga_weak_field.json',profiles)
    return rows, orders


def critical_path():
    config = load(CONFIG_PATH)['critical']
    source = [r for r in read(CRITICAL/'data/window_scan.csv') if r['protocol']=='fixed_period' and r['amplitude_kind']=='absolute' and r['lambda']==config['lambda']]
    L = config['L']
    uniform, refined = Uniform(L), Uniform(L,2)
    rows, profiles = [], []
    for r in source:
        p = load(CRITICAL/'profiles'/f'{r["id"]}_{r["amplitude_kind"]}.json')
        v, exact = np.array(p['v']), np.array(p['rho_exact'])
        g = r['g0']+r['lambda']*v
        f = GGA(L,r['g0'],uniform)
        rho = float(homogeneous(r['g0'])[1])
        direction = np.fft.ifft(np.fft.fft(v)/f.hessian()).real
        sol = optimize(f,g,rho+r['lambda']*direction)
        x = np.array(sol['rho'])
        # Grid refinement and an independent initial density control the new solver.
        fine = optimize(GGA(L,r['g0'],refined),g,x)
        restart = optimize(f,g,np.full(L,rho))
        error = rms(x-exact)
        row = dict(id=r['id'],L=L,m=r['m'],side=r['side'],g0=r['g0'],family=r['family'],
                   method='GGA',J_eff=J_EFF,**{'lambda':r['lambda']},signal_rms=r['signal_rms'],
                   error_rms=error,relative_error=error/r['signal_rms'],gradient=sol['projected_gradient'],
                   nrc_error_rms=r['error_rms'],nrc_relative_error=r['relative_error'],
                   lda_error_rms=rms(np.array(p['rho_LDA'])-exact),
                   lda_relative_error=rms(np.array(p['rho_LDA'])-exact)/r['signal_rms'],
                   refinement_shift_over_signal=rms(x-np.array(fine['rho']))/r['signal_rms'],
                   restart_density_difference=rms(x-np.array(restart['rho'])),
                   control_gradient=max(fine['projected_gradient'],restart['projected_gradient']),
                   solver_valid=sol['projected_gradient']<1e-10 and not sol['boundary_active'])
        assert row['solver_valid'] and row['control_gradient']<1e-10
        rows.append(row)
        profiles.append(dict(**row,rho_GGA=x.tolist(),rho_exact=exact.tolist(),v=v.tolist(),solver=sol,refined_solver=fine,restart_solver=restart))
    write(CRITICAL/'data/gga_critical_path.csv',rows)
    # Keep the original one-file-per-field window records intact.
    save(CRITICAL/'profiles/gga/fixed_amplitude.json',profiles)
    print('GGA critical path complete',len(rows),flush=True)
    return rows


def verify():
    manifest = load(WEAK/'logs/gga_control.json')
    assert manifest['config_sha256']==sha(CONFIG_PATH)
    assert all(sha(ROOT/p)==h for p,h in manifest['inputs'].items())
    assert all(sha(ROOT/p)==h for p,h in manifest['outputs'].items())
    assert all(sha(ROOT/p)==h for p,h in manifest['code'].items())
    rows = read(WEAK/'data/gga_weak_field.csv')
    profiles = load(WEAK/'profiles/gga_weak_field.json')
    assert len(rows)==len(profiles)==96
    originals = {(p['L'],p['side'],p['family'],p['ratio']):p for p in load(WEAK/'profiles/weak_field_profiles.json') if p['method']=='NRC-LDA'}
    for r,p in zip(rows,profiles):
        original = originals[p['L'],p['side'],p['family'],p['ratio']]
        assert p['exact_plus']==original['exact_plus'] and p['exact_minus']==original['exact_minus']
        assert abs(rms(np.array(p['rho_plus'])-p['exact_plus'])-r['density_rms'])<1e-20
        assert r['solver_valid'] and r['J_eff']==J_EFF
        f = GGA(int(r['L']),r['g0'],UNIFORM)
        g = r['g0']+r['lambda']*np.array(p['v'])
        assert abs(rms(f.value_grad(np.array(p['exact_plus']))[1]-g)-r['euler_defect_rms'])<1e-16
        assert abs(relaxation_penalty(f,g,np.array(p['rho_plus']),np.array(p['exact_plus']),8)-r['relaxation_penalty'])<1e-20
    orders = read(WEAK/'data/gga_orders.csv')
    assert len(orders)==72
    for order in orders:
        group = [r for r in rows if (r['L'],r['side'],r['family'])==(order['L'],order['side'],order['family']) and r['primary_window']]
        slope = np.polyfit(np.log([r['lambda'] for r in group]),np.log([r[order['metric']] for r in group]),1)[0]
        assert abs(slope-order['slope'])<1e-12 and order['count']==3
    critical = read(CRITICAL/'data/gga_critical_path.csv')
    cp = load(CRITICAL/'profiles/gga/fixed_amplitude.json')
    assert len(critical)==len(cp)==28
    for r,p in zip(critical,cp):
        old = load(CRITICAL/'profiles'/f'{r["id"]}_absolute.json')
        assert p['rho_exact']==old['rho_exact'] and p['v']==old['v']
        error = rms(np.array(p['rho_GGA'])-p['rho_exact'])
        assert abs(error-r['error_rms'])<1e-20 and abs(error/r['signal_rms']-r['relative_error'])<1e-14
        assert r['solver_valid'] and r['refinement_shift_over_signal']<1e-5 and r['restart_density_difference']<1e-9
    tests = read(MINIMAL/'data/gga_structural_checks.csv')
    assert len(tests)==12 and all(r['passed'] and r['error']<r['tolerance'] for r in tests)
    return dict(status='passed',weak_cases=len(rows),order_fits=len(orders),critical_cases=len(critical),structural_checks=len(tests))


def run():
    config = load(CONFIG_PATH)
    assert config['J_eff']==J_EFF and not config['refit']
    input_paths = [WEAK/'profiles/weak_field_profiles.json',WEAK/'config/frozen_field_shapes.json',WEAK/'data/weak_field_results.csv',CRITICAL/'data/window_scan.csv']
    input_paths += sorted((WEAK/'cache').glob('*nodes25.npz'))
    input_paths += [CRITICAL/'profiles'/f'{r["id"]}_absolute.json' for r in read(CRITICAL/'data/window_scan.csv') if r['protocol']=='fixed_period' and r['amplitude_kind']=='absolute' and r['lambda']==config['critical']['lambda']]
    inputs = {p.relative_to(ROOT).as_posix():sha(p) for p in input_paths}
    checks = structural_checks()
    weak, orders = weak_field()
    critical = critical_path()
    assert inputs=={p.relative_to(ROOT).as_posix():sha(p) for p in input_paths}
    summary = dict(reference=config['reference'],J_eff=J_EFF,refit=False,weak_cases=len(weak),critical_cases=len(critical),
                   order_ranges={metric:[min(r['slope'] for r in orders if r['metric']==metric),max(r['slope'] for r in orders if r['metric']==metric)] for metric in ['density_rms','euler_defect_rms','relaxation_penalty']},
                   max_gradient=max(r['projected_gradient'] for r in weak),
                   max_primary_linear_coefficient_difference=max(r['linear_coefficient_relative_difference'] for r in weak if r['ratio']==.0003),
                   max_critical_gradient=max(r['gradient'] for r in critical),
                   max_refinement_shift_over_signal=max(r['refinement_shift_over_signal'] for r in critical),
                   max_restart_density_difference=max(r['restart_density_difference'] for r in critical),
                   critical_worst_by_mass=[dict(m=m,lda=max(r['lda_relative_error'] for r in critical if r['m']==m),gga=max(r['relative_error'] for r in critical if r['m']==m),nrc_lda=max(r['nrc_relative_error'] for r in critical if r['m']==m)) for m in config['critical']['masses']],
                   structural_max_error=max(r['error'] for r in checks))
    save(WEAK/'data/gga_summary.json',summary)
    output_paths = [MINIMAL/'data/gga_structural_checks.csv',WEAK/'data/gga_weak_field.csv',WEAK/'data/gga_orders.csv',WEAK/'data/gga_summary.json',WEAK/'profiles/gga_weak_field.json',CRITICAL/'data/gga_critical_path.csv',CRITICAL/'profiles/gga/fixed_amplitude.json']
    code_paths = [ROOT/'tfim'/p for p in ['gga.py','gga_study.py','uniform_ising.py','functional.py','weak_field/functional.py','weak_field/optimization.py','weak_field/experiment.py']]
    save(WEAK/'logs/gga_control.json',dict(config_sha256=sha(CONFIG_PATH),inputs=inputs,outputs={p.relative_to(ROOT).as_posix():sha(p) for p in output_paths},code={p.relative_to(ROOT).as_posix():sha(p) for p in code_paths},reference_inputs_unchanged=True))
    result = verify()
    save(WEAK/'logs/gga_verification.json',result)
    print('GGA verified',result,flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        run()
