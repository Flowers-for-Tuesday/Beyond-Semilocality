import csv,json,re,platform,hashlib,zipfile
from datetime import datetime,timezone
import numpy as np
import fitz,scipy,matplotlib
from .common import ROOT,NEW,PARTS,CONFIG,read,save,sha,rms,fields,matches_recorded_hash
from .analysis import groupkey,accepted

def run():
    checks=[]
    def check(name,condition,**details):
        checks.append(dict(check=name,passed=bool(condition),**details))
        if not condition:raise AssertionError(name+' '+str(details))
    rows=read(NEW/'data/window_scan.csv');coarse=[r for r in rows if r['amplitude_kind']in ['scaled','absolute']];windows=read(NEW/'data/applicability_windows.csv');env=read(NEW/'data/conservative_windows.csv');summary=json.loads((NEW/'data/summary.json').read_text())
    check('obsolete_project_removed',not (ROOT.parent/'tfim_critical_dft').exists())
    from .workspace import verify
    from .common import WORK_DIRS
    for number,part in enumerate(PARTS,1):
        actual={p.name for p in (ROOT/part).iterdir() if p.is_dir() and p.name!='__pycache__'}
        check('uniform_workspace_'+str(number),actual==set(WORK_DIRS) and verify(number)['status']=='passed')
    check('identical_local_entry_points',len({sha(ROOT/part/'run.py') for part in PARTS})==1)
    migration=json.loads((ROOT/'layout_migration.json').read_text(encoding='utf8'))
    check('layout_migration_preserved_data',len(migration['moves'])==24 and all(not (ROOT/r['old_path']).exists() and matches_recorded_hash(ROOT/r['new_path'],r['sha256']) for r in migration['moves']))
    naming=json.loads((ROOT/'naming_migration.json').read_text(encoding='utf8'))
    check('root_readme_preserved',all(sha(ROOT.parent/p)==h for p,h in naming['readme_sha256'].items()))
    check('terminology_numeric_payload_unchanged',naming['scientific_data_validation']['passed'])
    replay=json.loads((ROOT/PARTS[2]/'logs/reproduction_check.json').read_text(encoding='utf8'))
    check('retained_weak_field_independently_reproduced',replay['status']=='passed' and replay['cases']==288 and replay['maximum_density_error_difference']<1e-11)
    check('retained_weak_field_kernel_inputs',len(list((ROOT/PARTS[2]/'cache').glob('*nodes25.npz')))==12)
    branch_profiles=json.loads((NEW/'profiles/retained/retained_branch_profiles.json').read_text(encoding='utf8'))
    check('raw_branch_counterexample_retained',len(branch_profiles)==48 and all(r['method']=='Oracle' for r in branch_profiles))
    check('complete_coarse_scan',len(coarse)==620)
    check('forty_groups',len(set(groupkey(r) for r in rows))==40)
    check('unique_field_records',len({(r['id'],r['amplitude_kind']) for r in rows})==len(rows))
    check('adaptive_records_accounted',len(rows)==620+summary['boundary_refinement_rows'])
    check('no_superseded_density_profiles',len(list((NEW/'profiles').glob('*.json')))==len(rows))
    check('no_superseded_exact_caches',len(list((NEW/'cache').glob('exact_*.json')))==len({r['id'] for r in rows}))
    check('all_exact_quadratures_converged',all(r['quadrature_success'] for r in rows))
    errors=[];classifier=[];shape_map={(L,s['family']):np.array(s['v']) for L in [CONFIG['fixed_period']]+CONFIG['crossover_periods'] for s in fields(L)}
    for r in rows:
        p=json.loads((NEW/'profiles'/f'{r["id"]}_{r["amplitude_kind"]}.json').read_text());x=np.array(p['rho_NRC_LDA']);e=np.array(p['rho_exact']);err=rms(x-e);errors.append(abs(err-r['error_rms']));assert len(x)==r['L']
        assert np.array_equal(p['v'],shape_map[(int(r['L']),r['family'])])
        cache=json.loads((NEW/'cache'/f'exact_{r["id"]}.json').read_text());assert np.array_equal(cache['cell'],r['g0']+r['lambda']*np.array(p['v']))
        check_fields=[r['relative_error'],r['signal_rms'],r['error_rms'],r['gradient']]
        assert np.all(np.isfinite(check_fields))
        classifier.append(abs(err/r['signal_rms']-r['relative_error']))
    check('recomputed_density_errors',max(errors)<1e-18)
    check('same_dimensionless_multimode_shape',all(np.max(abs(np.fft.fft(shape_map[(L,'four_low_modes')])[1:5]/L-np.fft.fft(shape_map[(32,'four_low_modes')])[1:5]/32))<1e-14 for L in CONFIG['crossover_periods']))
    check('recomputed_response_normalization',max(classifier)<1e-14)
    check('eighty_tolerance_windows',len(windows)==80)
    for w in windows:
        rr=sorted([r for r in rows if groupkey(r)==groupkey(w)],key=lambda r:r['lambda']);passed=[];failure=None
        for r in rr:
            if accepted(r,w['tolerance']):passed.append(r)
            else:failure=r;break
        check('window_'+str(groupkey(w))+'_'+str(w['tolerance']),bool(passed) and passed[-1]['lambda']==w['lambda_verified'] and (failure['lambda'] if failure else None)==w['lambda_first_failure'])
        if w['boundary_status']=='error_threshold':check('resolved_bracket_'+str(groupkey(w))+'_'+str(w['tolerance']),w['lambda_first_failure']/w['lambda_verified']<=1.1200001)
    check('twenty_conservative_envelopes',len(env)==20)
    for e in env:
        ww=[w for w in windows if (w['protocol'],w['L'],w['m'],w['tolerance'])==(e['protocol'],e['L'],e['m'],e['tolerance'])];assert len(ww)==4;assert e['lambda_verified_all']==min(w['lambda_verified'] for w in ww)
    bc=read(NEW/'data/boundary_controls.csv');qc=read(NEW/'data/exact_quadrature_controls.csv');uc=read(NEW/'data/uniform_quadrature_controls.csv')
    required={w[k] for w in windows for k in ['limit_id','failure_id'] if w[k]};check('all_boundary_points_controlled',{r['id'] for r in bc}==required)
    check('refinement_preserves_threshold_classification',all(r['classification_1pct_agrees'] and r['classification_5pct_agrees'] for r in bc))
    check('refinement_error_small_against_response',max(r['refinement_shift_over_signal'] for r in bc)<.0005,maximum=max(r['refinement_shift_over_signal'] for r in bc))
    check('boundary_local_Hessians_positive',min(r['minimum_Hessian'] for r in bc)>0,minimum=min(r['minimum_Hessian'] for r in bc))
    check('boundary_Hessians_symmetric',max(r['Hessian_asymmetry'] for r in bc)<1e-4)
    check('alternative_start_consistency',max(r['alternative_start_density_difference'] for r in bc)<1e-7,maximum=max(r['alternative_start_density_difference'] for r in bc))
    check('tight_exact_quadrature',all(r['success'] for r in qc) and max(r['shift_over_signal'] for r in qc)<1e-7)
    check('independent_uniform_quadrature',max(r['relative_difference'] for r in uc)<1e-8)
    tests=read(ROOT/PARTS[1]/'data/structural_checks.csv');check('new_solver_and_gradient_tests',len(tests)==12 and max(r['error'] for r in tests)<1e-7)
    oldrows=read(ROOT/PARTS[2]/'data/weak_field_results.csv');oldprofiles=json.loads((ROOT/PARTS[2]/'profiles/weak_field_profiles.json').read_text());check('core_evidence_count',len(oldrows)==288 and len(oldprofiles)==288 and {r['method'] for r in oldrows}=={'LDA','NRC-LDA-R8','NRC-LDA'})
    check('core_profiles_recomputed',all(abs(rms(np.array(p['rho_plus'])-p['exact_plus'])-p['density_rms'])<1e-20 for p in oldprofiles))
    check('retained_structural_checks',len(read(ROOT/PARTS[1]/'data/structural_validation.csv'))==10)
    provenance=json.loads((ROOT/'provenance.json').read_text());check('retained_snapshot_hashes',all(sha(ROOT/p['output'])==p['output_sha256'] for p in provenance))
    for p in (NEW/'cache').glob('kernel*.npz'):
        z=np.load(p);check('finite_kernel_'+p.stem,np.all(np.isfinite(z['K'])) and int(z['quadrature_version'])==2)
    first=json.loads(sorted((NEW/'logs').glob('*.json'))[0].read_text());check('original_grid_config_unchanged',matches_recorded_hash(NEW/'config/experiment.json',first['config_sha256']))
    report=ROOT/'report'
    layout=json.loads((report/'report_layout_validation.json').read_text(encoding='utf8'))
    evidence=json.loads((report/'table_provenance.json').read_text(encoding='utf8'))
    check('report_balanced_four_part_layout',layout['actual_pages']==layout['planned_pages']==15 and layout['section_pages']=={'1':2,'2':3,'3':4,'4':4})
    check('report_input_data_unchanged',layout['scientific_files_unchanged'] and all(sha(ROOT/p)==h for p,h in layout['scientific_input_hashes'].items()))
    check('report_manifest_hash',sha(report/'table_provenance.json')==layout['source_table_manifest_sha256'])
    sources={s['id']:s for s in evidence['sources']};loaded={}
    for sid,s in sources.items():
        path=ROOT/s['path'];check('report_source_hash_'+sid,sha(path)==s['sha256'])
        loaded[sid]=read(path) if path.suffix=='.csv' else np.load(path) if path.suffix=='.npz' else json.loads(path.read_text(encoding='utf8'))
    # Independently map displayed numeric columns to their source observables.
    mappings={
        '1.1':dict(q='q',K='K_exact',Klda='K_LDA'),
        '1.2':dict(m='m',side='side',exact='exact_scaled',limit='theory',residual='exact_residual'),
        '1.3':dict(m='mass',side='side',R='R',norm='coefficient_l1',prediction='scaled_prediction_sup',error='scaled_error_sup'),
        '2.2':dict(side='side',error='error',energy='energy_error'),
        '2.3':dict(m='m',side='side',error='error',energy='energy_error'),
        '2.4':dict(L='L',side='side',R='R',cut='truncated_train_error',fit='train_error',valid='validation_error'),
        '2.5':dict(L='L',side='side',pred='predicted_C1_max',measured='measured_C1_max',pmin='refit_p_min',pmax='refit_p_max',smin='nrc_lda_p_min',smax='nrc_lda_p_max'),
        '3.1':dict(method='method',ratio='ratio',density='density_rms',euler='euler_defect_rms',G='relaxation_penalty',gradient='projected_gradient'),
        '3.3':dict(side='side',R='R',scaledR='R_over_xi',pred='predicted_linear_RMS',measured='measured_odd_over_lambda',relative='relative_coefficient_difference',resolved='linear_signal_resolved'),
        '3.5':dict(ratio='ratio',GL8='relaxation_penalty',GL4='penalty_GL4',direct='penalty_direct_subtraction',quadratic_ratio='penalty_over_quadratic'),
        '4.1':dict(m='m',side='side',signal='signal_rms',error='error_rms',epsilon='relative_error',gradient='gradient'),
        '4.2':dict(L='L',m='m',tolerance='tolerance',alpha='alpha_verified_all',censored='right_censored',**{'pass':'lambda_verified_all','fail':'lambda_first_failure_any'}),
        '4.3':dict(alpha='alpha',epsilon='relative_error',upper='relative_error_upper',gradient='gradient',**{'lambda':'lambda'}),
        '4.4':dict(refined='relative_error_refined',shift='refinement_shift_over_signal',eigen='minimum_Hessian',restart='alternative_start_density_difference'),
        '4.5':dict(m='m',epsilon='relative_error',gradient='gradient',valid='solver_valid',**{'lambda':'lambda'}),
        '4.6':dict(m='m',count='points',slope='slope',C2='C2',A1='linear_signal_coefficient'),
        '4.8':dict(optimizer='optimizer',cluster='cluster',energy='energy',gradient='projected_gradient',hessian='hessian_min')}
    table_counts={}
    for t in evidence['tables']:
        tid=t['id'];table_counts[tid]=len(t['records'])
        with (report/t['export']).open(encoding='utf8',newline='') as f:export=list(csv.DictReader(f))
        assert len(export)==len(t['records'])
        for r,out in zip(t['records'],export):
            assert all(out[k]==('' if r[k] is None else str(r[k])) for k in t['keys']), (tid,'export')
            originals=[]
            for ref in r['_refs']:
                raw=loaded[ref['source']]
                if ref['kind']=='csv':
                    original=raw[ref['row']-2];assert original==ref['snapshot'],(tid,ref['source'],ref['row']);originals.append(original)
                    if 'line' in r:assert r['line']==ref['row']
                elif ref['kind']=='json':
                    p=raw[ref['index']];i=ref['site']
                    assert r['exact']==p['exact_plus'][i] and r['NRC_LDA']==p['rho_plus'][i] and r['Euler']==p['euler_defect'][i]
                    assert r['difference']==p['rho_plus'][i]-p['exact_plus'][i]
                elif ref['kind']=='npz':
                    node,k=ref['index']
                    assert r['g']==raw['fields'][node] and r['rho']==raw['rho'][node] and r['K']==raw['K'][node,k] and r['K0']==raw['K'][node,0]
                    assert r['delta']==r['K']-r['K0']
            if tid in mappings:assert all(r[k]==originals[0].get(v) for k,v in mappings[tid].items()),tid
            if tid=='1.1':assert r['ratio']==originals[0]['K_exact']*np.log(1/originals[0]['q'])/np.pi
            if tid=='3.2':assert all(r[k]==o['slope'] for k,o in zip(['pLDA','pR8','pS','pz','pG'],originals)) and all((r['L'],r['side'])==(o['L'],o['side']) and o['count']==3 and o['window']=='primary' for o in originals)
            if tid=='4.4':assert r['same']==(originals[0]['classification_1pct_agrees'] and originals[0]['classification_5pct_agrees'])
            if tid=='4.7':assert [r[k] for k in ['exact1','exact2','app1','app2']]==[o[k] for o in originals for k in ['A1','A2']]
            if tid=='4.8':
                group=[x for x in loaded[r['_refs'][0]['source']] if x['optimizer']==r['optimizer'] and x['cluster']==r['cluster']]
                assert r['count']==len(group) and r['energy']==min(x['energy'] for x in group)
        check('report_table_verified_'+tid,True,rows=len(t['records']))
    check('report_complete_declared_subsets',table_counts['3.2']==24 and table_counts['4.1']==28 and table_counts['4.2']==20 and table_counts['4.5']==1 and table_counts['3.4']==16)
    check('report_refit_tables_complete',table_counts['2.4']==24 and table_counts['2.5']==6)
    check('report_scientific_table_totals',sum(n for t,n in table_counts.items() if t!='D.1')==layout['scientific_table_rows']==250 and len(table_counts)-1==21)
    pdf=ROOT/'report/TFIM_核心研究与临界适用窗口_含重拟合对照.pdf';doc=fitz.open(pdf);pages=[]
    for i,page in enumerate(doc):
        text=page.get_text();check('pdf_text_'+str(i+1),'\ufffd'not in text and len(text)>100)
        spans=[s for b in page.get_text('dict')['blocks'] if 'lines'in b for l in b['lines'] for s in l['spans']];check('pdf_bounds_'+str(i+1),all(s['bbox'][0]>=0 and s['bbox'][2]<=page.rect.width+.5 and s['bbox'][1]>=0 and s['bbox'][3]<=page.rect.height+.5 for s in spans));pages.append(dict(page=i+1,characters=len(text),first_line=text.splitlines()[0]))
    (ROOT/'report/pdf_text_extraction.txt').write_text('\n\n'.join(p.get_text() for p in doc),encoding='utf8')
    for i in range(len(doc)):
        if i<len(doc):doc[i].get_pixmap(matrix=fitz.Matrix(1.2,1.2)).save(ROOT/'report'/f'preview_{i+1}.png')
    for p in [ROOT/'README.md',ROOT/'report/TFIM_report.md']+[ROOT/part/'README.md' for part in PARTS]:
        content=re.sub(r'\\\[.*?\\\]','',p.read_text(encoding='utf8'),flags=re.S)
        for link in re.findall(r'\]\(([^)]+)\)',content):check('link_'+str(p.relative_to(ROOT))+'_'+link,(p.parent/link).exists())
    result=dict(status='passed',created_utc=datetime.now(timezone.utc).isoformat(),checks=checks,pages=pages,new_main_cases=620,new_boundary_cases=summary['boundary_refinement_rows'],new_total_cases=len(rows),retained_core_rows=288,versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,pymupdf=fitz.VersionBind),scope='Standalone TFIM directory; historical source hashes recorded without requiring old workspace files.')
    save(ROOT/'audit.json',result)
    files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__'not in p.parts and p.suffix not in ['.pyc','.zip'] and not p.name.startswith('preview_') and p.name not in ['artifact_manifest.json','package_validation.json','TFIM_核心研究与临界适用窗口.pdf']]
    manifest={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted(files)};save(ROOT/'artifact_manifest.json',dict(sha256=manifest));files.append(ROOT/'artifact_manifest.json');zip_path=ROOT/'report/TFIM_核心实验.zip'
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,str(p.relative_to(ROOT.parent)))
    with zipfile.ZipFile(zip_path) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read('TFIM/'+p)).hexdigest()==h for p,h in manifest.items())
    save(ROOT/'report/package_validation.json',dict(sha256=sha(zip_path),bytes=zip_path.stat().st_size,files=len(files),all_entry_hashes_verified=True))
    print('AUDIT PASSED',len(checks),'checks;',len(doc),'pages;',len(rows),'new fields;',len(files),'standalone package files',flush=True)
