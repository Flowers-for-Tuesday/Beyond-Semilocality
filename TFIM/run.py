import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,contextlib,json,time,sys,importlib
from datetime import datetime,timezone
from threadpoolctl import threadpool_limits
from tfim.common import ROOT,NEW,init,save,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--stages',nargs='+',default=['tests','scan','analyze','refine','controls','gga','report','audit']);args=p.parse_args();init();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S');manifest=dict(start=stamp,config_sha256=sha(NEW/'config/experiment.json'),sources={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'tfim').glob('*.py')},stages=args.stages,completed=[])
    class Tee:
        def __init__(self,*files):self.files=files
        def write(self,s):
            for f in self.files:f.write(s);f.flush()
            return len(s)
        def flush(self):
            for f in self.files:f.flush()
    path=ROOT/'logs'/f'{stamp}.json';save(path,manifest)
    with (ROOT/'logs'/f'{stamp}.txt').open('w',encoding='utf8') as logfile,contextlib.redirect_stdout(Tee(sys.stdout,logfile)),threadpool_limits(limits=1):
        for stage in args.stages:
            if stage=='audit':continue # Package only after the execution log is closed.
            t=time.perf_counter();mod,fun=dict(tests=('study','tests'),scan=('study','run'),analyze=('analysis','run'),refine=('refinement','run'),controls=('controls','run'),gga=('gga_study','run'),report=('report','run'),audit=('audit','run'))[stage];getattr(importlib.import_module('tfim.'+mod),fun)();manifest['completed'].append(dict(stage=stage,seconds=time.perf_counter()-t));save(path,manifest)
    manifest['finished']=datetime.now(timezone.utc).isoformat();manifest['audit_after_log_close']='audit' in args.stages;save(path,manifest)
    if 'audit' in args.stages:importlib.import_module('tfim.audit').run()
if __name__=='__main__':main()
