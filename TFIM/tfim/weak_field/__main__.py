"""Recompute all 288 retained cases without leaving duplicate result trees."""
import tempfile,shutil,sys
from pathlib import Path
from threadpoolctl import threadpool_limits
from . import common
from ..common import read,save

def cleanup_stale():
    parent=(common.BASE/'cache/replay').resolve()
    for candidate in parent.glob('verify_*'):
        resolved=candidate.resolve()
        assert resolved.parent==parent and resolved.is_relative_to(common.ROOT.resolve()) and not candidate.is_symlink()
        if candidate.is_dir():shutil.rmtree(resolved)

def main():
    if '--cleanup-only' in sys.argv:
        cleanup_stale();print('Stale verification output cleaned',flush=True);return
    parent=common.BASE/'cache/replay';parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='verify_',dir=parent) as directory,threadpool_limits(limits=1):
        target=Path(directory).resolve()
        assert target.is_relative_to(parent.resolve()),'Temporary output escaped the experiment directory'
        common.RESULTS=target
        from .experiment import run
        run()
        original=read(common.BASE/'data/weak_field_results.csv');actual=common.read('new_experiment')
        key=lambda r:(r['L'],r['side'],r['family'],r['method'],r['ratio'])
        index={key(r):r for r in original};assert len(actual)==len(index)==288
        worst=max(abs(r['density_rms']-index[key(r)]['density_rms']) for r in actual)
        assert worst<1e-11
        result=dict(status='passed',cases=288,maximum_density_error_difference=worst,methods=common.METHODS,old_workspace_required=False,temporary_outputs_removed=True)
    save(common.BASE/'logs/reproduction_check.json',result)
    print('Independent weak-field replay passed:',result,flush=True)
if __name__=='__main__':main()
