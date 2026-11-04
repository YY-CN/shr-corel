"""Owned group execution. Default CLI never executes real data without Permit."""
import argparse,contextlib,copy,ctypes,dataclasses,json,os,signal,sys,time
from pathlib import Path
import numpy as np
import torch
from .config import SPEC,ScopeError,canonical_hash
from .authorization import Permit,require_execution
from .trainer import TrainingSession,QueryArtifact,tensor_digest
from .storage import atomic_json,finish,load_final

@contextlib.contextmanager
def exclusive(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);f=path.open('a+b')
    try:
        if os.name=='nt':
            import msvcrt
            f.seek(0);f.write(b'0');f.flush();f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    finally:f.close()

def restore_final(session,result):
    session.model.load_state_dict(result['model']);session.best_state=copy.deepcopy(result['model']);session.done=True
    session.epoch=result['completed_epoch'];session.updates=result['updates'];session.history=result['history'];session.batch_trace=result['batch_trace']
    session.stopper.__dict__.update(result['stopper']);session.model.eval();return session

def run_fit(data,scope,branch,seed,unit,query=None,device='cuda',outer_epochs=None,permit=None,synthetic_cap=None,interrupt_after=None):
    if permit is not None:permit.validate()
    require_execution(data,'run_fit',permit);unit=Path(unit)
    s=TrainingSession(data,scope,branch,seed,query,device=device,outer_inner_epochs=outer_epochs,permit=permit)
    if synthetic_cap is not None:
        if data.kind!='SYNTHETIC':raise ScopeError('synthetic budget cannot alter real training')
        s.max_epochs=min(s.max_epochs,synthetic_cap);s.binding['synthetic_cap']=synthetic_cap
    with exclusive(unit.parent/'locks'/(unit.name+'.lock')):
        if (unit/'COMPLETE.json').exists():
            final=load_final(unit,s.binding);finish(unit,final);return restore_final(s,final)
        if (unit/'latest.json').exists():s.resume(unit)
        else:s.save(unit)
        start=time.perf_counter();timings=[];last_epoch=s.epoch
        try:
            while not s.done:
                if device.startswith('cuda'):torch.cuda.synchronize()
                tick=time.perf_counter();s.step()
                if device.startswith('cuda'):torch.cuda.synchronize()
                timings.append(time.perf_counter()-tick)
                if s.epoch!=last_epoch:
                    s.save(unit);last_epoch=s.epoch
                    if interrupt_after is not None and s.epoch>=interrupt_after:raise InterruptedError('synthetic fit interruption after epoch commit')
            s.select_best()
            final={'binding':s.binding,'done':True,'model':copy.deepcopy(s.best_state),'selected_epoch':s.max_epochs if s.outer else s.stopper.best_epoch,
                'completed_epoch':s.epoch,'updates':s.updates,'normalizer':s.normalizer.state(),'center_value':s.center.value,
                'center_source_ids':s.center.source_ids,'other_records':s.other.records,'history':s.history,'batch_trace':s.batch_trace,
                'stopper':copy.deepcopy(s.stopper.__dict__),'model_digest':tensor_digest(s.best_state),
                'query_artifact':dataclasses.asdict(s.query_artifact()) if s.stage=='A' else None}
            finish(unit,final)
            atomic_json(unit/'TIMING.json',{'fit_wall_seconds':time.perf_counter()-start,'step_seconds':timings,'epochs':s.epoch,'updates':s.updates,
                'synthetic_cap':synthetic_cap,'final_bytes':(unit/'FINAL.pt').stat().st_size,
                'gpu_allocated_peak':torch.cuda.max_memory_allocated() if device.startswith('cuda') else None,
                'gpu_reserved_peak':torch.cuda.max_memory_reserved() if device.startswith('cuda') else None})
            return s
        except BaseException as e:
            atomic_json(unit/('FAILURE_'+str(time.time_ns())+'.json'),{'type':type(e).__name__,'message':str(e),'epoch':s.epoch,'binding':s.binding});raise

def run_group(data,scopes,outer,seed,output,device='cpu',permit=None,synthetic_cap=None,branches=('M0','M2','C1-0')):
    require_execution(data,'group execution',permit);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with exclusive(output/'GROUP.lock'):
        settings={'branches':list(branches),'data_hash':data.fingerprint,'outer':outer,'seed':seed}
        settings_path=output/'GROUP_SETTINGS.json'
        if settings_path.exists() and json.loads(settings_path.read_text())!=settings:
            raise ScopeError('output directory belongs to another experiment configuration')
        atomic_json(settings_path,settings)
        results={};best={k:[] for k in branches};digests={}
        targets=[next(s for s in scopes if s.outer_fold==outer and s.inner_fold==i) for i in [0,1,2,3,None]]
        for sc in targets:
            artifacts={}
            for branch in best:
                family='M1' if branch in ['M3','C1-1'] else 'M0'
                unit=output/(f'inner{sc.inner_fold}' if sc.inner_fold is not None else 'refit')/branch
                s=run_fit(data,sc,branch,seed,unit,artifacts.get(family) if branch not in ['M0','M1'] else None,
                    device,best[branch] if sc.inner_fold is None else None,permit,synthetic_cap)
                if sc.inner_fold is not None:best[branch].append(s.stopper.best_epoch)
                if branch in ['M0','M1']:artifacts[branch]=s.query_artifact()
                digests[str(unit.relative_to(output))]=tensor_digest(s.best_state)
                if sc.inner_fold is None:
                    # Raw outer scores are sealed; no real-label metric evaluation here.
                    scores=s.predictions(sc.test_ids)
                    atomic_json(output/(branch+'_SEALED.json'),scores)
                    if s.stage=='B':
                        from .trainer import common_counterfactual_roles
                        common,_,coverage=common_counterfactual_roles(s,sc.test_ids)
                        controls={mode:s.predictions(sc.test_ids,mode,common,evaluation_kind='COUNTERFACTUAL') for mode in ['SELF','NULL','OTHER']}
                        atomic_json(output/(branch+'_FIXED_CONTROLS_SEALED.json'),{'coverage':coverage,'predictions':controls})
                        if data.dataset in ['DOLOS','SYN_DOLOS']:
                            subset,source,coverage=common_counterfactual_roles(s,sc.test_ids,True)
                            controls={mode:s.predictions(sc.test_ids,mode,subset,source if mode=='SOURCE_OTHER' else None,evaluation_kind='COUNTERFACTUAL') for mode in ['SELF','NULL','OTHER','SOURCE_OTHER']} if subset else {}
                            atomic_json(output/(branch+'_SOURCE_CONTROLS_SEALED.json'),{'coverage':coverage,'predictions':controls})
                del s
        from .storage import sha
        atomic_json(output/'GROUP_COMPLETE.json',{'dataset':data.dataset,'outer':outer,'seed':seed,'fits':5*len(branches),'branches':list(branches),'digests':digests,
            'sealed_files_sha256':{p.name:sha(p) for p in output.glob('*_SEALED.json')},'real_metrics_unsealed':False})
        return digests
