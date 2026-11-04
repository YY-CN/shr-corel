"""Independent NumPy score evaluator. No model, optimizer or feature fitting."""
from collections import defaultdict
import numpy as np
from .config import SPEC, ScopeError, RealDataForbidden
from .routing import PRIMARY_CONTROLS,validate_identity

METRICS=('AP','AUROC','BCE','ACC','Balanced_ACC','F1','Precision','Recall')
def binary_metrics(y,score):
    y=np.asarray(y,dtype=np.int64);s=np.asarray(score,dtype=np.float64)
    if y.shape!=s.shape or not len(y) or not np.isfinite(s).all() or not set(y)<= {0,1}:raise ScopeError('invalid evaluation input')
    pos=int(y.sum());neg=len(y)-pos;pred=s>=0
    tp=int(np.sum(pred&(y==1)));fp=int(np.sum(pred&(y==0)));fn=pos-tp;tn=neg-fp
    precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/pos if pos else np.nan
    ap=np.nan
    if pos:
        order=np.argsort(-s,kind='stable');ys=y[order];ss=s[order]
        ends=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(y)-1]
        cumulative=np.cumsum(ys)[ends];new_pos=np.diff(np.r_[0,cumulative])
        ap=float(np.sum(new_pos/pos*cumulative/(ends+1)))
    auc=float(np.mean((s[y==1,None]>s[y==0][None,:])+.5*(s[y==1,None]==s[y==0][None,:]))) if pos and neg else np.nan
    return {'AP':ap,'AUROC':auc,'BCE':float(np.mean(np.logaddexp(0,s)-y*s)),
        'ACC':(tp+tn)/len(y),'Balanced_ACC':.5*(tp/pos+tn/neg) if pos and neg else np.nan,
        'F1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,'Precision':precision,'Recall':recall}

def checked_rows(rows):
    rows=list(rows);keys=[];experiments=set()
    if not rows:raise ScopeError('empty prediction collection')
    for r in rows:
        if r.get('domain') not in ['REAL_AUDIT_ONLY','SYNTHETIC']:raise ScopeError('unknown prediction provenance')
        experiments.add(validate_identity(r))
        keys.append((r['dataset'],r['participant_id'],r['reference_id'],int(r['seed']),r['query_id']))
        if r['query_id']==r['reference_id']:raise ScopeError('self query in evaluator')
    if len(set(keys))!=len(keys):raise ScopeError('duplicated prediction role')
    if len(experiments)!=1:raise ScopeError('mixed models, modes or evaluation kinds in one prediction collection')
    return rows

def require_common_queries(*collections):
    checked=[checked_rows(rows) for rows in collections]
    key=lambda r:(r['dataset'],r['participant_id'],r['reference_id'],r['query_id'],int(r['seed']))
    baseline={key(r):(int(r['label']),r.get('outer_fold'),r.get('scope')) for r in checked[0]}
    for rows in checked[1:]:
        if {key(r):(int(r['label']),r.get('outer_fold'),r.get('scope')) for r in rows}!=baseline:raise ScopeError('common p/r/q/seed, labels or fold/scope mismatch')
    return checked

def hierarchy(rows,require_all_seeds=True):
    rows=checked_rows(rows)
    datasets={r['dataset'] for r in rows}
    if len(datasets)!=1:raise ScopeError('datasets must be evaluated separately')
    groups=defaultdict(list)
    for r in rows:groups[(r['participant_id'],r['reference_id'],r['seed'])].append(r)
    seed_roles=defaultdict(lambda:defaultdict(set))
    for r in rows:seed_roles[r['participant_id']][r['seed']].add((r['reference_id'],r['query_id'],int(r['label'])))
    if any(any(g!=next(iter(v.values())) for g in v.values()) for v in seed_roles.values()):raise ScopeError('query/reference/label collection differs between seeds')
    condition=[];per_seed=defaultdict(list)
    for (p,ref,seed),g in sorted(groups.items()):
        m=binary_metrics([x['label'] for x in g],[x['logit'] for x in g]);condition.append({'participant_id':p,'reference_id':ref,'seed':seed,'queries':len(g),**m});per_seed[(p,seed)].append(m)
    per_person=defaultdict(list);seed_counts=defaultdict(set)
    avg=lambda vals:float(np.mean([x for x in vals if np.isfinite(x)])) if any(np.isfinite(x) for x in vals) else np.nan
    for (p,seed),g in per_seed.items():
        seed_counts[p].add(int(seed));per_person[p].append({k:avg([m[k] for m in g]) for k in METRICS})
    if require_all_seeds and any(s!=set(SPEC.seeds) for s in seed_counts.values()):raise ScopeError('missing or extra seed; no implicit ensemble')
    people=[{'participant_id':p,**{k:avg([m[k] for m in g]) for k in METRICS}} for p,g in sorted(per_person.items())]
    macro={k:avg([r[k] for r in people]) for k in METRICS}
    return {'conditions':condition,'participants':people,'macro':macro,'effective_participants':{k:sum(np.isfinite(r[k]) for r in people) for k in METRICS},'seeds_are_independent_units':False}

def paired_bootstrap(left,right,seed=SPEC.bootstrap_seed):
    a,b=require_common_queries(left,right)
    key=lambda r:(r['dataset'],r['participant_id'],r['reference_id'],r['query_id'],int(r['seed']))
    aa={key(r):r for r in a};bb={key(r):r for r in b}
    if set(aa)!=set(bb) or any(aa[k]['label']!=bb[k]['label'] for k in aa):raise ScopeError('paired models must have identical p/r/q/seed/labels')
    pa=hierarchy(a)['participants'];pb=hierarchy(b)['participants'];assert [r['participant_id'] for r in pa]==[r['participant_id'] for r in pb]
    rng=np.random.default_rng(seed);indices=rng.integers(0,len(pa),size=(5000,len(pa)))
    out={}
    for metric in METRICS:
        x=np.array([r[metric] for r in pa])-np.array([r[metric] for r in pb])
        if not np.isfinite(x).all():
            out[metric]={'status':'NOT_ESTIMABLE','participants':int(np.isfinite(x).sum())};continue
        low,high=np.percentile(x[indices].mean(axis=1),[2.5,97.5])
        out[metric]={'difference':float(x.mean()),'ci_low':float(low),'ci_high':float(high),'participants':len(x),'status':'DESCRIPTIVE'}
    return {'metrics':out,'bootstrap_replicates':5000,'seed':seed,'multiplicity_adjusted':False,'unit':'participant retaining all references/seeds'}

def margin_diagnostics(left,right):
    a,b=require_common_queries(left,right)
    if len({r['dataset'] for r in a+b})!=1:raise ScopeError('counterfactual datasets must match')
    key=lambda r:(r['participant_id'],r['reference_id'],r['seed'],r['query_id'])
    aa={key(r):r for r in a};bb={key(r):r for r in b}
    if set(aa)!=set(bb):raise ScopeError('counterfactual common query set mismatch')
    group=defaultdict(list)
    for k in aa:
        if aa[k]['label']!=bb[k]['label']:raise ScopeError('counterfactual label mismatch')
        group[k[:3]].append(k)
    result=[]
    for (p,r,seed),keys in sorted(group.items()):
        s=np.array([aa[k]['logit'] for k in keys]);o=np.array([bb[k]['logit'] for k in keys]);y=np.array([aa[k]['label'] for k in keys]);delta=s-o
        row={'participant_id':p,'reference_id':r,'seed':seed,'common_shift':float(delta.mean()),
             'common_shift_energy':float(delta.mean()**2),'centered_difference_energy':float(np.mean((delta-delta.mean())**2)),'total_difference_energy':float(np.mean(delta**2))}
        if not (y==0).any() or not (y==1).any():row['status']='NOT_ESTIMABLE'
        else:
            sm=s[y==1,None]-s[y==0][None,:];om=o[y==1,None]-o[y==0][None,:]
            row.update({'status':'DESCRIPTIVE','pairs':int(sm.size),'self_margin':float(sm.mean()),'other_margin':float(om.mean()),
                'rescue':float(((sm>0)&(om<0)).mean()),'damage':float(((sm<0)&(om>0)).mean()),
                'strict_flip':float((np.sign(sm)*np.sign(om)<0).mean()),'self_win':float(((sm>0)+.5*(sm==0)).mean()),
                'other_win':float(((om>0)+.5*(om==0)).mean()),'any_tie':float(((sm==0)|(om==0)).mean())})
        result.append(row)
    return result

def mu3d_condition_subset(rows,conditions,group):
    rules={'A_SAME_QUERY_VALENCE':{'PT':('NT','NL'),'NT':('PT','PL')},'B_SAME_QUERY_REFERENT':{'PT':('NT','PL'),'NT':('PT','NL')}}
    if group not in rules:raise ScopeError('unknown MU3D condition group')
    out=[]
    for row in checked_rows(rows):
        rc=conditions[row['reference_id']];qc=conditions[row['query_id']]
        if rc not in rules[group]:raise ScopeError('reference is not a truthful MU3D condition')
        if qc in rules[group][rc]:out.append(row)
    return out

def near_tie_audit(rows,atol=1e-7,rtol=1e-7):
    """Descriptive numerical sensitivity only; never modifies scores or metric ties."""
    groups=defaultdict(list)
    for r in checked_rows(rows):groups[(r['dataset'],r['participant_id'],r['reference_id'],r['seed'])].append(r['logit'])
    out=[]
    for key,values in sorted(groups.items()):
        s=np.asarray(values,float);i,j=np.triu_indices(len(s),1);gap=abs(s[i]-s[j]);limit=atol+rtol*np.maximum(abs(s[i]),abs(s[j]))
        out.append({'dataset':key[0],'participant_id':key[1],'reference_id':key[2],'seed':key[3],
            'exact_tie_pairs':int(np.sum(gap==0)),'near_nonexact_pairs':int(np.sum((gap>0)&(gap<=limit))),
            'atol':atol,'rtol':rtol,'scores_modified':False})
    return out

def planned_model_comparisons(predictions_by_model):
    if set(predictions_by_model)!=set(PRIMARY_CONTROLS):raise ScopeError('six primary model collections required')
    for model,rows in predictions_by_model.items():
        rows=checked_rows(rows)
        if any(r['model']!=model or r['control']!=PRIMARY_CONTROLS[model] or r['evaluation_kind']!='PRIMARY' for r in rows):raise ScopeError('primary model collection mislabeled or counterfactual passed as main result')
    require_common_queries(*predictions_by_model.values())
    pairs=[('M2','M0'),('M3','M1'),('M1','M0'),('M2','C1-0'),('M3','C1-1')]
    return {a+'-'+b:paired_bootstrap(predictions_by_model[a],predictions_by_model[b]) for a,b in pairs}

def counterfactual_summary(predictions_by_control):
    if not {'SELF','NULL','OTHER'}<=set(predictions_by_control):raise ScopeError('SELF/NULL/OTHER required')
    models=set()
    for mode,rows in predictions_by_control.items():
        rows=checked_rows(rows);models.update(r['model'] for r in rows)
        if any(r['control']!=mode or r['evaluation_kind']!='COUNTERFACTUAL' for r in rows):raise ScopeError('counterfactual mode key or explicit kind mismatch')
    if len(models)!=1:raise ScopeError('fixed-weight counterfactuals must belong to one model')
    require_common_queries(*predictions_by_control.values())
    absolute={k:hierarchy(v) for k,v in predictions_by_control.items()}
    comparisons={k:{'paired':paired_bootstrap(predictions_by_control['SELF'],v),
        'margin':margin_diagnostics(predictions_by_control['SELF'],v)} for k,v in predictions_by_control.items() if k!='SELF'}
    return {'absolute':absolute,'self_minus_control':comparisons,
        'near_ties':{k:near_tie_audit(v) for k,v in predictions_by_control.items()},'status':'DESCRIPTIVE'}

