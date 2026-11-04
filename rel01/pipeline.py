from .config import SPEC, ScopeError
from .data import require_synthetic, roles_for
from .trainer import TrainingSession
from .routing import PRIMARY_CONTROLS
from .evaluation import require_common_queries
BRANCHES=('M0','M1','M2','M3','C1-0','C1-1')

def run_synthetic_six(data,scope,seed,outer_epoch_choices=None):
    require_synthetic(data,'six-branch engineering integration');artifacts={};sessions={}
    for branch in BRANCHES:
        qfamily='M1' if branch in ['M3','C1-1'] else 'M0'
        query=artifacts[qfamily] if branch not in ['M0','M1'] else None
        epochs=None if scope.inner_fold is not None else (outer_epoch_choices or {}).get(branch)
        session=TrainingSession(data,scope,branch,seed,query,outer_inner_epochs=epochs);session.run();session.select_best();sessions[branch]=session
        if branch in ['M0','M1']:artifacts[branch]=session.query_artifact()
    return sessions

def primary_predictions(sessions,people):
    """One explicit common original-query set; C1 supplies OTHER as main result."""
    if set(sessions)!=set(BRANCHES):raise ScopeError('all six sessions required')
    anchor=sessions['M0']
    for model,s in sessions.items():
        if s.branch!=model or s.scope.fingerprint!=anchor.scope.fingerprint or s.data.fingerprint!=anchor.data.fingerprint or s.seed!=anchor.seed:raise ScopeError('model/session scope or seed mismatch')
    roles=roles_for(anchor.data,anchor.scope,people)
    common=[r for r in roles if all(sessions[k].other.resolve(anchor.data,anchor.scope,r) is not None for k in ['C1-0','C1-1'])]
    if not common:raise ScopeError('no common legal main-evaluation queries')
    if sessions['C1-0'].other.sha!=sessions['C1-1'].other.sha:raise ScopeError('C1 branches must share the same OTHER map')
    predictions={k:s.predictions(people,mode=PRIMARY_CONTROLS[k],roles=common,evaluation_kind='PRIMARY') for k,s in sessions.items()}
    require_common_queries(*predictions.values())
    return {'predictions':predictions,'candidate_roles':len(roles),'common_roles':len(common),'unmatched_roles':len(roles)-len(common),
        'primary_controls':dict(PRIMARY_CONTROLS),'evaluation_kind':'PRIMARY'}

