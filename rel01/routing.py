"""Explicit experiment identity; no network, loss, seed or role changes."""
from .config import ScopeError,canonical_hash

PRIMARY_CONTROLS={'M0':'QUERY_ONLY','M1':'QUERY_ONLY','M2':'SELF','M3':'SELF','C1-0':'OTHER','C1-1':'OTHER'}
COUNTERFACTUAL_CONTROLS=frozenset({'SELF','NULL','OTHER','SOURCE_OTHER'})
ROUTING_POLICY={'version':'REL01_PRIMARY_CONTROL_V1','primary_controls':PRIMARY_CONTROLS,
    'counterfactual_requires_explicit_kind':True,'require_single_experiment_per_collection':True}

def routing_hash():return canonical_hash(ROUTING_POLICY)

def resolve_control(model,mode=None,evaluation_kind='PRIMARY'):
    if model not in PRIMARY_CONTROLS:raise ScopeError('unknown model identifier')
    if evaluation_kind=='PRIMARY':
        expected=PRIMARY_CONTROLS[model]
        if mode is not None and mode!=expected:raise ScopeError(f'{model} PRIMARY requires {expected}, received {mode}')
        return expected
    if evaluation_kind!='COUNTERFACTUAL':raise ScopeError('unknown evaluation kind')
    if model in ['M0','M1'] or mode not in COUNTERFACTUAL_CONTROLS:raise ScopeError('counterfactual requires a relation model and an explicit valid mode')
    return mode

def validate_identity(row):
    for k in ['model','control','evaluation_kind']:
        if k not in row:raise ScopeError(f'missing experiment identity field: {k}')
    resolve_control(row['model'],row['control'],row['evaluation_kind'])
    return row['model'],row['control'],row['evaluation_kind']
