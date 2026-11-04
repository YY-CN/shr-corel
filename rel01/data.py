from dataclasses import dataclass
import hashlib, math
from functools import cached_property
import numpy as np
from .config import SPEC, ScopeError, RealDataForbidden, ImplementationBlocked, canonical_hash

@dataclass(frozen=True)
class Answers:
    dataset:str
    ids:tuple
    participants:tuple
    labels:np.ndarray
    cls:np.ndarray
    psy:np.ndarray
    mask:np.ndarray
    reason:np.ndarray
    nlex:np.ndarray
    sources:tuple
    kind:str='REAL_AUDIT_ONLY'

    def __post_init__(self):
        n=len(self.ids)
        if len(set(self.ids))!=n or len(self.participants)!=n or len(self.sources)!=n:raise ScopeError('duplicate IDs or inconsistent metadata')
        expected={'labels':(n,),'cls':(n,768),'psy':(n,36),'mask':(n,36),'reason':(n,36),'nlex':(n,)}
        for k,shape in expected.items():
            a=np.array(getattr(self,k),copy=True)
            if a.shape!=shape:raise ScopeError(f'{k} wrong shape {a.shape}')
            a.setflags(write=False);object.__setattr__(self,k,a)
        if self.cls.dtype!=np.float32:raise ScopeError('CLS must retain original float32 representation')
        if self.mask.dtype!=bool or not set(np.unique(self.labels))<= {0,1}:raise ScopeError('bad masks or labels')
        if not np.isfinite(self.cls).all() or np.max(np.abs(np.linalg.norm(self.cls.astype(np.float64),axis=1)-1))>SPEC.norm_tolerance:raise ImplementationBlocked('CLS norms/finite gate failed; no silent normalization')
        if not np.isfinite(self.psy[self.mask]).all() or np.isinf(self.psy).any():raise ImplementationBlocked('invalid Psy numeric measurement')
        if not np.isfinite(self.nlex).all() or (self.nlex<0).any():raise ScopeError('invalid frozen Nlex')
        if self.kind not in ['SYNTHETIC','REAL_AUDIT_ONLY']:raise ScopeError('unknown provenance domain')
        if self.kind=='SYNTHETIC' and (not self.dataset.startswith('SYN_') or not all(x.startswith('SYN_') for x in self.ids)):raise RealDataForbidden('synthetic namespace required')

    @property
    def index(self):return {v:i for i,v in enumerate(self.ids)}
    @cached_property
    def fingerprint(self):
        h=hashlib.sha256(canonical_hash([self.dataset,self.kind,self.ids,self.participants,self.sources]).encode())
        for key in ['labels','cls','psy','mask','reason','nlex']:
            a=getattr(self,key);h.update(str(a.dtype).encode());h.update(a.tobytes())
        return h.hexdigest()

def require_synthetic(data,action):
    if not isinstance(data,Answers) or data.kind!='SYNTHETIC' or not data.dataset.startswith('SYN_'):raise RealDataForbidden(f'{action}: real supervised training and real model inference are not authorized')

@dataclass(frozen=True)
class Scope:
    dataset:str
    outer_fold:int
    inner_fold:int|None
    fit_ids:tuple
    validation_ids:tuple=()
    test_ids:tuple=()
    def __post_init__(self):
        for k in ['fit_ids','validation_ids','test_ids']:
            value=tuple(sorted(getattr(self,k)))
            if len(value)!=len(set(value)):raise ScopeError('duplicate identity in scope')
            object.__setattr__(self,k,value)
        a,b,c=map(set,(self.fit_ids,self.validation_ids,self.test_ids))
        if not a or a&b or a&c or b&c:raise ScopeError('fit/validation/test identity overlap or empty fit')
        if self.inner_fold is None and b:raise ScopeError('outer refit cannot have inner validation identities')
        if self.inner_fold is not None and not b:raise ScopeError('inner fit requires validation identities')
    @property
    def key(self):return f'{self.dataset}|outer={self.outer_fold}|inner={self.inner_fold if self.inner_fold is not None else "refit"}'
    @cached_property
    def fingerprint(self):return canonical_hash([self.key,self.fit_ids,self.validation_ids,self.test_ids])
    @property
    def evaluation_ids(self):return self.validation_ids if self.inner_fold is not None else self.test_ids
    def check(self,data):
        if data.dataset!=self.dataset or not (set(self.fit_ids)|set(self.validation_ids)|set(self.test_ids))<=set(data.participants):raise ScopeError('scope/data binding mismatch')
    def indices(self,data,people=None):
        self.check(data);people=self.fit_ids if people is None else tuple(people)
        return np.array([i for i,p in enumerate(data.participants) if p in set(people)],dtype=np.int64)

@dataclass(frozen=True)
class Role:
    participant:str
    reference:int
    query:int
    def key(self,data,scope):return (scope.key,self.participant,data.ids[self.reference],data.ids[self.query])

def roles_for(data,scope,people):
    scope.check(data);people=tuple(sorted(people))
    if not set(people)<=set(scope.fit_ids+scope.evaluation_ids):raise ScopeError('role targets outside permitted fit/evaluation scope')
    result=[]
    for p in people:
        ii=scope.indices(data,[p]);tt=ii[data.labels[ii]==0]
        if len(tt)<2 or not (data.labels[ii]==1).any():raise ScopeError('K1 requires >=2 truths and >=1 lie')
        for r in tt:
            for q in ii:
                if q!=r:result.append(Role(p,int(r),int(q)))
    return result

def query_weights(data,scope):
    scope.check(data);w=np.zeros(len(data.ids),np.float64)
    for p in scope.fit_ids:
        ii=scope.indices(data,[p]);n=len(ii);t=int((data.labels[ii]==0).sum())
        if t<2 or n<=t:raise ScopeError('invalid K1 target population')
        w[ii]=np.where(data.labels[ii]==0,(t-1)/(t*(n-1)),1/(n-1))/len(scope.fit_ids)
        if abs(w[ii].sum()-1/len(scope.fit_ids))>1e-12:raise ScopeError('person weight sum mismatch')
    if abs(w.sum()-1)>1e-12:raise ScopeError('scope weight sum mismatch')
    return w

def role_weights(data,scope,people):
    roles=roles_for(data,scope,people);out=[]
    for z in roles:
        ii=scope.indices(data,[z.participant]);t=int((data.labels[ii]==0).sum())
        out.append(1/(len(people)*t*(len(ii)-1)))
    return roles,np.array(out,np.float64)

