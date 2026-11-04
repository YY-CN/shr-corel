from dataclasses import dataclass
import numpy as np
from .config import SPEC, ScopeError, ImplementationBlocked, canonical_hash
from .data import query_weights

@dataclass
class PsyNormalizer:
    mean:np.ndarray
    scale:np.ndarray
    available:np.ndarray
    status:tuple
    fit_ids:tuple
    scope_hash:str
    data_hash:str
    @classmethod
    def fit(cls,data,scope,fit_ids=None):
        scope.check(data)
        if fit_ids is not None and set(fit_ids)!=set(scope.fit_ids):raise ScopeError('Normalizer attempted to fit unauthorized identities')
        w=query_weights(data,scope);mu=np.zeros(36,np.float64);scale=np.ones(36,np.float64);avail=np.zeros(36,bool);status=[]
        for j in range(36):
            good=(w>0)&data.mask[:,j]
            if not good.any():status.append('ALL_MISSING');continue
            ww=w[good]/w[good].sum();xx=data.psy[good,j].astype(np.float64)
            mu[j]=np.sum(ww*xx);var=np.sum(ww*(xx-mu[j])**2);avail[j]=True
            if not np.isfinite(var) or var<0:raise ImplementationBlocked('invalid training variance')
            if var<SPEC.variance_floor:status.append('NEAR_CONSTANT')
            else:scale[j]=np.sqrt(var);status.append('AVAILABLE')
        return cls(mu,scale,avail,tuple(status),scope.fit_ids,scope.fingerprint,data.fingerprint)
    def transform(self,data,scope):
        if scope.fingerprint!=self.scope_hash or data.fingerprint!=self.data_hash:raise ScopeError('Normalizer binding mismatch')
        effective=data.mask&self.available[None,:]
        z=np.zeros(data.psy.shape,np.float64)
        rows,cols=np.nonzero(effective)
        z[rows,cols]=(data.psy[rows,cols]-self.mean[cols])/self.scale[cols]
        if not np.isfinite(z).all():raise ImplementationBlocked('nonfinite normalized Psy')
        x=np.concatenate([z,effective.astype(np.float64)],axis=1).astype(np.float32)
        if not np.isfinite(x).all():raise ImplementationBlocked('float32 Psy overflow')
        return x
    def state(self):return {'mean':self.mean.tolist(),'scale':self.scale.tolist(),'available':self.available.tolist(),'status':self.status,'fit_ids':self.fit_ids,'scope_hash':self.scope_hash,'data_hash':self.data_hash}

@dataclass
class Center:
    value:np.ndarray
    source_ids:tuple
    fit_ids:tuple
    scope_hash:str
    data_hash:str
    norm_before:float
    @classmethod
    def fit(cls,data,scope,fit_ids=None):
        scope.check(data)
        if fit_ids is not None and set(fit_ids)!=set(scope.fit_ids):raise ScopeError('center attempted to use evaluation identities')
        means=[];sources=[]
        for p in scope.fit_ids:
            ii=scope.indices(data,[p]);tt=ii[data.labels[ii]==0]
            if not len(tt):raise ScopeError('no truthful answer for center donor')
            means.append(data.cls[tt].astype(np.float64).mean(axis=0));sources.extend(data.ids[i] for i in tt)
        raw=np.stack(means).mean(axis=0);den=float(np.linalg.norm(raw))
        if not np.isfinite(den) or den<=SPEC.center_min_norm:raise ImplementationBlocked('training center has too small norm')
        value=(raw/den).astype(np.float32)
        if abs(np.linalg.norm(value.astype(float))-1)>SPEC.norm_tolerance:raise ImplementationBlocked('invalid normalized center')
        return cls(value,tuple(sorted(sources)),scope.fit_ids,scope.fingerprint,data.fingerprint,den)
    @property
    def sha(self):return canonical_hash({'scope':self.scope_hash,'source_ids':self.source_ids,'value_float32_bytes':self.value.tobytes().hex()})

