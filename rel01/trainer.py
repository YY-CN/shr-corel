"""Contract-faithful two-stage framework, hard gated to synthetic data for this round."""
import copy,hashlib,math,random
from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import functional as F
from .config import SPEC,ScopeError,ImplementationBlocked,canonical_hash,specification_hash,contract_sha,derived_seed,deterministic,code_fingerprint,runtime_binding
from .data import require_synthetic,query_weights,roles_for,role_weights
from .preprocessing import PsyNormalizer,Center
from .other import build_other
from .models import make_query,AnchoredModel
from .sampling import HierarchicalSampler
from .checkpoint import save_transaction,load_transaction
from .routing import resolve_control
from .authorization import require_execution

def tensor_digest(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

@dataclass
class QueryArtifact:
    use_psy:bool
    seed:int
    scope_hash:str
    data_hash:str
    normalizer_hash:str
    state:dict
    state_sha:str
    provenance:dict

@dataclass
class EarlyStop:
    patience:int
    min_delta:float=SPEC.min_delta
    best:float=math.inf
    best_epoch:int|None=None
    bad_epochs:int=0
    def observe(self,value,epoch):
        if not math.isfinite(value):raise ImplementationBlocked('nonfinite validation BCE')
        improved=value<self.best-self.min_delta
        if improved:self.best=float(value);self.best_epoch=int(epoch);self.bad_epochs=0
        else:self.bad_epochs+=1
        return improved,self.bad_epochs>=self.patience

def refit_epochs(values,stage):
    if len(values)!=4 or any(int(x)!=x for x in values):raise ScopeError('exactly four integer inner epochs required')
    low,high=(1,100) if stage=='A' else (0,80)
    if not all(low<=x<=high for x in values):raise ScopeError('best epoch outside stage contract')
    return int(math.ceil(float(np.median(values))))

def optimizer_for(model,stage):
    params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    wd=[p for n,p in params if p.ndim==2 and n.endswith('weight')]
    no=[p for n,p in params if not(p.ndim==2 and n.endswith('weight'))]
    return torch.optim.AdamW([{'params':wd,'weight_decay':.001},{'params':no,'weight_decay':0.}],
        lr=SPEC.query_lr if stage=='A' else SPEC.relation_lr,betas=SPEC.adam_betas,eps=SPEC.adam_eps,amsgrad=False,foreach=False,fused=False)

class TrainingSession:
    def __init__(self,data,scope,branch,seed,query_artifact=None,device='cpu',outer_inner_epochs=None,permit=None):
        require_execution(data,'TrainingSession construction',permit);self.permit=permit
        if branch not in ['M0','M1','M2','M3','C1-0','C1-1'] or seed not in SPEC.seeds:raise ScopeError('unregistered branch or training seed')
        scope.check(data);self.data=data;self.scope=scope;self.branch=branch;self.seed=seed;self.device=torch.device(device)
        deterministic(seed);self.stage='A' if branch in ['M0','M1'] else 'B';self.use_psy=branch in ['M1','M3','C1-1']
        self.normalizer=PsyNormalizer.fit(data,scope);self.center=Center.fit(data,scope)
        self.psy=torch.from_numpy(self.normalizer.transform(data,scope)).to(self.device)
        self.emb=torch.tensor(data.cls.copy(),device=self.device);self.labels=torch.tensor(data.labels.copy(),dtype=torch.float32,device=self.device)
        self.weights=torch.tensor(query_weights(data,scope),dtype=torch.float32,device=self.device)
        self.fit_index=scope.indices(data);self.other=build_other(data,scope)
        query=make_query(self.use_psy,seed)
        if self.stage=='A':self.model=query
        else:
            a=query_artifact
            expected_provenance={'contract_sha':contract_sha(),'spec_sha':specification_hash(),'code_sha':code_fingerprint(),'runtime':runtime_binding(),'completed':True}
            if a is None or a.use_psy!=self.use_psy or a.seed!=seed or a.scope_hash!=scope.fingerprint or a.data_hash!=data.fingerprint or a.normalizer_hash!=canonical_hash(self.normalizer.state()) or tensor_digest(a.state)!=a.state_sha or any(a.provenance.get(k)!=v for k,v in expected_provenance.items()):raise ScopeError('Query checkpoint does not belong to this completed fit scope/seed/family/code/runtime')
            query.load_state_dict(a.state);self.model=AnchoredModel(query,self.center.value,seed)
        self.model=self.model.to(self.device);self.optimizer=optimizer_for(self.model,self.stage)
        self.sampler=HierarchicalSampler(data,scope,seed) if self.stage=='B' else None
        self.shuffle=np.random.default_rng(derived_seed(seed,scope.key,'QUERY_SHUFFLE'))
        self.steps_epoch=math.ceil(len(self.fit_index)/32) if self.stage=='A' else self.sampler.steps_per_epoch
        self.max_epochs=100 if self.stage=='A' else 80
        self.outer=scope.inner_fold is None
        if self.outer:
            if outer_inner_epochs is None:raise ScopeError('outer fixed epoch must derive from four inner epochs')
            self.max_epochs=refit_epochs(outer_inner_epochs,self.stage)
        elif outer_inner_epochs is not None:raise ScopeError('inner fit cannot use outer fixed epochs')
        self.stopper=EarlyStop(15 if self.stage=='A' else 12)
        self.epoch=0;self.cursor=0;self.updates=0;self.permutation=None;self.done=False;self.history=[];self.batch_trace=[]
        self.best_state=None;self.latest_loss=None
        if self.stage=='B':
            if not self.outer:
                diagnostics=self.diagnostics(scope.validation_ids);self.stopper.observe(diagnostics['BCE'],0)
                self.best_state=cpu_state(self.model);self.history.append({'epoch':0,'validation':diagnostics,'training':self.diagnostics(scope.fit_ids),'best':True})
            else:
                self.history.append({'epoch':0,'relation_updates':0,'outer_fixed_epochs':self.max_epochs,'training':self.diagnostics(scope.fit_ids)})
            if self.max_epochs==0:self.done=True;self.best_state=cpu_state(self.model)
        self.binding={'contract_sha':contract_sha(),'spec_sha':specification_hash(),'data_hash':data.fingerprint,'scope_hash':scope.fingerprint,
            'scope_key':scope.key,'branch':branch,'seed':seed,'stage':self.stage,'center_sha':self.center.sha,
            'normalizer_hash':canonical_hash(self.normalizer.state()),'other_sha':self.other.sha,
            'query_artifact_sha':query_artifact.state_sha if query_artifact is not None else None,
            'query_artifact_provenance_sha':canonical_hash(query_artifact.provenance) if query_artifact is not None else None,'max_epochs':self.max_epochs,'device':str(self.device),
            'code_sha':code_fingerprint(),'runtime':runtime_binding()}
    def _reference_indices(self,roles,mode):
        if mode=='SELF':return [r.reference for r in roles]
        if mode=='OTHER':
            ii=[self.other.resolve(self.data,self.scope,r) for r in roles]
            if any(i is None for i in ii):raise ScopeError('UNMATCHED main OTHER: explicit common-set contract required')
            return ii
        raise ScopeError('unknown reference mode')
    def step(self):
        require_execution(self.data,'optimizer step',self.permit)
        if self.done:return None
        self.model.train();self.optimizer.zero_grad(set_to_none=True)
        if self.stage=='A':
            if self.permutation is None:self.permutation=self.shuffle.permutation(self.fit_index)
            ii=self.permutation[self.cursor*32:(self.cursor+1)*32];ix=torch.as_tensor(ii,device=self.device)
            z=self.model(self.emb[ix],self.psy[ix]);loss=len(self.fit_index)/len(ii)*torch.sum(self.weights[ix]*F.binary_cross_entropy_with_logits(z,self.labels[ix],reduction='none'))
            trace=[self.data.ids[int(i)] for i in ii]
        else:
            roles=self.sampler.sample();q=[r.query for r in roles];rr=self._reference_indices(roles,'OTHER' if self.branch.startswith('C1') else 'SELF')
            z=self.model(self.emb[q],self.psy[q],self.emb[rr]);loss=F.binary_cross_entropy_with_logits(z,self.labels[q])
            trace=[r.key(self.data,self.scope) for r in roles]
        if not bool(torch.isfinite(loss)):raise ImplementationBlocked('nonfinite optimization loss')
        loss.backward()
        grads=[p.grad for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        if any(not bool(torch.isfinite(g).all()) for g in grads):raise ImplementationBlocked('nonfinite gradient')
        torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        self.optimizer.step();self.latest_loss=float(loss.detach());self.updates+=1;self.cursor+=1
        self.batch_trace.append(canonical_hash(trace))
        if self.cursor==self.steps_epoch:
            self.epoch+=1;self.cursor=0;self.permutation=None
            if self.outer:
                self.history.append({'epoch':self.epoch,'train_loss_last_batch':self.latest_loss,'training':self.diagnostics(self.scope.fit_ids)})
                self.best_state=cpu_state(self.model);self.done=self.epoch>=self.max_epochs
            else:
                val=self.diagnostics(self.scope.validation_ids);train=self.diagnostics(self.scope.fit_ids)
                better,stop=self.stopper.observe(val['BCE'],self.epoch)
                if better:self.best_state=cpu_state(self.model)
                self.history.append({'epoch':self.epoch,'validation':val,'training':train,'best':better})
                self.done=stop or self.epoch>=self.max_epochs
        return self.latest_loss
    def run(self,max_updates=None):
        count=0
        while not self.done and (max_updates is None or count<max_updates):self.step();count+=1
        return {'done':self.done,'updates':self.updates,'epoch':self.epoch,'best_epoch':self.stopper.best_epoch if not self.outer else self.max_epochs}
    def diagnostics(self,people):
        require_execution(self.data,'diagnostics',self.permit);self.model.eval()
        roles,w=role_weights(self.data,self.scope,people);q=[r.query for r in roles]
        with torch.no_grad():
            if self.stage=='A':base=self.model(self.emb[q],self.psy[q]);z=base;res=torch.zeros_like(z)
            else:
                rr=self._reference_indices(roles,'OTHER' if self.branch.startswith('C1') else 'SELF')
                v=self.model.decomposition(self.emb[q],self.psy[q],self.emb[rr]);z=v['logit'];base=v['base'];res=v['residual']
            loss=F.binary_cross_entropy_with_logits(z,self.labels[q],reduction='none').double().cpu().numpy()
            bl=F.binary_cross_entropy_with_logits(base,self.labels[q],reduction='none').double().cpu().numpy()
        zz=z.cpu().numpy();rr=res.cpu().numpy()
        return {'BCE':float(loss@w),'base_BCE':float(bl@w),'logit_p10_p50_p90':np.quantile(zz,[.1,.5,.9]).tolist(),
            'residual_p10_p50_p90':np.quantile(rr,[.1,.5,.9]).tolist(),'residual_weighted_mean':float(rr@w),'role_count':len(roles)}
    def query_artifact(self):
        if self.stage!='A' or self.best_state is None or not self.done:raise ScopeError('only a completed stage A can provide a selected Query checkpoint')
        state=copy.deepcopy(self.best_state)
        provenance={'contract_sha':contract_sha(),'spec_sha':specification_hash(),'code_sha':code_fingerprint(),'runtime':runtime_binding(),'completed':True,
            'selected_epoch':self.max_epochs if self.outer else self.stopper.best_epoch}
        return QueryArtifact(self.use_psy,self.seed,self.scope.fingerprint,self.data.fingerprint,canonical_hash(self.normalizer.state()),state,tensor_digest(state),provenance)
    def select_best(self):
        if self.best_state is None or not self.done:raise ScopeError('finish the fit before selecting an inference checkpoint')
        self.model.load_state_dict(self.best_state);self.model.eval()
    def predictions(self,people,mode=None,roles=None,other_map=None,*,evaluation_kind='PRIMARY'):
        require_execution(self.data,'model predictions',self.permit)
        mode=resolve_control(self.branch,mode,evaluation_kind)
        if not self.done:raise ScopeError('complete the fit before model prediction; use diagnostics for ongoing validation')
        self.select_best();self.model.eval()
        roles=roles_for(self.data,self.scope,people) if roles is None else list(roles)
        allowed={r.key(self.data,self.scope) for r in roles_for(self.data,self.scope,people)}
        keys=[r.key(self.data,self.scope) for r in roles]
        if not keys or len(set(keys))!=len(keys) or any(k not in allowed for k in keys):raise ScopeError('empty, duplicated or out-of-scope prediction role collection')
        if other_map is not None and mode!='SOURCE_OTHER':raise ScopeError('main OTHER must use the bound fit-scope map')
        if mode=='SOURCE_OTHER' and (other_map is None or other_map.mode!='SOURCE_LENGTH'):raise ScopeError('SOURCE_OTHER requires an explicit strict source map')
        rr=[]
        if self.stage=='B':
            for role in roles:
                if mode=='SELF':rr.append(role.reference)
                elif mode in ['OTHER','SOURCE_OTHER']:
                    mapping=other_map or self.other;o=mapping.resolve(self.data,self.scope,role)
                    if o is None:raise ScopeError('filter common counterfactual roles before inference')
                    rr.append(o)
                elif mode!='NULL':raise ScopeError('unknown inference control')
        q=[r.query for r in roles]
        with torch.no_grad():
            if self.stage=='A':z=self.model(self.emb[q],self.psy[q])
            else:
                r=self.model.center.reshape(1,768).expand(len(q),768) if mode=='NULL' else self.emb[rr]
                z=self.model(self.emb[q],self.psy[q],r)
        other_sha=(other_map or self.other).sha if mode in ['OTHER','SOURCE_OTHER'] else None
        return [{'domain':self.data.kind,'dataset':self.data.dataset,'scope':self.scope.key,'outer_fold':self.scope.outer_fold,
            'participant_id':r.participant,'reference_id':self.data.ids[r.reference],'query_id':self.data.ids[r.query],
            'seed':self.seed,'label':int(self.data.labels[r.query]),'logit':float(value),'control':mode,'model':self.branch,
            'evaluation_kind':evaluation_kind,
            'reference_input_id':None if mode=='QUERY_ONLY' else ('CENTER:'+self.center.sha if mode=='NULL' else self.data.ids[rr[j]]),
            'other_map_sha':other_sha}
            for j,(r,value) in enumerate(zip(roles,z.detach().cpu().numpy()))]
    def save(self,directory,fail_at=None):
        require_execution(self.data,'checkpoint write',self.permit)
        state={'binding':self.binding,'model':cpu_state(self.model),'optimizer':self.optimizer.state_dict(),
            'epoch':self.epoch,'cursor':self.cursor,'updates':self.updates,'permutation':self.permutation,
            'shuffle':copy.deepcopy(self.shuffle.bit_generator.state),'sampler':self.sampler.state() if self.sampler else None,
            'stopper':copy.deepcopy(self.stopper.__dict__),'done':self.done,'history':self.history,'batch_trace':self.batch_trace,
            'best_state':self.best_state,'latest_loss':self.latest_loss,
            'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
            'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'normalizer':self.normalizer.state(),'center_source_ids':self.center.source_ids,'center_value':self.center.value,
            'other_records':self.other.records}
        return save_transaction(directory,state,fail_at)
    def resume(self,directory):
        require_execution(self.data,'checkpoint resume',self.permit);s=load_transaction(directory,'cpu')
        if s['binding']!=self.binding:raise ScopeError('checkpoint scope/data/contract/seed/family/center/OTHER mismatch')
        self.model.load_state_dict(s['model']);self.optimizer.load_state_dict(s['optimizer'])
        for k in ['epoch','cursor','updates','permutation','done','history','batch_trace','best_state','latest_loss']:setattr(self,k,s[k])
        self.shuffle.bit_generator.state=s['shuffle']
        if self.sampler:self.sampler.load_state(s['sampler'])
        self.stopper.__dict__.update(s['stopper']);torch.set_rng_state(s['torch_rng'].cpu());np.random.set_state(s['numpy_rng']);random.setstate(s['python_rng'])
        if s['cuda_rng'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(s['cuda_rng'])
        self.model.train(not self.done);return self

def common_counterfactual_roles(session,people,include_source=False):
    require_execution(session.data,'counterfactual inference routing',session.permit)
    roles=roles_for(session.data,session.scope,people)
    source=build_other(session.data,session.scope,people,same_source=True,length_cap=.25) if include_source else None
    common=[r for r in roles if session.other.resolve(session.data,session.scope,r) is not None and (source is None or source.resolve(session.data,session.scope,r) is not None)]
    return common,source,{'candidate_roles':len(roles),'common_roles':len(common),'unmatched_roles':len(roles)-len(common)}

