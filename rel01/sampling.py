import copy,math
import numpy as np
from .config import SPEC, ScopeError, derived_seed
from .data import Role

class HierarchicalSampler:
    def __init__(self,data,scope,seed):
        scope.check(data)
        if len(scope.fit_ids)<8:raise ScopeError('B sampler requires 8 distinct fit identities')
        self.data=data;self.scope=scope;self.people=list(scope.fit_ids)
        self.answers={p:scope.indices(data,[p]) for p in self.people}
        self.truth={p:ii[data.labels[ii]==0] for p,ii in self.answers.items()}
        if any(len(t)<2 or len(self.answers[p])<=len(t) for p,t in self.truth.items()):raise ScopeError('ineligible K1 participant')
        self.rng=np.random.default_rng(derived_seed(seed,scope.key,'ROLE_SAMPLING'))
        self.steps_per_epoch=math.ceil(sum(map(len,self.truth.values()))/8)
        self.steps=0;self.duplicate_queries=0;self.query_draws=0
    def sample(self):
        result=[]
        for p in self.rng.choice(self.people,size=8,replace=False):
            r=int(self.rng.choice(self.truth[p]));pool=self.answers[p][self.answers[p]!=r]
            q=self.rng.choice(pool,size=4,replace=True)
            self.duplicate_queries+=4-len(set(q));self.query_draws+=4
            result.extend(Role(str(p),r,int(i)) for i in q)
        self.steps+=1;return result
    def state(self):return {'rng':copy.deepcopy(self.rng.bit_generator.state),'steps':self.steps,'duplicate_queries':self.duplicate_queries,'query_draws':self.query_draws}
    def load_state(self,s):
        self.rng.bit_generator.state=copy.deepcopy(s['rng']);self.steps=s['steps'];self.duplicate_queries=s['duplicate_queries'];self.query_draws=s['query_draws']

