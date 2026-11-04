import copy
import torch
from torch import nn
from .config import SPEC, ImplementationBlocked, derived_seed

EXPECTED={'M0':51457,'M1':53169,'M2':109090,'M3':110802,'C1-0':109090,'C1-1':110802}
def validate_cls(x):
    if x.ndim!=2 or x.shape[1]!=768 or x.dtype!=torch.float32 or not bool(torch.isfinite(x).all()):raise ImplementationBlocked('invalid CLS tensor')
    if bool(((torch.linalg.vector_norm(x.detach().double(),dim=-1)-1).abs()>SPEC.norm_tolerance).any()):raise ImplementationBlocked('CLS is not within frozen L2 tolerance')

def projection(a,b):return nn.Sequential(nn.Linear(a,b,bias=True),nn.GELU(approximate='none'),nn.LayerNorm(b,eps=1e-5,elementwise_affine=True))

class QueryModel(nn.Module):
    def __init__(self,use_psy=False):
        super().__init__();self.use_psy=use_psy
        self.semantic=projection(768,64)
        self.psy=projection(72,16) if use_psy else None
        self.head=nn.Sequential(nn.Linear(80 if use_psy else 64,32,bias=True),nn.GELU(approximate='none'),nn.Linear(32,1,bias=True))
    def forward(self,e,x_psy=None):
        validate_cls(e);h=self.semantic(e)
        if self.use_psy:
            if x_psy is None or x_psy.shape!=(len(e),72) or not bool(torch.isfinite(x_psy).all()):raise ImplementationBlocked('finite Psy72 required')
            h=torch.cat([h,self.psy(x_psy)],dim=-1)
        return self.head(h).squeeze(-1)

class Relation(nn.Module):
    def __init__(self):
        super().__init__();self.projection=projection(768,64)
        self.mlp=nn.Sequential(nn.Linear(257,32,bias=True),nn.GELU(approximate='none'),nn.Linear(32,1,bias=True))
        nn.init.zeros_(self.mlp[2].weight);nn.init.zeros_(self.mlp[2].bias)
    def components(self,q,r,c):
        validate_cls(q);validate_cls(r)
        cc=c.reshape(1,768).expand_as(q);validate_cls(cc)
        uq=self.projection(q);ur=self.projection(r);uc=self.projection(cc)
        pr=torch.cat([uq,ur,(uq-ur).abs(),uq*ur,(q*r).sum(-1,keepdim=True)],dim=-1)
        pc=torch.cat([uq,uc,(uq-uc).abs(),uq*uc,(q*cc).sum(-1,keepdim=True)],dim=-1)
        return self.mlp(pr).squeeze(-1),self.mlp(pc).squeeze(-1)
    def forward(self,q,r,c):
        fr,fc=self.components(q,r,c);return fr-fc

class AnchoredModel(nn.Module):
    def __init__(self,query,center,relation_seed):
        super().__init__();self.query=copy.deepcopy(query)
        for p in self.query.parameters():p.requires_grad_(False)
        self.query.eval()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(derived_seed(relation_seed,'RELATION_INIT'));self.relation=Relation()
        self.register_buffer('center',torch.as_tensor(center,dtype=torch.float32).clone())
        validate_cls(self.center.reshape(1,-1))
    def train(self,mode=True):
        super().train(mode);self.query.eval();return self
    def forward(self,q,x_psy,r):return self.query(q,x_psy)+self.relation(q,r,self.center)
    def decomposition(self,q,x_psy,r):
        base=self.query(q,x_psy);fr,fc=self.relation.components(q,r,self.center)
        return {'base':base,'F_reference':fr,'F_center':fc,'residual':fr-fc,'logit':base+(fr-fc)}

def make_query(use_psy,seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(derived_seed(seed,'QUERY_INIT',int(use_psy)));return QueryModel(use_psy)

def parameter_count(model,trainable_only=False):return sum(p.numel() for p in model.parameters() if not trainable_only or p.requires_grad)

