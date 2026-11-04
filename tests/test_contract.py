import copy,json,math,tempfile,unittest
from dataclasses import replace
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
from scipy.special import erf
from scipy.stats import rankdata

from rel01.config import SPEC,ScopeError,ImplementationBlocked,RealDataForbidden,deterministic,canonical_hash
from rel01.data import Answers,Scope,roles_for,query_weights,role_weights
from rel01.preprocessing import PsyNormalizer,Center
from rel01.other import build_other
from rel01.models import EXPECTED,make_query,AnchoredModel,parameter_count
from rel01.sampling import HierarchicalSampler
from rel01.synthetic import synthetic_answers
from rel01.trainer import TrainingSession,EarlyStop,refit_epochs,tensor_digest,common_counterfactual_roles,optimizer_for
from rel01.evaluation import binary_metrics,hierarchy,paired_bootstrap,margin_diagnostics,mu3d_condition_subset

ROOT=Path(__file__).resolve().parents[1]
OBS={}
def record(name,value):OBS[name]=value
def outer_scope(scope):return Scope(scope.dataset,scope.outer_fold,None,scope.fit_ids,(),scope.validation_ids+scope.test_ids)
def assert_nested(test,a,b):
    if torch.is_tensor(a):test.assertTrue(torch.equal(a,b))
    elif isinstance(a,np.ndarray):test.assertTrue(np.array_equal(a,b,equal_nan=True))
    elif isinstance(a,dict):
        test.assertEqual(a.keys(),b.keys())
        for k in a:assert_nested(test,a[k],b[k])
    elif isinstance(a,(list,tuple)):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b):assert_nested(test,x,y)
    else:test.assertEqual(a,b)

class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        deterministic(SPEC.seeds[0]);cls.data,cls.scope=synthetic_answers(fit_people=12)
        cls.norm=PsyNormalizer.fit(cls.data,cls.scope);cls.center=Center.fit(cls.data,cls.scope)
        cls.seed=SPEC.seeds[0]
    def test_01_exact_six_parameter_counts(self):
        counts={}
        for name in EXPECTED:
            psy=name in ['M1','M3','C1-1'];q=make_query(psy,self.seed)
            model=q if name in ['M0','M1'] else AnchoredModel(q,self.center.value,self.seed)
            counts[name]=parameter_count(model);self.assertEqual(counts[name],EXPECTED[name])
            if name not in ['M0','M1']:self.assertEqual(parameter_count(model,True),57633)
            self.assertFalse(any(isinstance(m,(torch.nn.Dropout,torch.nn.BatchNorm1d)) for m in model.modules()))
        record('parameter_counts',counts)
    def test_02_shapes_finite_and_psy_branch(self):
        e=torch.tensor(self.data.cls[:5].copy());x=torch.tensor(self.norm.transform(self.data,self.scope)[:5]);r=e.flip(0)
        for psy in [False,True]:
            q=make_query(psy,self.seed);model=AnchoredModel(q,self.center.value,self.seed)
            self.assertEqual(model(e,x,r).shape,(5,));self.assertTrue(torch.isfinite(model(e,x,r)).all())
        with self.assertRaises(ImplementationBlocked):make_query(True,self.seed)(e)
    def test_03_null_identity_after_nonzero_weights(self):
        model=AnchoredModel(make_query(True,self.seed),self.center.value,self.seed).eval()
        with torch.no_grad():model.relation.mlp[2].weight.normal_(0,.2);model.relation.mlp[2].bias.fill_(.73)
        q=torch.tensor(self.data.cls[:11].copy());x=torch.tensor(self.norm.transform(self.data,self.scope)[:11]);c=model.center.reshape(1,-1).expand_as(q)
        with torch.no_grad():actual=model(q,x,c);expected=model.query(q,x)
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
        record('NULL_max_abs_error',float((actual-expected).abs().max()))
    def test_04_zero_init_then_upstream_gradient(self):
        model=AnchoredModel(make_query(False,self.seed),self.center.value,self.seed)
        q=torch.tensor(self.data.cls[:8].copy());r=torch.tensor(self.data.cls[8:16].copy());y=torch.tensor(self.data.labels[:8],dtype=torch.float32)
        opt=optimizer_for(model,'B');before=tensor_digest(model.query.state_dict())
        loss=F.binary_cross_entropy_with_logits(model(q,None,r),y);loss.backward()
        self.assertGreater(float(model.relation.mlp[2].weight.grad.norm()),1e-8)
        self.assertEqual(float(model.relation.projection[0].weight.grad.abs().max()),0.)
        self.assertEqual(float(model.relation.mlp[0].weight.grad.abs().max()),0.)
        self.assertEqual(float(model.relation.mlp[2].bias.grad.abs().max()),0.)
        opt.step();opt.zero_grad(set_to_none=True);F.binary_cross_entropy_with_logits(model(q,None,r),y).backward()
        g=float(model.relation.projection[0].weight.grad.norm());self.assertGreater(g,1e-8)
        self.assertEqual(tensor_digest(model.query.state_dict()),before)
        record('upstream_gradient_after_first_update',g)
    def test_05_center_path_gradient_not_detached(self):
        model=AnchoredModel(make_query(False,self.seed),self.center.value,self.seed)
        with torch.no_grad():model.relation.mlp[2].weight.fill_(.03)
        q=torch.tensor(self.data.cls[:6].copy());r=torch.tensor(self.data.cls[10:16].copy());captured=[]
        def hook(module,inputs,output):output.retain_grad();captured.append(output)
        h=model.relation.projection.register_forward_hook(hook)
        model(q,None,r).sum().backward();h.remove()
        self.assertEqual(len(captured),3);self.assertGreater(float(captured[2].grad.norm()),0)
        true=model.relation.projection[0].weight.grad.clone();model.zero_grad()
        fr,fc=model.relation.components(q,r,model.center);(fr-fc.detach()).sum().backward()
        self.assertGreater(float((true-model.relation.projection[0].weight.grad).norm()),1e-7)
        record('center_projected_gradient_norm',float(captured[2].grad.norm()))
    def test_06_constructed_nonadditive_rank_flip(self):
        c=np.zeros(768,np.float32);c[2]=1;model=AnchoredModel(make_query(False,self.seed),c,self.seed)
        with torch.no_grad():
            for p in model.query.parameters():p.zero_()
            model.relation.mlp[0].weight.zero_();model.relation.mlp[0].bias.zero_();model.relation.mlp[0].weight[0,256]=1
            model.relation.mlp[2].weight.zero_();model.relation.mlp[2].weight[0,0]=1;model.relation.mlp[2].bias.zero_()
        q=torch.zeros(2,768);q[0,0]=1;q[1,1]=1
        s0=model(q,None,q[0].repeat(2,1));s1=model(q,None,q[1].repeat(2,1))
        m0=float((s0[0]-s0[1]).detach());m1=float((s1[0]-s1[1]).detach())
        self.assertGreater(m0,0);self.assertLess(m1,0);self.assertGreater(abs(m0-m1),.5)
        record('constructed_nonadditive_margins',[m0,m1])
    def test_07_numpy_independent_forward(self):
        model=AnchoredModel(make_query(True,self.seed),self.center.value,self.seed).eval()
        with torch.no_grad():model.relation.mlp[2].weight.normal_(0,.1)
        s={k:v.detach().numpy().astype(float) for k,v in model.state_dict().items()}
        gelu=lambda x:.5*x*(1+erf(x/np.sqrt(2)))
        def linear(x,p):return x@s[p+'.weight'].T+s[p+'.bias']
        def proj(x,p):
            a=gelu(linear(x,p+'.0'));return (a-a.mean(-1,keepdims=True))/np.sqrt(a.var(-1,keepdims=True)+1e-5)*s[p+'.2.weight']+s[p+'.2.bias']
        eq=self.data.cls[:7].astype(float);er=self.data.cls[7:14].astype(float);xp=self.norm.transform(self.data,self.scope)[:7].astype(float);c=self.center.value.astype(float)[None,:].repeat(7,0)
        base=linear(gelu(linear(np.c_[proj(eq,'query.semantic'),proj(xp,'query.psy')],'query.head.0')),'query.head.2').ravel()
        uq,ur,uc=[proj(x,'relation.projection') for x in [eq,er,c]]
        def ff(u,ex):
            phi=np.c_[uq,u,abs(uq-u),uq*u,np.sum(eq*ex,axis=1)]
            return linear(gelu(linear(phi,'relation.mlp.0')),'relation.mlp.2').ravel()
        independent=base+ff(ur,er)-ff(uc,c)
        actual=model(torch.tensor(eq,dtype=torch.float32),torch.tensor(xp,dtype=torch.float32),torch.tensor(er,dtype=torch.float32)).detach().numpy()
        np.testing.assert_allclose(actual,independent,atol=3e-6,rtol=2e-5)
        record('independent_numpy_forward_max_abs_error',float(np.max(abs(actual-independent))))
    def test_08_norm_and_center_block_gates(self):
        with self.assertRaises(ImplementationBlocked):replace(self.data,cls=(self.data.cls*2).astype(np.float32))
        xx=np.zeros_like(self.data.cls)
        for p in set(self.data.participants):
            ii=[i for i,v in enumerate(self.data.participants) if v==p]
            for j,i in enumerate(ii):xx[i,0]=1 if j%2==0 else -1
        bad=replace(self.data,cls=xx)
        with self.assertRaises(ImplementationBlocked):Center.fit(bad,self.scope)
    def test_09_exact_marginal_weights_and_mass(self):
        w=query_weights(self.data,self.scope);roles,rw=role_weights(self.data,self.scope,self.scope.fit_ids);expanded=np.zeros_like(w)
        for r,a in zip(roles,rw):expanded[r.query]+=a
        np.testing.assert_allclose(w,expanded,atol=1e-15,rtol=1e-14)
        self.assertAlmostEqual(w[self.data.labels==1].sum(),2/3,places=14)
        for p in self.scope.fit_ids:self.assertAlmostEqual(w[self.scope.indices(self.data,[p])].sum(),1/12,places=14)
    def test_10_query_batch_gradient_equivalence(self):
        data,scope=synthetic_answers(vary_counts=True);norm=PsyNormalizer.fit(data,scope);q=make_query(True,self.seed)
        ix=scope.indices(data);x=torch.tensor(data.cls.copy());xp=torch.tensor(norm.transform(data,scope));y=torch.tensor(data.labels.copy(),dtype=torch.float32);w=torch.tensor(query_weights(data,scope),dtype=torch.float32)
        q.zero_grad();full=(w[ix]*F.binary_cross_entropy_with_logits(q(x[ix],xp[ix]),y[ix],reduction='none')).sum();full.backward()
        expected=[p.grad.clone() for p in q.parameters()];q.zero_grad();n=len(ix)
        for start in range(0,n,32):
            ii=ix[start:start+32];b=len(ii);loss=n/b*(w[ii]*F.binary_cross_entropy_with_logits(q(x[ii],xp[ii]),y[ii],reduction='none')).sum()
            (b/n*loss).backward()
        for p,g in zip(q.parameters(),expected):torch.testing.assert_close(p.grad,g,atol=2e-6,rtol=2e-5)
    def test_11_normalizer_missing_true_zero_and_near_constant(self):
        v=self.data.psy.copy();m=self.data.mask.copy();fit=self.scope.indices(self.data);val=self.scope.indices(self.data,self.scope.validation_ids)
        v[:,0]=np.nan;m[:,0]=False;v[val,0]=5;m[val,0]=True
        v[:,1]=7;m[:,1]=True;v[:,2]=0;m[:,2]=True;v[fit[0],2]=10
        v[:,3]=3;m[:,3]=True;v[fit[0],3]+=1e-7
        data=replace(self.data,psy=v,mask=m,reason=(~m).astype(int));raw=v.copy();norm=PsyNormalizer.fit(data,self.scope);out=norm.transform(data,self.scope)
        self.assertEqual(norm.status[0],'ALL_MISSING');self.assertTrue(np.all(out[:,0]==0));self.assertTrue(np.all(out[:,36]==0))
        self.assertEqual(norm.status[1],'NEAR_CONSTANT');self.assertEqual(norm.scale[1],1)
        self.assertEqual(norm.status[3],'NEAR_CONSTANT');self.assertGreater(norm.mean[2],0);self.assertLess(norm.mean[2],10)
        self.assertTrue(np.isfinite(out).all());self.assertEqual(out.dtype,np.float32);self.assertTrue(np.array_equal(data.psy,raw,equal_nan=True))
    def test_12_independent_normalizer_and_center(self):
        from fractions import Fraction
        ii=self.scope.indices(self.data);pcount=len(self.scope.fit_ids);w=[]
        for i in ii:
            person=self.scope.indices(self.data,[self.data.participants[i]]);n=len(person);t=int((self.data.labels[person]==0).sum())
            w.append(float(Fraction(t-1,t*(n-1)*pcount) if self.data.labels[i]==0 else Fraction(1,(n-1)*pcount)))
        w=np.array(w)
        for j in range(36):
            valid=self.data.mask[ii,j];z=self.data.psy[ii,j][valid];ww=w[valid]/w[valid].sum();mu=np.dot(ww,z);var=np.dot(ww,(z-mu)**2)
            self.assertAlmostEqual(self.norm.mean[j],mu,places=12);self.assertAlmostEqual(self.norm.scale[j],np.sqrt(var) if var>=1e-12 else 1,places=12)
        means=[]
        for p in self.scope.fit_ids:
            ids=[i for i,v in enumerate(self.data.participants) if v==p and self.data.labels[i]==0];means.append(np.mean(self.data.cls[ids].astype(float),axis=0))
        c=np.mean(means,axis=0);c=c/np.linalg.norm(c);np.testing.assert_allclose(c,self.center.value,atol=2e-8,rtol=2e-7)
    def test_13_scope_poisoning_cannot_change_fit_statistics(self):
        v=self.data.psy.copy();e=self.data.cls.copy();notfit=np.array([p not in self.scope.fit_ids for p in self.data.participants]);v[notfit]=np.where(self.data.mask[notfit],1e12,np.nan);e[notfit]=np.roll(e[notfit],1,axis=1)
        other=replace(self.data,psy=v,cls=e);n=PsyNormalizer.fit(other,self.scope);c=Center.fit(other,self.scope)
        np.testing.assert_array_equal(n.mean,self.norm.mean);np.testing.assert_array_equal(n.scale,self.norm.scale);np.testing.assert_array_equal(c.value,self.center.value)
        with self.assertRaises(ScopeError):PsyNormalizer.fit(self.data,self.scope,self.scope.fit_ids+self.scope.validation_ids)
        with self.assertRaises(ScopeError):Center.fit(self.data,self.scope,self.scope.fit_ids+self.scope.test_ids)
        with self.assertRaises(ScopeError):Scope(self.data.dataset,0,0,self.scope.fit_ids,self.scope.fit_ids[:1],())
    def test_14_other_train_pool_and_original_role_keys(self):
        other=build_other(self.data,self.scope);roles=roles_for(self.data,self.scope,self.scope.validation_ids)
        before=[r.key(self.data,self.scope) for r in roles]
        for r in roles:
            o=other.resolve(self.data,self.scope,r);self.assertIn(self.data.participants[o],self.scope.fit_ids);self.assertNotEqual(self.data.participants[o],r.participant)
        self.assertEqual(before,[r.key(self.data,self.scope) for r in roles]);self.assertEqual(len(roles),4*2*3)
        poisoned=copy.deepcopy(other);r=roles[0];poisoned.mapping[r.reference]=r.query
        with self.assertRaises(ScopeError):poisoned.resolve(self.data,self.scope,r)
    def test_15_other_independent_hash_selection_and_unmatched(self):
        import hashlib
        other=build_other(self.data,self.scope)
        pool=[i for i,p in enumerate(self.data.participants) if p in self.scope.fit_ids and self.data.labels[i]==0]
        for row in other.records:
            r=self.data.index[row['reference_id']];p=row['participant_id'];keys=[]
            for o in pool:
                if self.data.participants[o]==p:continue
                gap=abs(math.log1p(int(self.data.nlex[o]))-math.log1p(int(self.data.nlex[r])))
                # NumPy and scalar libm can differ by an ulp; compare candidate identity on the prescribed NumPy gap.
                gap=float(abs(np.log1p(self.data.nlex[o])-np.log1p(self.data.nlex[r])))
                key=hashlib.sha256(f'20260722|{self.scope.key}|{p}|{self.data.ids[r]}|{self.data.ids[o]}'.encode()).hexdigest();keys.append((gap,key,self.data.ids[o]))
            self.assertEqual(row['other_id'],min(keys)[2])
        unique=replace(self.data,sources=tuple('UNIQUE_'+x for x in self.data.ids));ss=build_other(unique,self.scope,same_source=True,length_cap=.25)
        self.assertTrue(all(x is None for x in ss.mapping.values()));self.assertTrue(all(r['status']=='UNMATCHED' for r in ss.records))
    def test_16_sampler_distinct_people_replacement_and_shared_seed(self):
        a=HierarchicalSampler(self.data,self.scope,self.seed);b=HierarchicalSampler(self.data,self.scope,self.seed);c=HierarchicalSampler(self.data,self.scope,SPEC.seeds[1])
        seen=[];different=False
        for _ in range(20):
            x=a.sample();y=b.sample();z=c.sample();self.assertEqual(x,y);different |= x!=z
            counts=Counter(r.participant for r in x);self.assertEqual(len(counts),8);self.assertEqual(set(counts.values()),{4})
            for p in counts:self.assertEqual(len({r.reference for r in x if r.participant==p}),1)
            self.assertTrue(all(r.query!=r.reference for r in x));seen.extend(x)
        self.assertTrue(different);self.assertGreater(a.duplicate_queries,0);self.assertEqual(a.steps_per_epoch,3)
    def test_17_monte_carlo_actual_relation_gradient(self):
        deterministic(self.seed);data,scope=synthetic_answers(fit_people=8);norm=PsyNormalizer.fit(data,scope);center=Center.fit(data,scope)
        model=AnchoredModel(make_query(False,self.seed),center.value,self.seed)
        with torch.no_grad():model.relation.mlp[2].weight.normal_(0,.02)
        roles,weights=role_weights(data,scope,scope.fit_ids);params=[p for p in model.parameters() if p.requires_grad]
        e=torch.tensor(data.cls.copy());y=torch.tensor(data.labels.copy(),dtype=torch.float32)
        rows=[]
        for r in roles:
            loss=F.binary_cross_entropy_with_logits(model(e[r.query:r.query+1],None,e[r.reference:r.reference+1]),y[r.query:r.query+1])
            g=torch.autograd.grad(loss,params);rows.append(torch.cat([x.ravel() for x in g]).detach().numpy())
        G=np.array(rows,dtype=np.float64);expected=weights@G
        sampler=HierarchicalSampler(data,scope,self.seed);index={(r.reference,r.query):i for i,r in enumerate(roles)};counts=np.zeros(len(roles),int)
        for _ in range(3000):
            for r in sampler.sample():counts[index[(r.reference,r.query)]]+=1
        empirical=(counts/counts.sum())@G;relative=float(np.linalg.norm(empirical-expected)/np.linalg.norm(expected))
        self.assertLess(relative,.12)
        # Independent complete objective autograd also agrees with rolewise gradient assembly.
        q=[r.query for r in roles];r=[r.reference for r in roles]
        loss=(torch.tensor(weights,dtype=torch.float32)*F.binary_cross_entropy_with_logits(model(e[q],None,e[r]),y[q],reduction='none')).sum()
        full=torch.cat([x.ravel() for x in torch.autograd.grad(loss,params)]).detach().numpy()
        np.testing.assert_allclose(full,expected,atol=3e-6,rtol=3e-4)
        record('MC_gradient',{'draws':int(counts.sum()),'parameter_coordinates':len(expected),'relative_L2_error':relative,'tolerance':.12})
    def test_19_optimizer_weight_decay_and_freeze(self):
        model=AnchoredModel(make_query(True,self.seed),self.center.value,self.seed);model.train()
        self.assertFalse(model.query.training);opt=optimizer_for(model,'B')
        active={id(p) for g in opt.param_groups for p in g['params']}
        self.assertTrue(all(id(p) not in active and not p.requires_grad for p in model.query.parameters()))
        for group in opt.param_groups:
            for p in group['params']:self.assertEqual(group['weight_decay'],.001 if p.ndim==2 else 0.)
        self.assertEqual(opt.defaults['betas'],(.9,.999));self.assertEqual(opt.defaults['eps'],1e-8)
    def test_20_early_stop_ties_epoch0_and_patience(self):
        e=EarlyStop(12);self.assertEqual(e.observe(1.,0),(True,False))
        self.assertFalse(e.observe(1.-1e-6,1)[0]);self.assertEqual(e.best_epoch,0)
        for epoch in range(2,13):improved,stop=e.observe(1.,epoch)
        self.assertTrue(stop);self.assertEqual(e.bad_epochs,12)
        a=EarlyStop(15);a.observe(.7,1);self.assertTrue(a.observe(.7-2e-6,2)[0]);self.assertEqual(a.best_epoch,2)
        with self.assertRaises(ImplementationBlocked):a.observe(float('nan'),3)
    def test_21_outer_median_ceil_and_zero_relation(self):
        self.assertEqual(refit_epochs([1,2,3,10],'A'),3);self.assertEqual(refit_epochs([0,0,0,8],'B'),0)
        with self.assertRaises(ScopeError):refit_epochs([0,0,0,0],'A')
        data,sc=synthetic_answers();outer=outer_scope(sc);a=TrainingSession(data,outer,'M0',self.seed,outer_inner_epochs=[1]*4);a.run()
        b=TrainingSession(data,outer,'M2',self.seed,a.query_artifact(),outer_inner_epochs=[0]*4)
        self.assertTrue(b.done);self.assertEqual(b.updates,0);self.assertEqual(len(b.optimizer.state),0)
        x=b.predictions(outer.test_ids,'SELF');n=b.predictions(outer.test_ids,'NULL',evaluation_kind='COUNTERFACTUAL')
        self.assertEqual([r['logit'] for r in x],[r['logit'] for r in n])
    def test_22_query_checkpoint_scope_and_seed_binding(self):
        a=TrainingSession(self.data,self.scope,'M0',self.seed);a.run();artifact=a.query_artifact()
        with self.assertRaises(ScopeError):TrainingSession(self.data,self.scope,'M2',SPEC.seeds[1],artifact)
        with self.assertRaises(ScopeError):TrainingSession(self.data,outer_scope(self.scope),'M2',self.seed,artifact,outer_inner_epochs=[1]*4)
        with self.assertRaises(ScopeError):TrainingSession(self.data,self.scope,'M3',self.seed,artifact)
    def test_23_self_c1_reuse_and_exact_role_stream(self):
        a=TrainingSession(self.data,self.scope,'M0',self.seed);a.run();artifact=a.query_artifact()
        s=TrainingSession(self.data,self.scope,'M2',self.seed,artifact);c=TrainingSession(self.data,self.scope,'C1-0',self.seed,artifact)
        self.assertEqual(tensor_digest(s.model.relation.state_dict()),tensor_digest(c.model.relation.state_dict()))
        self.assertEqual(s.other.sha,c.other.sha)
        s.run(max_updates=4);c.run(max_updates=4);self.assertEqual(s.batch_trace,c.batch_trace)
        self.assertEqual(tensor_digest(s.model.query.state_dict()),artifact.state_sha);self.assertEqual(tensor_digest(c.model.query.state_dict()),artifact.state_sha)
    def _resume_case(self,branch,cut):
        data,sc=synthetic_answers(vary_counts=True);outer=outer_scope(sc);artifact=None
        if branch=='M2':
            a=TrainingSession(data,outer,'M0',self.seed,outer_inner_epochs=[1]*4);a.run();artifact=a.query_artifact()
        make=lambda:TrainingSession(data,outer,branch,self.seed,artifact,outer_inner_epochs=[3]*4)
        continuous=make();continuous.run();first=make();first.run(max_updates=cut)
        with tempfile.TemporaryDirectory() as tmp:
            first.save(tmp);resumed=make();resumed.resume(tmp);resumed.run()
            assert_nested(self,continuous.model.state_dict(),resumed.model.state_dict());assert_nested(self,continuous.optimizer.state_dict(),resumed.optimizer.state_dict())
            self.assertEqual(continuous.batch_trace,resumed.batch_trace);self.assertEqual(continuous.history,resumed.history)
            if continuous.sampler:assert_nested(self,continuous.sampler.state(),resumed.sampler.state())
    def test_24_resume_A_mid_epoch(self):self._resume_case('M0',1)
    def test_25_resume_A_epoch_boundary(self):self._resume_case('M0',3)
    def test_26_resume_B_mid_epoch(self):self._resume_case('M2',1)
    def test_27_resume_B_epoch_boundary(self):self._resume_case('M2',4)
    def test_28_checkpoint_commit_failures_and_sha(self):
        sc=outer_scope(self.scope);make=lambda:TrainingSession(self.data,sc,'M0',self.seed,outer_inner_epochs=[3]*4)
        session=make();session.run(max_updates=1)
        with tempfile.TemporaryDirectory() as tmp:
            session.save(tmp);session.run(max_updates=1)
            with self.assertRaises(InterruptedError):session.save(tmp,fail_at='after_payload')
            r=make().resume(tmp);self.assertEqual(r.updates,1)
            with self.assertRaises(InterruptedError):session.save(tmp,fail_at='after_pointer')
            r=make().resume(tmp);self.assertEqual(r.updates,2)
            pointer=json.loads((Path(tmp)/'latest.json').read_text());payload=Path(tmp)/pointer['payload'];payload.write_bytes(payload.read_bytes()+b'bad')
            with self.assertRaises(ImplementationBlocked):make().resume(tmp)
    def test_29_checkpoint_wrong_scope_rejected(self):
        a=TrainingSession(self.data,self.scope,'M0',self.seed);a.run(max_updates=1)
        with tempfile.TemporaryDirectory() as tmp:
            a.save(tmp)
            wrong=TrainingSession(self.data,self.scope,'M0',SPEC.seeds[1])
            with self.assertRaises(ScopeError):wrong.resume(tmp)
    def test_30_metric_ties_threshold_and_independent_rank(self):
        y=np.array([0,1,1]);s=np.array([0.,0.,0.]);m=binary_metrics(y,s)
        self.assertAlmostEqual(m['AP'],2/3);self.assertEqual(m['AUROC'],.5);self.assertEqual(m['Recall'],1)
        y=np.array([0,1,0,1,1,0]);s=np.array([-.7,.2,.2,.8,.2,-.7]);m=binary_metrics(y,s)
        ranks=rankdata(s,method='average');auc=(ranks[y==1].sum()-3*4/2)/(3*3);self.assertEqual(m['AUROC'],auc)
        ap=sum(((y[s>=v]).sum()/sum(s>=v))*sum((s==v)&(y==1))/sum(y) for v in set(s));self.assertAlmostEqual(m['AP'],ap)
    def _evaluation_rows(self):
        rows=[]
        for i in range(3):
            for ref in range(2):
                for j,seed in enumerate(SPEC.seeds):
                    for q,y in enumerate([0,1,1]):
                        rows.append({'domain':'SYNTHETIC','dataset':'SYN_EVAL','participant_id':f'SYN_P{i}','reference_id':f'SYN_P{i}_R{ref}',
                            'query_id':f'SYN_P{i}_Q{ref}_{q}','seed':seed,'label':y,'logit':float([.2,-.5,.8][(q+j+i)%3]),'outer_fold':i%2,
                            'model':'M2','control':'SELF','evaluation_kind':'PRIMARY'})
        return rows
    def test_31_hierarchy_no_ensemble_no_fold_weighting(self):
        rows=self._evaluation_rows();got=hierarchy(rows);manual=[]
        for i in range(3):
            values=[]
            for ref in range(2):
                for seed in SPEC.seeds:
                    g=[r for r in rows if r['participant_id']==f'SYN_P{i}' and r['reference_id']==f'SYN_P{i}_R{ref}' and r['seed']==seed]
                    values.append(binary_metrics([r['label'] for r in g],[r['logit'] for r in g])['AP'])
            manual.append(float(np.mean(values)))
        self.assertAlmostEqual(got['macro']['AP'],np.mean(manual));self.assertEqual(len(got['participants']),3)
        with self.assertRaises(ScopeError):hierarchy([r for r in rows if r['seed']!=SPEC.seeds[0]])
    def test_32_paired_bootstrap_and_wrong_pairing(self):
        a=self._evaluation_rows();b=[{**r,'logit':r['logit']+.9} for r in a]
        out=paired_bootstrap(a,b);self.assertEqual(out['bootstrap_replicates'],5000)
        self.assertAlmostEqual(out['metrics']['AP']['difference'],0);self.assertAlmostEqual(out['metrics']['AUROC']['difference'],0)
        with self.assertRaises(ScopeError):paired_bootstrap(a,b[:-1])
        self.assertEqual(out,paired_bootstrap(a,b))
    def test_33_common_shift_diagnostics_preserve_ranking(self):
        a=self._evaluation_rows();b=[{**r,'logit':r['logit']+2.} for r in a]
        d=margin_diagnostics(a,b)
        for r in d:self.assertEqual(r['strict_flip'],0);self.assertAlmostEqual(r['centered_difference_energy'],0,places=25)
        self.assertNotEqual(hierarchy(a)['macro']['BCE'],hierarchy(b)['macro']['BCE'])
    def test_34_counterfactual_common_subset(self):
        sc=outer_scope(self.scope);a=TrainingSession(self.data,sc,'M0',self.seed,outer_inner_epochs=[1]*4);a.run()
        b=TrainingSession(self.data,sc,'M2',self.seed,a.query_artifact(),outer_inner_epochs=[1]*4);b.run()
        common,source,coverage=common_counterfactual_roles(b,sc.test_ids,True)
        self.assertGreater(len(common),0)
        results=[b.predictions(sc.test_ids,mode,common,source if mode=='SOURCE_OTHER' else None,evaluation_kind='COUNTERFACTUAL') for mode in ['SELF','NULL','OTHER','SOURCE_OTHER']]
        keys=lambda rr:[(x['participant_id'],x['reference_id'],x['query_id']) for x in rr]
        self.assertTrue(all(keys(z)==keys(results[0]) for z in results));self.assertEqual(coverage['common_roles'],len(common))
    def test_35_mu3d_condition_symmetry(self):
        conditions={};rows=[]
        for p in ['SYN_P0','SYN_P1']:
            for c in ['PT','NT','PL','NL']:conditions[p+'_'+c]=c
            for r in ['PT','NT']:
                for q in ['PT','NT','PL','NL']:
                    if q!=r:rows.append({'domain':'SYNTHETIC','dataset':'SYN_MU','participant_id':p,'reference_id':p+'_'+r,'query_id':p+'_'+q,'seed':SPEC.seeds[0],'label':int(q.endswith('L')),'logit':.1,'model':'M2','control':'SELF','evaluation_kind':'PRIMARY'})
        for g in ['A_SAME_QUERY_VALENCE','B_SAME_QUERY_REFERENT']:
            out=mu3d_condition_subset(rows,conditions,g);self.assertEqual(len(out),8)
            self.assertEqual(Counter(x['reference_id'] for x in out),Counter({p+'_'+r:2 for p in ['SYN_P0','SYN_P1'] for r in ['PT','NT']}))
    def test_36_six_model_outer_framework_smoke(self):
        data,inner=synthetic_answers(fit_people=8);sc=outer_scope(inner);base={};counts={}
        for branch in ['M0','M1']:
            s=TrainingSession(data,sc,branch,self.seed,outer_inner_epochs=[1]*4);s.run();base[branch]=s.query_artifact();counts[branch]=s.updates
        for branch in ['M2','M3','C1-0','C1-1']:
            q='M1' if branch in ['M3','C1-1'] else 'M0';s=TrainingSession(data,sc,branch,self.seed,base[q],outer_inner_epochs=[1]*4);s.run();counts[branch]=s.updates
            self.assertTrue(s.done);self.assertTrue(all(np.isfinite(r['logit']) for r in s.predictions(sc.test_ids)))
            self.assertFalse(s.model.query.training)
        record('six_branch_synthetic_updates',counts)
    def test_37_inner_B_resume_best_and_patience(self):
        a=TrainingSession(self.data,self.scope,'M0',self.seed);a.run();artifact=a.query_artifact()
        make=lambda:TrainingSession(self.data,self.scope,'M2',self.seed,artifact)
        full=make();full.run();part=make();part.run(max_updates=part.steps_epoch+1)
        with tempfile.TemporaryDirectory() as tmp:
            part.save(tmp);resumed=make().resume(tmp);resumed.run()
            assert_nested(self,full.model.state_dict(),resumed.model.state_dict());assert_nested(self,full.best_state,resumed.best_state)
            self.assertEqual(full.stopper.__dict__,resumed.stopper.__dict__);self.assertEqual(full.history,resumed.history)
            pred=resumed.predictions(self.scope.validation_ids);self.assertTrue(pred)
            self.assertEqual(tensor_digest(resumed.model.state_dict()),tensor_digest(resumed.best_state))
        record('inner_relation_best_epoch',full.stopper.best_epoch)
    def test_38_unfinished_query_and_runtime_resume_gates(self):
        a=TrainingSession(self.data,self.scope,'M0',self.seed);a.run(max_updates=a.steps_epoch)
        with self.assertRaises(ScopeError):a.query_artifact()
        with self.assertRaises(ScopeError):a.predictions(self.scope.validation_ids)
        with tempfile.TemporaryDirectory() as tmp:
            a.save(tmp);other=TrainingSession(self.data,self.scope,'M0',self.seed)
            other.binding['runtime']={**other.binding['runtime'],'torch':'unverified-version'}
            with self.assertRaises(ScopeError):other.resume(tmp)

    def test_40_independent_nonzero_bootstrap_and_seed_role_gate(self):
        rows=self._evaluation_rows();other=[{**r,'logit':(2.*r['label']-1 if r['participant_id'] in ['SYN_P0','SYN_P1'] else r['logit'])} for r in rows]
        out=paired_bootstrap(rows,other);deltas=[]
        for p in sorted({r['participant_id'] for r in rows}):
            per=[]
            for ref in sorted({r['reference_id'] for r in rows if r['participant_id']==p}):
                for seed in SPEC.seeds:
                    ms=[]
                    for rr in [rows,other]:
                        g=[r for r in rr if r['participant_id']==p and r['reference_id']==ref and r['seed']==seed]
                        y=np.array([r['label'] for r in g]);s=np.array([r['logit'] for r in g]);rank=rankdata(s,method='average');n1=sum(y);n0=len(y)-n1
                        ms.append((sum(rank[y==1])-n1*(n1+1)/2)/(n1*n0))
                    per.append(ms[0]-ms[1])
            deltas.append(np.mean(per))
        self.assertGreater(np.var(deltas),0);self.assertTrue(any(abs(x)>0 for x in deltas))
        rng=np.random.default_rng(SPEC.bootstrap_seed);sample=rng.choice(np.array(deltas),size=(5000,len(deltas)),replace=True).mean(axis=1)
        actual=out['metrics']['AUROC'];self.assertAlmostEqual(actual['difference'],np.mean(deltas))
        np.testing.assert_allclose([actual['ci_low'],actual['ci_high']],np.quantile(sample,[.025,.975]),atol=1e-15)
        with self.assertRaises(ScopeError):hierarchy(rows[:-1])
        record('independent_bootstrap_AUROC',actual)
    def test_41_near_ties_and_fixed_control_report(self):
        from rel01.evaluation import near_tie_audit,counterfactual_summary,planned_model_comparisons
        a=self._evaluation_rows();b=[{**r,'logit':r['logit']+2.} for r in a]
        out=counterfactual_summary({k:[{**r,'control':k,'evaluation_kind':'COUNTERFACTUAL'} for r in rows] for k,rows in [('SELF',a),('NULL',b),('OTHER',b)]})
        self.assertEqual(set(out['absolute']),{'SELF','NULL','OTHER'})
        self.assertEqual(out['self_minus_control']['NULL']['paired']['metrics']['AP']['difference'],0)
        near=[{**r,'logit':float(i%3)*1e-8} for i,r in enumerate(a)]
        self.assertTrue(all(r['near_nonexact_pairs']==3 and r['exact_tie_pairs']==0 for r in near_tie_audit(near)))
        from rel01.routing import PRIMARY_CONTROLS
        comp=planned_model_comparisons({k:[{**r,'model':k,'control':PRIMARY_CONTROLS[k]} for r in a] for k in PRIMARY_CONTROLS});self.assertEqual(len(comp),5)

    def test_42_outer_training_diagnostics_without_test_selection(self):
        sc=outer_scope(self.scope);a=TrainingSession(self.data,sc,'M0',self.seed,outer_inner_epochs=[1]*4);a.run()
        self.assertIn('base_BCE',a.history[-1]['training']);self.assertNotIn('validation',a.history[-1])
        b=TrainingSession(self.data,sc,'M2',self.seed,a.query_artifact(),outer_inner_epochs=[0]*4)
        self.assertEqual(b.updates,0);self.assertIn('residual_p10_p50_p90',b.history[0]['training'])
        self.assertEqual(b.history[0]['training']['residual_weighted_mean'],0)
        self.assertEqual(b.history[0]['training']['BCE'],b.history[0]['training']['base_BCE'])
        self.assertTrue(all('validation' not in r for r in b.history))

    def _six_sessions(self):
        from rel01.pipeline import run_synthetic_six
        sc=outer_scope(self.scope)
        return run_synthetic_six(self.data,sc,self.seed,{k:[1]*4 for k in EXPECTED}),sc
    def test_43_primary_mode_defaults_and_actual_C1_other_forward(self):
        from rel01.routing import PRIMARY_CONTROLS
        sessions,sc=self._six_sessions();differences={}
        for branch,s in sessions.items():
            rows=s.predictions(sc.test_ids)
            self.assertTrue(all(r['control']==PRIMARY_CONTROLS[branch] and r['evaluation_kind']=='PRIMARY' and r['model']==branch for r in rows))
            if branch.startswith('C1'):
                roles=roles_for(self.data,sc,sc.test_ids);q=[r.query for r in roles];o=[s.other.resolve(self.data,sc,r) for r in roles]
                with torch.no_grad():expected=s.model(s.emb[q],s.psy[q],s.emb[o]).cpu().numpy()
                np.testing.assert_array_equal([r['logit'] for r in rows],expected)
                self.assertEqual([r['reference_input_id'] for r in rows],[self.data.ids[i] for i in o])
                alternate=s.predictions(sc.test_ids,'SELF',evaluation_kind='COUNTERFACTUAL')
                differences[branch]=max(abs(a['logit']-b['logit']) for a,b in zip(rows,alternate));self.assertGreater(differences[branch],1e-7)
        record('primary_controls',PRIMARY_CONTROLS);record('C1_other_vs_self_max_difference',differences)
    def test_44_explicit_counterfactual_and_query_mode_rejections(self):
        from rel01.routing import resolve_control
        for branch in ['C1-0','C1-1']:
            with self.assertRaises(ScopeError):resolve_control(branch,'SELF')
            self.assertEqual(resolve_control(branch,'SELF','COUNTERFACTUAL'),'SELF')
            with self.assertRaises(ScopeError):resolve_control(branch,None,'COUNTERFACTUAL')
        for branch in ['M0','M1']:
            for mode in ['SELF','OTHER','NULL']:
                with self.assertRaises(ScopeError):resolve_control(branch,mode)
                with self.assertRaises(ScopeError):resolve_control(branch,mode,'COUNTERFACTUAL')
        sessions,sc=self._six_sessions()
        for branch in ['C1-0','C1-1']:
            with self.assertRaises(ScopeError):sessions[branch].predictions(sc.test_ids,'SELF')
        with self.assertRaises(ScopeError):sessions['M2'].predictions(sc.test_ids,'SOURCE_OTHER',evaluation_kind='COUNTERFACTUAL')
    def test_45_evaluator_model_mode_kind_and_common_query_gates(self):
        from rel01.evaluation import checked_rows,planned_model_comparisons,counterfactual_summary
        from rel01.routing import PRIMARY_CONTROLS
        a=self._evaluation_rows()
        for field in ['model','control','evaluation_kind']:
            bad=copy.deepcopy(a);bad[0].pop(field)
            with self.assertRaises(ScopeError):hierarchy(bad)
        for changes in [{'model':'C1-0'},{'control':'OTHER'},{'model':'UNKNOWN'},{'evaluation_kind':'COUNTERFACTUAL'}]:
            bad=copy.deepcopy(a);bad[0].update(changes)
            with self.assertRaises(ScopeError):checked_rows(bad)
        good={k:[{**r,'model':k,'control':mode} for r in a] for k,mode in PRIMARY_CONTROLS.items()}
        self.assertEqual(len(planned_model_comparisons(good)),5)
        bad=copy.deepcopy(good);bad['C1-0']=[{**r,'control':'SELF','evaluation_kind':'COUNTERFACTUAL'} for r in bad['C1-0']]
        with self.assertRaises(ScopeError):planned_model_comparisons(bad)
        bad=copy.deepcopy(good);bad['M3']=bad['M2']
        with self.assertRaises(ScopeError):planned_model_comparisons(bad)
        bad=copy.deepcopy(good);bad['C1-1']=bad['C1-1'][:-1]
        with self.assertRaises(ScopeError):planned_model_comparisons(bad)
        bad=[{**r,'outer_fold':999} for r in a]
        with self.assertRaises(ScopeError):paired_bootstrap(a,bad)
        cf={k:[{**r,'model':'C1-0','control':k,'evaluation_kind':'COUNTERFACTUAL'} for r in a] for k in ['SELF','OTHER','NULL']}
        self.assertEqual(counterfactual_summary(cf)['status'],'DESCRIPTIVE')
        cf['OTHER']=[{**r,'model':'M2'} for r in cf['OTHER']]
        with self.assertRaises(ScopeError):counterfactual_summary(cf)
    def test_46_six_model_pipeline_common_roles_and_binding(self):
        from rel01.pipeline import primary_predictions
        from rel01.evaluation import require_common_queries
        sessions,sc=self._six_sessions();out=primary_predictions(sessions,sc.test_ids)
        self.assertEqual(out['unmatched_roles'],0);require_common_queries(*out['predictions'].values())
        for k in ['C1-0','C1-1']:self.assertTrue(all(r['control']=='OTHER' for r in out['predictions'][k]))
        bad=dict(sessions);bad['C1-0']=sessions['M2']
        with self.assertRaises(ScopeError):primary_predictions(bad,sc.test_ids)
        role=roles_for(self.data,sc,sc.test_ids)[0];saved=sessions['C1-0'].other.mapping[role.reference]
        for k in ['C1-0','C1-1']:sessions[k].other.mapping[role.reference]=None
        reduced=primary_predictions(sessions,sc.test_ids)
        self.assertGreater(reduced['unmatched_roles'],0);require_common_queries(*reduced['predictions'].values())
        for rr in reduced['predictions'].values():self.assertTrue(all(r['reference_id']!=self.data.ids[role.reference] for r in rr))

if __name__=='__main__':unittest.main()
