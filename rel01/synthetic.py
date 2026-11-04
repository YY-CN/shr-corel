import numpy as np
from .data import Answers,Scope

def synthetic_answers(seed=701,fit_people=12,validation_people=4,test_people=4,vary_counts=False):
    rng=np.random.default_rng(seed);ids=[];pids=[];labels=[];length=[];sources=[]
    total=fit_people+validation_people+test_people
    people=[f'SYN_P{i:03d}' for i in range(total)]
    for i,p in enumerate(people):
        t=2+(i%2 if vary_counts else 0);l=2+(i%3 if vary_counts else 0)
        for j in range(t+l):
            ids.append(f'{p}_A{j:02d}');pids.append(p);labels.append(int(j>=t));length.append(5+(i*3+j)%23);sources.append(f'SYN_SOURCE_{(i+j)%4}')
    n=len(ids);e=rng.normal(size=(n,768));e/=np.linalg.norm(e,axis=1,keepdims=True)
    values=rng.normal(size=(n,36));mask=rng.random((n,36))>.15
    values[mask&(rng.random((n,36))<.3)]=0.;values[~mask]=np.nan
    reason=(~mask).astype(np.int64)
    data=Answers('SYN_DATA',tuple(ids),tuple(pids),np.array(labels),e.astype(np.float32),values,mask,reason,np.array(length),tuple(sources),'SYNTHETIC')
    scope=Scope(data.dataset,0,0,tuple(people[:fit_people]),tuple(people[fit_people:fit_people+validation_people]),tuple(people[fit_people+validation_people:]))
    return data,scope

