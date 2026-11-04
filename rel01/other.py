from dataclasses import dataclass
import hashlib
import numpy as np
from .config import ScopeError, SPEC, canonical_hash
from .data import roles_for

@dataclass
class OtherMap:
    mapping:dict
    records:list
    scope_hash:str
    data_hash:str
    mode:str
    @property
    def sha(self):return canonical_hash({'records':self.records,'scope':self.scope_hash,'data':self.data_hash,'mode':self.mode})
    def resolve(self,data,scope,role):
        if self.scope_hash!=scope.fingerprint or self.data_hash!=data.fingerprint:raise ScopeError('OTHER artifact binding mismatch')
        if data.participants[role.query]!=role.participant or data.participants[role.reference]!=role.participant or role.query==role.reference or data.labels[role.reference]!=0:raise ScopeError('illegal original SELF role')
        if role.participant not in set(scope.fit_ids+scope.evaluation_ids):raise ScopeError('OTHER target outside scope')
        o=self.mapping.get(role.reference)
        if o is None:return None
        if data.participants[o] not in scope.fit_ids or data.participants[o]==role.participant or data.labels[o]!=0 or o==role.query:raise ScopeError('illegal OTHER donor identity/label')
        return o

def build_other(data,scope,people=None,same_source=False,length_cap=None):
    scope.check(data);people=tuple(scope.fit_ids+scope.evaluation_ids) if people is None else tuple(people)
    roles=roles_for(data,scope,people);refs=sorted({z.reference for z in roles},key=lambda i:data.ids[i])
    pool=scope.indices(data);pool=pool[data.labels[pool]==0];records=[];mapping={}
    for r in refs:
        p=data.participants[r]
        candidates=[]
        for o in pool:
            if data.participants[o]==p:continue
            if same_source and data.sources[o]!=data.sources[r]:continue
            gap=abs(float(np.log1p(data.nlex[o])-np.log1p(data.nlex[r])))
            if length_cap is not None and gap>length_cap:continue
            tie=hashlib.sha256(f'{SPEC.other_seed}|{scope.key}|{p}|{data.ids[r]}|{data.ids[o]}'.encode()).hexdigest()
            candidates.append((gap,tie,data.ids[o],int(o)))
        candidates.sort();o=candidates[0][3] if candidates else None;mapping[r]=o
        records.append({'scope':scope.key,'participant_id':p,'reference_id':data.ids[r],
            'other_id':data.ids[o] if o is not None else None,'other_participant_id':data.participants[o] if o is not None else None,
            'source_id':data.sources[r],'other_source_id':data.sources[o] if o is not None else None,
            'reference_length':int(data.nlex[r]),'other_length':int(data.nlex[o]) if o is not None else None,
            'candidate_count':len(candidates),'log_length_gap':candidates[0][0] if candidates else None,
            'tie_sha256':candidates[0][1] if candidates else None,'status':'MATCHED' if candidates else 'UNMATCHED'})
    return OtherMap(mapping,records,scope.fingerprint,data.fingerprint,'SOURCE_LENGTH' if same_source else 'LENGTH')

