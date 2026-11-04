"""Fit-owned atomic latest-two recovery and verified final retirement."""
import hashlib,json,os,re,uuid
from pathlib import Path
import torch
from .config import ImplementationBlocked
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sync_dir(p):
    if os.name!='nt':
        fd=os.open(p,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
def atomic_json(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:json.dump(value,f,sort_keys=True);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path);sync_dir(path.parent)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def owned(directory,binding=None,create=False):
    p=Path(directory).absolute()
    if any(x.is_symlink() for x in [p,*p.parents]):raise ImplementationBlocked('checkpoint symlink escape')
    p.mkdir(parents=True,exist_ok=True);owner=p/'OWNER.json'
    if not owner.exists():
        if not create or any(p.iterdir()):raise ImplementationBlocked('unowned checkpoint directory')
        atomic_json(owner,{'format':'REL01_STORAGE_V1','binding':binding,'absolute_directory':str(p.resolve())})
    v=read(owner)
    if v.get('absolute_directory')!=str(p.resolve()) or (binding is not None and v['binding']!=binding):raise ImplementationBlocked('owner binding mismatch')
    return p,v['binding']
def payload_file(directory,pointer):
    name=pointer['payload']
    if not re.fullmatch(r'checkpoint-[a-f0-9]{64}\.pt',name):raise ImplementationBlocked('invalid checkpoint filename')
    p=directory/name
    if p.is_symlink() or p.resolve().parent!=directory.resolve() or p.stat().st_size!=pointer['bytes'] or sha(p)!=pointer['sha256']:raise ImplementationBlocked('checkpoint integrity mismatch')
    return p
def checked_payload(p,binding):
    if p.is_symlink() or not re.fullmatch(r'checkpoint-[a-f0-9]{64}\.pt',p.name) or sha(p)!=p.stem.split('-')[1]:raise ImplementationBlocked('unverified recovery file')
    x=torch.load(p,map_location='cpu',weights_only=False)
    if x['binding']!=binding:raise ImplementationBlocked('foreign fit checkpoint')
    return x
def save_transaction(directory,payload,fail_at=None):
    p,b=owned(directory,payload['binding'],True)
    if (p/'COMPLETE.json').exists():raise ImplementationBlocked('completed fit cannot create recovery')
    old=read(p/'latest.json') if (p/'latest.json').exists() else None
    history=old.get('history',[{k:old[k] for k in ['payload','bytes','sha256']}]) if old else []
    for h in history:payload_file(p,h)
    temp=p/('pending-'+uuid.uuid4().hex+'.tmp')
    with temp.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    digest=sha(temp);name='checkpoint-'+digest+'.pt';size=temp.stat().st_size;os.replace(temp,p/name);sync_dir(p)
    if fail_at=='after_payload':raise InterruptedError('after payload before pointer')
    item={'payload':name,'bytes':size,'sha256':digest};history=([item]+[h for h in history if h['payload']!=name])[:2]
    pointer={**item,'history':history};atomic_json(p/'latest.json',pointer)
    if fail_at=='after_pointer':raise InterruptedError('after pointer before retention')
    keep={h['payload'] for h in history};candidates=[x for x in p.glob('checkpoint-*.pt') if x.name not in keep]
    for x in candidates:checked_payload(x,b)
    removed=[]
    for x in candidates:removed.append({'file':x.name,'sha256':sha(x),'bytes':x.stat().st_size});x.unlink()
    if removed:
        rows=read(p/'RETENTION_LOG.json') if (p/'RETENTION_LOG.json').exists() else []
        atomic_json(p/'RETENTION_LOG.json',rows+[{'committed':name,'removed':removed}]);sync_dir(p)
    return pointer
def load_transaction(directory,device='cpu'):
    p,b=owned(directory)
    if (p/'COMPLETE.json').exists():raise ImplementationBlocked('fit completed; load final instead')
    pointer=read(p/'latest.json');x=torch.load(payload_file(p,pointer),map_location=device,weights_only=False)
    if x['binding']!=b:raise ImplementationBlocked('recovery owner mismatch')
    return x
def load_final(directory,binding=None):
    p,b=owned(directory,binding);receipt=read(p/'COMPLETE.json');f=p/'FINAL.pt'
    if f.is_symlink() or f.resolve().parent!=p.resolve() or sha(f)!=receipt['sha256'] or f.stat().st_size!=receipt['bytes'] or receipt['binding']!=b:raise ImplementationBlocked('final verification failed')
    x=torch.load(f,map_location='cpu',weights_only=False)
    if x['binding']!=b or 'optimizer' in x or not x['done']:raise ImplementationBlocked('invalid final state')
    return x
def finish(directory,final,fail_at=None):
    p,b=owned(directory,final['binding'],True)
    if not final['done'] or 'optimizer' in final:raise ImplementationBlocked('unfinished fit cannot finalize')
    if not (p/'COMPLETE.json').exists():
        tmp=p/'FINAL.writing'
        with tmp.open('wb') as f:torch.save(final,f);f.flush();os.fsync(f.fileno())
        os.replace(tmp,p/'FINAL.pt');sync_dir(p)
        if fail_at=='after_final_payload':raise InterruptedError('after final payload')
        atomic_json(p/'COMPLETE.json',{'binding':b,'sha256':sha(p/'FINAL.pt'),'bytes':(p/'FINAL.pt').stat().st_size})
    result=load_final(p,b)
    if fail_at=='after_complete':raise InterruptedError('after complete')
    pending=p/'CLEANUP_PENDING.json'
    if not pending.exists():
        files=list(p.glob('checkpoint-*.pt'))
        for f in files:checked_payload(f,b)
        atomic_json(pending,{'binding':b,'final_sha256':sha(p/'FINAL.pt'),'files':[{'file':f.name,'sha256':sha(f),'bytes':f.stat().st_size} for f in files]})
    cleanup=read(pending)
    if cleanup['binding']!=b or cleanup['final_sha256']!=sha(p/'FINAL.pt'):raise ImplementationBlocked('cleanup identity changed')
    if (p/'latest.json').exists():os.replace(p/'latest.json',p/'LATEST_RETIRED.json');sync_dir(p)
    if fail_at=='after_retire':raise InterruptedError('after retirement')
    for item in cleanup['files']:
        f=p/item['file']
        if not re.fullmatch(r'checkpoint-[a-f0-9]{64}\.pt',item['file']) or f.resolve().parent!=p.resolve() or f.is_symlink():raise ImplementationBlocked('cleanup path escape')
        if f.exists():
            if sha(f)!=item['sha256'] or f.stat().st_size!=item['bytes']:raise ImplementationBlocked('cleanup file changed')
            f.unlink();sync_dir(p)
        if fail_at=='during_cleanup':raise InterruptedError('during cleanup')
    atomic_json(p/'CLEANUP_COMPLETE.json',{'binding':b,'final_sha256':sha(p/'FINAL.pt'),'released_bytes':sum(x['bytes'] for x in cleanup['files'])})
    return result
