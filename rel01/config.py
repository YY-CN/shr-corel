from dataclasses import dataclass, asdict
import hashlib, json, os, random, platform
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=Path(__file__).resolve().parent/'protocol/implementation_contract.md'
class ImplementationBlocked(RuntimeError):pass
class ScopeError(ImplementationBlocked):pass
class RealDataForbidden(ImplementationBlocked):pass

@dataclass(frozen=True)
class Specification:
    version:str='TRV2_K1_DESIGN_REL01_V1_1'
    cls_dim:int=768
    core_dim:int=36
    ln_eps:float=1e-5
    norm_tolerance:float=1e-5
    center_min_norm:float=1e-12
    variance_floor:float=1e-12
    query_batch:int=32
    people_per_step:int=8
    queries_per_reference:int=4
    query_lr:float=3e-4
    relation_lr:float=1e-3
    weight_decay:float=1e-3
    adam_betas:tuple=(.9,.999)
    adam_eps:float=1e-8
    query_max_epochs:int=100
    relation_max_epochs:int=80
    query_patience:int=15
    relation_patience:int=12
    min_delta:float=1e-6
    grad_clip:float=1.0
    seeds:tuple=(20260722,20260723,20260724)
    other_seed:int=20260722
    bootstrap_seed:int=20260722
    bootstrap_replicates:int=5000

SPEC=Specification()
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def contract_sha():return sha_bytes(CONTRACT.read_bytes())
def canonical_hash(obj):return sha_bytes(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8'))
def specification_hash():return canonical_hash({'contract':contract_sha(),'implementation':asdict(SPEC)})
def code_fingerprint():return canonical_hash({p.name:sha_bytes(p.read_bytes()) for p in sorted((ROOT/'rel01').glob('*.py'))})
def runtime_binding():return {'torch':str(torch.__version__),'numpy':np.__version__,'python':platform.python_version(),
    'system':platform.system(),'machine':platform.machine(),'torch_threads':torch.get_num_threads(),'deterministic':torch.are_deterministic_algorithms_enabled(),
    'cuda_build':torch.version.cuda}
def derived_seed(seed,*parts):return int.from_bytes(hashlib.sha256('|'.join(map(str,(seed,*parts))).encode()).digest()[:8],'big')%(2**63-1)
def deterministic(seed,threads=1):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

