"""Immutable experiment bookkeeping and chunked, deterministic inference."""
from datetime import datetime,timezone
import hashlib,json,shutil
from pathlib import Path
import numpy as np
import torch
from .policies import positions


def sha(file):return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def dump(path,payload):Path(path).write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def begin(output,plan):
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    snapshot=output/'source_snapshot'
    shutil.copytree(Path(__file__).parent,snapshot,ignore=shutil.ignore_patterns('__pycache__'))
    plan={**plan,'created_at':datetime.now(timezone.utc).isoformat(),
          'source_sha256':{p.name:sha(p) for p in snapshot.glob('*.py')}}
    dump(output/'experiment_plan.json',plan)
    return plan


def freeze(output,selections,**extra):
    manifest={'frozen_at':datetime.now(timezone.utc).isoformat(),'selections':selections,
              'checkpoint_sha256':{str(p.relative_to(output)):sha(p) for p in output.rglob('*.pt')},
              'test_generated_before_freeze':False,**extra}
    dump(output/'frozen_before_test.json',manifest)
    return manifest


def save_batch(path,batch):np.savez_compressed(path,**{k:v.detach().cpu().numpy() for k,v in batch.items()})


def load_batch(path):
    with np.load(path) as values:return {k:torch.from_numpy(values[k]) for k in values.files}


def infer(model,batch,config,chunk=1024):
    values=[]
    with torch.no_grad():
        for start in range(0,len(batch['spot']),chunk):
            values.append(positions(model,{k:v[start:start+chunk] for k,v in batch.items()},config))
    return torch.cat(values)


def save_execution(path,result,h):
    np.savez_compressed(path,positions=h.detach().numpy(),**{k:v.detach().numpy() for k,v in result.items()})
