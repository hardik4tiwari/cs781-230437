"""Distributed feature/logit caches; full-document CRF decoding without sentence omissions."""
import argparse,contextlib,json,os
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from rr_data import Blocks,collate,to_device,atomic_torch,atomic_json,digest,file_hash,metric,read_json
from rr_models import build_model,CRF
def setup():
    rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    device=torch.device('cuda',local) if torch.cuda.is_available() else torch.device('cpu')
    if device.type=='cuda':torch.cuda.set_device(local)
    if world>1 and not dist.is_initialized():
        if device.type=='cuda':dist.init_process_group('nccl',device_id=device)
        else:dist.init_process_group('gloo')
    return rank,world,device
def autocast(device):return torch.autocast('cuda',dtype=torch.float16) if device.type=='cuda' else contextlib.nullcontext()
@torch.inference_mode()
def sentence_features(model,cache,did,device):
    doc=cache['docs'][did];n=len(doc['text'])
    if not n:return torch.zeros(0,model.pool_dim)
    local=dict(cache);local['infer_blocks']=[(did,0,n)];dataset=Blocks(local,'infer')
    batch=to_device(collate([dataset[0]]),device)
    with autocast(device):values=model.encode(batch)[0]
    return values.float().cpu()
@torch.inference_mode()
def logits_for_documents(model,cache,ids,device,features=None):
    model.eval();output={}
    for did in ids:
        n=len(cache['docs'][did]['text'])
        if not n:output[did]=torch.zeros(0,len(cache['labels']));continue
        vector=features[did].float() if features is not None else sentence_features(model,cache,did,device)
        local=dict(cache);local['infer_blocks']=[row for row in cache['all_blocks']+cache['test_blocks'] if row[0]==did]
        dataset=Blocks(local,'infer',features={did:vector});emissions=np.zeros((n,len(cache['labels'])),np.float32);counts=np.zeros(n,np.float32)
        for first in range(0,len(dataset),model.cfg.get('head_batch',8)):
            batch=to_device(collate([dataset[i] for i in range(first,min(first+model.cfg.get('head_batch',8),len(dataset)))]),device)
            with autocast(device):scores=model(batch)['logits'].float().cpu().numpy()
            for j,row in enumerate(batch['rows']):
                a,b=row['start'],row['end'];length=b-a
                weight=np.minimum(np.arange(1,length+1),np.arange(length,0,-1)).astype(np.float32)
                weight=np.minimum(weight,cache['block_overlap']+1)
                emissions[a:b]+=scores[j,:length]*weight[:,None];counts[a:b]+=weight
        if np.any(counts<=0):raise ValueError('Inference omitted sentences.')
        output[did]=torch.from_numpy(emissions/counts[:,None])
    return output
def decode(logits,arrays,labels,calibration=None):
    cal=calibration or {'transition_scale':1.,'bias':[0.]*len(labels)}
    return {did:[labels[i] for i in CRF.decode(values.float().numpy() if isinstance(values,torch.Tensor) else values,
        arrays,cal['transition_scale'],cal['bias'])] for did,values in logits.items()}
def select_calibration(logits,arrays,cache):
    labels=cache['labels'];truth={did:cache['docs'][did]['gold'] for did in cache['splits']['dev']}
    counts=np.array([cache['support']['train'][label]+1 for label in labels],np.float32)
    log_prior=np.log(counts/counts.sum());best=None
    for scale in (1.,.5,0.):
        for amount in (0.,.15,.30):
            bias=(-amount*(log_prior-log_prior.mean())).tolist();cal={'transition_scale':scale,'bias':bias}
            pred=decode(logits,arrays,labels,cal);score=metric(pred,truth,labels)
            if best is None or score['macro_f1']>best[0]['macro_f1']+1e-9:best=(score,cal,pred)
    return best
def domain_metrics(pred,cache):
    output={}
    for domain in ('CL','IT'):
        truth={did:cache['docs'][did]['gold'] for did in pred if cache['docs'][did]['domain']==domain}
        output[domain]=metric({did:pred[did] for did in truth},truth,cache['labels'])
    return output
def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=['features','logits'],required=True)
    p.add_argument('--config');p.add_argument('--checkpoint');p.add_argument('--cache',required=True)
    p.add_argument('--split',choices=['train','dev','all','test','every'],default='every');p.add_argument('--output',required=True)
    args=p.parse_args();rank,world,device=setup();cache=torch.load(args.cache,map_location='cpu',weights_only=False)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False) if args.checkpoint else None
    cfg=saved['config'] if saved else json.loads(Path(args.config).read_text());model,_=build_model(cfg,args.checkpoint);model.to(device).eval()
    if saved and saved['cache_fingerprint']!=cache['fingerprint']:raise ValueError('Logit cache data mismatch.')
    ids=list(cache['docs']) if args.split=='every' else cache['splits'][args.split]
    origin=file_hash(args.checkpoint) if args.checkpoint else digest([file_hash(cfg['parent']),cfg.get('student_layers') if cfg['method']=='kd' else None,cfg.get('token_bidirectional',False)])
    identity=digest([args.action,origin,cache['fingerprint'],ids,world])
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True);path=root/f'part_{rank}.pt'
    marker=root/'complete.json'
    if marker.exists() and (root/'complete.pt').exists():
        if read_json(marker)['identity']!=identity:raise ValueError('Completed feature/logit cache identity changed.')
        if rank==0:print('Already cached:',root,flush=True)
        if world>1:dist.destroy_process_group()
        return
    previous=torch.load(path,map_location='cpu',weights_only=False) if path.exists() else {'identity':identity,'values':{}}
    if previous['identity']!=identity:raise ValueError('Feature/logit cache identity changed. Use a new PROJECT directory.')
    values=previous['values']
    pending=0
    for did in ids[rank::world]:
        if did in values:continue
        value=sentence_features(model,cache,did,device) if args.action=='features' else logits_for_documents(model,cache,[did],device)[did]
        if not bool(torch.isfinite(value).all()):raise FloatingPointError('Non-finite cached representation.')
        values[did]=value.half();pending+=1
        if pending>=8:atomic_torch(path,{'identity':identity,'values':values});pending=0
    atomic_torch(path,{'identity':identity,'values':values})
    if world>1:dist.barrier()
    if rank==0:
        merged={}
        for i in range(world):merged.update(torch.load(root/f'part_{i}.pt',map_location='cpu',weights_only=False)['values'])
        if set(merged)!=set(ids):raise ValueError('Cache is incomplete.')
        payload={'identity':identity,'values':merged,'cache_fingerprint':cache['fingerprint'],'source_sha256':origin,
            'split':args.split,'kind':args.action,'crf':model.crf.arrays() if args.action=='logits' else None,'complete':True}
        atomic_torch(root/'complete.pt',payload)
        atomic_json(marker,{'identity':identity,'complete':True,'documents':len(ids)})
        for i in range(world):(root/f'part_{i}.pt').unlink(missing_ok=True)
    if world>1:dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
