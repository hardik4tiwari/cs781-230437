"""Long-document inference, fixed BIO constraints, character boundaries, teacher caches."""
import argparse, contextlib, json, os, time, zipfile
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from ner_data import NERWindows, collate_windows, atomic_json, file_hash, digest, validate_spans, strict_metrics
from ner_models import build_model

def device_setup():
    rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    device=torch.device('cuda',local) if torch.cuda.is_available() else torch.device('cpu')
    if device.type=='cuda':torch.cuda.set_device(local)
    if world>1 and not dist.is_initialized():dist.init_process_group('nccl' if device.type=='cuda' else 'gloo')
    return rank,world,device

def autocast(device):
    return torch.autocast('cuda',dtype=torch.float16) if device.type=='cuda' else contextlib.nullcontext()

def bio_viterbi(logits):
    """Fixed legality constraints, not a learned CRF. Input: tokens x (O,B1,I1,...)."""
    n,c=logits.shape
    if not n:return []
    transition=np.zeros((c,c),dtype=np.float32)
    for curr in range(2,c,2):
        transition[:,curr]=-1e9;transition[curr-1,curr]=0;transition[curr,curr]=0
    score=logits[0].copy();score[2::2]=-1e9;back=[]
    for row in logits[1:]:
        options=score[:,None]+transition;previous=options.argmax(0)
        score=options[previous,np.arange(c)]+row;back.append(previous)
    tag=int(score.argmax());tags=[tag]
    for previous in reversed(back):tag=int(previous[tag]);tags.append(tag)
    return tags[::-1]

def character_span(offsets,i,j,label,left,right,labels):
    a,b=offsets[i];c,d=offsets[j]
    # The boundary head predicts trims within the original BPE token's character span.
    l=int(np.argmax(left[i,label,:b-a]));r=int(np.argmax(right[j,label,:d-c]))
    start,end=a+l,d-r
    return [start,end,labels[label]] if start<end else None

def decode_bio(logits,offsets,left,right,labels):
    tags=bio_viterbi(logits);spans=[];i=0
    while i<len(tags):
        tag=tags[i]
        if tag==0:i+=1;continue
        label=(tag-1)//2;j=i
        while j+1<len(tags) and tags[j+1]==2+2*label:j+=1
        span=character_span(offsets,i,j,label,left,right,labels)
        if span is not None:spans.append(span)
        i=j+1
    return sorted({tuple(s) for s in spans})

@torch.inference_mode()
def predict_documents(model,cache,doc_ids,batch_size,device,threshold=0.,progress=None):
    model.eval();predictions={};labels=cache['labels'];L=len(labels);C=cache['char_classes']
    for did in doc_ids:
        if progress is not None and did in progress:predictions[did]=progress[did];continue
        doc=cache['encoded'][did];n=len(doc['ids'])
        if not n:predictions[did]=[];continue
        # One document at a time bounds RAM; all windows cover the entire input.
        windows=[w for w in cache['test_windows']+cache['all_windows'] if w['did']==did]
        local=dict(cache);local['infer_windows']=windows;ds=NERWindows(local,'infer',model.cfg['head'])
        if model.cfg['head']=='bio':
            emissions=np.zeros((n,1+2*L),np.float32);left=np.zeros((n,L,C),np.float32);right=np.zeros_like(left);counts=np.zeros(n,np.float32)
        candidates={}
        for start in range(0,len(ds),batch_size):
            batch=collate_windows([ds[i] for i in range(start,min(start+batch_size,len(ds)))])
            with autocast(device):out=model(batch['ids'].to(device),batch['valid'].to(device))
            values={k:v.float().cpu().numpy() for k,v in out.items() if k!='span_valid'}
            for row,item in enumerate(batch['items']):
                w=item['window'];a,b=w['start'],w['end'];length=b-a
                weight=np.minimum(np.arange(1,length+1),np.arange(length,0,-1)).astype(np.float32)
                weight=np.minimum(weight,cache['overlap']+1)
                if model.cfg['head']=='bio':
                    emissions[a:b]+=values['logits'][row,:length]*weight[:,None]
                    left[a:b]+=values['left'][row,:length]*weight[:,None,None];right[a:b]+=values['right'][row,:length]*weight[:,None,None]
                    counts[a:b]+=weight
                else:
                    scores=values['logits'][row,:,:length,:length];hits=np.argwhere(scores>threshold)
                    # A computational guard, recorded in config, rather than inventing entities.
                    limit=model.cfg.get('max_spans_per_window',1024)
                    if len(hits)>limit:
                        hit_scores=scores[tuple(hits.T)];hits=hits[np.argpartition(hit_scores,-limit)[-limit:]]
                    for label,i,j in hits:
                        span=character_span(item['offsets'],int(i),int(j),int(label),values['left'][row],values['right'][row],labels)
                        if span is not None:
                            key=tuple(span);candidates[key]=max(candidates.get(key,-1e9),float(scores[label,i,j]))
        if model.cfg['head']=='bio':
            if np.any(counts==0):raise ValueError('Inference omitted tokens.')
            spans=decode_bio(emissions/counts[:,None],doc['offsets'],left/counts[:,None,None],right/counts[:,None,None],labels)
        else:spans=sorted(candidates)
        predictions[did]=[list(s) for s in spans]
        if progress is not None:progress[did]=predictions[did]
    return predictions

@torch.inference_mode()
def cache_teacher(model,cache,split,output,device,batch_size,checkpoint_hash):
    if split not in ('train','all') or model.cfg['head']!='bio':raise ValueError('KD uses only labelled training windows and a BIO teacher.')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    identity={'cache':cache['fingerprint'],'teacher_sha256':checkpoint_hash,'split':split,'windows':digest(cache[split+'_windows'])}
    manifest=output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['identity']!=identity:raise ValueError('Teacher cache identity mismatch.')
    atomic_json(manifest,{'identity':identity,'complete':False});ds=NERWindows(cache,split,'bio');model.eval()
    for start in range(0,len(ds),batch_size):
        indices=[i for i in range(start,min(start+batch_size,len(ds))) if not (output/f'{i:06d}.npz').exists()]
        if not indices:continue
        batch=collate_windows([ds[i] for i in indices])
        with autocast(device):logits=model(batch['ids'].to(device),batch['valid'].to(device))['logits'].half().cpu().numpy()
        for row,index in enumerate(indices):
            target=output/f'{index:06d}.npz';temp=target.with_suffix('.tmp.npz')
            np.savez_compressed(temp,logits=logits[row,:len(batch['items'][row]['ids'])]);temp.replace(target)
    for i in range(len(ds)):
        values=np.load(output/f'{i:06d}.npz')['logits']
        if values.shape!=(len(ds[i]['ids']),1+2*len(cache['labels'])) or not np.isfinite(values).all():raise ValueError('Incomplete/invalid teacher logits.')
    atomic_json(manifest,{'identity':identity,'complete':True,'windows':len(ds)})

def submission_zip(predictions,texts,labels,path):
    validate_spans(predictions,texts,labels);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(predictions,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
    temp=path.with_suffix('.tmp.zip')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as archive:
        info=zipfile.ZipInfo('prediction.json',date_time=(2026,10,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
        archive.writestr(info,data)
    temp.replace(path)
    with zipfile.ZipFile(path) as archive:
        assert archive.namelist()==['prediction.json'];validate_spans(json.loads(archive.read('prediction.json')),texts,labels)
    return digest(predictions)

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--cache',required=True)
    p.add_argument('--output',required=True);p.add_argument('--split',choices=['test','val','train','all'],default='test')
    p.add_argument('--teacher-cache',action='store_true');p.add_argument('--batch-size',type=int,default=4)
    args=p.parse_args();rank,world,device=device_setup()
    cache=torch.load(args.cache,map_location='cpu',weights_only=False);saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model,_=build_model(saved['config'],cache['labels'],cache['char_classes'],cache['tokenizer_sha256'],args.checkpoint);model.to(device)
    if saved['cache_fingerprint']!=cache['fingerprint']:raise ValueError('Prediction checkpoint/data mismatch.')
    if args.teacher_cache:
        if rank==0:cache_teacher(model,cache,args.split,args.output,device,args.batch_size,file_hash(args.checkpoint))
    else:
        ids=cache['split']['val_ids'] if args.split=='val' else sorted({w['did'] for w in cache[args.split+'_windows']})
        if args.split=='test':ids=[k for k in cache['texts'] if k not in cache['gold']]
        root=Path(args.output);root.mkdir(parents=True,exist_ok=True);partial=root/f'part_{rank}.json'
        identity=digest([file_hash(args.checkpoint),cache['fingerprint'],args.split,ids,world])
        old=json.loads(partial.read_text()) if partial.exists() else {'identity':identity,'predictions':{}}
        if old['identity']!=identity:raise ValueError('Prediction resume identity changed; use a new output directory.')
        progress=old['predictions']
        for did in ids[rank::world]:
            progress.update(predict_documents(model,cache,[did],args.batch_size,device,progress=progress))
            atomic_json(partial,{'identity':identity,'predictions':progress})
        if world>1:dist.barrier()
        if rank==0:
            predictions={}
            for i in range(world):predictions.update(json.loads((root/f'part_{i}.json').read_text())['predictions'])
            texts={k:cache['texts'][k] for k in ids};validate_spans(predictions,texts,cache['labels'])
            atomic_json(root/'prediction.json',predictions)
            if args.split=='test':submission_zip(predictions,texts,cache['labels'],root/'submission.zip')
            if args.split=='val':atomic_json(root/'metrics.json',strict_metrics(predictions,{k:cache['gold'][k] for k in ids},cache['labels']))
    if world>1:dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
