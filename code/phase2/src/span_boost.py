"""Own-foundation bidirectional span NER; imported only by the additional cell."""
import argparse, bisect, json, math, os, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from ner_models import NERModel, load_foundation, ner_loss as original_loss
from ner_data import NERWindows, collate_windows, windows_for, digest, atomic_json, file_hash
from ner_predict import autocast, character_span

def setup_device():
    rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    device=torch.device('cuda',local) if torch.cuda.is_available() else torch.device('cpu')
    if device.type=='cuda':torch.cuda.set_device(local)
    if world>1 and not dist.is_initialized():
        if device.type=='cuda':dist.init_process_group('nccl',device_id=device)
        else:dist.init_process_group('gloo')
    return rank,world,device

class SpanWindows(NERWindows):
    def __getitem__(self,index):
        item=super().__getitem__(index);w=item['window']
        item['tags']=torch.from_numpy(self.cache['aligned'][w['did']]['tags'][w['start']:w['end']].copy())
        return item

class BidirectionalSpanNER(NERModel):
    def __init__(self,backbone,cfg,labels,char_classes):
        super().__init__(backbone,cfg,labels,char_classes)
        if backbone.config.architecture!='decoder' or self.two_directions or cfg['head']!='span':
            raise ValueError('This extension expects the original decoder, one bidirectional pass, and a span head.')
        for attention in self.attention_modules:
            attention.causal=False;attention.window=0;attention.dropout=cfg.get('backbone_dropout',.1)
        for module in self.backbone.modules():
            if isinstance(module,nn.Dropout):module.p=cfg.get('backbone_dropout',.1)
        self.aux_bio=nn.Linear(backbone.config.dim,1+2*len(labels))

    def forward(self,ids,valid):
        h=self.features(ids,valid);b,t,_=h.shape;L=len(self.labels);C=self.char_classes
        d=self.cfg.get('span_dim',64)
        pairs=self.span_projection(h).view(b,t,L,2,d)
        q=self.span_rotary(pairs[:,:,:,0].transpose(1,2))
        k=self.span_rotary(pairs[:,:,:,1].transpose(1,2))
        scores=torch.einsum('blid,bljd->blij',q,k)/math.sqrt(d)
        i=torch.arange(t,device=ids.device)[:,None];j=torch.arange(t,device=ids.device)[None,:]
        allowed=((j>=i)&(j-i<self.cfg.get('max_span_tokens',t)))[None,None]
        allowed=allowed&valid[:,None,:,None]&valid[:,None,None,:]
        return {'logits':scores.masked_fill(~allowed,-1e4),'span_valid':allowed,
                'left':self.left(h).view(b,t,L,C),'right':self.right(h).view(b,t,L,C),
                'aux_bio':self.aux_bio(h)}

def build_boost(cfg,labels,char_classes,tokenizer_hash,restore=None):
    backbone,provenance=load_foundation(cfg['phase1_checkpoint'],tokenizer_hash,'decoder',allow_tiny=cfg.get('allow_tiny',False))
    backbone.config.gradient_checkpointing=cfg.get('gradient_checkpointing',False)
    model=BidirectionalSpanNER(backbone,cfg,labels,char_classes)
    if restore:model.load_state_dict(torch.load(restore,map_location='cpu',weights_only=False)['model'],strict=True)
    return model,provenance

def boost_loss(result,batch,cfg,weights):
    aux=F.cross_entropy(result['aux_bio'].float().flatten(0,1),batch['tags'].flatten(),
                        weight=weights,ignore_index=-100,label_smoothing=.03)
    return original_loss(result,batch,cfg,weights)+cfg.get('aux_bio_weight',.3)*aux

def trim_whitespace(text,start,end):
    while start<end and text[start].isspace():start+=1
    while end>start and text[end-1].isspace():end-=1
    return start,end

@torch.inference_mode()
def raw_documents(model,cache,doc_ids,batch_size,device,floor=-2.):
    model.eval();result={};labels=cache['labels'];cap=model.cfg.get('candidate_cap',4096)
    for did in doc_ids:
        doc=cache['encoded'][did];text=cache['texts'][did]
        local=dict(cache);local['infer_windows']=windows_for(cache['encoded'],[did],cache['seq_len'],cache['overlap'])
        dataset=NERWindows(local,'infer','span');maps={'learned':{},'whitespace':{}}
        for first in range(0,len(dataset),batch_size):
            batch=collate_windows([dataset[j] for j in range(first,min(first+batch_size,len(dataset)))])
            with autocast(device):out=model(batch['ids'].to(device),batch['valid'].to(device))
            for row,item in enumerate(batch['items']):
                n=len(item['ids']);scores=out['logits'][row,:,:n,:n].float()
                hits=(scores>floor).nonzero()
                if hits.size(0)>cap:
                    values=scores[hits[:,0],hits[:,1],hits[:,2]]
                    hits=hits[values.topk(cap).indices]
                values=scores[hits[:,0],hits[:,1],hits[:,2]].cpu().tolist()
                hits=hits.cpu().tolist()
                left=out['left'][row,:n].float().cpu().numpy();right=out['right'][row,:n].float().cpu().numpy()
                offsets=item['offsets']
                for (label,i,j),score in zip(hits,values):
                    span=character_span(offsets,i,j,label,left,right,labels)
                    if span is not None:
                        key=tuple(span);maps['learned'][key]=max(maps['learned'].get(key,-1e9),score)
                    a,b=trim_whitespace(text,offsets[i][0],offsets[j][1])
                    if a<b:
                        key=(a,b,labels[label]);maps['whitespace'][key]=max(maps['whitespace'].get(key,-1e9),score)
        result[did]={mode:[[*key,float(value)] for key,value in sorted(candidates.items())] for mode,candidates in maps.items()}
    return result

def suppress(spans,allow_nested=False):
    unique={}
    for s in spans:
        key=tuple(s[:3]);unique[key]=max(unique.get(key,-1e9),float(s[3]))
    ordered=sorted(unique,key=lambda key:(-unique[key],key[1]-key[0],key))
    if allow_nested:return sorted(ordered)
    chosen=[];starts=[];ends=[]
    for key in ordered:
        a,b,_=key;pos=bisect.bisect_left(starts,a)
        if (pos and ends[pos-1]>a) or (pos<len(starts) and starts[pos]<b):continue
        starts.insert(pos,a);ends.insert(pos,b);chosen.append(key)
    return sorted(chosen)

def training_nested(cache,ids):
    flags={label:False for label in cache['labels']}
    for did in ids:
        for label in cache['labels']:
            spans=sorted({tuple(e[:2]) for e in cache['gold'][did] if e[2]==label})
            end=-1
            for a,b in spans:
                if a<end:flags[label]=True
                end=max(end,b)
    return flags

def decode_raw(raw,labels,calibration):
    mode=calibration['boundary_mode'];thresholds=calibration['thresholds'];nested=calibration['allow_nested']
    predictions={}
    for did,options in raw.items():
        rows=options[mode];entities=[]
        for label in labels:
            candidates=[s for s in rows if s[2]==label and s[3]>thresholds[label]]
            entities.extend(suppress(candidates,nested.get(label,False)))
        predictions[did]=[list(e) for e in sorted(set(entities))]
    return predictions

def default_calibration(cache):
    return {'boundary_mode':'learned','thresholds':{label:0. for label in cache['labels']},
            'allow_nested':training_nested(cache,cache['split']['train_ids'])}

def predict_boost(model,cache,doc_ids,batch_size,device,**kwargs):
    raw=raw_documents(model,cache,doc_ids,batch_size,device,floor=0.)
    return decode_raw(raw,cache['labels'],default_calibration(cache))

def blend_raw(all_raw,floor=-2.):
    if not all_raw:return {}
    if any(set(raw)!=set(all_raw[0]) for raw in all_raw):raise ValueError('Ensemble document IDs differ.')
    merged={};count=len(all_raw)
    for did in all_raw[0]:
        merged[did]={}
        for mode in ('learned','whitespace'):
            sums={}
            for raw in all_raw:
                for row in raw[did][mode]:
                    key=tuple(row[:3]);sums[key]=sums.get(key,floor*count)+(float(row[3])-floor)
            merged[did][mode]=[[*key,value/count] for key,value in sorted(sums.items())]
    return merged

def calibrated_decoder(raw,cache,grid=(0.,.5,-.5,1.,-1.,1.5,2.),min_support=5):
    from ner_data import strict_metrics
    labels=cache['labels'];gold={did:cache['gold'][did] for did in cache['split']['val_ids']}
    nested=training_nested(cache,cache['split']['train_ids']);best=None
    for mode in ('learned','whitespace'):
        for threshold in grid:
            cal={'boundary_mode':mode,'thresholds':dict.fromkeys(labels,threshold),'allow_nested':nested}
            pred=decode_raw(raw,labels,cal);metrics=strict_metrics(pred,gold,labels)
            quality=(metrics['macro_f1'],metrics['macro_precision'],-sum(len(e) for e in pred.values()))
            if best is None or quality>best[0]:best=(quality,cal,metrics)
    cal=best[1];counts={l:sum(e[2]==l for spans in gold.values() for e in spans) for l in labels}
    for label in labels:
        if counts[label]<min_support:continue
        one_gold={did:[e for e in spans if e[2]==label] for did,spans in gold.items()}
        best_quality=None;chosen=cal['thresholds'][label]
        for threshold in grid:
            one=dict(cal);one['thresholds']={**cal['thresholds'],label:threshold}
            pred=decode_raw(raw,[label],one)
            score=strict_metrics(pred,one_gold,[label])
            quality=(score['macro_f1'],score['macro_precision'],-sum(len(e) for e in pred.values()))
            if best_quality is None or quality>best_quality:best_quality=quality;chosen=threshold
        cal['thresholds'][label]=chosen
    predictions=decode_raw(raw,labels,cal)
    return cal,strict_metrics(predictions,gold,labels),predictions

def main():
    if len(sys.argv)>1 and sys.argv[1]=='train':
        sys.argv=[sys.argv[0],*sys.argv[2:]]
        import train_ner
        train_ner.build_model=build_boost;train_ner.NERWindows=SpanWindows
        train_ner.ner_loss=boost_loss;train_ner.predict_documents=predict_boost;train_ner.device_setup=setup_device
        train_ner.main();return
    p=argparse.ArgumentParser();p.add_argument('action',choices=['raw'])
    p.add_argument('--checkpoint',required=True);p.add_argument('--cache',required=True)
    p.add_argument('--split',choices=['val','test'],required=True);p.add_argument('--output',required=True)
    p.add_argument('--batch-size',type=int,default=2);args=p.parse_args()
    rank,world,device=setup_device();cache=torch.load(args.cache,map_location='cpu',weights_only=False)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if saved['cache_fingerprint']!=cache['fingerprint']:raise ValueError('Checkpoint/cache mismatch.')
    model,_=build_boost(saved['config'],cache['labels'],cache['char_classes'],cache['tokenizer_sha256'],args.checkpoint)
    model.to(device);root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    ids=cache['split']['val_ids'] if args.split=='val' else sorted(set(cache['texts'])-set(cache['gold']))
    identity=digest([file_hash(args.checkpoint),cache['fingerprint'],args.split,world,'raw-v1'])
    path=root/f'part_{rank}.json'
    saved_raw=json.loads(path.read_text()) if path.exists() else {'identity':identity,'raw':{}}
    if saved_raw['identity']!=identity:raise ValueError('Raw prediction identity changed; choose another output directory.')
    progress=saved_raw['raw']
    for did in ids[rank::world]:
        if did in progress:continue
        progress.update(raw_documents(model,cache,[did],args.batch_size,device))
        atomic_json(path,{'identity':identity,'raw':progress})
    if world>1:dist.barrier()
    if rank==0:
        combined={}
        for i in range(world):combined.update(json.loads((root/f'part_{i}.json').read_text())['raw'])
        if set(combined)!=set(ids):raise ValueError('Raw prediction IDs missing.')
        atomic_json(root/'raw.json',combined)
    if world>1:dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
