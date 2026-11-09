"""Dynamically batched conditional LM with chunked CE/KD and greedy KV decoding."""
import math
from collections import defaultdict

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from tokenizers import Tokenizer

from s6_model import Summarizer, load_parent, merged_adapters

def pad(rows,fill,width=None):
    if width is None:width=max(map(len,rows))
    return torch.tensor([row+[fill]*(width-len(row)) for row in rows],dtype=torch.long)

def align(n,bound,multiple=32):
    return min(bound,math.ceil(n/multiple)*multiple)

def collation(rows,cache,device,train=True,native=False):
    pad_id=cache['special']['pad'];bos=cache['special']['bos']
    ss=[r['source'] for r in rows];lengths=torch.tensor(list(map(len,ss)),dtype=torch.long)
    if train:
        ys=[r['target'] for r in rows]
        if any(not y for y in ys):raise ValueError('Missing target in supervised batch.')
        target=torch.tensor([tok for y in ys for tok in y],dtype=torch.long).to(device)
        row_index=torch.tensor([i for i,y in enumerate(ys) for _ in y],dtype=torch.long).to(device)
        inputs=[[bos]+y[:-1] for y in ys]
        if not native:
            seq=[s+x for s,x in zip(ss,inputs)]
            width=align(max(map(len,seq)),cache['source_tokens']+cache['target_tokens'])
            positions=torch.tensor([j for src,y in zip(ss,ys) for j in range(len(src),len(src)+len(y))],
                                   dtype=torch.long).to(device)
            return {'ids':pad(seq,pad_id,width).to(device),'target':target,'rows':row_index,
                    'positions':positions,'token_count':target.numel()}
        ti=pad(inputs,pad_id).to(device)
        positions=torch.tensor([j for y in ys for j in range(len(y))],dtype=torch.long).to(device)
    width=align(max(map(len,ss)),cache['source_tokens'])
    source=pad(ss,pad_id,width).to(device)
    valid=(torch.arange(width)[None]<lengths[:,None]).to(device)
    batch={'source':source,'source_valid':valid,'lengths':lengths}
    if train:batch.update(ids=ti,target=target,rows=row_index,positions=positions,token_count=target.numel())
    return batch

class FastSummarizer(Summarizer):
    def target_states(self,batch):
        if self.native:
            h,_=self.hidden(batch['ids'],None,batch['source'],batch['source_valid'],compact=True)
        else:
            h,_=self.hidden(batch['ids'],None,compact=True)
        offset=self.prompt.shape[0] if hasattr(self,'prompt') else 0
        return h[batch['rows'],batch['positions']+offset]
    def forward(self,batch,teacher_states=None,teacher_weight=None):
        h=self.target_states(batch);y=batch['target']
        step=self.cfg['projection_chunk'];loss=h.new_zeros((),dtype=torch.float32)
        temp=self.cfg['kd_temperature'];weight=self.cfg['kd_weight']
        def ce(hidden,gold,other=None):
            logits=self.backbone.project(hidden).float()
            with torch.autocast(device_type=hidden.device.type,enabled=False):
                nll=F.cross_entropy(logits,gold,reduction='sum',label_smoothing=self.cfg['label_smoothing'])
            if other is None:return nll
            with torch.no_grad():tl=F.linear(other,teacher_weight).float()
            with torch.autocast(device_type=hidden.device.type,enabled=False):
                kl=F.kl_div(F.log_softmax(logits/temp,dim=-1),F.softmax(tl/temp,dim=-1),reduction='sum')
            return (1.-weight)*nll+weight*temp**2*kl
        for start in range(0,len(y),step):
            stop=start+step
            args=(h[start:stop],y[start:stop],None if teacher_states is None else teacher_states[start:stop])
            loss=loss+(checkpoint(ce,*args,use_reentrant=False) if self.training else ce(*args))
        return loss/batch['token_count']

def build(cfg,cache,device):
    parent=load_parent(cfg['parent'],cache['tokenizer_sha256'],cfg.get('allow_tiny',False))
    if parent.config.architecture!=cfg['architecture']:raise ValueError('Parent architecture mismatch.')
    s,t=cache['source_tokens'],cache['target_tokens']
    needed=max(s,t+1) if cfg['architecture']=='encoder_decoder' else s+t+cfg.get('prompt_tokens',0)+1
    if needed>parent.config.max_positions:raise ValueError(f'Need {needed} positions; Phase 1 parent supports {parent.config.max_positions}.')
    return FastSummarizer(parent,cfg).to(device)

@torch.inference_mode()
def greedy(model,rows,cache,device,decode,first_blocked=()):
    batch=collation(rows,cache,device,train=False);src=batch['source'];sv=batch['source_valid']
    bos=torch.full((len(rows),1),cache['special']['bos'],device=device,dtype=torch.long)
    ids=bos if model.native else torch.cat([src,bos],1)
    valid=torch.ones_like(ids,dtype=torch.bool) if model.native else torch.cat([sv,torch.ones_like(bos,dtype=torch.bool)],1)
    blocked=cache['generation_blocked_ids'];eos=cache['special']['eos']
    max_new=decode['max_new_tokens'];min_new=decode['min_new_tokens'];n=decode['no_repeat_ngram']
    sequences=[[] for _ in rows];grams=[defaultdict(set) for _ in rows];done=[False]*len(rows);state=None
    for step in range(max_new):
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            h,state=model.hidden(ids,valid,src if model.native else None,sv if model.native else None,state,True,
                                 source_width=src.shape[1],source_lengths=batch['lengths'])
            logits=model.backbone.project(h[:,-1])
        if blocked:logits[:,blocked]=float('-inf')
        if step==0 and first_blocked:logits[:,list(first_blocked)]=float('-inf')
        if step<min_new:logits[:,eos]=float('-inf')
        if n>=2:
            ban=[(i,t) for i,seq in enumerate(sequences) if not done[i] and len(seq)>=n-1
                 for t in grams[i].get(tuple(seq[-(n-1):]),())]
            if ban:
                index,values=zip(*ban);logits[list(index),list(values)]=float('-inf')
        ids=logits.argmax(-1,keepdim=True)
        for i,token in enumerate(ids[:,0].tolist()):
            if done[i]:continue
            if token==eos:done[i]=True;continue
            seq=sequences[i]
            if n>=2 and len(seq)>=n-1:grams[i][tuple(seq[-(n-1):])].add(token)
            seq.append(token)
        if all(done):break
        valid=torch.ones_like(ids,dtype=torch.bool)
    return sequences

@torch.inference_mode()
def predict(model,rows,cache,device,decode,batch_size):
    tok=Tokenizer.from_file(cache['tokenizer_path']);result=[];first_blocked=None
    rows=sorted(rows,key=lambda r:(len(r['source']),r['id']))
    model.eval()
    with merged_adapters(model):
        for start in range(0,len(rows),batch_size):
            local=rows[start:start+batch_size]
            while True:
                # A smaller generation batch does not change the IDs covered.
                try:
                    seqs=greedy(model,local,cache,device,decode)
                    break
                except torch.OutOfMemoryError:
                    if len(local)==1:raise
                    seqs=None
                    break
            if seqs is None:
                if device.type=='cuda':torch.cuda.empty_cache()
                seqs=[greedy(model,[r],cache,device,decode)[0] for r in local]
            for row,seq in zip(local,seqs):
                text=tok.decode(seq,skip_special_tokens=True).strip()
                if not text:
                    if first_blocked is None:
                        first_blocked=[i for i in range(tok.get_vocab_size()) if not tok.decode([i],skip_special_tokens=True).strip()]
                    retry=greedy(model,[row],cache,device,decode,first_blocked)
                    text=tok.decode(retry[0],skip_special_tokens=True).strip()
                if not text:raise ValueError(f'Empty generated summary for {row["id"]!r}. Check the Phase 1 tokenizer and model.')
                result.append({'id':row['id'],'summary':text})
            if (start//batch_size+1)%25==0:print(f'Generated {len(result):,}/{len(rows):,} summaries.',flush=True)
    return result
