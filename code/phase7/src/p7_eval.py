"""Shortlist reranking, exact official Micro-F1@7, and checked submission archives."""
import contextlib,json,math,time,zipfile
from pathlib import Path
import numpy as np
import torch
from evaluate_phase7_official import score,validate
from p7_data import atomic_json,digest,read_json
from p7_model import encode_batch,pairs_batch,similarity,merged_adapters

def metrics(store,split,pred):
    queries=store.splits[split]['queries'];gold={q:r['relevant'] for q,r in queries.items() if q in pred}
    if split=='test':raise ValueError('Test labels are never used for scoring.')
    errors=validate(pred,gold,set(store.splits[split]['candidates']))
    if errors:raise ValueError('; '.join(errors[:5]))
    raw=score(gold,pred)
    return {'micro_f1_at7':raw['primary_metric_1'],'micro_precision_at7':raw['secondary_metric_1'],'micro_recall_at7':raw['secondary_metric_2'],'tp':raw['tp'],'fp':raw['fp'],'fn':raw['fn'],'queries':len(gold)}
def normalized(x):
    x=np.asarray(x,dtype=np.float32);std=max(float(x.std()),1e-6);return (x-x.mean())/std
def predictions(raw,weight):
    return {q:ids[:7] for q,ids in ranking_lists(raw,weight).items()}
def ranking_lists(raw,weight):
    pred={}
    for q,r in raw.items():
        score=weight*normalized(r['neural'])+(1-weight)*normalized(r['lexical'])
        order=np.argsort(-score,kind='stable');pred[q]=[r['ids'][i] for i in order]
    return pred
def select_fusion(store,raw,weights=(.1,.25,.5,.75,1.)):
    candidates=[]
    for w in weights:
        pred=predictions(raw,w);m=metrics(store,'dev',pred);candidates.append({'neural_weight':float(w),'metrics':m})
    # Prefer the stronger neural contribution on an exact tie; the sparse-only baseline is reported separately.
    best=max(candidates,key=lambda x:(x['metrics']['micro_f1_at7'],x['neural_weight']))
    return {**best,'candidates':candidates,'scope':'Validation selection estimate, not an independent test score.'}
def query_ids(store,split,limit=None,seed=781):
    ids=sorted(store.splits[split]['queries'],key=lambda q:digest([seed,q]))
    return ids[:limit] if limit else ids
def shortlist(store,split,qid,cfg,base_rankings=None):
    lex=store.lexical[split][qid];ids=lex['ids'];values=dict(zip(ids,lex['scores']))
    if cfg['kind']!='cross':return ids,lex['scores']
    k=min(cfg['rerank_topk'],len(ids))
    # Cross-encoder reranks a union of lexical and trained first-stage rankings, all inside this split's lexical pool.
    dense=base_rankings.get(qid,ids) if base_rankings else ids
    selected=list(dict.fromkeys(ids[:max(7,k//2)]+[i for i in dense[:k//2] if i in values]+ids))[:k]
    return selected,[values[i] for i in selected]
@torch.inference_mode()
def teacher_scores(model,store,qids,cfg,device,count=16):
    # Gold injection is restricted to TRAIN teacher supervision, never validation/test retrieval.
    model.eval();out={};amp=device.type=='cuda';selection_path=Path(cfg['run'])/'full_dev_selection.json'
    weight=read_json(selection_path)['neural_weight'] if selection_path.is_file() else 1.
    with merged_adapters(model):
        for qid in qids:
            row=store.splits['train']['queries'][qid];lex=store.lexical['train'][qid];gold=row['relevant']
            if not gold:continue
            ids=list(dict.fromkeys(gold+lex['ids']))[:max(count,len(gold)+1)]
            if not set(ids)-set(gold):continue
            scores=dict(zip(lex['ids'],lex['scores']));scores.update(lex.get('extra_scores',{}));qkey=row['key'];dkeys=[store.splits['train']['candidates'][c]['key'] for c in ids]
            if cfg['kind']=='cross':
                batch=pairs_batch(store,[qkey]*len(ids),dkeys,[scores.get(c,0.) for c in ids],cfg,device)
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):values=model.rank_pairs(batch)
            else:
                batch=encode_batch(store,[qkey]+dkeys,[0]+[1]*len(ids),cfg,device)
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):g,ch,m=model.encode(batch)
                values=similarity((g[:1],ch[:1],m[:1]),(g[1:],ch[1:],m[1:]),cfg['kind']).squeeze(0)/cfg['temperature']
            neural=values.cpu().float().numpy();lexical=np.asarray([scores.get(c,0.) for c in ids],dtype=np.float32)
            fused=4*(weight*normalized(neural)+(1-weight)*normalized(lexical))
            out[qid]={'ids':ids,'scores':fused.tolist()}
    return out
@torch.inference_mode()
def rank(model,store,split,qids,cfg,device,base_rankings=None):
    model.eval();raw={};started=time.perf_counter();amp=device.type=='cuda';pools={q:shortlist(store,split,q,cfg,base_rankings) for q in qids}
    with merged_adapters(model):
        if cfg['kind']=='cross':
            for qi,qid in enumerate(qids):
                ids,lex=pools[qid];scores=[];qkey=store.splits[split]['queries'][qid]['key']
                for a in range(0,len(ids),cfg['inference_pairs']):
                    local=ids[a:a+cfg['inference_pairs']];dkeys=[store.splits[split]['candidates'][c]['key'] for c in local]
                    batch=pairs_batch(store,[qkey]*len(local),dkeys,lex[a:a+len(local)],cfg,device)
                    with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):s=model.rank_pairs(batch)
                    scores.extend(s.cpu().float().tolist())
                raw[qid]={'ids':ids,'lexical':lex,'neural':scores}
                if (qi+1)%25==0:print(json.dumps({'reranked_queries':qi+1,'total':len(qids),'seconds':time.perf_counter()-started}),flush=True)
        else:
            # Encode each candidate only once per rank/validation pass, not once per query–candidate pair.
            cids=sorted({c for q in qids for c in pools[q][0]});vectors=[];chunks=[];masks=[];batch_size=cfg['index_batch']
            for a in range(0,len(cids),batch_size):
                local=cids[a:a+batch_size];keys=[store.splits[split]['candidates'][c]['key'] for c in local];batch=encode_batch(store,keys,[1]*len(local),cfg,device)
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):g,ch,ma=model.encode(batch)
                vectors.append(g.cpu().half());chunks.append(ch.cpu().half());masks.append(ma.cpu())
            if not cids:return raw
            g=torch.cat(vectors);ch=torch.cat(chunks);ma=torch.cat(masks);lookup={c:i for i,c in enumerate(cids)}
            for qi,qid in enumerate(qids):
                ids,lex=pools[qid];key=store.splits[split]['queries'][qid]['key'];batch=encode_batch(store,[key],[0],cfg,device)
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):q=model.encode(batch)
                idx=torch.tensor([lookup[c] for c in ids]);d=(g[idx].to(device).float(),ch[idx].to(device).float(),ma[idx].to(device))
                with torch.autocast(device_type=device.type,enabled=False):scores=similarity(q,d,cfg['kind']).squeeze(0).cpu().float().tolist()
                raw[qid]={'ids':ids,'lexical':lex,'neural':scores}
                if (qi+1)%100==0:print(json.dumps({'ranked_queries':qi+1,'total':len(qids),'seconds':time.perf_counter()-started}),flush=True)
    if any(not np.isfinite(r['neural']).all() for r in raw.values()):raise FloatingPointError('Nonfinite retrieval scores.')
    return raw
def pack(project,name,pred,store):
    queries=store.splits['test']['queries'];gold={q:[] for q in queries};pool=set(store.splits['test']['candidates']);errors=validate(pred,gold,pool)
    if errors:raise ValueError('; '.join(errors[:10]))
    folder=Path(project)/'submissions'/name;atomic_json(folder/'prediction.json',pred);archive=folder.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:z.write(folder/'prediction.json','prediction.json')
    with zipfile.ZipFile(archive) as z:
        if z.namelist()!=['prediction.json']:raise ValueError('ZIP must contain prediction.json at its root.')
    return {'name':name,'submission_zip':str(archive),'prediction_sha256':digest(pred),'queries':len(pred)}
