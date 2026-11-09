"""Official L-NER data, unchanged text, document split, and character alignment."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

OFFICIAL_LABELS = {'APP','RESP','A.COUNSEL','R.COUNSEL','JUDGE','WIT','AUTH','COURT','STAT','PREC','DATE','CASENO'}

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)

def no_duplicates(pairs):
    data={}
    for key,value in pairs:
        if key in data: raise ValueError(f'Duplicate JSON key: {key}')
        data[key]=value
    return data

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),object_pairs_hook=no_duplicates)

def validate_spans(predictions,texts,labels):
    if not isinstance(predictions,dict) or set(predictions)!=set(texts): raise ValueError('Prediction IDs must match document IDs exactly.')
    for did,entities in predictions.items():
        if not isinstance(entities,list): raise ValueError(f'{did}: entities must be a list.')
        seen=set()
        for span in entities:
            if not isinstance(span,list) or len(span) not in (3,4): raise ValueError(f'{did}: invalid span schema.')
            start,end,label=span[:3]
            if type(start) is not int or type(end) is not int or not 0<=start<end<=len(texts[did]):
                raise ValueError(f'{did}: invalid character offsets {span[:2]}.')
            if label not in labels: raise ValueError(f'{did}: unknown label {label}.')
            if len(span)==4 and span[3]!=texts[did][start:end]: raise ValueError(f'{did}: entity text differs from its exact source slice.')
            key=tuple(span[:3])
            if key in seen: raise ValueError(f'{did}: repeated entity.')
            seen.add(key)
    return True

def load_official(root):
    root=Path(root)
    if (root/'Phase_2_LNER').is_dir(): root=root/'Phase_2_LNER'
    labels=[x.strip() for x in (root/'Train/labels.txt').read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    if len(labels)!=12 or set(labels)!=OFFICIAL_LABELS: raise ValueError('Official 12-label vocabulary mismatch.')
    texts,gold,folds={},{},{}
    for fold in (1,2):
        docs=read_json(root/f'Train/fold{fold}-text.json');annotations=read_json(root/f'Train/fold{fold}-gold.json')
        if set(docs)!=set(annotations): raise ValueError(f'Fold {fold}: document/gold ID mismatch.')
        if set(texts)&set(docs): raise ValueError('Repeated IDs across training folds.')
        if not all(isinstance(k,str) and isinstance(v,str) for k,v in docs.items()): raise ValueError('Expected ID -> original document string.')
        # The official evaluator ignores the optional fourth field. Its wording may
        # differ from the document slice; character offsets and label remain authoritative.
        for key in annotations:
            if annotations[key] is None:annotations[key]=[]
        validate_spans({k:[e[:3] for e in v] for k,v in annotations.items()},docs,labels)
        texts.update(docs);gold.update(annotations);folds.update({key:fold for key in docs})
    test=read_json(root/'Test/fold3-text.json')
    if not test or not all(isinstance(k,str) and isinstance(v,str) for k,v in test.items()): raise ValueError('Official test document schema mismatch.')
    if set(texts)&set(test): raise ValueError('Train/test ID overlap.')
    return texts,gold,test,labels,folds

def grouped_split(texts,gold,labels,fraction=.2,seed=230437):
    groups={}
    for did,text in texts.items():
        # Group whitespace/case duplicates for split hygiene without altering model input.
        key=hashlib.sha256(' '.join(text.split()).casefold().encode()).hexdigest()
        groups.setdefault(key,[]).append(did)
    keys=sorted(groups,key=lambda k:digest([seed,k]))
    if len(keys)<2 or not 0<fraction<1:raise ValueError('Need two distinct groups and a validation fraction between zero and one.')
    n=min(len(keys)-1,max(1,round(len(keys)*fraction)))
    presence={k:np.array([any(e[2]==label for did in groups[k] for e in gold[did]) for label in labels],dtype=np.int64) for k in keys}
    total=sum(presence.values());target=np.maximum(1,np.round(total*fraction));target[total<2]=0
    chosen=[];counts=np.zeros(len(labels),dtype=np.int64)
    for _ in range(n):
        options=[k for k in keys if k not in chosen and np.all((total-counts-presence[k]>=1)|(total==0))]
        if not options:break
        def value(k):
            before=np.sum(np.abs(target-counts)/np.maximum(total,1))
            after=np.sum(np.abs(target-counts-presence[k])/np.maximum(total,1))
            return before-after
        best=max(options,key=value);chosen.append(best);counts+=presence[best]
    if not chosen:
        # If every group owns a singleton label, a nonempty holdout necessarily
        # removes some training support. Choose the least damaging group and report it.
        best=min(keys,key=lambda k:int(presence[k][total==1].sum()));chosen=[best];counts=presence[best].copy()
    val_ids=sorted(did for k in chosen for did in groups[k]);train_ids=sorted(set(texts)-set(val_ids))
    if not train_ids or not val_ids: raise ValueError('Need at least two distinct document groups.')
    report={'seed':seed,'validation_fraction_requested':fraction,'train_ids':train_ids,'val_ids':val_ids,
        'duplicate_groups':len(texts)-len(groups),'validation_groups_by_label':dict(zip(labels,map(int,counts))),
        'training_groups_by_label':dict(zip(labels,map(int,total-counts))),
        'actual_validation_group_fraction':len(chosen)/len(keys),
        'training_absent_labels':[l for l,c in zip(labels,total-counts) if c==0],
        'validation_absent_labels':[l for l,c in zip(labels,counts) if c==0]}
    return train_ids,val_ids,report

def encode_documents(texts,tokenizer_path):
    tokenizer=Tokenizer.from_file(str(tokenizer_path));tokenizer.no_truncation();tokenizer.no_padding()
    encoded={}
    for did,text in texts.items():
        result=tokenizer.encode(text,add_special_tokens=False)
        ids,offsets=[],[]
        for tid,(a,b) in zip(result.ids,result.offsets):
            if a==b: continue
            if not 0<=a<b<=len(text): raise ValueError(f'{did}: tokenizer offsets are not Python character offsets.')
            ids.append(tid);offsets.append((a,b))
        encoded[did]={'ids':ids,'offsets':offsets}
    return encoded

def align_entities(encoded,gold,labels):
    aligned={};audit={'entities':0,'max_entity_tokens':0,'max_token_chars':0,'bio_dropped_overlaps':0,'uncovered':[]}
    for did,doc in encoded.items():
        offsets=doc['offsets'];n=len(offsets)
        audit['max_token_chars']=max(audit['max_token_chars'],max((b-a for a,b in offsets),default=1))
        starts=np.array([a for a,b in offsets]);ends=np.array([b for a,b in offsets])
        entities=[]
        for span in gold.get(did,[]):
            a,b,label=span[:3];hits=np.flatnonzero((ends>a)&(starts<b))
            if not len(hits): audit['uncovered'].append([did,*span[:3]]);continue
            i,j=int(hits[0]),int(hits[-1])
            if not offsets[i][0]<=a<offsets[i][1] or not offsets[j][0]<b<=offsets[j][1]:
                raise ValueError(f'{did}: gold boundary cannot be represented by tokenizer offsets.')
            entities.append((i,j,labels.index(label),a-offsets[i][0],offsets[j][1]-b))
            audit['entities']+=1;audit['max_entity_tokens']=max(audit['max_entity_tokens'],j-i+1)
        tags=np.zeros(n,dtype=np.int64);occupied=np.zeros(n,dtype=bool)
        kept=[]
        for i,j,l,left,right in sorted(entities,key=lambda x:(-(x[1]-x[0]),x[0],x[2])):
            if occupied[i:j+1].any():audit['bio_dropped_overlaps']+=1;continue
            tags[i]=1+2*l;tags[i+1:j+1]=2+2*l;occupied[i:j+1]=True;kept.append((i,j,l,left,right))
        aligned[did]={'entities':entities,'bio_entities':kept,'tags':tags}
    if audit['uncovered']: raise ValueError(f'Uncovered annotations: {audit["uncovered"][:3]}')
    return aligned,audit

def windows_for(encoded,ids,seq_len,overlap):
    if not 0<=overlap<seq_len:raise ValueError('Invalid context overlap.')
    windows=[]
    for did in ids:
        n=len(encoded[did]['ids']);start=0
        if not n:continue
        while True:
            end=min(start+seq_len,n);windows.append({'did':did,'start':start,'end':end})
            if end==n:break
            start=end-overlap
    return windows

def create_cache(texts,gold,test,labels,tokenizer_path,path,seq_len=512,overlap=128,seed=230437,val_fraction=.2):
    path=Path(path);source_id=digest({'texts':texts,'gold':gold,'test':test,'labels':labels,'tokenizer':file_hash(tokenizer_path),
                                   'seq_len':seq_len,'overlap':overlap,'seed':seed,'val_fraction':val_fraction,'version':3})
    if path.exists():
        old=torch.load(path,map_location='cpu',weights_only=False)
        if old['fingerprint']==source_id:return old
        raise ValueError('Token cache configuration changed; choose a new cache filename.')
    encoded=encode_documents({**texts,**test},tokenizer_path);aligned,audit=align_entities(encoded,gold,labels)
    if audit['max_entity_tokens']>seq_len:raise ValueError(f'Increase SEQ_LEN: longest gold entity uses {audit["max_entity_tokens"]} tokens.')
    effective_overlap=max(overlap,audit['max_entity_tokens']-1)
    if effective_overlap>=seq_len:raise ValueError('Context window too short for annotated entities.')
    char_classes=max(8,audit['max_token_chars']+1)
    if char_classes>256:raise ValueError('Tokenizer has unusually long tokens; inspect before allocating boundary heads.')
    train_ids,val_ids,split=grouped_split(texts,gold,labels,val_fraction,seed)
    cache={'fingerprint':source_id,'texts':{**texts,**test},'gold':gold,'labels':labels,'encoded':encoded,'aligned':aligned,
        'split':split,'audit':audit,'seq_len':seq_len,'overlap':effective_overlap,'char_classes':char_classes,
        'tokenizer_sha256':file_hash(tokenizer_path)}
    for name,doc_ids in [('train',train_ids),('val',val_ids),('all',sorted(texts)),('test',list(test))]:
        cache[name+'_windows']=windows_for(encoded,doc_ids,seq_len,effective_overlap)
    # Every labelled entity must fit in at least one inference-compatible training window.
    for did,alignment in aligned.items():
        if did not in gold:continue
        local=[w for w in cache['all_windows'] if w['did']==did]
        for i,j,*_ in alignment['entities']:
            if not any(w['start']<=i<=j<w['end'] for w in local):raise ValueError('A gold span falls between windows.')
    audit['gold_entity_text_mismatches']=sum(len(e)==4 and e[3]!=texts[k][e[0]:e[1]] for k,entities in gold.items() for e in entities)
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp.pt');torch.save(cache,temp);temp.replace(path)
    return cache

class NERWindows(Dataset):
    def __init__(self,cache,split,head,teacher_dir=None):
        self.cache,self.windows,self.head=cache,cache[split+'_windows'],head
        self.teacher_dir=Path(teacher_dir) if teacher_dir else None
    def __len__(self):return len(self.windows)
    def __getitem__(self,index):
        window=self.windows[index];did,start,end=window['did'],window['start'],window['end']
        n=end-start;L=len(self.cache['labels']);doc=self.cache['encoded'][did]
        item={'ids':torch.tensor(doc['ids'][start:end]),'offsets':doc['offsets'][start:end],
              'index':index,'window':window,'tags':torch.zeros(n,dtype=torch.long),
              'left':torch.full((n,L),-100,dtype=torch.long),'right':torch.full((n,L),-100,dtype=torch.long),'entities':[]}
        alignment=self.cache['aligned'][did]
        if self.head=='bio':item['tags']=torch.from_numpy(alignment['tags'][start:end].copy())
        use=alignment['bio_entities'] if self.head=='bio' else alignment['entities']
        for i,j,label,left,right in use:
            if start<=i<=j<end:
                item['entities'].append((i-start,j-start,label))
                item['left'][i-start,label]=left;item['right'][j-start,label]=right
        if self.teacher_dir:
            item['teacher']=torch.from_numpy(np.load(self.teacher_dir/f'{index:06d}.npz')['logits'].astype(np.float32))
        return item

def collate_windows(items,pad_id=0):
    size=max(len(x['ids']) for x in items);batch=len(items);L=items[0]['left'].size(1)
    ids=torch.full((batch,size),pad_id,dtype=torch.long);valid=torch.zeros(batch,size,dtype=torch.bool)
    tags=torch.full((batch,size),-100,dtype=torch.long);left=torch.full((batch,size,L),-100,dtype=torch.long);right=left.clone()
    teacher=None
    if 'teacher' in items[0]:teacher=torch.zeros(batch,size,items[0]['teacher'].size(-1))
    for row,item in enumerate(items):
        n=len(item['ids']);ids[row,:n]=item['ids'];valid[row,:n]=True;tags[row,:n]=item['tags']
        left[row,:n]=item['left'];right[row,:n]=item['right']
        if teacher is not None:teacher[row,:n]=item['teacher']
    return {'ids':ids,'valid':valid,'tags':tags,'left':left,'right':right,'teacher':teacher,'items':items}

def class_weights(cache,split):
    counts=np.ones(1+2*len(cache['labels']),dtype=np.float64)
    for did in cache['split']['train_ids'] if split=='train' else cache['gold']:
        counts+=np.bincount(cache['aligned'][did]['tags'],minlength=len(counts))
    weights=np.sqrt(counts.sum()/counts);weights/=weights[1:].mean();weights=np.clip(weights,.15,5.)
    return torch.tensor(weights,dtype=torch.float32)

def strict_metrics(predictions,gold,labels):
    if set(predictions)!=set(gold):raise ValueError('Metric ID mismatch.')
    rows=[]
    for label in labels:
        tp=fp=fn=0
        for did in gold:
            p={tuple(x[:3]) for x in predictions[did] if x[2]==label};g={tuple(x[:3]) for x in gold[did] if x[2]==label}
            tp+=len(p&g);fp+=len(p-g);fn+=len(g-p)
        precision=tp/(tp+fp) if tp+fp else 0;recall=tp/(tp+fn) if tp+fn else 0
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
        rows.append({'label':label,'tp':tp,'fp':fp,'fn':fn,'precision':precision,'recall':recall,'f1':f1})
    return {'macro_f1':sum(x['f1'] for x in rows)/len(labels),'macro_precision':sum(x['precision'] for x in rows)/len(labels),
            'macro_recall':sum(x['recall'] for x in rows)/len(labels),'per_label':rows}
