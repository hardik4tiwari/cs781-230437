"""Phase-5 schemas, leakage protection, all-token cache, sparse-label metrics and calibration."""
import hashlib,json,math,os,shutil,zipfile

from pathlib import Path

import numpy as np

import torch

from torch.utils.data import Dataset

from tokenizers import Tokenizer

from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as handle:
        for chunk in iter(lambda:handle.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def read_json(path):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError(f'Duplicate JSON key: {key}')
            result[key]=value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),object_pairs_hook=unique)

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)

def atomic_torch(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');torch.save(value,tmp);tmp.replace(path)

def alias(row,names):
    keys=[name for name in names if name in row]
    if len(keys)>1 and any(row[k]!=row[keys[0]] for k in keys[1:]):raise ValueError(f'Conflicting fields: {keys}')
    return row[keys[0]] if keys else None

def shape(value):
    if isinstance(value,dict):return {str(k):type(v).__name__ for k,v in list(value.items())[:8]}
    return {'type':type(value).__name__,'length':len(value) if isinstance(value,list) else None}

def to_device(batch,device):return {k:v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}

import re,bisect
DATA_FILES=['Train/train.json','Train/dev.json','Test/test.json']

def records(value,split,audit=None):
    if isinstance(value,dict) and set(value) in ({'data'},{'records'},{'documents'}):value=next(iter(value.values()))
    if isinstance(value,list):rows=[(None,row) for row in value]
    elif isinstance(value,dict):
        texts=alias(value,('text','sentences','document'));ids=alias(value,('id','ids','case_id','doc_id'))
        labels=None if split=='test' else alias(value,('labels','label','gold','statutes','statute_ids'))
        if isinstance(ids,str) and texts is not None:rows=[(None,{'id':ids,'text':texts,'labels':labels})]
        elif isinstance(ids,list) and isinstance(texts,list):
            if len(ids)!=len(texts) or (labels is not None and (not isinstance(labels,list) or len(labels)!=len(ids))):raise ValueError(f'{split}: column lengths differ.')
            rows=[(None,{'id':did,'text':texts[i],'labels':None if labels is None else labels[i]}) for i,did in enumerate(ids)]
        elif isinstance(texts,dict):
            for column in (ids,labels):
                if column is not None and (not isinstance(column,dict) or set(column)!=set(texts)):raise ValueError('Column keys differ.')
            rows=[(None,{'id':key if ids is None else ids[key],'text':body,'labels':None if labels is None else labels[key]}) for key,body in texts.items()]
        else:rows=list(value.items())
    else:raise ValueError(f'{split}: unsupported input structure: {shape(value)}')
    grouped={};exact=set();copies=0;different=0
    for key,row in rows:
        if isinstance(row,str) and key is not None:row={'id':key,'text':row}
        if not isinstance(row,dict):raise ValueError(f'{split}: document must be an object.')
        did=alias(row,('id','case_id','doc_id'));did=key if did is None else did
        if not isinstance(did,str) or not did.strip():raise ValueError(f'{split}: original nonempty string IDs are required.')
        if key is not None and key!=did:raise ValueError('Mapping ID and explicit case ID differ.')
        text=alias(row,('text','sentences','document'))
        if isinstance(text,list) and all(isinstance(x,str) for x in text):text='\n'.join(text)
        if not isinstance(text,str):raise ValueError(f'{did}: expected text or an ordered sentence-string list.')
        gold=None if split=='test' else alias(row,('labels','label','gold','statutes','statute_ids'))
        if split!='test':
            if not isinstance(gold,list) or any(type(x) is not int or x<0 for x in gold):raise ValueError(f'{did}: labels must be a list of nonnegative integer statute IDs.')
            gold=sorted(set(gold))
        signature=(did,digest(text),None if gold is None else tuple(gold))
        if signature in exact:copies+=1;continue
        exact.add(signature)
        entry=grouped.setdefault(did,{'texts':[],'text_hashes':set(),'gold':set(),'rows':0,'label_sets':set()})
        entry['rows']+=1;sha=digest(text)
        if sha not in entry['text_hashes']:entry['text_hashes'].add(sha);entry['texts'].append(text)
        if gold is not None:entry['gold'].update(gold);entry['label_sets'].add(tuple(gold))
    result=[{'id':did,'source_id':did,'text':'\n\n'.join(v['texts']),
             'gold':None if split=='test' else sorted(v['gold']),'split':split} for did,v in grouped.items()]
    if not result:raise ValueError(f'{split}: empty split.')
    if audit is not None:audit.update(raw_rows=len(rows),case_count=len(result),exact_copies_removed=copies,
        repeated_cases_combined=sum(v['rows']>1 for v in grouped.values()),
        cases_with_different_label_sets=sum(len(v['label_sets'])>1 for v in grouped.values()),
        policy='One original ID per case; distinct texts concatenate in source order; supplied train/dev label sets union. Test labels are discarded.')
    return result

def load_records(root,return_audit=False):
    root=Path(root);audit={};all_rows={}
    for split,name in zip(('train','dev','test'),DATA_FILES):
        audit[split]={};all_rows[split]=records(read_json(root/name),split,audit[split])
    test_ids={r['id'] for r in all_rows['test']};dev_ids={r['id'] for r in all_rows['dev']}
    audit['dev_test_source_overlap_removed']=sorted(dev_ids&test_ids)
    all_rows['dev']=[r for r in all_rows['dev'] if r['id'] not in test_ids]
    held_ids=dev_ids|test_ids
    held_text={digest(' '.join(r['text'].split())) for s in ('dev','test') for r in all_rows[s]}
    removed=[r['id'] for r in all_rows['train'] if r['id'] in held_ids or digest(' '.join(r['text'].split())) in held_text]
    remove=set(removed);all_rows['train']=[r for r in all_rows['train'] if r['id'] not in remove]
    audit['training_held_out_cases_or_text_copies_removed']=removed
    if any(not all_rows[s] for s in ('train','dev','test')):raise ValueError('Leakage filtering left an empty split; inspect IDs and schema before training.')
    result=sum([all_rows[s] for s in ('train','dev','test')],[])
    if len({r['id'] for r in result})!=len(result):raise ValueError('IDs overlap after case filtering.')
    return (result,audit) if return_audit else result

def prepare_data(root,tokenizer_path,output,chunk_tokens=384,overlap=48):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if not 0<=overlap<chunk_tokens:raise ValueError('Require 0 <= overlap < chunk_tokens.')
    identity=digest({'files':{name:file_hash(root/name) for name in DATA_FILES},'tokenizer':file_hash(tokenizer_path),
        'chunk_tokens':chunk_tokens,'overlap':overlap,'version':1})
    manifest=output/'manifest.json'
    if manifest.exists():
        old=read_json(manifest)
        if old['fingerprint']!=identity:raise ValueError('Input/cache changed. Use a new PROJECT.')
        if any(old['docs'][i]['gold'] is not None for i in old['splits']['test']):raise ValueError('Unsafe cache contains test targets.')
        if (output/'tokens.i32').stat().st_size!=4*old['total_tokens']:raise ValueError('Incomplete token cache.')
        return str(manifest)
    rows,audit=load_records(root,True)
    label_ids=sorted({y for r in rows if r['split']!='test' for y in r['gold']})
    if not label_ids:raise ValueError('No public training/development labels.')
    label_count=max(label_ids)+1
    if label_count>50000:raise ValueError('Label dimension exceeds 50000. Verify that statutory sections are already mapped to evaluator class IDs.')
    tokenizer=Tokenizer.from_file(str(tokenizer_path));tokenizer.no_padding();tokenizer.no_truncation()
    unk=tokenizer.token_to_id('<unk>');unk=0 if unk is None else unk
    docs=[];total=chunks=empty=0;temporary=output/'tokens.i32.tmp'
    cue=re.compile(r'\b(?:sections?|articles?|acts?|statutes?|codes?|provisions?|ipc|crpc|constitution|rules?)\b',re.I)
    with temporary.open('wb') as handle:
        for index,row in enumerate(rows):
            encoded=tokenizer.encode(row['text'],add_special_tokens=False);ids=encoded.ids;offsets=encoded.offsets
            empty+=int(not ids);ids=ids or [unk];offsets=offsets or [(0,0)]
            np.asarray(ids,dtype='<i4').tofile(handle);n=len(ids);final=max(0,n-chunk_tokens)
            starts=list(range(0,final+1,chunk_tokens-overlap))
            if starts[-1]!=final:starts.append(final)
            windows=[];scores=[]
            for start in starts:
                length=min(chunk_tokens,n-start);windows.append([total+start,length,start/max(1,n-1)])
                left=offsets[start][0];right=offsets[start+length-1][1]
                scores.append(len(cue.findall(row['text'][left:right])))
            order=list(dict.fromkeys([0,len(windows)-1]+sorted(range(len(windows)),key=lambda j:(-scores[j],j))))
            docs.append({'id':row['id'],'source_id':row['id'],'gold':row['gold'],'split':row['split'],
                'token_count':n,'chunks':windows,'evidence_order':order,'feature_start':chunks,'text_sha256':digest(row['text'])})
            total+=n;chunks+=len(windows)
            if (index+1)%2000==0:print('Tokenized cases:',index+1,flush=True)
    temporary.replace(output/'tokens.i32')
    splits={s:[i for i,d in enumerate(docs) if d['split']==s] for s in ('train','dev','test')}
    splits['all']=splits['train']+splits['dev'];splits['every']=list(range(len(docs)))
    support=np.zeros(label_count,dtype=np.int64)
    for i in splits['train']:support[docs[i]['gold']]+=1
    value={'fingerprint':identity,'docs':docs,'splits':splits,'total_tokens':total,'total_chunks':chunks,
        'chunk_tokens':chunk_tokens,'overlap':overlap,'label_ids':label_ids,'label_count':label_count,'train_support':support.tolist(),
        'tokenizer_sha256':file_hash(tokenizer_path),'audit':{**audit,'empty_documents':empty,'all_tokens_retained':True,
        'case_counts':{s:len(v) for s,v in splits.items()},'label_count_including_id_gaps':label_count,
        'labels_without_training_positives':np.flatnonzero(support==0).tolist(),
        'adaptation_view':'First, last, then statute-cue windows; full branch retains all windows.',
        'full_feature_gib_per_512d_parent':chunks*1024*2/1024**3}}
    atomic_json(manifest,value);return str(manifest)

def target_matrix(docs,label_count):
    y=np.zeros((len(docs),label_count),dtype=np.float32)
    for i,doc in enumerate(docs):
        if doc['gold'] is None:raise ValueError('Test targets cannot be used for fitting or selection.')
        y[i,doc['gold']]=1
    return y

class Bags(Dataset):
    def __init__(self,manifest,split,features=None,teacher=None,view='all',adapt_chunks=3):
        self.path=Path(manifest);self.data=read_json(self.path);self.rows=self.data['splits'][split]
        self.tokens=np.memmap(self.path.parent/'tokens.i32',dtype='<i4',mode='r');self.features=None
        if features:
            info=read_json(Path(features)/'manifest.json')
            if not info['complete'] or info['data_fingerprint']!=self.data['fingerprint']:raise ValueError('Feature cache mismatch/incomplete.')
            self.features=np.memmap(Path(features)/'features.f16',dtype='<f2',mode='r',shape=tuple(info['shape']))
        if view not in ('all','evidence'):raise ValueError('Unknown context view.')
        self.teacher,self.view,self.adapt_chunks=teacher,view,adapt_chunks
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):return self.document(self.rows[index])
    def document(self,index):
        doc=self.data['docs'][index];windows=doc['chunks'];keep=list(range(len(windows)))
        if self.view=='evidence':keep=sorted(doc['evidence_order'][:self.adapt_chunks])
        position=[[windows[j][2],1-windows[j][2],windows[j][2]**2,float(j==0),float(j==len(windows)-1),math.log1p(windows[j][1])/8] for j in keep]
        target=None
        if doc['gold'] is not None:
            target=np.zeros(self.data['label_count'],dtype=np.float32);target[doc['gold']]=1
        return {'index':index,'id':doc['id'],'chunks':[np.array(self.tokens[windows[j][0]:windows[j][0]+windows[j][1]],dtype=np.int64) for j in keep] if self.features is None else None,
            'features':torch.from_numpy(np.array(self.features[[doc['feature_start']+j for j in keep]],dtype=np.float32)) if self.features is not None else None,
            'position':position,'target':target,'teacher':self.teacher[doc['id']] if self.teacher is not None else None}

def collate(rows):
    B=len(rows);S=max(len(r['position']) for r in rows);positions=torch.zeros(B,S,6);mask=torch.zeros(B,S,dtype=torch.bool)
    owners=[];chunks=[];features=None
    if rows[0]['features'] is not None:features=torch.zeros(B,S,rows[0]['features'].shape[-1])
    for b,row in enumerate(rows):
        n=len(row['position']);positions[b,:n]=torch.tensor(row['position']);mask[b,:n]=True
        if features is not None:features[b,:n]=row['features']
        else:chunks.extend(row['chunks']);owners.extend(b*S+j for j in range(n))
    targets=None if rows[0]['target'] is None else torch.from_numpy(np.stack([r['target'] for r in rows]))
    return {'chunks':chunks,'owners':owners,'features':features,'position':positions,'mask':mask,'target':targets,
        'teacher':torch.stack([torch.as_tensor(r['teacher']) for r in rows]) if rows[0]['teacher'] is not None else None,'rows':rows}

def probabilities(logits):
    x=np.asarray(logits,dtype=np.float64);return 1/(1+np.exp(-np.clip(x,-40,40)))

def eval_dimension(y):
    active=np.flatnonzero(np.asarray(y).sum(0));return int(active.max()+1) if len(active) else np.asarray(y).shape[1]

def metric(y,probability,threshold=.5):
    y=np.asarray(y,dtype=np.uint8);p=np.asarray(probability);pred=p>=np.asarray(threshold)
    n=eval_dimension(y);y=y[:,:n];pred=pred[:,:n]
    return {'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
        'micro_f1':float(f1_score(y,pred,average='micro',zero_division=0)),
        'subset_accuracy':float(np.all(y==pred,axis=1).mean()),'label_dimension_from_truth':n}

def fit_thresholds(y,p,shrink=20.):
    y=np.asarray(y);p=np.asarray(p);grid=np.array([.03,.05,.075,.1,.15,.2,.25,.3,.35,.4,.45,.5,.6,.7,.8])
    global_t=max(grid,key=lambda t:metric(y,p,t)['macro_f1']);threshold=np.full(y.shape[1],global_t,dtype=float)
    support=y.sum(0)
    for j in range(y.shape[1]):
        if support[j]<3:continue
        pred=p[:,j,None]>=grid[None,:];tp=(pred*y[:,j,None]).sum(0);denom=pred.sum(0)+support[j]
        f1=np.divide(2*tp,denom,out=np.zeros(len(grid)),where=denom>0)
        tied=np.flatnonzero(f1>=f1.max()-1e-12);best=tied[np.argmin(abs(grid[tied]-global_t))]
        strength=support[j]/(support[j]+shrink);threshold[j]=(1-strength)*global_t+strength*grid[best]
    if metric(y,p,threshold)['macro_f1']<metric(y,p,global_t)['macro_f1']:threshold[:]=global_t
    return threshold.tolist()

def predict_labels(probability,threshold,allowed_ids):
    p=np.asarray(probability);binary=p>=np.asarray(threshold);allowed=set(allowed_ids)
    return [[int(j) for j in np.flatnonzero(row) if j in allowed] for row in binary]

def write_submission(ids,preds,path):
    if len(ids)!=len(set(ids)) or len(ids)!=len(preds):raise ValueError('Submission IDs/count mismatch.')
    if any(not isinstance(row,list) or any(type(y) is not int or y<0 for y in row) or len(row)!=len(set(row)) for row in preds):raise ValueError('Predictions require unique nonnegative integer label IDs.')
    rows=[{'id':did,'pred':sorted(p)} for did,p in zip(ids,preds)];raw=json.dumps(rows,ensure_ascii=False).encode()
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as archive:archive.writestr('prediction.json',raw)
    tmp.replace(path);return rows,hashlib.sha256(raw).hexdigest()
