"""Phase-4 input validation, complete token coverage, disk-backed bags and exact metrics."""
import hashlib,json,math,os,shutil,zipfile
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix

DATA_FILES=['Train/train.json','Train/dev.json','Test/test.json']
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
def resolve_training_rows(rows,audit=None,reserved_ids=None,split="train"):
    # Repeated source-case IDs can describe distinct training instances. IDs below are internal only.
    raw_counts={};groups={};unique=[];seen=set();case_labels={};text_labels={}
    for row in rows:
        source=row['id'];raw_counts[source]=raw_counts.get(source,0)+1
        text_sha=hashlib.sha256(row['text'].encode('utf-8')).hexdigest();key=(source,text_sha,row['gold'])
        case_labels.setdefault(source,set()).add(row['gold']);text_labels.setdefault((source,text_sha),set()).add(row['gold'])
        if key in seen:continue
        seen.add(key);groups[source]=groups.get(source,0)+1;unique.append((row,text_sha))
    external=set(reserved_ids or ());reserved=set(raw_counts)|external;result=[]
    for row,text_sha in unique:
        source=row['id'];did=source
        if groups[source]>1 or source in external:
            stem=source+'::'+split+'::'+digest([text_sha,row['gold']])[:16];did=stem;counter=0
            while did in reserved:counter+=1;did=stem+'::'+str(counter)
            reserved.add(did)
        result.append({**row,'id':did,'source_id':source})
    if audit is not None:
        disagreements=[source for source,labels in case_labels.items() if len(labels)>1]
        audit.update(raw_rows=len(rows),unique_source_case_ids=len(raw_counts),retained_distinct_rows=len(result),
            repeated_source_id_groups=sum(n>1 for n in raw_counts.values()),exact_copies_removed=len(rows)-len(result),
            distinct_instance_case_groups=sum(n>1 for n in groups.values()),
            internally_named_training_rows=sum(row['id']!=row['source_id'] for row in result),
            source_cases_with_both_labels=len(disagreements),both_label_case_id_examples=disagreements[:20],
            same_text_conflicting_label_groups=sum(len(labels)>1 for labels in text_labels.values()),
            policy=f'Collapse exact (source ID, text, label) copies; preserve differing text/labels as distinct internal {split} rows. Never majority-vote labels.')
    return result
def records(value,split,audit=None,reserved_ids=None):
    if isinstance(value,dict) and set(value) in ({'data'},{'records'},{'documents'}):value=next(iter(value.values()))
    if isinstance(value,list):rows=[(None,r) for r in value]
    elif isinstance(value,dict):
        text=alias(value,('text','sentences'));ids=alias(value,('id','ids','case_id','doc_id'))
        labels=alias(value,('label','labels','gold','outcome'))
        if isinstance(ids,str) and text is not None:rows=[(None,{'id':ids,'text':text,'label':labels})]
        elif isinstance(ids,list) and isinstance(text,list):
            if len(ids)!=len(text) or (labels is not None and (not isinstance(labels,list) or len(labels)!=len(ids))):
                raise ValueError('ID, text and label columns have different lengths.')
            rows=[(None,{'id':did,'text':text[i],'label':None if labels is None else labels[i]}) for i,did in enumerate(ids)]
        elif isinstance(text,dict):
            for column in (ids,labels):
                if column is not None and (not isinstance(column,dict) or set(column)!=set(text)):raise ValueError('JSON column keys differ.')
            rows=[(None,{'id':key if ids is None else ids[key],'text':body,'label':None if labels is None else labels[key]}) for key,body in text.items()]
        else:rows=list(value.items())
    else:raise ValueError(f'{split}: unsupported JSON structure: {shape(value)}')
    result=[];seen=set()
    for key,row in rows:
        if isinstance(row,str) and key is not None:row={'id':key,'text':row}
        if not isinstance(row,dict):raise ValueError(f'{split}: document must be an object; structure={shape(value)}')
        did=alias(row,('id','case_id','doc_id'));did=key if did is None else did
        if not isinstance(did,str) or not did.strip():raise ValueError(f'{split}: original nonempty string IDs are required; IDs are never invented.')
        if key is not None and key!=did:raise ValueError('Mapping key and explicit case ID differ.')
        seen.add(did);text=alias(row,('text','sentences'))
        if isinstance(text,list) and all(isinstance(s,str) for s in text):text='\n'.join(text)
        if not isinstance(text,str):raise ValueError(f'{did}: text must be a string or an ordered list of strings.')
        gold=None if split=='test' else alias(row,('label','labels','gold','outcome'))
        if split=='test':
            gold=None  # Supplied test labels are read only by the separate post-training evaluator.
        elif type(gold) is not int or gold not in (0,1):raise ValueError(f'{did}: expected a scalar integer label 0 or 1.')
        result.append({'id':did,'text':text,'gold':gold,'split':split})
    if not result:raise ValueError(f'{split}: empty split.')
    if split in ('train','dev'):return resolve_training_rows(result,audit,reserved_ids,split=split)
    # The evaluator accepts one prediction per original test case, not one per source row.
    grouped={};copies=0
    for row in result:
        entry=grouped.setdefault(row['id'],{'row':row,'texts':[],'seen':set(),'rows':0})
        entry['rows']+=1
        if row['text'] in entry['seen']:copies+=1;continue
        entry['seen'].add(row['text']);entry['texts'].append(row['text'])
    merged=[{**v['row'],'source_id':did,'text':'\n\n'.join(v['texts'])} for did,v in grouped.items()]
    if audit is not None:
        audit.update(raw_rows=len(result),unique_source_case_ids=len(merged),exact_text_copies_removed=copies,
            repeated_source_id_groups=sum(v['rows']>1 for v in grouped.values()),
            cases_with_distinct_texts_combined=sum(len(v['texts'])>1 for v in grouped.values()),
            policy='One original test ID per prediction; collapse exact text copies and concatenate distinct case texts in source order. Test labels are excluded from model data; supplied labels are evaluated separately after training.')
    return merged
def load_records(root,return_audit=False):
    root=Path(root);test_audit={};dev_audit={};audit={}
    test=records(read_json(root/'Test/test.json'),'test',test_audit)
    test_ids={row['source_id'] for row in test}
    dev=records(read_json(root/'Train/dev.json'),'dev',dev_audit,reserved_ids=test_ids)
    dev_sources={row['source_id'] for row in dev};held_out=dev_sources|test_ids
    excluded_dev=[row for row in dev if row['source_id'] in test_ids]
    dev=[row for row in dev if row['source_id'] not in test_ids]
    if not dev:raise ValueError('No development records remain after test source-case protection.')
    dev_audit.update(test_overlap_rows_removed=len(excluded_dev),
        test_overlap_source_cases_removed=len({row['source_id'] for row in excluded_dev}),
        test_overlap_case_id_examples=sorted({row['source_id'] for row in excluded_dev})[:20],
        scoring_unit='Distinct (source case ID, text, label) development records; no label voting.',
        retained_development_rows=len(dev))
    train=records(read_json(root/'Train/train.json'),'train',audit,
        reserved_ids=held_out|{row['id'] for row in dev}|test_ids)
    removed=[row for row in train if row['source_id'] in held_out]
    audit['held_out_case_rows_removed']=len(removed)
    audit['held_out_source_case_ids_removed']=len({row['source_id'] for row in removed})
    audit['held_out_case_id_examples']=sorted({row['source_id'] for row in removed})[:20]
    train=[row for row in train if row['source_id'] not in held_out]
    if not train:raise ValueError('No training cases remain after held-out source-case filtering.')
    audit['training_rows_after_source_case_filter']=len(train)
    audit['development_id_resolution']=dev_audit;audit['test_id_resolution']=test_audit
    result=train+dev+test
    if len({row['id'] for row in result})!=len(result):raise ValueError('Internal record IDs unexpectedly collide.')
    return (result,audit) if return_audit else result
def prepare_data(root,tokenizer_path,output,chunk_tokens=512,overlap=64):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if not 0<=overlap<chunk_tokens:raise ValueError('Require 0 <= overlap < chunk_tokens.')
    identity=digest({'files':{name:file_hash(root/name) for name in DATA_FILES},'tokenizer':file_hash(tokenizer_path),
        'chunk_tokens':chunk_tokens,'overlap':overlap,'version':2})
    manifest=output/'manifest.json'
    if manifest.exists():
        old=read_json(manifest)
        if old['fingerprint']!=identity:raise ValueError('Data/tokenization changed; use a new PROJECT.')
        if any(old['docs'][i]['gold'] is not None for i in old['splits']['test']):raise ValueError('Unsafe cache contains test targets. Use a new PROJECT; never train or refit with test labels.')
        if (output/'tokens.i32').stat().st_size!=4*old['total_tokens']:raise ValueError('Incomplete token cache.')
        return str(manifest)
    rows,record_audit=load_records(root,return_audit=True);dev_hash={digest(' '.join(r['text'].split())) for r in rows if r['split']=='dev'}
    removed=[r['id'] for r in rows if r['split']=='train' and digest(' '.join(r['text'].split())) in dev_hash]
    tokenizer=Tokenizer.from_file(str(tokenizer_path));tokenizer.no_padding();tokenizer.no_truncation()
    unk=tokenizer.token_to_id('<unk>');unk=0 if unk is None else unk
    docs=[];token_count=chunk_count=empty=0;temporary=output/'tokens.i32.tmp'
    with temporary.open('wb') as handle:
        for index,row in enumerate(rows):
            ids=tokenizer.encode(row['text'],add_special_tokens=False).ids;empty+=int(not ids);ids=ids or [unk]
            np.asarray(ids,dtype='<i4').tofile(handle);n=len(ids);windows=[]
            final=max(0,n-chunk_tokens);starts=list(range(0,final+1,chunk_tokens-overlap))
            if starts[-1]!=final:starts.append(final)
            for start in starts:
                length=min(chunk_tokens,n-start);windows.append([token_count+start,length,start/max(1,n-1)])
            docs.append({'id':row['id'],'gold':row['gold'],'split':row['split'],'token_count':n,'chunks':windows,
                'feature_start':chunk_count,'text_sha256':digest(row['text']),'source_id':row.get('source_id',row['id'])})
            token_count+=n;chunk_count+=len(windows)
            if (index+1)%2000==0:print('Tokenized cases:',index+1,flush=True)
    temporary.replace(output/'tokens.i32')
    splits={s:[i for i,d in enumerate(docs) if d['split']==s and d['id'] not in removed] for s in ('train','dev','test')}
    splits['all']=splits['train']+splits['dev'];splits['every']=list(range(len(docs)))
    if any(not splits[s] for s in ('train','dev','test')):raise ValueError('Deduplication left an empty split.')
    if set(docs[i]['gold'] for i in splits['train'])!={0,1}:raise ValueError('Both binary classes are required in training.')
    value={'fingerprint':identity,'docs':docs,'splits':splits,'total_tokens':token_count,'total_chunks':chunk_count,
        'chunk_tokens':chunk_tokens,'overlap':overlap,'tokenizer_sha256':file_hash(tokenizer_path),
        'audit':{'empty_documents':empty,'training_dev_duplicates_removed':removed,'training_id_resolution':record_audit,'all_tokens_retained':True,
            'case_counts':{s:len(v) for s,v in splits.items()},'max_chunks_per_case':max(len(d['chunks']) for d in docs),
            'label_support':{s:{str(y):sum(docs[i]['gold']==y for i in splits[s]) for y in (0,1)} for s in ('train','dev')}}}
    atomic_json(manifest,value);return str(manifest)

class Bags(Dataset):
    def __init__(self,manifest,split,features=None,teacher=None,view='all'):
        self.path=Path(manifest);self.data=read_json(self.path);self.rows=self.data['splits'][split]
        self.tokens=np.memmap(self.path.parent/'tokens.i32',dtype='<i4',mode='r')
        self.features=None
        if features:
            info=read_json(Path(features)/'manifest.json')
            if not info['complete'] or info['data_fingerprint']!=self.data['fingerprint']:raise ValueError('Frozen feature cache identity/completion mismatch.')
            self.features=np.memmap(Path(features)/'features.f16',dtype='<f2',mode='r',shape=tuple(info['shape']))
        if view not in ('all','last'):raise ValueError('Unknown training context view.')
        self.teacher,self.view=teacher,view
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):return self.document(self.rows[index])
    def document(self,index):
        doc=self.data['docs'][index];windows=doc['chunks'];start=doc['feature_start']
        if self.view=='last':start+=len(windows)-1;windows=windows[-1:]
        return {'index':index,'id':doc['id'],'chunks':[np.array(self.tokens[a:a+n],dtype=np.int64) for a,n,p in windows] if self.features is None else None,
            'features':torch.from_numpy(np.array(self.features[start:start+len(windows)],dtype=np.float32)) if self.features is not None else None,
            'position':[[p,1-p,p*p,float(j==0),float(j==len(windows)-1),math.log1p(n)/8] for j,(a,n,p) in enumerate(windows)],
            'target':doc['gold'],'teacher':self.teacher[doc['id']] if self.teacher is not None else None}
def collate(rows):
    B=len(rows);S=max(len(r['position']) for r in rows);positions=torch.zeros(B,S,6);mask=torch.zeros(B,S,dtype=torch.bool)
    owners=[];chunks=[];features=None
    if rows[0]['features'] is not None:features=torch.zeros(B,S,rows[0]['features'].shape[-1])
    for b,row in enumerate(rows):
        n=len(row['position']);positions[b,:n]=torch.tensor(row['position']);mask[b,:n]=True
        if features is not None:features[b,:n]=row['features']
        else:
            chunks.extend(row['chunks']);owners.extend(b*S+j for j in range(n))
    return {'chunks':chunks,'owners':owners,'features':features,'position':positions,'mask':mask,
        'target':torch.tensor([r['target'] if r['target'] is not None else -100 for r in rows]),
        'teacher':torch.stack([torch.as_tensor(r['teacher']) for r in rows]) if rows[0]['teacher'] is not None else None,
        'rows':rows}
def to_device(batch,device):return {k:v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
def metric(y,probability,threshold=.5):
    pred=(np.asarray(probability)>=threshold).astype(int);y=np.asarray(y)
    return {'macro_f1':float(f1_score(y,pred,average='macro',labels=[0,1],zero_division=0)),
        'precision':float(precision_score(y,pred,average='binary',pos_label=1,zero_division=0)),
        'recall':float(recall_score(y,pred,average='binary',pos_label=1,zero_division=0)),
        'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist(),'threshold':float(threshold)}
def choose_calibration(y,neural,lexical=None):
    best=None
    for weight in ((0.,.15,.3) if lexical is not None else (0.,)):
        p=np.asarray(neural)*(1-weight)+(np.asarray(lexical)*weight if lexical is not None else 0)
        for threshold in (.5,.45,.55,.4,.6):
            scores=metric(y,p,threshold)
            if best is None or scores['macro_f1']>best[0]['macro_f1']+1e-10:best=(scores,{'lexical_weight':weight,'threshold':threshold})
    return best
def probabilities(values):
    x=np.asarray(values,dtype=np.float64);x-=x.max(-1,keepdims=True);e=np.exp(x);return e[:,1]/e.sum(-1)
def write_submission(ids,preds,path):
    if len(ids)!=len(set(ids)) or len(ids)!=len(preds):raise ValueError('Submission ID/count mismatch.')
    if any(type(y) is not int or y not in (0,1) for y in preds):raise ValueError('Predictions must be integer 0/1.')
    records=[{'id':did,'pred':y} for did,y in zip(ids,preds)];raw=json.dumps(records,ensure_ascii=False).encode()
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as archive:archive.writestr('prediction.json',raw)
    tmp.replace(path);return records,hashlib.sha256(raw).hexdigest()
