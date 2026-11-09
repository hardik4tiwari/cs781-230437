"""Official Phase-3 RR parsing, full-sentence token coverage and fixed dev scoring."""
import hashlib,json,math,zipfile
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

LABELS=['Fact','Issue','ArgumentPetitioner','ArgumentRespondent','Statute','Dissent','PrecedentReliedUpon',
        'PrecedentNotReliedUpon','PrecedentOverruled','RulingByLowerCourt','RatioOfTheDecision','RulingByPresentCourt','None']
DATA_FILES=['Train/CL_train.json','Train/CL_dev.json','Train/IT_train.json','Train/IT_dev.json','Train/label_vocab.json',
            'Test/CL_test.json','Test/IT_test.json']
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
def read_json(path):
    def pairs(rows):
        result={}
        for key,value in rows:
            if key in result:raise ValueError(f'Duplicate JSON key: {key}')
            result[key]=value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),object_pairs_hook=pairs)
def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
def atomic_torch(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    torch.save(data,tmp);tmp.replace(path)
def normalized_id(did):
    if not isinstance(did,str) or not did.strip():raise ValueError('Document IDs must be nonempty strings.')
    return did[:-4] if did.endswith('.txt') else did
def label_vocabulary(data):
    if isinstance(data,dict) and 'label2id' in data:data=data['label2id']
    elif isinstance(data,dict) and 'id2label' in data:data=data['id2label']
    if isinstance(data,list):labels=data;ids={i:l for i,l in enumerate(labels)}
    elif isinstance(data,dict) and all(type(v) is int for v in data.values()):
        if len(set(data.values()))!=len(data):raise ValueError('Repeated label IDs.')
        ids={v:k for k,v in data.items()};labels=[ids[i] for i in sorted(ids)]
    elif isinstance(data,dict) and all(isinstance(v,str) for v in data.values()):
        ids={int(k):v for k,v in data.items()};labels=[ids[i] for i in sorted(ids)]
    else:raise ValueError('Unsupported label_vocab.json format.')
    if len(labels)!=13 or len(set(labels))!=13 or set(labels)!=set(LABELS):raise ValueError('Official 13-label vocabulary mismatch.')
    return labels,ids
def _rr_shape(value,depth=0):
    # Diagnostics describe structure only; do not print judgment text.
    if isinstance(value,dict):
        fields=list(value.items())[:6]
        return {str(k):_rr_shape(v,depth+1) if depth<2 else type(v).__name__ for k,v in fields}
    if isinstance(value,list):return {'type':'list','length':len(value),'first':_rr_shape(value[0],depth+1) if value and depth<2 else None}
    return type(value).__name__
def _rr_alias(row,names):
    found=[name for name in names if name in row]
    if len(found)>1 and any(row[name]!=row[found[0]] for name in found[1:]):
        raise ValueError(f'Conflicting fields: {found}')
    return row[found[0]] if found else None
def _rr_document_rows(data):
    if isinstance(data,dict) and len(data)==1 and next(iter(data)) in ('data','documents','records'):
        return _rr_document_rows(next(iter(data.values())))
    if isinstance(data,list):return [(None,row) for row in data]
    if not isinstance(data,dict):raise ValueError(f'Unsupported document container: {_rr_shape(data)}')
    text=_rr_alias(data,('text','sentences'))
    if text is None:return list(data.items())
    did=_rr_alias(data,('id','ids','doc_id','document_id'))
    gold=_rr_alias(data,('labels','label','roles'))
    if isinstance(did,str):return [(None,{'id':did,'text':text,'labels':gold})]
    # Hugging Face Dataset.to_dict(): column name -> one value per document.
    if isinstance(text,list) and isinstance(did,list):
        if len(did)!=len(text) or (gold is not None and (not isinstance(gold,list) or len(gold)!=len(text))):
            raise ValueError('Document columns have different lengths.')
        return [(None,{'id':did[i],'text':sentences,'labels':None if gold is None else gold[i]}) for i,sentences in enumerate(text)]
    # ID-keyed text/label columns, or pandas columns keyed by row index with an explicit id column.
    if isinstance(text,dict):
        for name,column in (('id',did),('labels',gold)):
            if column is not None and (not isinstance(column,dict) or set(column)!=set(text)):
                raise ValueError(f'{name} and text column keys differ.')
        return [(key if did is None else None,{'id':key if did is None else did[key],
                    'text':sentences,'labels':None if gold is None else gold[key]}) for key,sentences in text.items()]
    raise ValueError(f'Document has no explicit string ID, or columns are unsupported: {_rr_shape(data)}')
def _rr_sentence_fields(payload,id_labels):
    if isinstance(payload,dict):
        text=_rr_alias(payload,('text','sentences'))
        gold=_rr_alias(payload,('labels','label','roles'))
    elif isinstance(payload,list):text,gold=payload,None
    else:raise ValueError(f'Document payload must be an object or sentence list; got {type(payload).__name__}.')
    known=lambda x:(type(x) is int and x in id_labels) or (isinstance(x,str) and x in LABELS)
    # Packed document: [sentence_strings, sentence_labels]. Do not confuse two sentence pairs with columns.
    if gold is None and isinstance(text,list) and len(text)==2 and all(isinstance(x,list) for x in text):
        if all(isinstance(x,str) for x in text[0]) and all(known(x) for x in text[1]):
            if all(len(x)==2 and isinstance(x[0],str) and known(x[1]) for x in text):
                raise ValueError('Ambiguous two-row payload; use explicit text and labels fields.')
            text,gold=text
    if not isinstance(text,list):raise ValueError('Expected sentence list; raw document strings are not resegmented.')
    if all(isinstance(x,str) for x in text):return text,gold
    # ID -> [{text/sentence: str, label/role: scalar}, ...] or ID -> [[sentence,label], ...].
    sentences=[];embedded=[];has_label=[]
    for index,item in enumerate(text):
        if isinstance(item,dict):
            sentence=_rr_alias(item,('text','sentence'))
            label=_rr_alias(item,('labels','label','role'))
            present=any(name in item for name in ('labels','label','role'))
        elif isinstance(item,list) and len(item)==2 and isinstance(item[0],str):
            sentence,label=item;present=True
        else:raise ValueError(f'Unsupported sentence entry at position {index}: {_rr_shape(item)}')
        if not isinstance(sentence,str):raise ValueError(f'Sentence {index} must be a string.')
        if present and not known(label):raise ValueError(f'Sentence {index} has an unknown or nonscalar role label.')
        sentences.append(sentence);embedded.append(label);has_label.append(present)
    if any(has_label) and not all(has_label):raise ValueError('Only some sentence entries contain labels.')
    if any(has_label):
        if gold is not None and gold!=embedded:raise ValueError('Sentence labels conflict with document labels.')
        gold=embedded
    return sentences,gold
def documents(data,domain,split,id_labels):
    result={};normalized=set()
    try:
        for key,row in _rr_document_rows(data):
            did=_rr_alias(row,('id','doc_id','document_id')) if isinstance(row,dict) else None
            did=key if did is None else did;nid=normalized_id(did)
            if key is not None and normalized_id(key)!=nid:raise ValueError('Mapping ID differs from document id.')
            if nid in normalized:raise ValueError('Duplicate IDs after official .txt normalization.')
            normalized.add(nid);text,gold=_rr_sentence_fields(row,id_labels)
            if split!='test':
                if not isinstance(gold,list) or len(gold)!=len(text):raise ValueError(f'{did}: label/sentence count mismatch.')
                gold=[id_labels[x] if type(x) is int and x in id_labels else x for x in gold]
                if any(not isinstance(x,str) or x not in LABELS for x in gold):raise ValueError(f'{did}: unknown rhetorical-role label.')
            elif gold is not None:raise ValueError('Public test records must not contain labels.')
            result[did]={'text':text,'gold':gold,'domain':domain,'split':split}
    except ValueError as error:
        raise ValueError(f'{domain}/{split}: {error} JSON structure: {_rr_shape(data)}') from error
    return result
def load_data(root):
    root=Path(root);labels,id_labels=label_vocabulary(read_json(root/'Train/label_vocab.json'))
    docs={};all_ids=set()
    for domain in ('CL','IT'):
        for split in ('train','dev','test'):
            folder='Test' if split=='test' else 'Train'
            local=documents(read_json(root/f'{folder}/{domain}_{split}.json'),domain,split,id_labels)
            norm={normalized_id(did) for did in local}
            if norm&all_ids:raise ValueError('Official normalized IDs overlap across domains/splits.')
            docs.update(local);all_ids|=norm
    return docs,labels
def sentence_blocks(docs,ids,size=12,overlap=4):
    if not 0<=overlap<size:raise ValueError('Invalid sentence-context overlap.')
    rows=[]
    for did in ids:
        n=len(docs[did]['text']);a=0
        while a<n:
            b=min(a+size,n);rows.append((did,a,b))
            if b==n:break
            a=b-overlap
    return rows
def prepare_cache(root,tokenizer_path,path,chunk_tokens=128,block_sentences=12,block_overlap=4):
    docs,labels=load_data(root);tok=Tokenizer.from_file(str(tokenizer_path));tok.no_padding();tok.no_truncation()
    identity=digest({'data':docs,'labels':labels,'tokenizer':file_hash(tokenizer_path),'chunk_tokens':chunk_tokens,
        'block_sentences':block_sentences,'block_overlap':block_overlap,'version':1})
    path=Path(path)
    if path.exists():
        old=torch.load(path,map_location='cpu',weights_only=False)
        if old['fingerprint']!=identity:raise ValueError('Dataset/cache changed; use a fresh PROJECT directory.')
        return old
    dev_hashes={digest(d['text']) for d in docs.values() if d['split']=='dev'}
    removed=[did for did,d in docs.items() if d['split']=='train' and digest(d['text']) in dev_hashes]
    splits={s:[did for did,d in docs.items() if d['split']==s and did not in removed] for s in ('train','dev','test')}
    splits['all']=splits['train']+splits['dev']
    if not splits['train'] or not splits['dev'] or not splits['test']:raise ValueError('Need nonempty public train/dev/test splits.')
    unk=tok.token_to_id('<unk>');unk=0 if unk is None else unk
    audit={'sentences':0,'tokens':0,'chunks':0,'long_sentences':0,'max_sentence_tokens':0,'empty_sentences':0,
           'training_duplicate_documents_removed':removed}
    for d in docs.values():
        d['chunks']=[];d['features']=[];n=len(d['text'])
        for index,sentence in enumerate(d['text']):
            tokens=tok.encode(sentence,add_special_tokens=False).ids
            audit['sentences']+=1;audit['tokens']+=len(tokens);audit['empty_sentences']+=int(not tokens)
            audit['max_sentence_tokens']=max(audit['max_sentence_tokens'],len(tokens));audit['long_sentences']+=int(len(tokens)>chunk_tokens)
            chunks=[tokens[i:i+chunk_tokens] for i in range(0,len(tokens),chunk_tokens)] or [[unk]]
            d['chunks'].append(chunks);audit['chunks']+=len(chunks)
            p=index/max(n-1,1)
            d['features'].append([p,1-p,p*p,float(index==0),float(index==n-1),min(math.log1p(len(tokens))/8,2.)])
        d['target']=None if d['gold'] is None else [labels.index(l) for l in d['gold']]
    cache={'fingerprint':identity,'docs':docs,'labels':labels,'splits':splits,'chunk_tokens':chunk_tokens,
        'tokenizer_sha256':file_hash(tokenizer_path),'block_sentences':block_sentences,'block_overlap':block_overlap,'audit':audit}
    for split,ids in splits.items():cache[split+'_blocks']=sentence_blocks(docs,ids,block_sentences,block_overlap)
    cache['support']={split:{l:sum(docs[did]['gold'].count(l) for did in splits[split]) for l in labels} for split in ('train','dev')}
    atomic_torch(path,cache);return cache

class Blocks(Dataset):
    def __init__(self,cache,split,features=None,teacher=None):
        self.cache,self.rows,self.features,self.teacher=cache,cache[split+'_blocks'],features,teacher
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        did,a,b=self.rows[index];doc=self.cache['docs'][did]
        return {'did':did,'start':a,'end':b,'chunks':doc['chunks'][a:b],
            'position':doc['features'][a:b],'domain':0 if doc['domain']=='CL' else 1,
            'target':doc['target'][a:b] if doc['target'] is not None else None,
            'features':self.features[did][a:b].float() if self.features is not None else None,
            'teacher':self.teacher[did][a:b].float() if self.teacher is not None else None}
def collate(rows):
    B=len(rows);S=max(x['end']-x['start'] for x in rows)
    positions=torch.zeros(B,S,6);mask=torch.zeros(B,S,dtype=torch.bool);targets=torch.full((B,S),-100,dtype=torch.long)
    chunks=[];owners=[];lengths=[];features=None;teacher=None
    if rows[0]['features'] is not None:features=torch.zeros(B,S,rows[0]['features'].size(-1))
    if rows[0]['teacher'] is not None:teacher=torch.zeros(B,S,rows[0]['teacher'].size(-1))
    for row,item in enumerate(rows):
        n=item['end']-item['start'];mask[row,:n]=True;positions[row,:n]=torch.tensor(item['position'])
        if item['target'] is not None:targets[row,:n]=torch.tensor(item['target'])
        if features is not None:features[row,:n]=item['features']
        else:
            for sentence,parts in enumerate(item['chunks']):
                for part in parts:chunks.append(part);owners.append(row*S+sentence);lengths.append(len(part))
        if teacher is not None:teacher[row,:n]=item['teacher']
    T=max(map(len,chunks),default=1);ids=torch.zeros(len(chunks),T,dtype=torch.long);valid=torch.zeros_like(ids,dtype=torch.bool)
    for i,chunk in enumerate(chunks):ids[i,:len(chunk)]=torch.tensor(chunk);valid[i,:len(chunk)]=True
    return {'ids':ids,'valid':valid,'owners':torch.tensor(owners,dtype=torch.long),'chunk_lengths':torch.tensor(lengths,dtype=torch.float32),
        'mask':mask,'position':positions,'domain':torch.tensor([r['domain'] for r in rows]),'targets':targets,
        'features':features,'teacher':teacher,'rows':rows}
def to_device(batch,device):return {k:v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
def metric(prediction,truth,labels):
    if set(prediction)!=set(truth):raise ValueError('Metric document IDs differ.')
    matrix=np.zeros((len(labels),len(labels)),dtype=np.int64);ids={l:i for i,l in enumerate(labels)}
    for did,gold in truth.items():
        pred=prediction[did]
        if len(pred)!=len(gold):raise ValueError('Sentence-count mismatch.')
        for g,p in zip(gold,pred):matrix[ids[g],ids[p]]+=1
    tp=np.diag(matrix);support=matrix.sum(1);predicted=matrix.sum(0)
    precision=tp/np.maximum(predicted,1);recall=tp/np.maximum(support,1);f1=2*tp/np.maximum(support+predicted,1)
    return {'macro_f1':float(f1.mean()),'macro_precision':float(precision.mean()),'macro_recall':float(recall.mean()),
        'per_label':[{'label':l,'support':int(support[i]),'predicted':int(predicted[i]),'f1':float(f1[i]),
         'precision':float(precision[i]),'recall':float(recall[i])} for i,l in enumerate(labels)],'confusion_matrix':matrix.tolist()}
def validate_prediction(pred,docs,labels):
    if set(pred)!=set(docs):raise ValueError('Predictions must include every official ID exactly once.')
    norm=[normalized_id(did) for did in pred]
    if len(set(norm))!=len(norm):raise ValueError('IDs collide after official normalization.')
    for did,values in pred.items():
        if not isinstance(values,list) or len(values)!=len(docs[did]['text']):raise ValueError(f'{did}: sentence-count mismatch.')
        if any(x not in labels for x in values):raise ValueError(f'{did}: invalid label.')
def submission_zip(pred,docs,labels,path):
    validate_prediction(pred,docs,labels);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(pred,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
    tmp=path.with_suffix('.tmp.zip')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('prediction.json',data)
    tmp.replace(path)
    with zipfile.ZipFile(path) as z:
        if z.namelist()!=['prediction.json']:raise ValueError('Incorrect ZIP root.')
        validate_prediction(json.loads(z.read('prediction.json')),docs,labels)
    return digest(pred)
