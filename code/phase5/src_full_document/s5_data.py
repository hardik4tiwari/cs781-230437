"""Full documents, train-only sparse vocabulary and test-target isolation."""
import gc, math, shutil, json
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from lsi_data import load_records, read_json, atomic_json, file_hash, digest, DATA_FILES

def prepare(root, tokenizer_path, output, cfg):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fingerprint=digest({'sources':{n:file_hash(Path(root)/n) for n in DATA_FILES},
        'tokenizer':file_hash(tokenizer_path),'chunk':cfg['chunk_tokens'],
        'word_features':cfg['word_features'],'char_features':cfg['char_features'],'version':3})
    manifest=output/'manifest.json'
    if manifest.exists():
        old=read_json(manifest)
        if old['fingerprint']!=fingerprint:raise ValueError('Inputs/settings changed. Use a new PROJECT.')
        for name,size in old['byte_sizes'].items():
            if not (output/name).is_file() or (output/name).stat().st_size!=size:raise ValueError('Incomplete cache: '+name)
        if any(d['gold'] is not None for d in old['docs'] if d['split']=='test'):raise ValueError('Unsafe test cache.')
        return str(manifest)
    rows,audit=load_records(root,True)
    splits={s:[i for i,r in enumerate(rows) if r['split']==s] for s in ('train','dev','test')}
    label_ids=sorted({j for r in rows if r['split']!='test' for j in r['gold']})
    C=max(label_ids)+1
    if C>5000:raise ValueError('Unexpected label count; verify mapped statute IDs.')
    support=np.zeros(C,dtype=np.int64)
    for i in splits['train']:support[rows[i]['gold']]+=1
    tok=Tokenizer.from_file(str(tokenizer_path));tok.no_padding();tok.no_truncation()
    unknown=tok.token_to_id('<unk>');unknown=0 if unknown is None else unknown
    docs=[];offset=0
    with (output/'tokens.i32').open('wb') as handle:
        for i,r in enumerate(rows):
            ids=tok.encode(r['text'],add_special_tokens=False).ids or [unknown]
            np.asarray(ids,dtype='<i4').tofile(handle)
            # Non-overlapping chunks cover each token exactly once, including the final partial chunk.
            windows=[[offset+a,min(cfg['chunk_tokens'],len(ids)-a)] for a in range(0,len(ids),cfg['chunk_tokens'])]
            docs.append({'id':r['id'],'gold':r['gold'],'split':r['split'],'chunks':windows,'token_count':len(ids)})
            offset+=len(ids)
            if (i+1)%5000==0:print('Tokenized',i+1,'cases',flush=True)
    train_text=[rows[i]['text'] for i in splits['train']]
    vectorizers=[];blocks=[]
    for kind,limit in [('word',cfg['word_features']),('char',cfg['char_features'])]:
        if not limit:continue
        vector=TfidfVectorizer(analyzer='word' if kind=='word' else 'char_wb',
            ngram_range=(1,2) if kind=='word' else (3,5),max_features=limit,
            min_df=min(3 if kind=='word' else 8,max(1,len(train_text)//4)),
            sublinear_tf=True,strip_accents='unicode',lowercase=True,dtype=np.float32)
        vector.fit(train_text) # No dev/test vocabulary or IDF fitting.
        print('Sparse vocabulary:',kind,len(vector.vocabulary_),flush=True)
        pieces=[vector.transform([r['text'] for r in rows[a:a+512]]) for a in range(0,len(rows),512)]
        blocks.append(sparse.vstack(pieces,format='csr'));vectorizers.append(vector);del pieces;gc.collect()
    matrix=sparse.hstack(blocks,format='csr',dtype=np.float32)/math.sqrt(len(blocks));matrix=matrix.tocsr()
    matrix.sort_indices();del blocks,train_text;gc.collect()
    np.save(output/'sparse_indices.npy',matrix.indices.astype(np.int64))
    np.save(output/'sparse_indptr.npy',matrix.indptr.astype(np.int64))
    np.save(output/'sparse_values.npy',matrix.data.astype(np.float32))
    joblib.dump(vectorizers,output/'vectorizers.joblib')
    value={'fingerprint':fingerprint,'tokenizer_sha256':file_hash(tokenizer_path),'docs':docs,'splits':splits,
        'label_count':C,'label_ids':label_ids,'support':support.tolist(),'sparse_dim':matrix.shape[1],
        'token_count':offset,'chunk_count':sum(len(d['chunks']) for d in docs),'audit':audit,
        'training_policy':'train only; full document; no test targets; no automatic train+dev refit'}
    names=['tokens.i32','sparse_indices.npy','sparse_indptr.npy','sparse_values.npy','vectorizers.joblib']
    value['byte_sizes']={n:(output/n).stat().st_size for n in names};atomic_json(manifest,value)
    print('Data counts:',{s:len(v) for s,v in splits.items()},'labels',C,'chunks',value['chunk_count'],flush=True)
    return str(manifest)

class Documents:
    def __init__(self,manifest):
        self.path=Path(manifest);self.data=read_json(manifest)
        self.tokens=np.memmap(self.path.parent/'tokens.i32',mode='r',dtype='<i4')
        self.indices=np.load(self.path.parent/'sparse_indices.npy',mmap_mode='r')
        self.indptr=np.load(self.path.parent/'sparse_indptr.npy',mmap_mode='r')
        self.values=np.load(self.path.parent/'sparse_values.npy',mmap_mode='r')
    def batch(self,indices,device):
        chunks=[];counts=[];features=[];values=[];offsets=[0];targets=[];positions=[]
        for i in indices:
            d=self.data['docs'][int(i)];counts.append(len(d['chunks']))
            for j,(a,n) in enumerate(d['chunks']):
                chunks.append(np.array(self.tokens[a:a+n],dtype=np.int64))
                positions.append([j/max(1,len(d['chunks'])-1),float(j==0),float(j==len(d['chunks'])-1)])
            a,b=self.indptr[i:i+2];features.extend(self.indices[a:b]);values.extend(self.values[a:b]);offsets.append(len(features))
            if d['gold'] is not None:
                y=np.zeros(self.data['label_count'],np.float32);y[d['gold']]=1;targets.append(y)
        if targets and len(targets)!=len(indices):raise ValueError('Mixed train and test batches are forbidden.')
        return {'indices':list(map(int,indices)),'chunks':chunks,'counts':counts,
            'position':torch.as_tensor(positions,dtype=torch.float32,device=device),
            'sparse_ids':torch.as_tensor(features,dtype=torch.long,device=device),
            'sparse_values':torch.as_tensor(values,dtype=torch.float32,device=device),
            'sparse_offsets':torch.as_tensor(offsets,dtype=torch.long,device=device),
            'target':torch.as_tensor(np.stack(targets),device=device) if targets else None}

def chunk_tensors(chunks,device):
    T=max(map(len,chunks));ids=torch.zeros(len(chunks),T,dtype=torch.long,device=device)
    valid=torch.zeros_like(ids,dtype=torch.bool)
    for i,row in enumerate(chunks):
        ids[i,:len(row)]=torch.as_tensor(row,device=device);valid[i,:len(row)]=True
    return ids,valid

