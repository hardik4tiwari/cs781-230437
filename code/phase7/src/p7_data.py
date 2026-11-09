"""Official split-aware PCR data; IDs stay strings, test relevance never enters the cache."""
import collections,functools,hashlib,json,math,pickle,re
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

FILES={s:(f'{folder}/{s}.json',f'{folder}/{s}_candidates.json') for s,folder in [('train','Train'),('dev','Dev'),('test','Test')]}
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def read_json(path):
    def unique(pairs):
        d={}
        for k,v in pairs:
            if k in d:raise ValueError(f'Duplicate JSON mapping key {k!r}.')
            d[k]=v
        return d
    def bad(x):raise ValueError(f'Nonstandard JSON number {x}.')
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),object_pairs_hook=unique,parse_constant=bad)
def atomic_json(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
def atomic_pickle(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:pickle.dump(x,f,pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
def load_pickle(path):
    with open(path,'rb') as f:return pickle.load(f)
def alias(row,keys):
    present=[k for k in keys if k in row]
    if len(present)>1 and any(row[k]!=row[present[0]] for k in present):raise ValueError(f'Conflicting aliases: {present}')
    return row[present[0]] if present else None
def records(value,split,queries=True):
    if isinstance(value,dict) and len(value)==1 and next(iter(value)) in ('data','records','documents'):value=next(iter(value.values()))
    if isinstance(value,list):items=[(None,r) for r in value]
    elif isinstance(value,dict):
        ids=alias(value,('id','ids','case_id','doc_id'));texts=alias(value,('text','document','sentences'))
        gold=None if not queries or split=='test' else alias(value,('relevant_candidates','relevant','labels'))
        if isinstance(ids,str) and texts is not None:items=[(None,value)]
        elif isinstance(ids,list) and isinstance(texts,list):
            if len(ids)!=len(texts) or (queries and split!='test' and (not isinstance(gold,list) or len(gold)!=len(ids))):raise ValueError('Column lengths differ.')
            items=[(None,{'id':did,'text':texts[i],**({'relevant_candidates':gold[i]} if queries and split!='test' else {})}) for i,did in enumerate(ids)]
        elif isinstance(texts,dict):
            if queries and split!='test' and (not isinstance(gold,dict) or set(gold)!=set(texts)):raise ValueError('Column keys differ.')
            items=[(None,{'id':k,'text':t,**({'relevant_candidates':gold[k]} if queries and split!='test' else {})}) for k,t in texts.items()]
        else:items=list(value.items())
    else:raise ValueError('Expected an ID mapping or list of record objects.')
    out={};duplicates=0;blank_labels=0
    for key,row in items:
        if key is not None and isinstance(row,(str,list)):row={'id':key,'text':row}
        if not isinstance(row,dict):raise ValueError(f'{split}: record must be an object, got {type(row).__name__}.')
        did=alias(row,('id','case_id','doc_id'));did=key if did is None else did
        if not isinstance(did,str) or not did.strip():raise ValueError('Original nonempty string IDs are required, including leading zeros.')
        if key is not None and did!=key:raise ValueError('Mapping key and explicit document ID differ.')
        text=alias(row,('text','document','sentences'))
        if isinstance(text,list) and all(isinstance(s,str) for s in text):text='\n'.join(text)
        if not isinstance(text,str) or not text.strip():raise ValueError(f'{split}/{did}: expected nonempty text or sentence strings.')
        rel=None
        if queries and split!='test':
            rel=alias(row,('relevant_candidates','relevant','labels'))
            if not isinstance(rel,list):raise ValueError(f'{split}/{did}: relevant_candidates must be a list.')
            clean=[];seen=set()
            for c in rel:
                # Same blank-source policy as official load_truth; no integer coercion or zero-padding guesses.
                if not isinstance(c,str) or not c.strip():blank_labels+=1;continue
                if c in seen:raise ValueError(f'{split}/{did}: duplicate relevance ID {c}.')
                seen.add(c);clean.append(c)
            rel=clean
        # Supplied test labels are deliberately never inspected.
        record={'id':did,'text':text.strip(),'relevant':rel}
        if did in out:
            if out[did]!=record:raise ValueError(f'Conflicting repeated {split} ID {did}; inspect source data.')
            duplicates+=1;continue
        out[did]=record
    if not out:raise ValueError(f'Empty {split} data.')
    return out,{'exact_duplicate_rows_removed':duplicates,'blank_source_labels_ignored':blank_labels}
def normhash(text):return digest(' '.join(text.lower().split()))
def load_data(root):
    raw={};audit={}
    for split,(qfile,cfile) in FILES.items():
        queries,qa=records(read_json(Path(root)/qfile),split,True);candidates,ca=records(read_json(Path(root)/cfile),split,False)
        if len(candidates)<7:raise ValueError(f'{split}: candidate pool must contain at least 7 cases.')
        for q in queries.values():
            missing=set(q['relevant'] or [])-set(candidates)
            if missing:raise ValueError(f'{split}/{q["id"]}: relevance IDs outside its corresponding pool: {sorted(missing)[:5]}')
        raw[split]={'queries':queries,'candidates':candidates};audit[split]={'queries':qa,'candidates':ca}
    # Candidate overlaps are legitimate shared precedents. Do not delete candidate documents across pools.
    test=raw['test']['queries'];test_hashes={normhash(r['text']) for r in test.values()}
    removed_dev=[q for q,r in raw['dev']['queries'].items() if q in test or normhash(r['text']) in test_hashes]
    for q in removed_dev:del raw['dev']['queries'][q]
    held_ids=set(raw['dev']['queries'])|set(test);held_hashes={normhash(r['text']) for s in ('dev','test') for r in raw[s]['queries'].values()}
    removed_train=[q for q,r in raw['train']['queries'].items() if q in held_ids or normhash(r['text']) in held_hashes]
    for q in removed_train:del raw['train']['queries'][q]
    if any(not raw[s]['queries'] for s in FILES):raise ValueError('Query overlap filtering left an empty split.')
    audit.update(train_held_query_copies_removed=removed_train,dev_test_query_copies_removed=removed_dev,
                 candidate_id_overlap_counts={f'{a}_{b}':len(set(raw[a]['candidates'])&set(raw[b]['candidates'])) for a,b in [('train','dev'),('train','test'),('dev','test')]},
                 policy='Only train queries/relevance drive training. Dev is for selection. Test relevance is discarded. Candidate pools remain split-specific; shared precedents are allowed.')
    return raw,audit

STOP=set('a an the and or but if to of in on for by with is are was were be been being it its this that these those as at from has have had not no he she they their we our you your i court judgment case appellant respondent petitioner learned honble said shall would could also'.split())
WORD=re.compile(r'[a-z][a-z0-9_]*|\d+',re.I)
def terms(text):return [w for w in WORD.findall(text.lower()) if w not in STOP and len(w)>1]
class BM25:
    """Sparse inverted BM25. IDF is computed on each permitted candidate corpus, without relevance labels."""
    def __init__(self,max_features=200000,k1=1.2,b=.75):self.max_features=max_features;self.k1=k1;self.b=b
    def fit(self,texts):
        from sklearn.feature_extraction.text import CountVectorizer
        from scipy import sparse
        self.vectorizer=CountVectorizer(analyzer=terms,max_features=self.max_features,dtype=np.float32)
        X=self.vectorizer.fit_transform(texts).tocsr();N=X.shape[0];df=np.asarray((X>0).sum(0)).ravel()
        self.idf=np.log1p((N-df+.5)/(df+.5)).astype(np.float32);length=np.asarray(X.sum(1)).ravel();avg=max(1.,float(length.mean()))
        denominator=self.k1*(1-self.b+self.b*length/avg)
        X.data=(X.data*(self.k1+1)/(X.data+np.repeat(denominator,np.diff(X.indptr))))*self.idf[X.indices]
        self.matrix=X.tocsr();return self
    def scores(self,text):
        q=self.vectorizer.transform([text]).tocsr();q.data=np.log1p(q.data)
        # Bound very long query term sets by candidate IDF×query-TF; no labels are used.
        if q.nnz>512:
            keep=np.argsort(-(q.data*self.idf[q.indices]),kind='stable')[:512];q.data[np.setdiff1d(np.arange(q.nnz),keep)]=0;q.eliminate_zeros()
        return (q@self.matrix.T).toarray().ravel().astype(np.float32)
def top_indices(scores,n):return np.argsort(-scores,kind='stable')[:min(n,len(scores))]

def prepare_cache(root,tokenizer_path,output,settings):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);manifest=output/'manifest.json'
    relevant_settings={k:settings[k] for k in ('chunk_tokens','chunk_overlap','max_chunks','lexical_topk','bm25_features')}
    identity=digest({'files':{name:file_hash(Path(root)/name) for pair in FILES.values() for name in pair},'tokenizer':file_hash(tokenizer_path),'settings':relevant_settings,'version':1})
    if manifest.exists():
        old=read_json(manifest)
        if old['fingerprint']!=identity:raise ValueError('Data/tokenization settings changed. Use a new PROJECT.')
        if (output/'tokens.i32').stat().st_size!=4*old['total_tokens']:raise ValueError('Incomplete token cache. Use a new PROJECT to rebuild safely.')
        if any(r['relevant'] is not None for r in old['splits']['test']['queries'].values()):raise ValueError('Unsafe cache contains test relevance.')
        return str(manifest)
    raw,audit=load_data(root);tok=Tokenizer.from_file(str(tokenizer_path));tok.no_padding();tok.no_truncation()
    pad=next((tok.token_to_id(x) for x in ('<pad>','[PAD]') if tok.token_to_id(x) is not None),None)
    bos=next((tok.token_to_id(x) for x in ('<bos>','[BOS]','<s>') if tok.token_to_id(x) is not None),None)
    if pad is None or bos is None or pad==bos:raise ValueError('Tokenizer must contain distinct existing PAD and BOS tokens.')
    unique={};docs={};splits={};total=0;df=np.zeros(tok.get_vocab_size(),dtype=np.int64);train_candidate_count=0
    tmp=output/'tokens.i32.tmp';span=settings['chunk_tokens'];stride=span-settings['chunk_overlap']
    if stride<=0:raise ValueError('Chunk overlap must be smaller than chunk size.')
    with tmp.open('wb') as f:
        for split,pool in raw.items():
            splits[split]={'queries':{},'candidates':{}}
            for kind in ('queries','candidates'):
                for did,r in sorted(pool[kind].items()):
                    sha=digest(r['text']);key=f'{split}/{kind}/{did}'
                    if sha not in unique:
                        ids=tok.encode(r['text'],add_special_tokens=False).ids or [bos];n=len(ids)
                        starts=list(range(0,max(1,n-span+1),stride));last=max(0,n-span)
                        if starts[-1]!=last:starts.append(last)
                        # Preserve all token IDs in the memmap; neural training uses a bounded selected subset.
                        np.asarray(ids,dtype='<i4').tofile(f);windows=[[total+a,min(span,n-a),a/max(1,n-1)] for a in starts]
                        unique[sha]={'offset':total,'tokens':n,'windows':windows};total+=n
                    meta=unique[sha]
                    docs[key]={**meta,'id':did,'text_sha256':sha}
                    splits[split][kind][did]={'key':key,**({'relevant':r['relevant']} if kind=='queries' else {})}
                    if split=='train' and kind=='candidates':
                        ids=tok.encode(r['text'],add_special_tokens=False).ids;df[np.unique(ids)]+=1;train_candidate_count+=1
            print('Tokenized split:',split,flush=True)
    tmp.replace(output/'tokens.i32');idf=np.log1p((train_candidate_count+.5)/(df+.5)).astype(np.float32)
    np.save(output/'token_idf.npy',idf);mm=np.memmap(output/'tokens.i32',mode='r',dtype='<i4')
    for d in docs.values():
        windows=d['windows'];scores=[float(idf[np.unique(mm[o:o+n])].mean()) for o,n,_ in windows]
        d['order']=list(dict.fromkeys([0,len(windows)-1]+sorted(range(len(windows)),key=lambda j:(-scores[j],j))))
    lexical={};baseline={};ceilings={}
    from evaluate_phase7_official import score
    for split,pool in raw.items():
        ids=sorted(pool['candidates']);lookup={c:i for i,c in enumerate(ids)};index=BM25(settings['bm25_features']).fit([pool['candidates'][i]['text'] for i in ids]);entries={};prediction={}
        for qid,r in sorted(pool['queries'].items()):
            scores=index.scores(r['text']);order=top_indices(scores,settings['lexical_topk']);entries[qid]={'ids':[ids[i] for i in order],'scores':[float(scores[i]) for i in order]};prediction[qid]=entries[qid]['ids'][:7]
            if split=='train':
                entries[qid]['extra_scores']={c:float(scores[lookup[c]]) for c in r['relevant']}
        lexical[split]=entries;atomic_pickle(output/f'bm25_{split}.pkl',index)
        if split!='test':
            gold={q:r['relevant'] for q,r in pool['queries'].items()};baseline[split]=score(gold,prediction)
            found=sum(len(set(gold[q])&set(entries[q]['ids'])) for q in gold);all_gold=sum(len(g) for g in gold.values());ceilings[split]={'lexical_shortlist_recall':found/max(1,all_gold),'micro_f1_at7_upper_bound_within_shortlist':2*sum(min(7,len(set(gold[q])&set(entries[q]['ids']))) for q in gold)/max(1,7*len(gold)+all_gold)}
        atomic_json(output/f'{split}_candidate_ids.json',ids)
        print('BM25 indexed:',split,'queries',len(entries),'candidates',len(ids),flush=True)
    atomic_json(output/'lexical.json',lexical);audit.update(counts={s:{k:len(v) for k,v in sp.items()} for s,sp in splits.items()},baseline_bm25=baseline,shortlist_ceilings=ceilings,
        total_unique_document_texts=len(unique),total_token_ids=total,neural_view=f'At most {settings["max_chunks"]} selected {span}-token windows per document. Full tokens are cached; not all are neurally encoded.')
    value={'fingerprint':identity,'tokenizer_path':str(tokenizer_path),'tokenizer_sha256':file_hash(tokenizer_path),'special':{'pad':pad,'bos':bos},'cache_dir':str(output),'docs':docs,'splits':splits,'total_tokens':total,'settings':relevant_settings,'audit':audit}
    atomic_json(manifest,value);atomic_json(output/'audit.json',audit);return str(manifest)

class Store:
    def __init__(self,manifest):
        self.manifest=read_json(manifest);self.root=Path(self.manifest['cache_dir']);self.docs=self.manifest['docs'];self.splits=self.manifest['splits']
        self.tokens=np.memmap(self.root/'tokens.i32',mode='r',dtype='<i4');self.idf=np.load(self.root/'token_idf.npy');self.lexical=read_json(self.root/'lexical.json')
        self.train_candidate_ids=sorted(self.splits['train']['candidates'])
    def chunks(self,key,count=None):
        d=self.docs[key];count=count or self.manifest['settings']['max_chunks'];order=sorted(d['order'][:count]);return [(self.tokens[d['windows'][i][0]:d['windows'][i][0]+d['windows'][i][1]].copy(),d['windows'][i][2]) for i in order]
    def pair(self,qkey,dkey,width):
        query,qset=self.query_view(qkey,width);best=None;bestscore=-1
        for offset,n,unique in self.window_terms(dkey):
            overlap=np.intersect1d(qset,unique,assume_unique=True);s=float(self.idf[overlap].sum())
            if s>bestscore:bestscore=s;best=self.tokens[offset:offset+n].copy()
        dcap=width-len(query)-3;doc=best[:dcap];bos=self.manifest['special']['bos']
        ids=np.concatenate([[bos],query,[bos],doc,[bos]]).astype(np.int64)
        segment=np.asarray([0]*(len(query)+2)+[1]*(len(doc)+1),dtype=np.int64)
        return ids,segment
    @functools.lru_cache(maxsize=2048)
    def query_view(self,key,width):
        flat=np.concatenate([a for a,_ in self.chunks(key)]);cap=(width-3)//2
        query=np.concatenate([flat[:cap//2],flat[-(cap-cap//2):]]) if len(flat)>cap else flat
        # Window selection can use all selected query chunks even though the joint neural pair is shorter.
        return query,np.unique(flat)
    @functools.lru_cache(maxsize=2048)
    def window_terms(self,key):
        return tuple((offset,n,np.unique(self.tokens[offset:offset+n])) for offset,n,_ in self.docs[key]['windows'])
