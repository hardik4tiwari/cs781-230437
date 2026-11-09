"""Official Phase-6 data, train-only salience selector, auditable cached contexts."""
import hashlib,json,re,os,pickle,collections
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

FILES=['Train/train.json','Test/test.json']
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def file_hash(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def read_json(p):
    def unique(pairs):
        d={}
        for k,v in pairs:
            if k in d:raise ValueError(f'Duplicate JSON mapping key: {k}')
            d[k]=v
        return d
    return json.loads(Path(p).read_text(encoding='utf-8-sig'),object_pairs_hook=unique)
def atomic_json(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def alias(r,names):
    keys=[k for k in names if k in r]
    if len(keys)>1 and any(r[k]!=r[keys[0]] for k in keys):raise ValueError(f'Conflicting fields: {keys}')
    return r[keys[0]] if keys else None
DOCUMENT_FIELDS=('document','text','sentences')
SUMMARY_FIELDS=('summary','summaries','abstract','reference_summary','reference_summaries','gold_summary','summary_text','abstractive_summary','reference','highlights','target')
def text_value(v,fields,depth=0):
    """Normalize text and explicit text containers, never stringify arbitrary metadata."""
    if depth>5:return None
    if isinstance(v,str):return v.strip() or None
    if isinstance(v,list):
        if all(isinstance(s,str) or s is None for s in v):
            return '\n'.join(s.strip() for s in v if isinstance(s,str) and s.strip()) or None
        # A list of structured references can contain alternatives; use the first usable one.
        for item in v:
            result=text_value(item,fields,depth+1)
            if result:return result
    elif isinstance(v,dict):
        for key in fields:
            if key in v:
                result=text_value(v[key],fields,depth+1)
                if result:return result
    return None
def records(v,split,audit=None):
    if isinstance(v,dict) and len(v)==1 and next(iter(v)) in ('data','records','documents'):v=next(iter(v.values()))
    if isinstance(v,list):items=[(None,r) for r in v]
    elif isinstance(v,dict):
        ids=alias(v,('id','ids','doc_id','case_id'));texts=alias(v,DOCUMENT_FIELDS)
        summary_columns={} if split=='test' else {k:v[k] for k in SUMMARY_FIELDS if k in v}
        if isinstance(ids,str) and texts is not None:items=[(None,v)]
        elif isinstance(ids,list) and isinstance(texts,list):
            if len(ids)!=len(texts) or any(v is not None and (not isinstance(v,list) or len(v)!=len(ids)) for v in summary_columns.values()):raise ValueError(f'{split}: column lengths differ; cannot align documents and summaries by position.')
            items=[(None,{'id':k,'document':texts[i],**{f:col[i] for f,col in summary_columns.items() if col is not None}}) for i,k in enumerate(ids)]
        elif isinstance(texts,dict) and ids is None:
            if any(col is not None and (not isinstance(col,dict) or set(col)-set(texts)) for col in summary_columns.values()):raise ValueError(f'{split}: summary columns contain unexpected IDs or are not mappings.')
            items=[(None,{'id':k,'document':t,**{f:col.get(k) for f,col in summary_columns.items() if col is not None}}) for k,t in texts.items()]
        else:items=list(v.items())
    else:raise ValueError('Expected list of records or ID mapping.')
    out={};duplicates=0;skipped=[];fields=collections.Counter()
    for key,r in items:
        if isinstance(r,(str,list)) and key is not None:r={'id':key,'document':r}
        if not isinstance(r,dict):raise ValueError(f'{split}: record must be an object; found {type(r).__name__}.')
        did=alias(r,('id','doc_id','case_id'));did=key if did is None else did
        if not isinstance(did,str) or not did.strip():raise ValueError('Original nonempty string IDs required; IDs are never invented.')
        if key is not None and key!=did:raise ValueError('Mapping key and explicit ID differ.')
        text=text_value(alias(r,DOCUMENT_FIELDS),DOCUMENT_FIELDS+('content','sentence'))
        # Deliberately do not even inspect supplied test reference fields.
        summary=None
        if split!='test':
            for field in SUMMARY_FIELDS:
                if field not in r:continue
                summary=text_value(r[field],SUMMARY_FIELDS+('text','content','sentences','sentence'))
                if summary:
                    fields[field]+=1;break
        if not text or (split!='test' and not summary):
            reason='missing or unusable document' if not text else 'missing or unusable reference summary'
            if split=='test':raise ValueError(f'test ID {did!r}: {reason}; cannot silently drop an official test ID.')
            skipped.append({'id':did,'reason':reason,'field_types':{k:type(val).__name__ for k,val in r.items()}})
            continue
        row={'id':did,'document':text,'summary':summary,'split':split}
        if did in out:
            if row!=out[did]:raise ValueError(f'{split}: conflicting duplicate ID {did}; inspect source instead of inventing an ID.')
            duplicates+=1;continue
        out[did]=row
    if audit is not None:audit.update(input_rows=len(items),usable_rows=len(out),skipped_rows=skipped,reference_fields_used=dict(fields))
    if not out:raise ValueError(f'No usable {split} records. Examples (IDs and field types only): {skipped[:5]}')
    return list(out.values()),duplicates
def normhash(text):return digest(' '.join(text.lower().split()))
def split_data(root,seed=781,dev_fraction=.08,dev_max=512):
    train_audit={};test_audit={}
    train,duptr=records(read_json(Path(root)/FILES[0]),'train',train_audit);test,dupte=records(read_json(Path(root)/FILES[1]),'test',test_audit)
    forbidden_ids={r['id'] for r in test};forbidden_text={normhash(r['document']) for r in test}
    removed=[r['id'] for r in train if r['id'] in forbidden_ids or normhash(r['document']) in forbidden_text]
    removed_ids=set(removed);train=[r for r in train if r['id'] not in removed_ids]
    # Group exact normalized document copies before splitting; retain one training reference.
    groups=collections.defaultdict(list)
    for r in train:groups[normhash(r['document'])].append(r)
    keys=sorted(groups);np.random.default_rng(seed).shuffle(keys)
    if len(keys)<4:raise ValueError(f'Need at least four distinct labeled train documents after cleanup; got {len(keys)}. Skipped rows (examples): {train_audit["skipped_rows"][:5]}')
    n=max(1,min(dev_max,round(len(keys)*dev_fraction),len(keys)-2));held=set(keys[:n])
    rows=[];conflicting=0
    for key in keys:
        group=groups[key];conflicting+=len({r['summary'] for r in group})>1
        r=dict(sorted(group,key=lambda x:x['id'])[0]);r['split']='dev' if key in held else 'train';rows.append(r)
    rows+=test
    audit={'duplicate_id_copies_removed':{'train':duptr,'test':dupte},'train_test_overlap_removed':removed,
           'train_records':train_audit,'test_records':test_audit,
           'normalized_document_copies_collapsed':sum(len(v)-1 for v in groups.values()),
           'copy_groups_with_multiple_references':conflicting,'dev_group_count':n,
           'test_references':'discarded without inspection; never cached','seed':seed}
    return rows,audit
def sentences(text):
    # Deterministic, no NLTK resource downloads. Keep sentence boundaries and source order.
    parts=[s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9(])|\n+',text) if s.strip()]
    # Cap individual fragments for bounded selector features; no head-only document truncation.
    return [s[i:i+1800] for s in parts for i in range(0,len(s),1800)] or [text]
def wordset(s):return set(re.findall(r'[a-z0-9]+',s.lower()))
CUE=re.compile(r'\b(held|hold|therefore|dismissed|allowed|appeal|issue|court|section|petitioner|respondent|conclusion|ordered|judgment)\b',re.I)
class Selector:
    """Hashing salience regression + position/cue features; fitted on training references only."""
    def __init__(self,seed=781):
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.linear_model import SGDRegressor
        self.vectorizer=HashingVectorizer(n_features=2**18,ngram_range=(1,2),alternate_sign=False,norm='l2')
        self.model=SGDRegressor(loss='huber',epsilon=.15,alpha=1e-5,learning_rate='invscaling',eta0=.05,random_state=seed)
    def features(self,ss):
        from scipy.sparse import hstack,csr_matrix
        n=len(ss);extra=np.asarray([[i/max(1,n-1),float(i<3),float(i>=n-3),min(5,len(CUE.findall(s)))/5,min(1,len(s)/600)] for i,s in enumerate(ss)],dtype=np.float32)
        return hstack([self.vectorizer.transform(ss),csr_matrix(extra)],format='csr')
    def fit(self,rows,passes=2):
        for _ in range(passes):
            for r in rows:
                ss=sentences(r['document']);ref=wordset(r['summary']);y=[]
                for s in ss:
                    w=wordset(s);overlap=len(w&ref)
                    # Sentence-level lexical F score is a cheap salience training signal, not the evaluation metric.
                    y.append(2*overlap/max(1,len(w)+min(len(ref),80)))
                self.model.partial_fit(self.features(ss),np.asarray(y,dtype=np.float32))
        return self
    def select(self,text,tok,budget):
        ss=sentences(text);encoded=tok.encode_batch(ss,add_special_tokens=False);ids=[e.ids for e in encoded]
        if sum(map(len,ids))+len(ss)<=budget:return tok.encode('\n'.join(ss),add_special_tokens=False).ids[:budget],len(ss),len(ss)
        scores=self.model.predict(self.features(ss));sets=list(map(wordset,ss));selected=[];used=0;remaining=set(range(len(ss)))
        # Reserve a little space for facts and final disposition; fill most of the context by learned salience + MMR.
        for i in list(dict.fromkeys([0,len(ss)-1])):
            take=min(len(ids[i]),max(1,budget//12))
            if take:selected.append((i,ids[i][:take]));used+=take+1;remaining.discard(i)
        while remaining and used<budget-8:
            def score(i):
                redundancy=max((len(sets[i]&sets[j])/max(1,len(sets[i]|sets[j])) for j,_ in selected),default=0)
                return float(scores[i])-.15*redundancy
            i=max(remaining,key=lambda j:(score(j),-j));remaining.remove(i)
            room=budget-used-1
            if len(ids[i])>room and selected:continue
            selected.append((i,ids[i][:room]));used+=min(room,len(ids[i]))+1
        sep=tok.encode('\n',add_special_tokens=False).ids
        result=[]
        for _,tokens in sorted(selected):result+=tokens+sep
        return result[:budget],len(selected),len(ss)
def special_ids(tok,overrides=None):
    overrides=overrides or {};out={}
    for name,candidates in {'pad':['<pad>','[PAD]'],'bos':['<bos>','[BOS]','<s>'],'eos':['<eos>','[EOS]','</s>']}.items():
        value=overrides.get(name)
        if value is None:value=next((tok.token_to_id(s) for s in candidates if tok.token_to_id(s) is not None),None)
        if value is None:raise ValueError(f'Cannot identify {name} token. Set SPECIAL_IDS explicitly from tokenizer vocabulary.')
        if not 0<=int(value)<tok.get_vocab_size():raise ValueError('Special ID outside tokenizer vocabulary.')
        out[name]=int(value)
    if len(set(out.values()))!=3:raise ValueError('PAD, BOS and EOS must be distinct existing tokens.')
    return out
def prepare_data(root,tokenizer_path,output,settings):
    import torch
    output=Path(output);output.mkdir(parents=True,exist_ok=True);path=output/'cache_v4.pt'
    identity=digest({'files':{n:file_hash(Path(root)/n) for n in FILES},'tokenizer':file_hash(tokenizer_path),'settings':settings,'version':4})
    if path.exists():
        cache=torch.load(path,map_location='cpu',weights_only=False)
        if cache['fingerprint']!=identity:raise ValueError('Inputs/settings changed. Choose a new PROJECT to rebuild cleanly.')
        if any(r['target'] is not None or r['summary'] is not None for r in cache['rows'] if r['split']=='test'):raise ValueError('Unsafe cache has test references.')
        return str(path)
    rows,audit=split_data(root,settings['seed'],settings['dev_fraction'],settings['dev_max'])
    skipped=audit['train_records']['skipped_rows']
    print(f'Training input audit: {audit["train_records"]["usable_rows"]} labeled records; {len(skipped)} unusable rows excluded; reference fields: {audit["train_records"]["reference_fields_used"]}',flush=True)
    if skipped:print(f'Excluded training IDs (first 10): {[r["id"] for r in skipped[:10]]}; full details in token_cache/audit.json.',flush=True)
    atomic_json(output/'audit.json',audit)
    tok=Tokenizer.from_file(str(tokenizer_path));tok.no_padding();tok.no_truncation();special=special_ids(tok,settings.get('special_ids'))
    train=[r for r in rows if r['split']=='train'];selector=Selector(settings['seed']).fit(train)
    tmp=output/'selector.pkl.tmp'
    with tmp.open('wb') as f:pickle.dump(selector,f)
    tmp.replace(output/'selector.pkl')
    target_lengths=[len(e.ids) for e in tok.encode_batch([r['summary'] for r in train],add_special_tokens=False)]
    target_cap=min(settings['target_tokens'],max(64,int(np.quantile(target_lengths,.95))+1))
    retained=[];clipped=0;cached=[]
    for i,r in enumerate(rows):
        source,nsel,ns=selector.select(r['document'],tok,settings['source_tokens']-64)
        prompt=tok.encode('Summarize the legal judgment, including the material facts, legal issue, reasoning and final decision.\nJudgment:\n',add_special_tokens=False).ids+source+tok.encode('\nSummary:\n',add_special_tokens=False).ids
        prompt=prompt[:settings['source_tokens']]
        target=None
        if r['split']!='test':
            target=tok.encode(r['summary'],add_special_tokens=False).ids;clipped+=len(target)>=target_cap
            target=target[:target_cap-1]+[special['eos']]
        cached.append({'id':r['id'],'split':r['split'],'source':prompt,'target':target,'summary':r['summary'],
                       'document_sha256':normhash(r['document']),'selected_sentences':nsel,'sentences':ns})
        retained.append(nsel/ns)
        if (i+1)%500==0:print('Cached documents:',i+1,flush=True)
    audit.update(counts={s:sum(r['split']==s for r in cached) for s in ('train','dev','test')},
                 target_token_quantiles_train={str(q):float(np.quantile(target_lengths,q)) for q in (.5,.9,.95,.99)},
                 target_cap=target_cap,references_clipped_at_target_cap=clipped,mean_sentence_retention=float(np.mean(retained)),
                 source_policy='Learned train-only sentence salience + MMR + first/last quota; neural input is a selected subset, not every document token.')
    value={'rows':cached,'fingerprint':identity,'special':special,'tokenizer_path':str(tokenizer_path),
           'generation_blocked_ids':[r['id'] for r in json.loads(tok.to_str()).get('added_tokens',[]) if r.get('special') and r['id']!=special['eos']],
           'tokenizer_sha256':file_hash(tokenizer_path),'source_tokens':settings['source_tokens'],'target_tokens':target_cap,'audit':audit}
    tmp=path.with_name(path.name+'.tmp');torch.save(value,tmp);tmp.replace(path);atomic_json(output/'audit.json',audit)
    return str(path)
