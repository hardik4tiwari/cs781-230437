"""One shared, bounded preprocessing pass. No test target access."""
import collections,math
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from s6_data import (FILES, Selector, atomic_json, digest, file_hash, normhash,
                     sentences, special_ids, split_data, wordset)
from s6_run import atomic_save

def choose_context(text,tokenizer,selector,budget,shortlist,sep):
    ss=sentences(text)
    ids=[e.ids for e in tokenizer.encode_batch(ss,add_special_tokens=False)]
    if sum(map(len,ids))+len(sep)*max(0,len(ids)-1)<=budget:
        return [t for i,row in enumerate(ids) for t in ((sep if i else [])+row)]
    score=selector.model.predict(selector.features(ss))
    selected={}
    # Keep some opening facts and the actual tail of the closing fragment.
    head=min(budget//10,len(ids[0]))
    tail=min(budget//6,len(ids[-1]))
    if len(ids)>1:
        if head:selected[0]=ids[0][:head]
        if tail:selected[len(ids)-1]=ids[-1][-tail:]
    candidates=sorted((i for i,ts in enumerate(ids) if ts and i not in selected),
                      key=lambda i:(-float(score[i]),i))[:shortlist]
    bags={i:wordset(ss[i]) for i in set(candidates)|set(selected)}
    def sim(i,j):
        a,b=bags[i],bags[j]
        return len(a&b)/max(1,len(a|b))
    redundancy={i:max((sim(i,j) for j in selected),default=0.) for i in candidates}
    remaining=budget-sum(len(x)+len(sep) for x in selected.values())
    while candidates and remaining>len(sep):
        i=max(candidates,key=lambda j:(float(score[j])-.15*redundancy[j],-j))
        candidates.remove(i)
        room=remaining-len(sep)
        if len(ids[i])>room and remaining<budget//4 and len(ids[i])>2*room:continue
        selected[i]=ids[i][:room]
        remaining-=len(selected[i])+len(sep)
        for j in candidates:redundancy[j]=max(redundancy[j],sim(i,j))
    if not selected:return ids[0][:budget]
    return [t for n,(_,row) in enumerate(sorted(selected.items())) for t in ((sep if n else [])+row)][:budget]

def encoded_rows(tok,texts):
    for start in range(0,len(texts),256):
        yield from (e.ids for e in tok.encode_batch(texts[start:start+256],add_special_tokens=False))

def prepare_fast(root,tokenizer_path,project,opts):
    root=Path(root);tokenizer_path=Path(tokenizer_path);project=Path(project)
    path=project/'cache'/'phase6_fast.pt'
    keys=('seed','source_tokens','target_tokens','dev_fraction','dev_max','train_pool','selector_fit_docs','selector_shortlist','special_ids')
    identity=digest({'data':{f:file_hash(root/f) for f in FILES},'tokenizer':file_hash(tokenizer_path),
                     'settings':{k:opts[k] for k in keys},'preprocessor':file_hash(__file__),
                     'parser':file_hash(Path(__file__).with_name('s6_data.py'))})
    if path.exists():
        value=torch.load(path,map_location='cpu',weights_only=False)
        if value['fingerprint']!=identity:raise ValueError('Cache inputs changed. Set a new PROJECT.')
        if any(r['target'] is not None or r['summary'] is not None for r in value['rows'] if r['split']=='test'):
            raise ValueError('Refusing test references in an existing cache.')
        if file_hash(tokenizer_path)!=value['tokenizer_sha256']:raise ValueError('Tokenizer changed.')
        if value['tokenizer_path']!=str(tokenizer_path):raise ValueError('Tokenizer mount changed. Use a new PROJECT.')
        atomic_json(project/'cache'/'audit.json',value['audit'])
        return path,value['audit']
    rows,audit=split_data(root,opts['seed'],opts['dev_fraction'],opts['dev_max'])
    train=sorted((r for r in rows if r['split']=='train'),key=lambda r:digest([opts['seed'],r['id']]))
    train=train[:opts['train_pool']] if opts['train_pool'] else train
    ids={r['id'] for r in train}
    rows=[r for r in rows if r['split']!='train' or r['id'] in ids]
    print(f'Preprocessing {len(train):,} training, {sum(r["split"]=="dev" for r in rows):,} dev and {sum(r["split"]=="test" for r in rows):,} test documents.',flush=True)
    tok=Tokenizer.from_file(str(tokenizer_path));tok.no_padding();tok.no_truncation()
    special=special_ids(tok,opts['special_ids'])
    blocked=sorted(set([special['pad'],special['bos']]+[r['id'] for r in __import__('json').loads(tok.to_str()).get('added_tokens',[]) if r.get('special') and r['id']!=special['eos']]))
    fit_rows=train[:opts['selector_fit_docs']]
    selector=Selector(opts['seed']).fit(fit_rows,passes=1)
    refs=[r for r in rows if r['split']!='test']
    labels={r['id']:ts for r,ts in zip(refs,encoded_rows(tok,[r['summary'] for r in refs]))}
    lengths=np.asarray([len(labels[r['id']]) for r in train])
    cap=min(opts['target_tokens'],max(64,int(np.quantile(lengths,.95))+1))
    prefix=tok.encode('Summarize the legal judgment, including the material facts, legal issue, reasoning and final decision.\nJudgment:\n',add_special_tokens=False).ids
    suffix=tok.encode('\nSummary:\n',add_special_tokens=False).ids
    sep=tok.encode('\n',add_special_tokens=False).ids
    room=opts['source_tokens']-len(prefix)-len(suffix)
    if room<32:raise ValueError('SOURCE_TOKENS is too small for the original tokenizer and prompt.')
    data=[]
    for i,r in enumerate(rows,1):
        sid=r['id'];target=None if r['split']=='test' else labels[sid][:cap-1]+[special['eos']]
        source=prefix+choose_context(r['document'],tok,selector,room,opts['selector_shortlist'],sep)+suffix
        data.append({'id':sid,'split':r['split'],'source':source,'target':target,'summary':r['summary'],'document_sha256':normhash(r['document'])})
        if i%500==0:print(f'Cached {i:,}/{len(rows):,} documents',flush=True)
    audit.update(counts=dict(collections.Counter(r['split'] for r in data)),
                 selector_fit_documents=len(fit_rows),selector_shortlist=opts['selector_shortlist'],
                 source_limit=opts['source_tokens'],target_limit=cap,
                 training_summary_tokens_p50_p90_p95=[round(float(np.quantile(lengths,q)),1) for q in (.5,.9,.95)],
                 train_references_clipped=int((lengths>=cap).sum()),
                 train_references_clipped_fraction=round(float((lengths>=cap).mean()),4))
    value={'rows':data,'fingerprint':identity,'tokenizer_path':str(tokenizer_path),'tokenizer_sha256':file_hash(tokenizer_path),
           'special':special,'generation_blocked_ids':blocked,'source_tokens':opts['source_tokens'],'target_tokens':cap,'audit':audit}
    atomic_save(path,value);atomic_json(project/'cache'/'audit.json',audit)
    return path,audit
