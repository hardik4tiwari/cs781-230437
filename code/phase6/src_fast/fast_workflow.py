"""Notebook orchestration: four experiments, independent submission ZIPs, optional dev metrics."""
import os,socket,subprocess,sys,time,zipfile
from pathlib import Path

import torch
from fast_run import lineage
from s6_data import atomic_json,digest,file_hash,read_json
from s6_workflow import pack_prediction

SPECS=[
    ('04_hybrid_lora','hybrid','lora',2e-4),
    ('05_decoder_prompt','decoder','prompt',3e-3),
    ('06_decoder_student_kd','decoder','kd',8e-5),
    ('07_encoder_decoder_cross_attention','encoder_decoder','cross_only',1.5e-4),
]
def configurations(project,phase1,cache_path,opts,chosen):
    project=Path(project);phase1=Path(phase1)
    cache=torch.load(cache_path,map_location='cpu',weights_only=False)
    known={x[0] for x in SPECS}
    if not chosen or len(set(chosen))!=len(chosen) or not set(chosen)<=known:raise ValueError(f'Select unique names from {sorted(known)}.')
    needed=set(chosen)
    if SPECS[2][0] in needed:needed.add(SPECS[0][0])
    configs={}
    for name,arch,method,lr in SPECS:
        if name not in needed:continue
        parent=phase1/'runs'/f'main_{arch}'/'best_model.pt'
        if not parent.is_file():raise FileNotFoundError(f'Missing original Phase 1 parent: {parent}')
        teacher=str(project/'configs'/f'{SPECS[0][0]}.json') if method=='kd' else None
        cfg={k:opts[k] for k in ('seed','epochs','steps_per_epoch','micro_batch','accumulation','generation_batch','max_train_minutes',
                                 'dev_loss_cases','kd_temperature','kd_weight','kd_every','checkpointing')}
        target=cache['target_tokens']
        cfg.update(name=name,architecture=arch,method=method,lr=lr,parent=str(parent),cache=str(cache_path),
                   run=str(project/'runs'/name),rank=16,alpha=32,prompt_tokens=32 if method=='prompt' else 0,
                   source_tokens=cache['source_tokens'],dropout=.05,projection_chunk=128,label_smoothing=.05,patience=1,
                   decode={'min_new_tokens':min(48,max(1,target//5)),'max_new_tokens':target,'no_repeat_ngram':4},
                   teacher_config=teacher,
                   warm_checkpoint=opts['warm_checkpoint'] if method=='lora' else None,
                   warm_cache=opts['warm_cache'] if method=='lora' else None)
        if opts.get('allow_tiny'):cfg['allow_tiny']=True
        path=project/'configs'/f'{name}.json'
        if path.exists() and read_json(path)!=cfg:raise ValueError(f'{name}: config changed. Use a new PROJECT or restore the old settings.')
        atomic_json(path,cfg);configs[name]=path
    return configs

def launch(path,action,world,**kwargs):
    cfg=read_json(path)
    worker=Path(__file__).with_name('fast_run.py')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    cmd=[sys.executable,'-m','torch.distributed.run','--nnodes=1','--master_addr=127.0.0.1',
         f'--master_port={port}',f'--nproc_per_node={world}',str(worker),
         '--config',str(path),'--action',action]
    for k,v in kwargs.items():cmd.extend(['--'+k.replace('_','-'),str(v)])
    env={**os.environ,'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'2','PYTHONUNBUFFERED':'1'}
    log=Path(cfg['run'])/f"{action}_{kwargs.get('split','')}.log";log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',encoding='utf-8') as f,subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env,bufsize=1) as p:
        for line in p.stdout:
            print(line,end='',flush=True);f.write(line);f.flush()
        if p.wait():raise RuntimeError(f'{action} failed for {cfg["name"]}. Log: {log}. Earlier ZIPs remain saved.')

def run_candidates(project,configs,chosen,world):
    project=Path(project);results=[];started=time.perf_counter()
    for name,path in configs.items():
        cfg=read_json(path);run=Path(cfg['run'])
        done=run/'train_complete.json'
        cache=torch.load(cfg['cache'],map_location='cpu',weights_only=False)
        signature=digest(lineage(cfg,cache,world));del cache
        if not done.exists():
            launch(path,'train',world)
        else:
            state=read_json(done)
            if state['signature']!=signature or state['checkpoint_sha256']!=file_hash(run/'best.pt'):
                raise ValueError(f'{name}: inputs or weights changed. Use a new PROJECT.')
        if name not in chosen:continue
        receipt=run/'submission_complete.json'
        if receipt.exists():
            old=read_json(receipt);result=old['result']
            if old['signature']!=signature or file_hash(result['submission_zip'])!=old['zip_sha256']:
                raise ValueError(f'{name}: previous ZIP or signature changed.')
            print('Reusing verified ZIP:',result['submission_zip'],flush=True)
        else:
            output=run/'test_predictions.json';start=time.perf_counter()
            launch(path,'generate',world,split='test',output=output)
            train=read_json(done)
            result={**pack_prediction(project,name,output,cfg['cache']),'method':cfg['method'],'architecture':cfg['architecture'],
                    'training_minutes':round(train['training_seconds']/60,2),'export_minutes':round((time.perf_counter()-start)/60,2),
                    'best_epoch':train['best_epoch'],'dev_token_loss':train['best_dev_loss'],'parameters':train['parameters']}
            atomic_json(receipt,{'signature':signature,'zip_sha256':file_hash(result['submission_zip']),'result':result})
        results.append(result);atomic_json(project/'results.json',results)
        print(f"Saved {name}: {result['submission_zip']}",flush=True)
    hashes={r['prediction_sha256'] for r in results}
    if len(hashes)<len(results):print('WARNING: identical prediction sets detected; they are not distinct submissions.')
    print(f'{len(results)} ZIPs, {len(hashes)} distinct prediction sets. This invocation: {(time.perf_counter()-started)/60:.1f} min (excluding preparation).')
    return results

def validate_dev(project,configs,results,world,cases,metric_device):
    if cases<=0:return []
    import importlib.util
    if any(importlib.util.find_spec(name) is None for name in ('bert_score','rouge_score')):
        subprocess.check_call([sys.executable,'-m','pip','install','-q','bert-score','rouge-score'])
    cache=torch.load(read_json(next(iter(configs.values())))['cache'],map_location='cpu',weights_only=False)
    truth={r['id']:r['summary'] for r in cache['rows'] if r['split']=='dev'};del cache
    from s6_metrics import score,release
    generated={}
    for result in results:
        name=result['name'];cfg=read_json(configs[name]);run=Path(cfg['run'])
        path=run/f'dev_{cases}_predictions.json'
        if not path.exists():launch(configs[name],'generate',world,split='dev',output=path,limit=cases)
        generated[name]=read_json(path)
    ranked=[]
    try:
        for result in results:
            data=generated[result['name']]
            metrics=score([x['summary'] for x in data],[truth[x['id']] for x in data],metric_device)
            ranked.append({**result,'heldout_probe':metrics})
    finally:release()
    ranked.sort(key=lambda r:r['heldout_probe']['bertscore_f1'],reverse=True)
    atomic_json(Path(project)/'dev_probe_leaderboard.json',ranked)
    for r in ranked:print(r['name'],r['heldout_probe'])
    return ranked

def bundle_submissions(project,results):
    project=Path(project);out=project/'phase6_fast_four_submissions.zip'
    with zipfile.ZipFile(out,'w',zipfile.ZIP_STORED) as z:
        for result in results:
            p=Path(result['submission_zip'])
            z.write(p,p.name)
    return out
