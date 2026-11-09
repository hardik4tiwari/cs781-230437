"""Notebook orchestration; saves every completed submission and ranks on held-out data only."""
import gc,json,math,os,shutil,subprocess,sys,time,zipfile
from pathlib import Path
import torch
from tokenizers import Tokenizer
from s6_data import FILES,atomic_json,read_json,file_hash,digest

def find_phase1(explicit=None):
    if explicit:
        p=Path(explicit)
        if not (p/'tokenizer.json').is_file():raise FileNotFoundError(f'No tokenizer.json under {p}')
        return p
    preferred=Path('/kaggle/input/datasets/hardik4tiwari/phase-1-outputs/capstone_phase1')
    candidates=[preferred] if preferred.is_dir() else sorted({p.parent for p in Path('/kaggle/input').rglob('tokenizer.json') if p.parent.name=='capstone_phase1'})
    if len(candidates)!=1:raise FileNotFoundError('Set PHASE1_ROOT to the unique original capstone_phase1 directory. Attach the Phase-1 outputs dataset.')
    return candidates[0]
def obtain_data(project,local_root=None,revision=None):
    if not local_root:
        candidates=sorted({p for p in Path('/kaggle/input').rglob('Phase_6_Summ') if (p/FILES[0]).is_file() and (p/FILES[1]).is_file()})
        if len(candidates)==1:local_root=candidates[0]
        elif len(candidates)>1:raise ValueError('Multiple Phase-6 datasets: set DATA_ROOT explicitly.')
    if local_root:
        source=Path(local_root)
        if (source/'Phase_6_Summ').is_dir():source/='Phase_6_Summ'
        if any(not (source/f).is_file() for f in FILES):raise FileNotFoundError('Need Phase_6_Summ/Train/train.json and Test/test.json.')
        atomic_json(Path(project)/'data_source.json',{'kind':'attached_official_phase6','root':str(source),'sha256':{n:file_hash(source/n) for n in FILES}})
        return str(source)
    from huggingface_hub import HfApi,hf_hub_download
    token=os.environ.get('HF_TOKEN')
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            token=UserSecretsClient().get_secret('HF_TOKEN')
        except Exception:token=None
    root=Path(project)/'data';manifest=root/'source.json';old=read_json(manifest) if manifest.exists() else None
    repo='Exploration-Lab/CS-781-Capstone'
    try:
        commit=old['revision'] if old else HfApi(token=token).dataset_info(repo,revision=revision).sha
        if old and revision and revision!=commit:raise ValueError('Pinned revision differs. Choose a new PROJECT.')
        for name in FILES:
            target=root/name
            if target.exists() and old and old.get('sha256',{}).get(name)==file_hash(target):continue
            source=hf_hub_download(repo,'Phase_6_Summ/'+name,repo_type='dataset',revision=commit,token=token)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        atomic_json(manifest,{'repo':repo,'revision':commit,'sha256':{n:file_hash(root/n) for n in FILES}})
    except Exception as e:raise RuntimeError('Attach official Phase_6_Summ data, or accept its Hugging Face terms, enable Kaggle Internet and add an HF_TOKEN secret.') from e
    return str(root)
def configurations(project,phase1,cache_path,settings):
    project=Path(project);phase1=Path(phase1);cache=torch.load(cache_path,map_location='cpu',weights_only=False)
    specs=[('01_encoder_decoder_full','encoder_decoder','full',settings['full_epochs'],8e-5),
           ('02_decoder_lorma','decoder','lorma',settings['peft_epochs'],2e-4),
           ('03_local_gqa_adapter','local_gqa','adapter',settings['peft_epochs'],3e-4),
           ('04_hybrid_lora','hybrid','lora',settings['peft_epochs'],2e-4),
           ('05_decoder_prompt','decoder','prompt',settings['peft_epochs'],3e-3),
           ('06_decoder_student_kd','decoder','kd',settings['student_epochs'],8e-5)]
    result=[]
    for name,arch,method,epochs,lr in specs:
        parent=phase1/'runs'/('main_'+arch)/'best_model.pt'
        if not parent.is_file():raise FileNotFoundError(f'Missing original Phase-1 parent: {parent}')
        cfg={'name':name,'architecture':arch,'method':method,'parent':str(parent),'cache':str(cache_path),
             'run':str(project/'runs'/name),'seed':settings['seed'],'epochs':epochs,'lr':lr,
             'source_tokens':cache['source_tokens'],
             'weight_decay':.01,'warmup_fraction':.06,'patience':settings['patience'],'min_delta':.0002,
             'micro_batch':settings['micro_batch'],'accumulation':settings['accumulation'],
             'generation_batch':settings['generation_batch'],'dev_probe':settings['dev_probe'],
             'rank':16,'alpha':32,'adapter_dim':64,'prompt_tokens':32 if method=='prompt' else 0,
             'globalize_local':arch=='local_gqa','checkpointing':True,'dropout':.05,
             'projection_chunk':128,'label_smoothing':.05,'log_every':25,
             'kd_cache':str(project/'kd_cache.json'),
             'decode':{'beams':1,'min_new_tokens':min(64,cache['target_tokens']//4),
                       'max_new_tokens':cache['target_tokens'],'length_penalty':.8,'no_repeat_ngram':4}}
        path=project/'configs'/f'{name}.json'
        if path.exists() and read_json(path)!=cfg:raise ValueError(f'Config changed for {name}. Use a new PROJECT to preserve resume integrity.')
        atomic_json(path,cfg);result.append(str(path))
    return result
def launch(config,action='train',world=2,**kwargs):
    module=Path(__file__).parent/'s6_run.py'
    command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',str(module),'--config',str(config),'--action',action]
    for key,value in kwargs.items():command+=['--'+key.replace('_','-'),str(value)]
    env=os.environ.copy();env['TOKENIZERS_PARALLELISM']='false';env['OMP_NUM_THREADS']='2';env['PYTHONUNBUFFERED']='1'
    env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True';env['PYTHONPATH']=str(module.parent)+os.pathsep+env.get('PYTHONPATH','')
    cfg=read_json(config);log=Path(cfg['run'])/(action+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',encoding='utf-8') as handle:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env)
        for line in process.stdout:print(line,end='');handle.write(line);handle.flush()
        status=process.wait()
    if status:raise RuntimeError(f'{action} failed for {cfg["name"]}; inspect {log}. Completed runs remain saved.')
def forecast(configs,world=2):
    rows=[];total=0
    for path in configs:
        cfg=read_json(path);p=Path(cfg['run'])/'preflight.json';pre=read_json(p);cache=torch.load(cfg['cache'],map_location='cpu',weights_only=False)
        n=sum(r['split']=='train' for r in cache['rows']);batches=math.ceil(math.ceil(n/world)/cfg['micro_batch'])
        # Preflight uses batch=1 with longest target, and synchronizes every step (conservative for no_sync accumulation).
        hours=pre['seconds_per_microbatch_per_rank']*cfg['micro_batch']*batches*cfg['epochs']/3600
        total+=hours;rows.append({'name':cfg['name'],'full_epoch_training_hours_conservative':hours/cfg['epochs'],'max_epochs':cfg['epochs'],'training_hours_conservative':hours,'peak_preflight_gib':pre['peak_reserved_gib_per_rank']})
    print(json.dumps(rows,indent=2));print(f'Training-only conservative forecast: {total:.2f} h. Excludes preprocessing, BERTScore, generation, teacher cache and disk I/O; early stopping can reduce it.')
    return rows
def kd_cache(project,teacher_config,world=2,limit=2048):
    project=Path(project);cfg=read_json(teacher_config);cache=torch.load(cfg['cache'],map_location='cpu',weights_only=False)
    checkpoint=Path(cfg['run'])/'best.pt';dest=project/'kd_cache.json';identity={'data_fingerprint':cache['fingerprint'],'teacher_checkpoint':str(checkpoint),'teacher_checkpoint_sha256':file_hash(checkpoint),'decode':cfg['decode'],'limit':limit}
    if dest.exists():
        old=read_json(dest)
        if any(old.get(k)!=v for k,v in identity.items()):raise ValueError('KD teacher/cache changed; use a new PROJECT.')
        return dest
    output=project/'teacher_train_predictions.json'
    # Only training source documents are decoded; no dev/test IDs are permitted in the KD cache.
    launch(teacher_config,'generate',world=world,split='train',output=output,limit=limit)
    tok=Tokenizer.from_file(cache['tokenizer_path']);allowed={r['id'] for r in cache['rows'] if r['split']=='train'};targets={}
    for r in read_json(output):
        if r['id'] not in allowed:raise ValueError('Held-out ID entered KD cache.')
        targets[r['id']]=tok.encode(r['summary'],add_special_tokens=False).ids[:cache['target_tokens']-1]+[cache['special']['eos']]
    atomic_json(dest,{**identity,'targets':targets,'objective':'Stochastic 50:50 gold vs greedy teacher sequence targets on cached training IDs; remaining train IDs use gold.'});return dest
def pack_prediction(project,name,pred_path,cache_path):
    cache=torch.load(cache_path,map_location='cpu',weights_only=False);expected={r['id'] for r in cache['rows'] if r['split']=='test'}
    pred=read_json(pred_path)
    if not isinstance(pred,list) or len(pred)!=len(expected):raise ValueError('Wrong number of test predictions.')
    if len({r['id'] for r in pred})!=len(pred) or {r['id'] for r in pred}!=expected:raise ValueError('Submission IDs mismatch.')
    if any(set(r)!={'id','summary'} or not isinstance(r['summary'],str) or not r['summary'].strip() for r in pred):raise ValueError('Invalid summary schema.')
    folder=Path(project)/'submissions'/name;atomic_json(folder/'prediction.json',pred);archive=folder.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:z.write(folder/'prediction.json','prediction.json')
    with zipfile.ZipFile(archive) as z:
        if z.namelist()!=['prediction.json']:raise ValueError('Incorrect ZIP root.')
    return {'name':name,'submission_zip':str(archive),'prediction_sha256':digest(sorted(pred,key=lambda r:r['id'])),'test_documents':len(pred)}
def run_all(project,configs,settings,world=2):
    project=Path(project);results=[];begin=time.perf_counter()
    for path in configs:
        cfg=read_json(path);run=Path(cfg['run'])
        completed=run/'completed.json'
        identity={'config_sha256':digest(cfg),'cache_sha256':file_hash(cfg['cache']),'parent_sha256':file_hash(cfg['parent']),
                  'code_sha256':{p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')},'world_size':world}
        if completed.exists():
            old=read_json(completed);item=old['result']
            if old['identity']!=identity or old['checkpoint_sha256']!=file_hash(run/'best.pt'):raise ValueError('Completed run inputs changed. Use a new PROJECT.')
            if not Path(item['submission_zip']).is_file():raise FileNotFoundError('Completed submission archive was removed.')
            results.append(item);atomic_json(project/'results.json',results);print('Reusing completed submission:',cfg['name']);continue
        if cfg['method']=='kd':
            teachers=[p for p in configs if read_json(p)['method']!='kd' and (Path(read_json(p)['run'])/'best_metrics.json').is_file()]
            teacher=max(teachers,key=lambda p:read_json(Path(read_json(p)['run'])/'best_metrics.json')['best'])
            kd_cache(project,teacher,world,settings['kd_cases'])
        launch(path,'train',world)
        # Choose greedy vs beam-2 using the same held-out probe; never inspect test references or leaderboard scores.
        cache=torch.load(cfg['cache'],map_location='cpu',weights_only=False)
        probe=sorted([r for r in cache['rows'] if r['split']=='dev'],key=lambda r:r['id'])[:settings['decode_tune_cases']]
        # A generation config changes decode only. Checkpoint signature must stay tied to original training config:
        # candidate selection is implemented outside worker using a local temporary model, then final inference uses chosen decode override.
        tuning=tune_decode(path,probe,cache,settings.get('tune_beams',True))
        atomic_json(run/'decode_selection.json',tuning)
        # Generate with an explicit decoding override file; it does not mutate the immutable training signature.
        decode_path=run/'selected_decode.json';atomic_json(decode_path,tuning['decode'])
        pred_path=run/'test_predictions.json';launch(path,'generate',world,split='test',output=pred_path,decode_file=decode_path)
        dev_path=run/'full_dev_predictions.json';launch(path,'generate',world,split='dev',output=dev_path,decode_file=decode_path)
        dev={r['id']:r['summary'] for r in read_json(dev_path)};held=[r for r in cache['rows'] if r['split']=='dev']
        from s6_metrics import score,release
        metrics=score([dev[r['id']] for r in held],[r['summary'] for r in held],torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'));release()
        atomic_json(run/'full_dev_metrics.json',metrics)
        item={**pack_prediction(project,cfg['name'],pred_path,cfg['cache']),'architecture':cfg['architecture'],'method':cfg['method'],'held_out':metrics,'best_epoch':read_json(run/'best_metrics.json')['best_epoch']}
        atomic_json(completed,{'identity':identity,'checkpoint_sha256':file_hash(run/'best.pt'),'result':item})
        results.append(item);atomic_json(project/'results.json',results)
        print(json.dumps({'completed':cfg['name'],'held_out':metrics,'elapsed_all_hours':(time.perf_counter()-begin)/3600}),flush=True)
    ranked=sorted(results,key=lambda x:x['held_out']['bertscore_f1'],reverse=True);hashes={r['prediction_sha256'] for r in results}
    atomic_json(project/'leaderboard.json',ranked)
    if ranked:shutil.copy2(ranked[0]['submission_zip'],project/'best_submission.zip')
    if len(hashes)<len(results):print('WARNING: Some models produced identical predictions; those do not count as distinct submissions. Inspect results before submission.')
    atomic_json(project/'compliance.json',{'completed':len(results),'distinct_prediction_files':len(hashes),'families':sorted({r['architecture'] for r in results}),'methods':sorted({r['method'] for r in results}),'satisfies_six_distinct_and_four_families':len(results)>=6 and len(hashes)>=6 and len({r['architecture'] for r in results})==4})
    return ranked
def tune_decode(path,rows,cache,enabled=True):
    # This parent process is otherwise CPU-only. All torchrun workers have exited before loading model + evaluation metric here.
    from s6_run import build,predict
    from s6_metrics import score,release
    cfg=read_json(path);device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');model=build(cfg,cache,device)
    saved=torch.load(Path(cfg['run'])/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(saved['model'],strict=True);del saved
    candidates=[]
    for beam in ([1,2] if enabled else [1]):
        decode={**cfg['decode'],'beams':beam};pred=predict(model,rows,cache,device,decode,cfg['generation_batch'])
        metric=score([p['summary'] for p in pred],[r['summary'] for r in rows],device);release()
        candidates.append({'decode':decode,'metrics':metric})
    winner=max(candidates,key=lambda x:x['metrics']['bertscore_f1']);del model;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return {**winner,'candidates':candidates,'selection':'held-out decode-tuning subset; full dev is not an independent unbiased estimate'}
def archive_all(project):
    project=Path(project);output=project.parent/(project.name+'_all_outputs.zip')
    size=sum(p.stat().st_size for p in project.rglob('*') if p.is_file())
    if shutil.disk_usage(project.parent).free<size+512*1024**2:raise OSError('Not enough free disk to duplicate all outputs into ZIP. Individual submission ZIPs and checkpoints are already saved; download them separately or free unrelated disk space.')
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(project.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.tmp'):z.write(p,str(p.relative_to(project.parent)))
    return output
