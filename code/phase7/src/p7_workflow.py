"""Kaggle orchestration: seven own-foundation experiments, cached outputs and optional fusion."""
import itertools,json,math,os,shutil,subprocess,sys,time,zipfile
from pathlib import Path
import numpy as np
from p7_data import FILES,Store,read_json,atomic_json,file_hash,digest
from p7_eval import rank,ranking_lists,predictions,select_fusion,metrics,pack

def find_phase1(explicit=None):
    if explicit:
        p=Path(explicit)
        if not (p/'tokenizer.json').is_file():raise FileNotFoundError(f'No tokenizer.json under {p}. Attach original Phase-1 outputs.')
        return p
    candidates=sorted({p.parent for p in Path('/kaggle/input').rglob('tokenizer.json') if p.parent.name=='capstone_phase1'})
    if len(candidates)!=1:raise ValueError('Set PHASE1_ROOT to your unique original capstone_phase1 directory.')
    return candidates[0]
def obtain_data(project,local_root=None,revision=None):
    names=sum([list(p) for p in FILES.values()],[])
    if not local_root:
        roots=sorted({p for p in Path('/kaggle/input').rglob('Phase_7_PCR') if all((p/n).is_file() for n in names)})
        if len(roots)==1:local_root=roots[0]
        elif len(roots)>1:raise ValueError('Multiple Phase-7 datasets. Set DATA_ROOT explicitly.')
    if local_root:
        root=Path(local_root)
        if (root/'Phase_7_PCR').is_dir():root/='Phase_7_PCR'
        missing=[n for n in names if not (root/n).is_file()]
        if missing:raise FileNotFoundError(f'Missing official Phase-7 files: {missing}')
        atomic_json(Path(project)/'data_source.json',{'kind':'attached_official_phase7','root':str(root),'sha256':{n:file_hash(root/n) for n in names}});return str(root)
    from huggingface_hub import HfApi,hf_hub_download
    token=os.environ.get('HF_TOKEN')
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            token=UserSecretsClient().get_secret('HF_TOKEN')
        except Exception:token=None
    repo='Exploration-Lab/CS-781-Capstone';root=Path(project)/'data';path=root/'source.json';old=read_json(path) if path.exists() else None
    try:
        commit=old['revision'] if old else HfApi(token=token).dataset_info(repo,revision=revision).sha
        if old and revision and revision!=commit:raise ValueError('Requested revision differs from cached official commit.')
        for name in names:
            target=root/name
            if target.is_file() and old:
                if old['sha256'].get(name)!=file_hash(target):raise ValueError('Cached official data changed. Choose a new PROJECT.')
                continue
            src=hf_hub_download(repo,'Phase_7_PCR/'+name,repo_type='dataset',revision=commit,token=token)
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if file_hash(target)!=file_hash(src):raise ValueError('Partial downloaded data conflicts with the pinned dataset.')
            else:
                # Share immutable HF bytes on the same filesystem instead of duplicating large candidate pools.
                try:os.link(Path(src).resolve(),target)
                except OSError:shutil.copy2(src,target)
        atomic_json(path,{'repo':repo,'revision':commit,'sha256':{n:file_hash(root/n) for n in names}})
    except Exception as e:raise RuntimeError('Attach official Phase_7_PCR or enable Internet, accept dataset terms, and add an HF_TOKEN Kaggle secret.') from e
    return str(root)
def configurations(project,phase1,manifest,settings):
    project=Path(project);phase1=Path(phase1);cache=read_json(manifest)
    specs=[('01_decoder_full','decoder','full','dense',settings['full_epochs'],5e-5),
           ('02_encoder_decoder_partial','encoder_decoder','partial','dense',settings['peft_epochs'],1e-4),
           ('03_local_gqa_lorma','local_gqa','lorma','late',settings['peft_epochs'],2e-4),
           ('04_hybrid_prompt','hybrid','prompt','dense',settings['peft_epochs'],2e-3),
           ('05_decoder_ia3','decoder','ia3','dense',settings['peft_epochs'],1e-3),
           ('06_decoder_cross_encoder','decoder','full','cross',settings['cross_epochs'],3e-5),
           ('07_decoder_student_kd','decoder','kd','dense',settings['student_epochs'],5e-5)]
    result=[]
    for name,arch,method,kind,epochs,lr in specs:
        parent=phase1/'runs'/('main_'+arch)/'best_model.pt'
        if not parent.is_file():raise FileNotFoundError(f'Missing original parent {parent}')
        cfg={'name':name,'architecture':arch,'method':method,'kind':kind,'parent':str(parent),'manifest':str(manifest),'run':str(project/'runs'/name),
             'seed':settings['seed'],'epochs':epochs,'lr':lr,'patience':settings['patience'],'min_delta':.0001,
             'micro_batch':settings['micro_batch'],'accumulation':settings['accumulation'],'negatives':settings['negatives'],
             'chunk_tokens':settings['chunk_tokens'],'max_chunks':settings['max_chunks'],'chunk_microbatch':settings['chunk_microbatch'],
             'pair_tokens':settings['pair_tokens'],'pair_microbatch':settings['pair_microbatch'],'index_batch':settings['index_batch'],
             'inference_pairs':settings['inference_pairs'],'rerank_topk':settings['rerank_topk'],'dev_probe':settings['dev_probe'],
             'teacher_candidates':settings['teacher_candidates'],'temperature':.05,'kd_temperature':2.,'kd_weight':.5,
             'rank':16,'alpha':32,'unfrozen_layers':2,'embedding_dim':256,'prompt_tokens':32 if method=='prompt' else 0,
             'checkpointing':True,'dropout':.05,'bos':cache['special']['bos'],'log_every':25,
             'hard_file':str(project/'hard_negatives.json') if settings['mine_hard_negatives'] and name!='01_decoder_full' else None,
             'base_run':str(project/'runs/01_decoder_full') if kind=='cross' else None,'kd_file':str(project/'teacher_scores.json')}
        path=project/'configs'/f'{name}.json'
        if path.exists() and read_json(path)!=cfg:raise ValueError(f'{name} configuration changed; choose a new PROJECT.')
        atomic_json(path,cfg);result.append(str(path))
    return result
def launch(config,action='train',world=2,**kwargs):
    module=Path(__file__).parent/'p7_run.py';command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',str(module),'--config',str(config),'--action',action]
    for k,v in kwargs.items():command+=['--'+k.replace('_','-'),str(v)]
    env=os.environ.copy();env.update(TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True');env['PYTHONPATH']=str(module.parent)+os.pathsep+env.get('PYTHONPATH','')
    cfg=read_json(config);path=Path(cfg['run'])/(action+'.log');path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env)
        for line in process.stdout:print(line,end='');log.write(line);log.flush()
        status=process.wait()
    if status:raise RuntimeError(f'{cfg["name"]} {action} failed; inspect {path}. Completed submissions remain saved.')
def forecast(configs,world=2):
    rows=[];total=0
    for path in configs:
        cfg=read_json(path);pre=read_json(Path(cfg['run'])/'preflight.json');store=Store(cfg['manifest']);n=sum(bool(r['relevant']) for r in store.splits['train']['queries'].values())
        batches=math.ceil(math.ceil(n/world)/cfg['micro_batch']);hours=batches*cfg['epochs']*pre['seconds_per_microbatch']/3600;total+=hours
        rows.append({'name':cfg['name'],'training_hours_at_max_epochs':hours,'seconds_per_microbatch':pre['seconds_per_microbatch'],'peak_preflight_gib':pre['peak_reserved_gib']})
    print(json.dumps(rows,indent=2));print(f'Measured training-only forecast: {total:.2f} hours. Additional preprocessing, validation/indexing, mining, distillation-target scoring, final inference and I/O are excluded. Example lengths vary; this is not a runtime guarantee.')
    return rows
def final_rankings(config,store,world):
    cfg=read_json(config);run=Path(cfg['run']);raws={}
    for split in ('dev','test'):
        path=run/f'{split}_raw.json';launch(config,'rank',world,split=split,output=path);raws[split]=read_json(path)
    selection=select_fusion(store,raws['dev']);weight=selection['neural_weight'];atomic_json(run/'full_dev_selection.json',selection)
    for split in ('dev','test'):
        lists=ranking_lists(raws[split],weight);atomic_json(run/f'{split}_rankings.json',lists);atomic_json(run/f'{split}_prediction.json',{q:r[:7] for q,r in lists.items()})
    return selection
def mine(config,store,project,world):
    cfg=read_json(config);run=Path(cfg['run']);path=run/'train_raw.json';launch(config,'rank',world,split='train',output=path)
    weight=read_json(run/'full_dev_selection.json')['neural_weight'];lists=ranking_lists(read_json(path),weight);atomic_json(run/'train_rankings.json',lists)
    # Known positives are filtered by the sampler; mining cannot use development/test relevance.
    atomic_json(Path(project)/'hard_negatives.json',{'data_fingerprint':store.manifest['fingerprint'],'source_checkpoint_sha256':file_hash(run/'best.pt'),'rankings':lists})
def make_teacher(config,store,project,world):
    cfg=read_json(config);project=Path(project);checkpoint=Path(cfg['run'])/'best.pt';path=project/'teacher_scores.json'
    identity={'data_fingerprint':store.manifest['fingerprint'],'teacher_checkpoint':str(checkpoint),'teacher_checkpoint_sha256':file_hash(checkpoint),
              'fusion_selection_sha256':file_hash(Path(cfg['run'])/'full_dev_selection.json')}
    if path.exists():
        old=read_json(path)
        if any(old.get(k)!=v for k,v in identity.items()):raise ValueError('Teacher cache inputs changed. Use a new PROJECT.')
        return
    scores=project/'teacher_train_raw.json';launch(config,'teach',world,output=scores)
    values=read_json(scores);allowed=store.splits['train']['queries']
    if not set(values)<=set(allowed):raise ValueError('Held-out queries entered the teacher cache.')
    if any(set(v['ids'])-set(store.splits['train']['candidates']) for v in values.values()):raise ValueError('Teacher candidates outside train pool.')
    atomic_json(path,{**identity,'scores':values,'objective':'Cached actual retrieval logits on train-only gold+shortlist candidate groups. Student uses supervised multi-positive loss plus KL distillation.'})
def make_ensemble(project,results,store):
    # Optional score candidate, not an additional trained architecture or mandatory-method experiment.
    ranked=sorted(results,key=lambda x:x['selection']['metrics']['micro_f1_at7'],reverse=True);candidates=[]
    for n in (2,3):
        chosen=ranked[:n]
        if len(chosen)<n:continue
        for primary_weight in (.5,1.,2.):
            weights=[primary_weight]+[1.]*(n-1);preds={}
            for split in ('dev','test'):
                rankings=[read_json(Path(project)/'runs'/r['name']/f'{split}_rankings.json') for r in chosen];pred={}
                for q in store.splits[split]['queries']:
                    scores={}
                    for weight,lists in zip(weights,rankings):
                        for i,c in enumerate(lists[q]):scores[c]=scores.get(c,0.)+weight/(60+i+1)
                    pred[q]=sorted(scores,key=lambda c:(-scores[c],c))[:7]
                preds[split]=pred
            m=metrics(store,'dev',preds['dev']);candidates.append({'members':[r['name'] for r in chosen],'weights':weights,'metrics':m,'prediction':preds['test']})
    if not candidates:return None
    best=max(candidates,key=lambda r:r['metrics']['micro_f1_at7']);prediction=best.pop('prediction');item=pack(project,'ensemble_rrf',prediction,store)
    atomic_json(Path(project)/'ensemble_selection.json',{**best,'candidate_metrics':[{k:v for k,v in r.items() if k!='prediction'} for r in candidates]})
    return {**item,'selection':{'metrics':best['metrics']},'kind':'ensemble','method':'rank_fusion','architecture':'multiple'}
def run_all(project,configs,settings,world=2):
    project=Path(project);store=Store(read_json(configs[0])['manifest']);results=[];started=time.perf_counter()
    for path in configs:
        cfg=read_json(path);run=Path(cfg['run']);completed=run/'completed.json'
        identity=run_identity(cfg,store,world)
        if completed.exists():
            old=read_json(completed);item=old['result']
            if old['identity']!=identity or old['checkpoint_sha256']!=file_hash(run/'best.pt'):raise ValueError('Completed run changed. Choose a new PROJECT.')
            if not Path(item['submission_zip']).is_file():raise FileNotFoundError('Completed submission ZIP missing.')
            results.append(item);print('Reusing completed submission:',cfg['name']);continue
        if cfg['method']=='kd':
            eligible=[p for p in configs if read_json(p)['method']!='kd' and (Path(read_json(p)['run'])/'full_dev_selection.json').is_file()]
            teacher=max(eligible,key=lambda p:read_json(Path(read_json(p)['run'])/'full_dev_selection.json')['metrics']['micro_f1_at7']);make_teacher(teacher,store,project,world)
        launch(path,'train',world);selection=final_rankings(path,store,world)
        item={**pack(project,cfg['name'],read_json(run/'test_prediction.json'),store),'architecture':cfg['architecture'],'kind':cfg['kind'],'method':cfg['method'],'selection':selection,'best_epoch':read_json(run/'best_metrics.json')['best_epoch']}
        if cfg['name']=='01_decoder_full':
            # The cross-encoder needs split-specific first-stage rankings; training rankings also enable one-time mining.
            mine(path,store,project,world)
        atomic_json(completed,{'identity':run_identity(cfg,store,world),'checkpoint_sha256':file_hash(run/'best.pt'),'result':item});results.append(item);atomic_json(project/'results.json',results)
        print(json.dumps({'completed':cfg['name'],'dev_metrics':selection['metrics'],'elapsed_all_hours':(time.perf_counter()-started)/3600}),flush=True)
    atomic_json(project/'results.json',results);ranked=sorted(results,key=lambda r:r['selection']['metrics']['micro_f1_at7'],reverse=True)
    if settings.get('ensemble',True):
        ensemble=make_ensemble(project,results,store)
        if ensemble:ranked=sorted(ranked+[ensemble],key=lambda r:r['selection']['metrics']['micro_f1_at7'],reverse=True)
    atomic_json(project/'leaderboard.json',ranked)
    if ranked:shutil.copy2(ranked[0]['submission_zip'],project/'best_submission.zip')
    distinct=len({r['prediction_sha256'] for r in results});compliance={'trained_models':len(results),'distinct_model_prediction_files':distinct,'methods':sorted({r['method'] for r in results}),'families':sorted({r['architecture'] for r in results}),'meets_six_distinct_submission_count':distinct>=6}
    atomic_json(project/'compliance.json',compliance)
    if distinct<len(results):print('WARNING: Some trained models have identical test rankings. Repeated predictions do not count as distinct submissions; inspect compliance.json.')
    return ranked
def run_identity(cfg,store,world):
    depends={}
    for key in ['hard_file']+(['kd_file'] if cfg['method']=='kd' else []):
        if cfg.get(key):depends[key]=file_hash(cfg[key]) if Path(cfg[key]).is_file() else None
    if cfg.get('base_run'):
        depends['base_run']={s:file_hash(Path(cfg['base_run'])/f'{s}_rankings.json') for s in ('train','dev','test') if (Path(cfg['base_run'])/f'{s}_rankings.json').is_file()}
    return {'config_sha256':digest(cfg),'data_fingerprint':store.manifest['fingerprint'],'parent_sha256':file_hash(cfg['parent']),
            'code_sha256':{p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')},'world_size':world,'dependencies':depends}
def archive_all(project):
    project=Path(project);output=project.parent/(project.name+'_all_outputs.zip');size=sum(p.stat().st_size for p in project.rglob('*') if p.is_file())
    if shutil.disk_usage(project.parent).free<size+512*1024**2:raise OSError('Insufficient free disk for the all-output ZIP. Individual submissions and checkpoints remain saved.')
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(project.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.tmp'):z.write(p,str(p.relative_to(project.parent)))
    return output
