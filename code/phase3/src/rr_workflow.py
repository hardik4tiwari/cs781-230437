"""Direct Phase-2 input restoration, official data, seven RR experiments and exports."""
import gc,json,math,os,shutil,socket,subprocess,sys,time,zipfile
from pathlib import Path
import torch
from legal_models import LegalLM,ModelConfig
from rr_data import DATA_FILES,atomic_json,atomic_torch,digest,file_hash,read_json,submission_zip

SOURCES={'decoder':'01_decoder_full','encoder_decoder':'02_encoder_decoder_full','local_gqa':'03_local_gqa_lora','hybrid':'04_hybrid_partial'}
def merge_phase2(saved):
    cfg=saved['config'];kind=cfg['method'];state={k[len('backbone.'):]:v.float() for k,v in saved['model'].items() if k.startswith('backbone.')}
    if not state:raise ValueError('No backbone in Phase-2 checkpoint.')
    if kind=='adapter':raise ValueError('Nonlinear Phase-2 adapters cannot be folded away. Select a full/partial/LoRA/LoRMA checkpoint.')
    if kind in ('lora','lorma'):
        dense={k:v for k,v in state.items() if not k.endswith(('.base.weight','.A.weight','.B.weight'))}
        for key,value in state.items():
            if not key.endswith('.base.weight'):continue
            name=key[:-len('.base.weight')];A=state[name+'.A.weight'];B=state[name+'.B.weight']
            scale=cfg.get('alpha',16)/cfg.get('rank',8)
            delta=B@(A@value) if kind=='lorma' else B@A
            dense[name+'.weight']=value+scale*delta
        state=dense
    elif kind not in ('full','partial'):raise ValueError('Use a full-size Phase-2 parent, not its compressed KD student.')
    return state
def restore_phase2(root,output,allow_selection=True,overrides=None,allow_tiny=False):
    root=Path(root);output=Path(output);overrides=overrides or {}
    if set(overrides)-set(SOURCES):raise ValueError('Unknown architecture in PARENT_OVERRIDES.')
    if not root.is_dir():raise FileNotFoundError(f'Attach the extracted capstone_phase2 directory: {root}')
    if 'all_kaggle_outputs' in root.parts:raise ValueError('Set PHASE2_ROOT to the direct capstone_phase2 folder.')
    manifest=read_json(root/'phase1/foundation_manifest.json');tokenizer=root/'phase1/tokenizer.json'
    if file_hash(tokenizer)!=manifest['tokenizer_sha256']:raise ValueError('Phase-2 tokenizer/lineage manifest mismatch.')
    output.mkdir(parents=True,exist_ok=True);copied=output/'tokenizer.json';shutil.copy2(tokenizer,copied)
    parents={};records={}
    for arch,run in SOURCES.items():
        refit=root/'runs'/run/'refit/best.pt';selection=root/'runs'/run/'selection/best.pt'
        source=Path(overrides[arch]) if arch in overrides else refit if refit.is_file() else selection if allow_selection else refit
        if not source.resolve().is_relative_to(root.resolve()):raise ValueError('Every Phase-2 checkpoint must be inside PHASE2_ROOT.')
        if not source.is_file():raise FileNotFoundError(f'Missing Phase-2 checkpoint: {source}')
        source_hash=file_hash(source);target=output/f'{arch}.pt'
        if target.exists():
            previous=torch.load(target,map_location='cpu',weights_only=False)
            if previous['provenance']['phase2_sha256']!=source_hash:raise ValueError('Phase-2 source changed; choose a new PROJECT directory.')
            del previous
        else:
            saved=torch.load(source,map_location='cpu',weights_only=False)
            if saved.get('training_origin')!='own_phase1_adapted' or saved['config']['architecture']!=arch:raise ValueError('Expected your own matching Phase-2 NER checkpoint.')
            provenance=saved['provenance'];config=provenance['phase1_model_config']
            if provenance['phase1_sha256']!=manifest['sha256'][arch] or config['architecture']!=arch:raise ValueError('Phase-1/2 lineage does not match.')
            if saved['config']['method']=='kd':raise ValueError('Select a full-size parent checkpoint.')
            state=merge_phase2(saved);model=LegalLM(ModelConfig(**config));model.load_state_dict(state,strict=True)
            count=sum(p.numel() for p in model.parameters())
            if not allow_tiny and not 50_000_000<=count<=300_000_000:raise ValueError('Parent violates the foundation parameter bounds.')
            atomic_torch(target,{'config':config,'model':state,'provenance':{'architecture':arch,'phase2_checkpoint':str(source),
                'phase2_sha256':source_hash,'phase1_sha256':provenance['phase1_sha256'],'tokenizer_sha256':manifest['tokenizer_sha256'],
                'source_stage':saved['stage'],'source_parameters':count}})
            del model,state,saved;gc.collect()
        parents[arch]=str(target);records[arch]={'source':str(source),'sha256':source_hash,'used_selection_fallback':source==selection}
    atomic_json(output/'input_manifest.json',{'input_root':str(root),'input_mode':'direct_directory_only','tokenizer_sha256':file_hash(copied),'parents':records})
    return parents,str(copied)

def obtain_data(project,local_root=None,revision=None):
    destination=Path(project)/'data';destination.mkdir(parents=True,exist_ok=True)
    provenance_path=destination/'source.json'
    if local_root:
        source=Path(local_root)
        if (source/'Phase_3_RR').is_dir():source/='Phase_3_RR'
        missing=[name for name in DATA_FILES if not (source/name).is_file()]
        if missing:raise FileNotFoundError('Missing official Phase-3 files: '+', '.join(missing))
        for name in DATA_FILES:
            target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
            if target.resolve()!=(source/name).resolve():shutil.copy2(source/name,target)
        origin={'kind':'attached_official_phase3','source':str(source)}
    else:
        from huggingface_hub import HfApi,hf_hub_download
        token=os.environ.get('HF_TOKEN')
        if not token:
            try:
                from kaggle_secrets import UserSecretsClient
                token=UserSecretsClient().get_secret('HF_TOKEN')
            except Exception:token=None
        repo='Exploration-Lab/CS-781-Capstone'
        old=read_json(provenance_path) if provenance_path.exists() else None
        commit=old['revision'] if old and old.get('kind')=='official_huggingface' else revision
        try:
            commit=HfApi(token=token).dataset_info(repo,revision=commit).sha
            origin={'kind':'official_huggingface','repo':repo,'revision':commit};atomic_json(provenance_path,origin)
            for name in DATA_FILES:
                target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():
                    source=hf_hub_download(repo,'Phase_3_RR/'+name,repo_type='dataset',revision=commit,token=token)
                    shutil.copy2(source,target)
        except Exception as exc:raise RuntimeError('Attach the seven official Phase-3 files, or accept the official dataset terms and add an HF_TOKEN Kaggle secret. Do not put tokens in notebook code.') from exc
    origin['sha256']={name:file_hash(destination/name) for name in DATA_FILES};atomic_json(provenance_path,origin)
    return str(destination)
def experiments(parents,settings):
    common={'label_count':13,'head_dim':settings['head_dim'],'dropout':.25,'encoder_dropout':.05,
        'chunk_batch':settings['chunk_batch'],'encoder_checkpointing':settings['encoder_checkpointing'],
        'head_batch':8,'batch_blocks':settings['batch_blocks'],'accumulation':settings['accumulation'],
        'encoder_lr':2e-5,'head_lr':2e-4,'weight_decay':.03,'ce_weight':.5,'shift_weight':.15,
        'epochs':settings['adapt_epochs'],'patience':2,'max_stage_hours':settings['adapt_stage_hours'],
        'rank':8,'alpha':16.,'seed':230437,'freeze_backbone':False,'checkpoint_every':100,'log_every':20,
        'token_bidirectional':settings.get('token_bidirectional',False)}
    specs=[('01_decoder_full','decoder','full','bilstm'),('02_encoder_decoder_full','encoder_decoder','full','transformer'),
        ('03_local_gqa_lora','local_gqa','lora','bilstm'),('04_hybrid_rslora','hybrid','rslora','bilstm'),
        ('05_decoder_lorma','decoder','lorma','bilstm'),('06_encoder_decoder_ia3','encoder_decoder','ia3','bilstm'),
        ('07_decoder_student_kd','decoder','kd','bilstm')]
    rows=[]
    for name,arch,method,head in specs:
        cfg={**common,'name':name,'architecture':arch,'method':method,'head_type':head,'parent':parents[arch]}
        if method in ('lora','rslora','lorma','ia3'):cfg['encoder_lr']=1e-4
        if method=='kd':cfg.update({'student_layers':settings['student_layers'],'temperature':2.,'kd_alpha':.5})
        rows.append(cfg)
    return rows
def launch(script,args,project,gpus=2,log=None):
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    cmd=[sys.executable,'-m','torch.distributed.run','--nnodes=1','--master_addr=127.0.0.1',f'--master_port={port}',
         f'--nproc_per_node={gpus}',str(Path(project)/'code'/script),*map(str,args)]
    env=dict(os.environ);env['PYTHONPATH']=str(Path(project)/'code')+os.pathsep+env.get('PYTHONPATH','');env['TOKENIZERS_PARALLELISM']='false'
    handle=Path(log).open('a',encoding='utf-8') if log else None
    try:
        process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env)
        for line in process.stdout:
            print(line,end='',flush=True)
            if handle:handle.write(line);handle.flush()
        if process.wait()!=0:raise RuntimeError(f'{script} failed. Inspect {log or "cell output"}; recovery checkpoints are retained.')
    finally:
        if handle:handle.close()
def warm_config(cfg,settings):
    return {**cfg,'freeze_backbone':True,'head_lr':6e-4,'epochs':settings['head_epochs'],'patience':4,
        'max_stage_hours':settings['head_stage_hours'],'batch_blocks':16,'accumulation':1}
def run_all(project,rows,cache_path,settings):
    from rr_predict import select_calibration,decode,domain_metrics
    project=Path(project);cache=torch.load(cache_path,map_location='cpu',weights_only=False);catalog=[];gpus=settings['gpus']
    for original in rows:
        cfg=dict(original);root=project/'runs'/cfg['name'];root.mkdir(parents=True,exist_ok=True)
        teacher_train=teacher_all=None
        if cfg['method']=='kd':
            teacher=max(catalog,key=lambda item:item['dev_metrics']['macro_f1']);cfg['teacher_parameters']=teacher['parameters']['total']
            for split,checkpoint in [('train',teacher['selection_checkpoint']),('all',teacher['test_checkpoint'])]:
                if split=='all' and not settings['refit_all']:continue
                destination=root/('teacher_'+split)
                launch('rr_predict.py',['--action','logits','--checkpoint',checkpoint,'--cache',cache_path,
                    '--split',split,'--output',destination],project,gpus,root/f'teacher_{split}.log')
                if split=='train':teacher_train=destination/'complete.pt'
                else:teacher_all=destination/'complete.pt'
        feature_key=cfg['architecture']+('_student_'+str(cfg['student_layers']) if cfg['method']=='kd' else '')
        feature_root=project/'features'/feature_key;feature_root.mkdir(parents=True,exist_ok=True)
        feature_cfg={**cfg,'freeze_backbone':True};feature_config=feature_root/'config.json';atomic_json(feature_config,feature_cfg)
        launch('rr_predict.py',['--action','features','--config',feature_config,'--cache',cache_path,'--split','every',
            '--output',feature_root],project,gpus,feature_root/'features.log')
        features=feature_root/'complete.pt';warm=warm_config(cfg,settings);warm_path=root/'warm_config.json';atomic_json(warm_path,warm)
        args=['--config',warm_path,'--cache',cache_path,'--output',root/'warm_selection','--split','train','--features',features]
        if teacher_train:args+=['--teacher',teacher_train]
        launch('rr_train.py',args,project,settings.get('head_gpus',1),root/'warm_selection.log')
        cfg['head_init']=str(root/'warm_selection/best.pt');config=root/'config.json';atomic_json(config,cfg)
        args=['--config',config,'--cache',cache_path,'--output',root/'selection','--split','train']
        if teacher_train:args+=['--teacher',teacher_train]
        launch('rr_train.py',args,project,gpus,root/'selection.log')
        selection=root/'selection/best.pt';summary=read_json(root/'selection/summary.json')
        launch('rr_predict.py',['--action','logits','--checkpoint',selection,'--cache',cache_path,'--split','dev',
            '--output',root/'selected_dev'],project,gpus,root/'selected_dev.log')
        raw=torch.load(root/'selected_dev/complete.pt',map_location='cpu',weights_only=False)
        metrics,calibration,pred=select_calibration(raw['values'],raw['crf'],cache)
        atomic_json(root/'dev_metrics.json',metrics);atomic_json(root/'domain_metrics.json',domain_metrics(pred,cache));atomic_json(root/'calibration.json',calibration)
        atomic_json(root/'dev_prediction.json',pred);test_checkpoint=selection
        if settings['refit_all']:
            best_warm=max(1,read_json(root/'warm_selection/summary.json')['best_epoch'])
            args=['--config',warm_path,'--cache',cache_path,'--output',root/'warm_refit','--split','all','--epochs',best_warm,'--features',features]
            if teacher_all:args+=['--teacher',teacher_all]
            launch('rr_train.py',args,project,settings.get('head_gpus',1),root/'warm_refit.log')
            refit_cfg={**cfg,'head_init':str(root/'warm_refit/best.pt')};refit_config=root/'refit_config.json';atomic_json(refit_config,refit_cfg)
            args=['--config',refit_config,'--cache',cache_path,'--output',root/'refit','--split','all','--epochs',max(1,summary['best_epoch'])]
            if teacher_all:args+=['--teacher',teacher_all]
            launch('rr_train.py',args,project,gpus,root/'refit.log');test_checkpoint=root/'refit/best.pt'
        launch('rr_predict.py',['--action','logits','--checkpoint',test_checkpoint,'--cache',cache_path,'--split','test',
            '--output',root/'test_prediction'],project,gpus,root/'test_prediction.log')
        test_raw=torch.load(root/'test_prediction/complete.pt',map_location='cpu',weights_only=False)
        prediction=decode(test_raw['values'],test_raw['crf'],cache['labels'],calibration)
        zip_path=project/'submissions'/f"{cfg['name']}.zip";sha=submission_zip(prediction,{did:cache['docs'][did] for did in cache['splits']['test']},cache['labels'],zip_path)
        atomic_json(root/'prediction.json',prediction)
        item={'name':cfg['name'],'architecture':cfg['architecture'],'method':cfg['method'],'dev_metrics':metrics,
            'parameters':summary['parameters'],'selection_epoch':summary['best_epoch'],'selection_checkpoint':str(selection),
            'test_checkpoint':str(test_checkpoint),'submission':str(zip_path),'prediction_sha256':sha,'dev_used_for_calibration':True}
        catalog.append(item);atomic_json(project/'catalog.json',catalog)
        print('READY TO DOWNLOAD:',zip_path,'development-tuned Macro-F1:',round(metrics['macro_f1'],5),flush=True)
    if len(catalog)!=7 or len({item['architecture'] for item in catalog})!=4:raise ValueError('Experiment coverage is incomplete.')
    if len({item['prediction_sha256'] for item in catalog})<7:
        print('Some predictions are identical. Renamed ZIPs do not count as distinct submissions; inspect metrics before uploading.',flush=True)
    return catalog
def smoke_all(project,rows,cache_path,settings):
    for cfg in rows:
        root=Path(project)/'smoke'/cfg['name'];root.mkdir(parents=True,exist_ok=True);config=root/'config.json';atomic_json(config,cfg)
        stamp=digest([cfg,file_hash(cache_path),settings['gpus'],file_hash(Path(project)/'code/rr_models.py'),file_hash(Path(project)/'code/rr_train.py')])
        marker=root/'passed.json'
        if marker.exists() and read_json(marker)['identity']==stamp:continue
        launch('rr_train.py',['--config',config,'--cache',cache_path,'--output',root/'run','--smoke'],project,settings['gpus'],root/'smoke.log')
        atomic_json(marker,{'identity':stamp,'gpus':settings['gpus'],'passed':True})
        for name in ('latest.pt','best.pt'):(root/'run'/name).unlink(missing_ok=True)

def benchmark(rows,cache,project,settings):
    from rr_data import Blocks,collate,to_device
    from rr_models import build_model,loss_function,parameter_report
    from rr_predict import autocast
    if not torch.cuda.is_available():raise RuntimeError('Run the throughput preflight on Kaggle GPUs.')
    device=torch.device('cuda',0);records=[];dataset=Blocks(cache,'train')
    longest=max(range(len(dataset)),key=lambda i:sum(len(part) for sentence in dataset[i]['chunks'] for part in sentence))
    for cfg in rows:
        while True:
            model=optimizer=result=loss=batch=scaler=None
            try:
                torch.cuda.empty_cache();model,_=build_model(cfg);model.to(device).train()
                batch=to_device(collate([dataset[longest] for _ in range(cfg['batch_blocks'])]),device)
                if cfg['method']=='kd':batch['teacher']=torch.zeros(*batch['mask'].shape,len(cache['labels']),device=device)
                optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,fused=True)
                scaler=torch.amp.GradScaler('cuda');weights=torch.ones(len(cache['labels']),device=device)
                torch.cuda.reset_peak_memory_stats(device);times=[];success=0
                for attempt in range(12):
                    torch.cuda.synchronize(device);start=time.monotonic();optimizer.zero_grad(set_to_none=True)
                    with autocast(device):result=model(batch);loss=loss_function(model,result,batch,weights,cfg)
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('Preflight loss is not finite.')
                    scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    old=scaler.get_scale();scaler.step(optimizer);scaler.update();torch.cuda.synchronize(device)
                    if scaler.get_scale()>=old:
                        success+=1
                        if success>1:times.append(time.monotonic()-start)
                    if success>=4:break
                if success<4:raise FloatingPointError('Could not obtain finite FP16 optimizer steps.')
                peak=torch.cuda.max_memory_reserved(device)/1024**3
                if peak>12.5:raise torch.cuda.OutOfMemoryError('Reserve space for distributed communication.')
                seconds=sum(times)/len(times);per_epoch=math.ceil(len(dataset)/(cfg['batch_blocks']*settings['gpus']))
                estimate=seconds*per_epoch*cfg['epochs']*(2 if settings['refit_all'] else 1)/3600
                record={'name':cfg['name'],'single_gpu_full_block_step_seconds':seconds,'peak_reserved_gib':peak,
                    'rough_adapt_and_refit_hours_excluding_warmup_validation_cache_io_ddp':estimate,'parameters':parameter_report(model),'chunk_batch':cfg['chunk_batch']}
                records.append(record);print(record,flush=True);break
            except torch.cuda.OutOfMemoryError:
                if cfg['chunk_batch']==1:raise RuntimeError('Preflight still exceeds T4 memory. Use a fresh PROJECT with fewer block_sentences.')
                cfg['chunk_batch']=max(1,cfg['chunk_batch']//2)
            finally:
                del model,optimizer,result,loss,batch,scaler;gc.collect();torch.cuda.empty_cache()
    atomic_json(Path(project)/'benchmark.json',records)
    return records
def bundle_working(working,output):
    working=Path(working);output=Path(output);tmp=output.with_suffix('.tmp.zip')
    files=[p for p in working.rglob('*') if p.is_file() and p.resolve() not in (output.resolve(),tmp.resolve()) and '__pycache__' not in p.parts]
    if shutil.disk_usage(working).free<sum(p.stat().st_size for p in files)+128*1024**2:raise RuntimeError('Download the seven small submission ZIPs first; there is insufficient disk space for a full working archive.')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for p in files:z.write(p,p.relative_to(working))
    tmp.replace(output)
    with zipfile.ZipFile(output) as z:
        if z.testzip():raise ValueError('Working archive CRC failure.')
    return str(output)
