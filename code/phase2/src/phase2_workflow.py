"""Restore own foundations, retrieve official data, run seven auditable experiments."""
import csv, gc, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
import torch
from ner_data import atomic_json,file_hash,load_official,create_cache,read_json,digest
from ner_predict import submission_zip

ARCHITECTURES=('decoder','encoder_decoder','local_gqa','hybrid')
REPO='Exploration-Lab/CS-781-Capstone'
REVISION='89edba6a44d6bb9593fd13b6895b5cd3b06eb82e'
DATA_FILES=('Train/fold1-text.json','Train/fold1-gold.json','Train/fold2-text.json','Train/fold2-gold.json','Train/labels.txt','Test/fold3-text.json')

DEFAULT_PHASE1_ROOT='/kaggle/input/datasets/hardik4tiwari/phase-1-outputs/capstone_phase1'

def restore_foundations(input_root,output,overrides=None,tokenizer_override=None,phase1_root=None):
    """Load five explicit files from the direct capstone_phase1 directory only.

    input_root remains in the signature for compatibility; it is never searched.
    No ZIP, alternate output tree, pilot or benchmark is inspected.
    """
    selected_root=Path(phase1_root or DEFAULT_PHASE1_ROOT)
    if 'all_kaggle_outputs' in selected_root.parts:
        raise ValueError(f'Use the direct capstone_phase1 input directory: {DEFAULT_PHASE1_ROOT}')
    if not selected_root.is_dir():raise FileNotFoundError(f'Phase-1 input folder does not exist: {selected_root}')
    overrides=overrides or {}
    if set(overrides)-set(ARCHITECTURES):raise ValueError('Unknown architecture in FOUNDATION_OVERRIDES.')
    chosen={arch:str(Path(overrides.get(arch,selected_root/'runs'/f'main_{arch}'/'best_model.pt'))) for arch in ARCHITECTURES}
    source_tokenizer=Path(tokenizer_override or selected_root/'tokenizer.json')
    required=[Path(path) for path in chosen.values()]+[source_tokenizer]
    for path in required:
        if not path.resolve().is_relative_to(selected_root.resolve()):
            raise ValueError(f'Phase-1 input must come from {selected_root}: {path}')
    missing=[str(path) for path in required if not path.is_file()]
    if missing:raise FileNotFoundError('Missing Phase-1 checkpoint/tokenizer files: '+', '.join(missing))
    tokenizer_hash=file_hash(source_tokenizer)
    for arch,path in chosen.items():
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved.get('model_config',{}).get('architecture')!=arch:
            raise ValueError(f'{path}: checkpoint architecture is not {arch}.')
        if saved.get('training_origin')!='random_initialization_official_phase1_only':
            raise ValueError(f'{path}: expected your own Phase-1 checkpoint.')
        if saved.get('tokenizer_sha256')!=tokenizer_hash:
            raise ValueError(f'{path}: checkpoint does not match the original tokenizer.')
        del saved;gc.collect()
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    tokenizer=output/'tokenizer.json'
    if source_tokenizer.resolve()!=tokenizer.resolve():shutil.copy2(source_tokenizer,tokenizer)
    source_checkpoints=dict(chosen)
    for arch,path in list(chosen.items()):
        target=output/'foundations'/arch/'best_model.pt';target.parent.mkdir(parents=True,exist_ok=True)
        if Path(path).resolve()!=target.resolve():
            if not target.exists() or file_hash(path)!=file_hash(target):shutil.copy2(path,target)
        chosen[arch]=str(target)
    atomic_json(output/'foundation_manifest.json',{'source_layout_root':str(selected_root),'input_mode':'direct_directory_only',
        'source_checkpoints':source_checkpoints,'source_tokenizer':str(source_tokenizer),
        'checkpoints':chosen,'tokenizer':str(tokenizer),'sha256':{a:file_hash(p) for a,p in chosen.items()},
        'tokenizer_sha256':tokenizer_hash})
    return chosen,str(tokenizer)

def obtain_data(project,input_root,local_root=None,revision=REVISION):
    destination=Path(project)/'data'
    if local_root is None:
        roots=[p.parent.parent for p in Path(input_root).rglob('fold1-text.json') if p.parent.name=='Train']
        roots=[p for p in roots if all((p/f).is_file() for f in DATA_FILES)]
        if len(roots)>1:raise ValueError('Multiple Phase-2 datasets found. Set DATA_ROOT explicitly.')
        local_root=roots[0] if roots else None
    if local_root:
        source=Path(local_root)
        if (source/'Phase_2_LNER').is_dir():source/= 'Phase_2_LNER'
        for name in DATA_FILES:
            target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
            if target.resolve()!=(source/name).resolve():shutil.copy2(source/name,target)
        origin={'source':'attached_official_data','source_root':str(source)}
    else:
        from huggingface_hub import hf_hub_download
        token=os.environ.get('HF_TOKEN')
        if not token:
            try:
                from kaggle_secrets import UserSecretsClient
                token=UserSecretsClient().get_secret('HF_TOKEN')
            except Exception:token=None
        for name in DATA_FILES:
            target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():continue
            try:downloaded=hf_hub_download(REPO,'Phase_2_LNER/'+name,repo_type='dataset',revision=revision,token=token)
            except Exception as exc:raise RuntimeError('Official data could not be fetched. Accept the dataset terms and add an HF_TOKEN Kaggle secret, or attach its six Phase-2 files. Do not put a token in notebook code.') from exc
            shutil.copy2(downloaded,target)
        origin={'source':'official_huggingface','repo':REPO,'revision':revision}
    origin['sha256']={name:file_hash(destination/name) for name in DATA_FILES};atomic_json(destination/'provenance.json',origin)
    return str(destination)

def experiments(foundations,settings):
    base={'epochs':settings['epochs'],'micro_batch':settings['micro_batch'],'accumulation':settings['accumulation'],
        'head_lr':5e-4,'backbone_lr':5e-5,'max_stage_hours':settings['max_stage_hours'],'patience':3,'workers':0,
        'head_dropout':.1,'boundary_weight':.25,'two_directions':True,'seed':230437,'rank':8,'alpha':16,
        'gradient_checkpointing':settings.get('gradient_checkpointing',False),'checkpoint_every':50,'log_every':20,
        'inference_batch':settings['micro_batch'],'allow_tiny':settings.get('allow_tiny',False)}
    specs=[('01_decoder_full','decoder','full','bio'),('02_encoder_decoder_full','encoder_decoder','full','bio'),
        ('03_local_gqa_lora','local_gqa','lora','bio'),('04_hybrid_partial','hybrid','partial','bio'),
        ('05_decoder_lorma_span','decoder','lorma','span'),('06_decoder_adapter','decoder','adapter','bio'),
        ('07_decoder_student_kd','decoder','kd','bio')]
    rows=[]
    for name,arch,method,head in specs:
        row={**base,'name':name,'architecture':arch,'method':method,'head':head,'phase1_checkpoint':foundations[arch]}
        if method in ('lora','lorma','adapter'):row['backbone_lr']=2e-4
        if method=='partial':row['partial_layers']=3
        if method=='adapter':row['adapter_dim']=64
        if method=='lorma':row.update({'span_dim':32,'max_spans_per_window':1024})
        if method=='kd':row.update({'student_layers':settings.get('student_layers',4),'kd_temperature':2.,'kd_alpha':.5})
        rows.append(row)
    return rows

def launch(script,args,project,gpus=2,log=None):
    # Explicit loopback avoids hostname-resolution failures in isolated containers.
    import socket
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    code=Path(project)/'code';cmd=[sys.executable,'-m','torch.distributed.run','--nnodes=1','--master_addr=127.0.0.1',f'--master_port={port}',f'--nproc_per_node={gpus}',str(code/script),*map(str,args)]
    env=dict(os.environ);env['PYTHONPATH']=str(code)+os.pathsep+env.get('PYTHONPATH','');env['TOKENIZERS_PARALLELISM']='false'
    handle=Path(log).open('a',encoding='utf-8') if log else None
    try:
        proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env)
        for line in proc.stdout:
            print(line,end='',flush=True)
            if handle:handle.write(line);handle.flush()
        if proc.wait()!=0:raise RuntimeError(f'{script} failed; inspect {log or "cell output"}. Recovery checkpoints are retained.')
    finally:
        if handle:handle.close()

def run_experiments(project,rows,cache_path,settings):
    project=Path(project);cache=torch.load(cache_path,map_location='cpu',weights_only=False);catalog=[];gpus=settings.get('gpus',2)
    for cfg in rows:
        name=cfg['name'];root=project/'runs'/name;root.mkdir(parents=True,exist_ok=True)
        if cfg['method']=='kd':
            eligible=[c for c in catalog if c['head']=='bio']
            teacher=max(eligible,key=lambda c:c['validation_macro_f1'])
            cfg.update({'teacher_experiment':teacher['name'],'teacher_parameters':teacher['parameters']['total_parameters']})
            for stage,split in [('selection','train'),('refit','all')]:
                if stage=='refit' and not settings['refit_all']:continue
                checkpoint=project/'runs'/teacher['name']/stage/'best.pt'
                target=root/('teacher_'+split)
                launch('ner_predict.py',['--checkpoint',checkpoint,'--cache',cache_path,'--split',split,'--teacher-cache','--output',target,
                    '--batch-size',cfg['micro_batch']],project,gpus=1,log=root/f'teacher_{split}.log')
        config=root/'config.json';atomic_json(config,cfg)
        train_args=['--config',config,'--cache',cache_path,'--output',root/'selection','--split','train']
        if cfg['method']=='kd':train_args+=['--teacher-dir',root/'teacher_train']
        launch('train_ner.py',train_args,project,gpus,root/'selection.log')
        selection=root/'selection'/'best.pt';summary=read_json(root/'selection'/'summary.json')
        # Decode the selected checkpoint, not the last epoch, for comparable reported metrics.
        launch('ner_predict.py',['--checkpoint',selection,'--cache',cache_path,'--split','val','--output',root/'selected_validation',
            '--batch-size',cfg['micro_batch']],project,gpus,root/'validation.log')
        metrics=read_json(root/'selected_validation'/'metrics.json');test_checkpoint=selection
        if settings['refit_all']:
            refit_args=['--config',config,'--cache',cache_path,'--output',root/'refit','--split','all','--epochs',max(1,summary['best_epoch'])]
            if cfg['method']=='kd':refit_args+=['--teacher-dir',root/'teacher_all']
            launch('train_ner.py',refit_args,project,gpus,root/'refit.log');test_checkpoint=root/'refit'/'best.pt'
        launch('ner_predict.py',['--checkpoint',test_checkpoint,'--cache',cache_path,'--split','test','--output',root/'test_prediction',
            '--batch-size',cfg['micro_batch']],project,gpus,root/'prediction.log')
        prediction=read_json(root/'test_prediction'/'prediction.json');texts={k:cache['texts'][k] for k in prediction}
        zip_path=project/'submissions'/f'{name}.zip';pred_hash=submission_zip(prediction,texts,cache['labels'],zip_path)
        item={'name':name,'architecture':cfg['architecture'],'method':cfg['method'],'head':cfg['head'],'validation_macro_f1':metrics['macro_f1'],
            'metrics':metrics,'parameters':summary['parameters'],'selection_epoch':summary['best_epoch'],
            'selection_seconds':summary['elapsed_seconds'],'test_checkpoint':str(test_checkpoint),'refit_all':settings['refit_all'],
            'submission':str(zip_path),'prediction_sha256':pred_hash,'entities':sum(map(len,prediction.values()))}
        catalog.append(item);atomic_json(project/'catalog.json',catalog)
        print('READY TO DOWNLOAD:',zip_path,'held-out strict Macro-F1:',round(metrics['macro_f1'],5),flush=True)
    if len(catalog)!=7 or len({c['architecture'] for c in catalog})!=4:raise ValueError('Experiment coverage is incomplete.')
    hashes={c['prediction_sha256'] for c in catalog}
    if len(hashes)<7:
        atomic_json(project/'distinctness_warning.json',{'unique_predictions':len(hashes),'experiments':7})
        print('WARNING: Seven experiments ran, but some test predictions are identical. Renaming ZIPs does not create distinct predictions. Inspect per-label metrics and improve the affected experiment before uploading duplicates.')
    report=project/'comparison.csv'
    columns=['name','architecture','method','head','validation_macro_f1','selection_epoch','selection_seconds','entities','prediction_sha256']
    with report.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows({k:c[k] for k in columns} for c in catalog)
    return catalog

def bundle_working(working,output):
    working=Path(working);output=Path(output);temp=output.with_suffix('.tmp.zip')
    files=[p for p in working.rglob('*') if p.is_file() and p.resolve() not in (output.resolve(),temp.resolve()) and '__pycache__' not in p.parts]
    total=sum(p.stat().st_size for p in files);free=shutil.disk_usage(working).free
    if free<total+256*1024**2:raise RuntimeError('Not enough free space for the complete working archive. Download the seven small submission ZIPs first; remove/re-export large recovery states deliberately, then rerun this cell.')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for path in files:z.write(path,path.relative_to(working))
    temp.replace(output)
    with zipfile.ZipFile(output) as z:
        bad=z.testzip()
        if bad:raise ValueError(f'Archive CRC failure: {bad}')
    return str(output)

def benchmark_suite(rows,cache,project,settings):
    """Actual T4 memory/step measurement; no benchmark weights are used for submission."""
    import math
    from ner_data import NERWindows,collate_windows,class_weights
    from ner_models import build_model,ner_loss,parameter_report
    from ner_predict import autocast
    if not torch.cuda.is_available():raise RuntimeError('Run this preflight on Kaggle T4 GPUs.')
    device=torch.device('cuda',0);reports=[];effective=settings['micro_batch']*settings['accumulation']
    for cfg in rows:
        while True:
            torch.cuda.empty_cache();model=None;optimizer=None;batch=None;out=None;loss=None;params=None;scaler=None;weights=None
            try:
                model,_=build_model(cfg,cache['labels'],cache['char_classes'],cache['tokenizer_sha256']);model.to(device).train()
                params=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=1e-4,fused=True)
                scaler=torch.amp.GradScaler('cuda');weights=class_weights(cache,'train').to(device)
                ds=NERWindows(cache,'train',cfg['head']);longest=max(range(len(ds)),key=lambda i:len(ds[i]['ids']))
                batch=collate_windows([ds[longest] for _ in range(cfg['micro_batch'])])
                for key in ('ids','valid','tags','left','right'):batch[key]=batch[key].to(device)
                torch.cuda.reset_peak_memory_stats();times=[]
                successful=attempts=0
                while successful<4 and attempts<16:
                    attempts+=1
                    torch.cuda.synchronize();start=time.monotonic();optimizer.zero_grad(set_to_none=True)
                    with autocast(device):out=model(batch['ids'],batch['valid']);loss=ner_loss(out,batch,cfg,weights)
                    if not bool(torch.isfinite(loss)):raise ValueError('Preflight loss is non-finite.')
                    scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
                    if scaler.get_scale()>=old_scale:
                        successful+=1
                        if successful>1:times.append(time.monotonic()-start)
                if successful<4:raise ValueError('Preflight could not obtain four finite FP16 updates after scale adjustment.')
                peak=torch.cuda.max_memory_reserved()/1024**3
                if peak>settings.get('memory_budget_gib',12.5):raise torch.cuda.OutOfMemoryError('Reserve headroom for DDP and full GPU execution.')
                mean=sum(times)/len(times);batches=math.ceil(math.ceil(len(ds)/settings['gpus'])/cfg['micro_batch'])
                estimate=mean*batches*cfg['epochs']*(2 if settings['refit_all'] else 1)
                reports.append({'name':cfg['name'],'micro_batch':cfg['micro_batch'],'accumulation':cfg['accumulation'],
                    'microbatch_seconds_single_gpu':mean,'peak_reserved_gib_single_gpu':peak,'parameters':parameter_report(model),
                    'rough_train_refit_hours_excluding_validation_ddp_io':estimate/3600})
                print(reports[-1],flush=True);break
            except torch.cuda.OutOfMemoryError:
                if cfg['micro_batch']==1:raise RuntimeError('Preflight still runs out of memory at micro-batch 1. Enable gradient checkpointing or reduce SEQ_LEN and rebuild the cache.')
                cfg['micro_batch']=max(1,cfg['micro_batch']//2);cfg['accumulation']=max(1,math.ceil(effective/cfg['micro_batch']))
                cfg['inference_batch']=cfg['micro_batch'];print('Reducing',cfg['name'],'micro-batch to',cfg['micro_batch'],flush=True)
            finally:
                del model,optimizer,batch,out,loss,params,scaler,weights;gc.collect();torch.cuda.empty_cache()
    atomic_json(Path(project)/'benchmark.json',reports)
    print('Rough measured training+refit forecast (excludes validation, KD caching, inference, DDP and IO):',round(sum(r['rough_train_refit_hours_excluding_validation_ddp_io'] for r in reports),2),'hours')
    return reports
