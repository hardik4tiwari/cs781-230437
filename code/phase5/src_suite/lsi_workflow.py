"""Seven Phase-5 methods: shared full-document banks plus adapted evidence and dev-only fusion."""
import contextlib,gc,json,math,os,shutil,socket,subprocess,sys,time,zipfile
from pathlib import Path
import numpy as np
import torch
from lsi_data import *
from lsi_model import restore_phase1,build_model,parameter_report,classification_loss
def obtain_data(project,local_root=None,revision=None):
    destination=Path(project)/'data';destination.mkdir(parents=True,exist_ok=True);provenance=destination/'source.json'
    if local_root:
        source=Path(local_root)
        if (source/'Phase_5_LSI').is_dir():source/='Phase_5_LSI'
        missing=[name for name in DATA_FILES if not (source/name).is_file()]
        if missing:raise FileNotFoundError('Missing official Phase-5 files: '+', '.join(missing))
        origin={'kind':'attached_official_phase5','source':str(source)}
        for name in DATA_FILES:
            target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
            if target.resolve()!=(source/name).resolve():shutil.copy2(source/name,target)
    else:
        old=read_json(provenance) if provenance.exists() else None
        if old and old.get('kind')=='official_huggingface' and old.get('sha256') and (revision is None or revision==old.get('revision')):
            if all((destination/name).is_file() for name in DATA_FILES):
                hashes={name:file_hash(destination/name) for name in DATA_FILES}
                if hashes!=old['sha256']:raise ValueError('Cached official data changed; restore it or use a new PROJECT.')
                return str(destination)
        from huggingface_hub import HfApi,hf_hub_download
        token=os.environ.get('HF_TOKEN')
        if not token:
            try:
                from kaggle_secrets import UserSecretsClient
                token=UserSecretsClient().get_secret('HF_TOKEN')
            except Exception:token=None
        repo='Exploration-Lab/CS-781-Capstone';commit=old['revision'] if old and old.get('kind')=='official_huggingface' else revision
        try:
            commit=HfApi(token=token).dataset_info(repo,revision=commit).sha
            origin={'kind':'official_huggingface','repo':repo,'revision':commit};atomic_json(provenance,origin)
            for name in DATA_FILES:
                target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():
                    source=hf_hub_download(repo,'Phase_5_LSI/'+name,repo_type='dataset',revision=commit,token=token)
                    shutil.copy2(source,target)
        except Exception as error:raise RuntimeError('Attach Phase_5_LSI, or accept the official dataset terms and enable the HF_TOKEN Kaggle secret. Never paste API tokens into code.') from error
    origin['sha256']={name:file_hash(destination/name) for name in DATA_FILES};atomic_json(provenance,origin)
    return str(destination)

def launch(action,args,project,gpus=2,log=None):
    script=Path(project)/'code/lsi_run.py'
    if gpus==1:cmd=[sys.executable,'-u',str(script),'--action',action,*map(str,args)]
    else:
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        cmd=[sys.executable,'-m','torch.distributed.run','--nnodes=1','--master_addr=127.0.0.1',f'--master_port={port}',
            f'--nproc_per_node={gpus}',str(script),'--action',action,*map(str,args)]
    env=dict(os.environ);env['PYTHONPATH']=str(Path(project)/'code')+os.pathsep+env.get('PYTHONPATH','')
    env['TOKENIZERS_PARALLELISM']='false';env['OMP_NUM_THREADS']='1'
    handle=Path(log).open('a',encoding='utf-8') if log else None
    try:
        process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env)
        for line in process.stdout:
            print(line,end='',flush=True)
            if handle:handle.write(line);handle.flush()
        if process.wait()!=0:raise RuntimeError(f'{action} failed. Inspect {log or "cell output"}. Earlier candidates and resume states remain available.')
    finally:
        if handle:handle.close()

def benchmark(rows,data_path,device=None):
    device=device or torch.device('cuda',0);data=read_json(data_path);dataset=Bags(data_path,'train');measurements=[]
    if device.type=='cuda':torch.cuda.set_device(device)
    for cfg in rows:
        dataset=Bags(data_path,'train',view=cfg.get('view','all'),adapt_chunks=cfg.get('adapt_chunks',3))
        torch.manual_seed(cfg['seed']);model,_=build_model(cfg);model.to(device).train();model.set_epoch(0)
        row=dataset[0];row['chunks']=row['chunks'][:4];row['position']=row['position'][:4]
        batch=to_device(collate([row]),device)
        if cfg['method']=='kd':batch['teacher']=torch.zeros(1,cfg['label_count'],device=device)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['head_lr'],fused=device.type=='cuda')
        scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda');times=[];peak=0
        for iteration in range(3):
            if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            start=time.monotonic();optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16) if device.type=='cuda' else contextlib.nullcontext():
                result=model(batch);loss,_=classification_loss(result['logits'],batch,cfg,torch.ones(2,device=device))
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
            if device.type=='cuda':torch.cuda.synchronize();peak=max(peak,torch.cuda.max_memory_reserved()/1024**3)
            if iteration:times.append(time.monotonic()-start)
        n=len(row['chunks']);seconds=sum(times)/len(times);total_chunks=sum(len(data['docs'][i]['chunks']) if cfg.get('view','all')=='all' else min(len(data['docs'][i]['chunks']),cfg.get('adapt_chunks',3)) for i in data['splits']['train'])
        result={'name':cfg['name'],'timed_chunks':n,'seconds_per_sample':seconds,'seconds_per_chunk':seconds/n,
            'peak_reserved_gib':peak,'parameters':parameter_report(model),'rough_adaptation_hours':total_chunks/n*seconds*cfg['epochs']/2/3600,
            'forecast_excludes':'feature extraction, heads, validation, KD caches, DDP, loading and I/O; variable-length bags differ'}
        measurements.append(result);print(json.dumps(result),flush=True)
        del model,optimizer,scaler,batch,result,loss;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
    return measurements

def smoke_all(project,rows,data_path,settings):
    project=Path(project);data=read_json(data_path);root=project/'preflight';root.mkdir(parents=True,exist_ok=True)
    teacher=root/'synthetic_teacher_for_smoke_only.pt'
    atomic_torch(teacher,{'data_fingerprint':data['fingerprint'],'split':'train','teacher_parameters':10**12,'smoke_only':True,
        'values':{data['docs'][i]['id']:torch.zeros(data['label_count']) for i in data['splits']['train']}})
    for cfg in rows:
        current={**cfg,'batch_docs':1,'accumulation':1,'epochs':1,'checkpoint_every':10**9};path=root/f"{cfg['name']}.json";atomic_json(path,current)
        destination=root/cfg['name'];args=['--config',path,'--data',data_path,'--output',destination,'--smoke']
        if cfg['method']=='kd':args+=['--teacher',teacher]
        launch('train',args,project,settings['gpus'],root/f"{cfg['name']}.log")
        if not read_json(destination/'summary.json')['complete']:raise ValueError('Incomplete GPU smoke test.')
        shutil.rmtree(destination)
    teacher.unlink();atomic_json(root/'passed.json',{'experiments':[r['name'] for r in rows],'data_fingerprint':data['fingerprint'],'world_size':settings['gpus']})

def bundle_working(working,destination):
    working=Path(working).resolve();destination=Path(destination).resolve();temporary=destination.with_suffix('.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for source in working.rglob('*'):
            if source.is_file() and source.resolve() not in (destination,temporary):archive.write(source,source.relative_to(working))
    temporary.replace(destination);return str(destination)

def experiments(parents,settings,data):
    support=np.asarray(data['train_support']);prior=((support+.5)/(len(data['splits']['train'])+1)).tolist()
    common={'head_dim':settings['head_dim'],'dropout':.2,'encoder_dropout':.05,'chunk_batch':settings['chunk_batch'],
        'encoder_checkpointing':True,'head_only':False,'batch_docs':settings['batch_docs'],'head_batch':settings['head_batch'],
        'accumulation':settings['accumulation'],'encoder_lr':1e-5,'head_lr':3e-4,'weight_decay':.01,
        'epochs':settings['adapt_epochs'],'patience':settings['adapt_patience'],'warmup_ratio':.08,
        'rank':8,'alpha':16.,'adapter_dim':48,'unfrozen_layers':2,'seed':230437,'checkpoint_every':100,'log_every':50,
        'temperature':2.,'kd_alpha':.5,'view':'evidence','adapt_chunks':settings['adapt_chunks'],'prompt_tokens':16,
        'label_count':data['label_count'],'label_prior':prior,'loss':'asl','gamma_negative':4.,'gamma_positive':0.,'asl_clip':.05}
    specs=[('01_decoder_full','decoder','full','bigru'),('02_encoder_decoder_partial','encoder_decoder','partial','transformer'),
        ('03_local_gqa_lorma','local_gqa','lorma','bigru'),('04_hybrid_adapter','hybrid','adapter','bigru'),
        ('05_decoder_prompt','decoder','prompt','bigru'),('06_encoder_decoder_gradual','encoder_decoder','gradual','bigru'),
        ('07_decoder_student_kd','decoder','kd','bigru')]
    rows=[]
    for name,architecture,method,head in specs:
        cfg={**common,'name':name,'architecture':architecture,'method':method,'head_type':head,'parent':parents[architecture]}
        if method in ('lorma','adapter'):cfg['encoder_lr']=1e-4
        if method=='prompt':cfg['encoder_lr']=5e-3
        if method=='kd':cfg['student_layers']=settings['student_layers']
        rows.append(cfg)
    return rows

def frozen_key(cfg):
    return digest({'parent':file_hash(cfg['parent']),'student_layers':cfg.get('student_layers') if cfg['method']=='kd' else None})[:16]

def warm_config(cfg,settings):
    return {**cfg,'name':'shared_warm_'+frozen_key(cfg)+'_'+cfg['head_type'],
        'method':'kd' if cfg['method']=='kd' else 'full','head_only':True,'view':'all','encoder_lr':1e-5,
        'epochs':settings['head_epochs'],'patience':settings['head_patience'],'batch_docs':settings['head_batch'],'accumulation':1}

def _predict(project,checkpoint,data_path,split,output,settings,features=None):
    args=['--checkpoint',checkpoint,'--data',data_path,'--split',split,'--output',output]
    if features:args+=['--features',features]
    launch('predict',args,project,1 if features else settings['gpus'],Path(output).parent/(Path(output).name+'.log'))
    return torch.load(Path(output)/'complete.pt',map_location='cpu',weights_only=False)

def _array(raw,data,split):
    return probabilities(torch.stack([raw['values'][data['docs'][i]['id']] for i in data['splits'][split]]).numpy())

def lexical_branch(project,data_root,data_path,max_features=80000):
    import joblib
    from joblib import Parallel,delayed
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    root=Path(project)/'lexical';root.mkdir(parents=True,exist_ok=True);data=read_json(data_path)
    identity=digest({'data':data['fingerprint'],'kind':'train_only_tfidf_sgd_ovr','max_features':max_features,'version':1})
    output=root/'probabilities.pt'
    if output.exists():
        old=torch.load(output,map_location='cpu',weights_only=False)
        if old['identity']!=identity:raise ValueError('Lexical setup changed; use a new PROJECT.')
        return old
    rows=load_records(data_root);byid={r['id']:r for r in rows};texts=[byid[d['id']]['text'] for d in data['docs']]
    train=data['splits']['train'];start=time.monotonic()
    vectorizer=TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=max_features,sublinear_tf=True,dtype=np.float32,token_pattern=r'(?u)\b\w+\b')
    vectorizer.fit([texts[i] for i in train]);X=vectorizer.transform(texts);y=target_matrix([data['docs'][i] for i in train],data['label_count'])
    def fit_one(j):
        target=y[:,j].astype(int)
        if target.min()==target.max():return None,np.full(X.shape[0],float(target[0]),dtype=np.float32)
        clf=SGDClassifier(loss='log_loss',alpha=1e-5,max_iter=20,tol=1e-3,class_weight='balanced',average=True,random_state=230437)
        clf.fit(X[train],target);return clf,clf.predict_proba(X)[:,1].astype(np.float32)
    fitted=Parallel(n_jobs=2,prefer='threads')(delayed(fit_one)(j) for j in range(data['label_count']))
    matrix=np.stack([v[1] for v in fitted],axis=1);values={doc['id']:matrix[i] for i,doc in enumerate(data['docs'])}
    result={'identity':identity,'values':values,'data_fingerprint':data['fingerprint'],'fit_split':'train_only',
        'elapsed_seconds':time.monotonic()-start,'label_count':data['label_count']}
    joblib.dump({'vectorizer':vectorizer,'classifiers':[v[0] for v in fitted],'constants':y.mean(0),'identity':identity},root/'model.joblib',compress=3)
    atomic_torch(output,result);print('Train-only lexical branch seconds:',round(result['elapsed_seconds'],1),flush=True);return result

def _fuse(warm,adapted,ids,lexical,calibration):
    alpha=calibration['adapt_weight'];p=(1-alpha)*np.asarray(warm)+alpha*np.asarray(adapted)
    weight=calibration.get('lexical_weight',0.)
    if weight:p=(1-weight)*p+weight*np.asarray([lexical['values'][did] for did in ids])
    return p

def calibrate(y,warm,adapted,ids,lexical):
    best=None
    for alpha in (.25,.5,.75,1.):
        for weight in ((0.,.2,.4) if lexical else (0.,)):
            cal={'adapt_weight':alpha,'lexical_weight':weight};p=_fuse(warm,adapted,ids,lexical,cal)
            threshold=fit_thresholds(y,p);scores=metric(y,p,threshold)
            if best is None or scores['macro_f1']>best[0]['macro_f1']+1e-10:best=(scores,{**cal,'thresholds':threshold})
    return best

def _teacher_cache(project,item,data_path,root,settings,lexical):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);data=read_json(data_path);target=root/'teacher.pt'
    identity=digest({'warm':file_hash(item['warm_checkpoint']),'adapted':file_hash(item['adapted_checkpoint']),
        'calibration':item['calibration'],'data':data['fingerprint'],'lexical':lexical['identity'] if lexical else None})
    if target.exists():
        old=torch.load(target,map_location='cpu',weights_only=False)
        if old['identity']!=identity:raise ValueError('Teacher changed. Use a new PROJECT.')
        return target
    warm=_predict(project,item['warm_checkpoint'],data_path,'train',root/'warm',settings,item['features'])
    adapted=_predict(project,item['adapted_checkpoint'],data_path,'train',root/'adapted',settings)
    ids=[data['docs'][i]['id'] for i in data['splits']['train']]
    p=_fuse(_array(warm,data,'train'),_array(adapted,data,'train'),ids,lexical,item['calibration']);p=np.clip(p,1e-6,1-1e-6)
    values={did:torch.tensor(np.log(q/(1-q)),dtype=torch.float32) for did,q in zip(ids,p)}
    atomic_torch(target,{'identity':identity,'data_fingerprint':data['fingerprint'],'split':'train','values':values,
        'teacher_parameters':item['parameters']['total'],'teacher_name':item['name'],'teacher_kind':'phase5_independent_bernoulli_logits',
        'label_count':data['label_count']})
    return target

def run_all(project,rows,data_path,settings,lexical=None):
    project=Path(project);data=read_json(data_path);catalog=[]
    dev=[data['docs'][i] for i in data['splits']['dev']];y=target_matrix(dev,data['label_count'])
    dev_ids=[r['id'] for r in dev];test_ids=[data['docs'][i]['id'] for i in data['splits']['test']]
    allowed=np.flatnonzero(np.asarray(data['train_support'])>0).tolist();dev_allowed=[j for j in allowed if j<eval_dimension(y)]
    atomic_json(project/'dev_truth.json',[{'id':r['id'],'gold':r['gold']} for r in dev])
    for original in rows:
        cfg=dict(original);root=project/'runs'/cfg['name'];root.mkdir(parents=True,exist_ok=True);teacher=None
        if cfg['method']=='kd':
            strongest=max(catalog,key=lambda item:item['dev_metrics']['macro_f1'])
            teacher=_teacher_cache(project,strongest,data_path,root/'teacher_train',settings,lexical);cfg['teacher_name']=strongest['name']
        feature_root=project/'features'/frozen_key(cfg);feature_root.mkdir(parents=True,exist_ok=True)
        warm=warm_config(cfg,settings);feature_cfg=feature_root/'encoder_config.json';atomic_json(feature_cfg,warm)
        launch('features',['--config',feature_cfg,'--data',data_path,'--output',feature_root],project,settings['gpus'],feature_root/'features.log')
        head_candidates=[]
        for loss_kind in settings.get('head_losses',('asl','bce')):
            candidate={**warm,'name':warm['name']+'_'+loss_kind,'loss':loss_kind}
            warm_path=feature_root/f"warm_{cfg['head_type']}_{loss_kind}.json";atomic_json(warm_path,candidate)
            candidate_root=feature_root/f"head_{cfg['head_type']}_{loss_kind}"
            args=['--config',warm_path,'--data',data_path,'--split','train','--output',candidate_root,'--features',feature_root]
            if teacher:args+=['--teacher',teacher]
            launch('train',args,project,1,candidate_root.with_suffix('.log'))
            summary=read_json(candidate_root/'summary.json');head_candidates.append((summary['best_f1'],candidate_root,candidate))
        _,warm_root,warm=max(head_candidates,key=lambda value:value[0]);warm_checkpoint=warm_root/'best.pt'
        cfg['loss']=warm['loss']
        cfg['head_init']=str(warm_checkpoint);cfg_path=root/'config.json';atomic_json(cfg_path,cfg)
        args=['--config',cfg_path,'--data',data_path,'--split','train','--output',root/'adaptation']
        if teacher:args+=['--teacher',teacher]
        launch('train',args,project,settings['gpus'],root/'adaptation.log');adapted_checkpoint=root/'adaptation/best.pt'
        warm_dev=_predict(project,warm_checkpoint,data_path,'dev',root/'warm_dev',settings,feature_root)
        adapted_dev=_predict(project,adapted_checkpoint,data_path,'dev',root/'adapted_dev',settings)
        wp,ap=_array(warm_dev,data,'dev'),_array(adapted_dev,data,'dev')
        wp[:,np.asarray(data['train_support'])==0]=0;ap[:,np.asarray(data['train_support'])==0]=0
        scores,cal=calibrate(y,wp,ap,dev_ids,lexical)
        dev_probability=_fuse(wp,ap,dev_ids,lexical,cal)
        dev_preds=predict_labels(dev_probability,cal['thresholds'],dev_allowed)
        atomic_json(root/'dev_prediction.json',[{'id':did,'pred':p} for did,p in zip(dev_ids,dev_preds)])
        atomic_json(root/'calibration.json',cal)
        warm_test=_predict(project,warm_checkpoint,data_path,'test',root/'warm_test',settings,feature_root)
        adapted_test=_predict(project,adapted_checkpoint,data_path,'test',root/'adapted_test',settings)
        p=_fuse(_array(warm_test,data,'test'),_array(adapted_test,data,'test'),test_ids,lexical,cal)
        pred=predict_labels(p,cal['thresholds'],allowed);submission=project/'submissions'/f"{cfg['name']}.zip"
        prediction,sha=write_submission(test_ids,pred,submission);atomic_json(root/'prediction.json',prediction)
        summary=read_json(adapted_checkpoint.parent/'summary.json');warm_summary=read_json(warm_root/'summary.json')
        item={'name':cfg['name'],'architecture':cfg['architecture'],'method':cfg['method'],'dev_metrics':scores,'calibration':cal,
            'warm_checkpoint':str(warm_checkpoint),'adapted_checkpoint':str(adapted_checkpoint),'features':str(feature_root),
            'parameters':summary['parameters'],'warm_parameters':warm_summary['parameters'],'teacher_name':cfg.get('teacher_name'),
            'submission':str(submission),'prediction_sha256':sha,'dev_used_for_selection':True,
            'head_epochs':warm_summary['best_epoch'],'adapt_epochs':summary['best_epoch'],
            'full_document_branch':cal['adapt_weight']<1.,'all_tokens_encoded_in_frozen_bank':True,'adapted_view':cfg['view'],'adapt_chunks':cfg['adapt_chunks'],
            'selected_loss':cfg['loss'],'warm_head_candidates':[{'loss':r[2]['loss'],'dev_macro_f1':r[0]} for r in head_candidates]}
        catalog.append(item);atomic_json(project/'catalog.json',catalog)
        print('READY:',submission,'dev Macro-F1:',round(scores['macro_f1'],5),'fusion:',cal['adapt_weight'],cal['lexical_weight'],flush=True)
    if len(catalog)!=7 or len({r['architecture'] for r in catalog})!=4:raise ValueError('Experiment coverage incomplete.')
    distinct=len({r['prediction_sha256'] for r in catalog})
    print('Distinct base prediction sets:',distinct,'; general course minimum:',6,flush=True)
    if distinct<6:print('The global distinct-submission requirement is not yet met. Inspect training and dev errors, then improve genuinely different experiments. Completed ZIPs are retained; predictions are not changed artificially.',flush=True)
    best_by_family=[max([r for r in catalog if r['architecture']==a],key=lambda r:r['dev_metrics']['macro_f1']) for a in ('decoder','encoder_decoder','local_gqa','hybrid')]
    atomic_json(project/'best_four_architectures.json',best_by_family)
    required={r['name']:r for r in best_by_family}
    for r in catalog:
        if r['method'] in ('kd','lorma'):required[r['name']]=r
    atomic_json(project/'required_submission_plan.json',{'all_seven':[r['submission'] for r in catalog],
        'architecture_lorma_kd_coverage_subset':[r['submission'] for r in required.values()],
        'global_minimum_distinct_submissions':6,'distinct_base_prediction_sets':len({r['prediction_sha256'] for r in catalog}),
        'global_count_met_by_base_predictions':len({r['prediction_sha256'] for r in catalog})>=6,
        'note':'Submit all seven candidates and verify at least six distinct prediction sets under the general instructions. The family/method subset is coverage analysis, not the overall submission-count requirement. Include KD even when another decoder is stronger. The optional ensemble is not a new adaptation method.'})
    return catalog

def bonus_ensemble(project,data_path,lexical=None):
    project=Path(project);data=read_json(data_path);catalog=read_json(project/'catalog.json')
    ranked=sorted(catalog,key=lambda r:r['dev_metrics']['macro_f1'],reverse=True);families=read_json(project/'best_four_architectures.json')
    dev_ids=[data['docs'][i]['id'] for i in data['splits']['dev']];test_ids=[data['docs'][i]['id'] for i in data['splits']['test']]
    y=target_matrix([data['docs'][i] for i in data['splits']['dev']],data['label_count']);allowed=np.flatnonzero(np.asarray(data['train_support'])>0).tolist()
    base={'name':ranked[0]['name'],'submission':ranked[0]['submission'],'dev_metrics':ranked[0]['dev_metrics'],'kind':'single_adaptation'}
    best=base['dev_metrics']['macro_f1'];winner=None
    for members in (ranked[:3],families):
        devs=[];tests=[]
        for item in members:
            root=project/'runs'/item['name'];cal=item['calibration']
            arrays={}
            for split,ids in (('dev',dev_ids),('test',test_ids)):
                warm=torch.load(root/f'warm_{split}/complete.pt',map_location='cpu',weights_only=False)
                adapted=torch.load(root/f'adapted_{split}/complete.pt',map_location='cpu',weights_only=False)
                raw=_fuse(_array(warm,data,split),_array(adapted,data,split),ids,lexical,cal)
                threshold=np.asarray(cal['thresholds']).clip(1e-5,1-1e-5);raw=raw.clip(1e-5,1-1e-5)
                arrays[split]=probabilities(np.log(raw/(1-raw))-np.log(threshold/(1-threshold)))
            devs.append(arrays['dev']);tests.append(arrays['test'])
        p=np.mean(devs,0);p[:,np.asarray(data['train_support'])==0]=0
        thresholds=fit_thresholds(y,p);scores=metric(y,p,thresholds)
        if scores['macro_f1']>best+1e-10:best=scores['macro_f1'];winner=(members,np.mean(tests,0),thresholds,scores)
    if winner:
        members,p,thresholds,scores=winner;path=project/'bonus/ensemble.zip'
        _,sha=write_submission(test_ids,predict_labels(p,thresholds,allowed),path)
        base={'name':'bonus_ensemble','submission':str(path),'dev_metrics':scores,'thresholds':thresholds,'prediction_sha256':sha,
            'kind':'development_selected_ensemble','members':[r['name'] for r in members],'counts_as_new_adaptation_method':False}
    atomic_json(project/'recommended_submission.json',base);return base

def export_manifest(project):
    project=Path(project).resolve();catalog=read_json(project/'catalog.json');entries=[]
    for item in catalog:
        row={k:item[k] for k in ('name','architecture','method','calibration','dev_metrics')}
        for key in ('warm_checkpoint','adapted_checkpoint'):
            path=Path(item[key]).resolve();saved=torch.load(path,map_location='cpu',weights_only=False)
            parent=Path(saved['config']['parent']).resolve()
            row[key]={'relative_path':str(path.relative_to(project)),'sha256':file_hash(path),
                'parent_relative_path':str(parent.relative_to(project)),'parent_sha256':file_hash(parent)}
        row['features_relative_path']=str(Path(item['features']).resolve().relative_to(project));entries.append(row)
    atomic_json(project/'phase5_model_manifest.json',{'format_version':1,'models':entries,'tokenizer':'parents/tokenizer.json',
        'restoration':'Resolve checkpoint/parent paths under extracted capstone_phase5; restore adapted head_init from warm_checkpoint; fusion uses full warm and selected-evidence adapted logits. Lexical artifacts are in lexical/.'})
    return project/'phase5_model_manifest.json'

def bundle_models(project,destination):
    """Portable learned artifacts; raw official data/feature banks and optimizer recovery are reproducible."""
    project=Path(project).resolve();destination=Path(destination).resolve();temporary=destination.with_suffix('.tmp')
    selected=[]
    for path in project.rglob('*'):
        if not path.is_file() or path.resolve() in (destination,temporary):continue
        relative=path.relative_to(project)
        if relative.parts[0] in ('data','token_cache','preflight'):continue
        if path.name in ('features.f16','done.u8','latest.pt'):continue
        if path.suffix in ('.zip','.tmp') or '__pycache__' in relative.parts:continue
        if path.suffix=='.pt' and path.name not in ('best.pt','decoder.pt','encoder_decoder.pt','local_gqa.pt','hybrid.pt'):continue
        selected.append(path)
    estimated=sum(p.stat().st_size for p in selected)
    if shutil.disk_usage(destination.parent).free<estimated+128*1024**2:raise RuntimeError('Not enough space for model archive. Download submissions first; keep parents/ and best.pt files.')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for path in selected:archive.write(path,Path(project.name)/path.relative_to(project))
    temporary.replace(destination);return str(destination)
