"""Kaggle workflow: official data, seven real adaptation methods, optional lexical fusion and exports."""
import contextlib,gc,json,math,os,shutil,socket,subprocess,sys,time,zipfile
from pathlib import Path
import numpy as np
import torch
from cj_data import DATA_FILES,atomic_json,atomic_torch,digest,file_hash,read_json,load_records,metric,probabilities,choose_calibration,write_submission,Bags,collate,to_device
from cj_model import restore_phase3,build_model,parameter_report,classification_loss

def obtain_data(project,local_root=None,revision=None):
    destination=Path(project)/'data';destination.mkdir(parents=True,exist_ok=True);provenance=destination/'source.json'
    if local_root:
        source=Path(local_root)
        if (source/'Phase_4_CJPE').is_dir():source/='Phase_4_CJPE'
        missing=[name for name in DATA_FILES if not (source/name).is_file()]
        if missing:raise FileNotFoundError('Missing official Phase-4 files: '+', '.join(missing))
        origin={'kind':'attached_official_phase4','source':str(source)}
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
                    source=hf_hub_download(repo,'Phase_4_CJPE/'+name,repo_type='dataset',revision=commit,token=token)
                    shutil.copy2(source,target)
        except Exception as error:raise RuntimeError('Attach Phase_4_CJPE, or accept the official dataset terms and enable the HF_TOKEN Kaggle secret. Never paste API tokens into code.') from error
    origin['sha256']={name:file_hash(destination/name) for name in DATA_FILES};atomic_json(provenance,origin)
    return str(destination)

def experiments(parents,settings):
    common={'head_dim':settings['head_dim'],'dropout':.2,'encoder_dropout':.05,'chunk_batch':settings['chunk_batch'],
        'encoder_checkpointing':True,'head_only':False,'batch_docs':settings['batch_docs'],'head_batch':settings['head_batch'],
        'accumulation':settings['accumulation'],'encoder_lr':1e-5,'head_lr':2e-4,'weight_decay':.01,
        'epochs':settings['adapt_epochs'],'patience':settings['adapt_patience'],'min_delta':0.,'warmup_ratio':.08,
        'rank':8,'alpha':16.,'adapter_dim':48,'unfrozen_layers':2,'label_smoothing':.02,'seed':230437,
        'checkpoint_every':100,'log_every':50,'temperature':2.,'kd_alpha':.5,'view':'last' if settings.get('two_stage') else 'all'}
    specs=[('01_decoder_full','decoder','full','bigru'),('02_encoder_decoder_partial','encoder_decoder','partial','transformer'),
        ('03_local_gqa_lorma','local_gqa','lorma','bigru'),('04_decoder_adapter','decoder','adapter','bigru'),
        ('05_hybrid_bitfit','hybrid','bitfit','bigru'),('06_encoder_decoder_gradual','encoder_decoder','gradual','bigru'),
        ('07_decoder_student_kd','decoder','kd','bigru')]
    rows=[]
    for name,architecture,method,head in specs:
        cfg={**common,'name':name,'architecture':architecture,'method':method,'head_type':head,'parent':parents[architecture]}
        if method in ('lorma','adapter','bitfit'):cfg['encoder_lr']=1e-4
        if method=='kd':cfg['student_layers']=settings['student_layers']
        rows.append(cfg)
    return rows
def launch(action,args,project,gpus=2,log=None):
    script=Path(project)/'code/cj_run.py'
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
def frozen_key(cfg):return cfg['architecture']+('_student_'+str(cfg['student_layers']) if cfg['method']=='kd' else '')
def warm_config(cfg,settings):return {**cfg,'head_only':True,'view':'all','head_lr':6e-4,'batch_docs':settings['head_batch'],'accumulation':1,
    'epochs':settings['head_epochs'],'patience':settings['head_patience']}

def lexical_branch(project,data_root,data_path,enabled=True):
    if not enabled:return None
    import joblib
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    project=Path(project);root=project/'lexical';root.mkdir(parents=True,exist_ok=True);data=read_json(data_path);result_path=root/'probabilities.pt'
    identity=digest({'data':data['fingerprint'],'feature':'word_1_2_full_and_last2048words','version':1})
    if result_path.exists():
        result=torch.load(result_path,map_location='cpu',weights_only=False)
        if result['identity']!=identity:raise ValueError('Lexical model data changed.')
        return result
    rows=load_records(data_root);byid={r['id']:r for r in rows};ids=[d['id'] for d in data['docs']]
    texts=[byid[did]['text'] for did in ids];tails=[' '.join(t.split()[-2048:]) for t in texts];train=data['splits']['train']
    vectorizer=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=150_000,sublinear_tf=True,dtype=np.float32)
    start=time.monotonic()
    try:vectorizer.fit([texts[i] for i in train])
    except ValueError as error:raise ValueError('Lexical vocabulary is empty. Disable USE_LEXICAL_FUSION for tiny/non-word smoke data.') from error
    matrix=hstack((vectorizer.transform(texts),vectorizer.transform(tails)),format='csr')
    labels=[data['docs'][i]['gold'] for i in train]
    classifier=LogisticRegression(C=2.,class_weight='balanced',solver='liblinear',max_iter=500,random_state=230437)
    classifier.fit(matrix[train],labels);p=classifier.predict_proba(matrix)[:,1].astype(np.float32)
    values={did:float(value) for did,value in zip(ids,p)};dev=data['splits']['dev']
    scores=metric([data['docs'][i]['gold'] for i in dev],p[dev])
    joblib.dump({'vectorizer':vectorizer,'classifier':classifier,'identity':identity},root/'model.joblib',compress=3)
    result={'identity':identity,'values':values,'data_fingerprint':data['fingerprint'],'dev_metrics':scores,
        'elapsed_seconds':time.monotonic()-start,'fit_split':'train_only','feature':'full_text_and_final2048_words'}
    atomic_torch(result_path,result);atomic_json(root/'metrics.json',{k:v for k,v in result.items() if k!='values'})
    print('Train-only lexical branch:',scores,'seconds:',round(result['elapsed_seconds'],1),flush=True)
    return result

def benchmark(rows,data_path,device=None):
    device=device or torch.device('cuda',0);data=read_json(data_path);dataset=Bags(data_path,'train');measurements=[]
    if device.type=='cuda':torch.cuda.set_device(device)
    for cfg in rows:
        dataset=Bags(data_path,'train',view=cfg.get('view','all'))
        torch.manual_seed(cfg['seed']);model,_=build_model(cfg);model.to(device).train();model.set_epoch(0)
        row=dataset[0];row['chunks']=row['chunks'][:4];row['position']=row['position'][:4]
        batch=to_device(collate([row]),device)
        if cfg['method']=='kd':batch['teacher']=torch.zeros(1,2,device=device)
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
        n=len(row['chunks']);seconds=sum(times)/len(times);total_chunks=sum(len(data['docs'][i]['chunks']) if cfg.get('view','all')=='all' else 1 for i in data['splits']['train'])
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
        'values':{data['docs'][i]['id']:torch.zeros(2) for i in data['splits']['train']}})
    for cfg in rows:
        current={**cfg,'batch_docs':1,'accumulation':1,'epochs':1,'checkpoint_every':10**9};path=root/f"{cfg['name']}.json";atomic_json(path,current)
        destination=root/cfg['name'];args=['--config',path,'--data',data_path,'--output',destination,'--smoke']
        if cfg['method']=='kd':args+=['--teacher',teacher]
        launch('train',args,project,settings['gpus'],root/f"{cfg['name']}.log")
        if not read_json(destination/'summary.json')['complete']:raise ValueError('Incomplete GPU smoke test.')
        shutil.rmtree(destination)
    teacher.unlink();atomic_json(root/'passed.json',{'experiments':[r['name'] for r in rows],'data_fingerprint':data['fingerprint'],'world_size':settings['gpus']})

def _predict(project,checkpoint,data_path,split,output,settings,features=None,sections='all'):
    args=['--checkpoint',checkpoint,'--data',data_path,'--split',split,'--output',output,'--sections',sections]
    if features:args+=['--features',features]
    launch('predict',args,project,settings['gpus'],Path(output).parent/(Path(output).name+'.log'))
    return torch.load(Path(output)/'complete.pt',map_location='cpu',weights_only=False)
def _array(raw,data,split):return probabilities(torch.stack([raw['values'][data['docs'][i]['id']] for i in data['splits'][split]]).numpy())
def _fuse(neural,ids,lexical,calibration):
    weight=calibration['lexical_weight']
    return np.asarray(neural)*(1-weight)+np.asarray([lexical['values'][did] for did in ids])*weight if weight else np.asarray(neural)
def _teacher_cache(project,item,data_path,split,root,settings,lexical):
    data=read_json(data_path);root=Path(root);root.mkdir(parents=True,exist_ok=True);target=root/'teacher.pt'
    identity=digest({'source':file_hash(item['selection_checkpoint']),'calibration':item['calibration'],'data':data['fingerprint'],
        'split':split,'lexical_identity':lexical['identity'] if lexical else None})
    if target.exists():
        saved=torch.load(target,map_location='cpu',weights_only=False)
        if saved['identity']!=identity:raise ValueError('Teacher changed; use a new PROJECT.')
        return target
    raw=_predict(project,item['selection_checkpoint'],data_path,split,root/'raw',settings,item.get('selected_features'))
    ids=[data['docs'][i]['id'] for i in data['splits'][split]];p=_fuse(_array(raw,data,split),ids,lexical,item['calibration'])
    p=np.clip(p,1e-6,1-1e-6);values={did:torch.tensor([math.log(1-q),math.log(q)],dtype=torch.float32) for did,q in zip(ids,p)}
    atomic_torch(target,{'identity':identity,'data_fingerprint':data['fingerprint'],'split':split,'values':values,
        'teacher_parameters':item['parameters']['total'],'teacher_checkpoint_sha256':file_hash(item['selection_checkpoint']),
        'teacher_name':item['name'],'teacher_kind':'selected_phase4_classifier_with_optional_train_only_lexical_branch'})
    return target

def run_all(project,rows,data_path,settings,lexical=None):
    project=Path(project);data=read_json(data_path);catalog=[];dev_ids=[data['docs'][i]['id'] for i in data['splits']['dev']]
    test_ids=[data['docs'][i]['id'] for i in data['splits']['test']];truth=[data['docs'][i]['gold'] for i in data['splits']['dev']]
    atomic_json(project/'dev_truth.json',[{'id':did,'gold':gold} for did,gold in zip(dev_ids,truth)])
    for original in rows:
        cfg=dict(original);root=project/'runs'/cfg['name'];root.mkdir(parents=True,exist_ok=True);teacher=None
        if cfg['method']=='kd':
            strongest=max(catalog,key=lambda item:item['dev_metrics']['macro_f1'])
            teacher=_teacher_cache(project,strongest,data_path,'train',root/'teacher_train',settings,lexical)
            cfg['teacher_name']=strongest['name']
        feature_root=project/'features'/frozen_key(cfg);feature_root.mkdir(parents=True,exist_ok=True)
        feature_cfg={**cfg,'head_only':True,'method':'kd' if cfg['method']=='kd' else 'full'}
        feature_path=feature_root/'config.json';atomic_json(feature_path,feature_cfg)
        launch('features',['--config',feature_path,'--data',data_path,'--output',feature_root],project,settings['gpus'],feature_root/'features.log')
        warm=warm_config(cfg,settings);warm_path=root/'warm_config.json';atomic_json(warm_path,warm)
        args=['--config',warm_path,'--data',data_path,'--split','train','--output',root/'warm_selection','--features',feature_root]
        if teacher:args+=['--teacher',teacher]
        launch('train',args,project,1,root/'warm_selection.log')
        cfg['head_init']=str(root/'warm_selection/best.pt');cfg_path=root/'config.json';atomic_json(cfg_path,cfg)
        args=['--config',cfg_path,'--data',data_path,'--split','train','--output',root/'selection']
        if teacher:args+=['--teacher',teacher]
        launch('train',args,project,settings['gpus'],root/'selection.log')
        adapted_checkpoint=root/'selection/best.pt';adapted_features=None;hierarchy_epoch=None
        if settings.get('two_stage'):
            hierarchy={**warm_config(cfg,settings),'backbone_init':str(adapted_checkpoint),'head_init':str(adapted_checkpoint)}
            hierarchy_path=root/'hierarchy_config.json';atomic_json(hierarchy_path,hierarchy)
            adapted_features=root/'adapted_features';adapted_features.mkdir(exist_ok=True)
            launch('features',['--config',hierarchy_path,'--data',data_path,'--output',adapted_features],project,settings['gpus'],root/'adapted_features.log')
            args=['--config',hierarchy_path,'--data',data_path,'--split','train','--output',root/'hierarchy_selection','--features',adapted_features]
            if teacher:args+=['--teacher',teacher]
            launch('train',args,project,1,root/'hierarchy_selection.log')
            adapted_checkpoint=root/'hierarchy_selection/best.pt';hierarchy_epoch=read_json(root/'hierarchy_selection/summary.json')['best_epoch']
        warm_raw=_predict(project,root/'warm_selection/best.pt',data_path,'dev',root/'warm_dev',settings,feature_root)
        adapted_raw=_predict(project,adapted_checkpoint,data_path,'dev',root/'adapted_dev',settings,adapted_features)
        lexical_dev=np.asarray([lexical['values'][did] for did in dev_ids]) if lexical else None
        warm_metrics,warm_cal=choose_calibration(truth,_array(warm_raw,data,'dev'),lexical_dev)
        adapt_metrics,adapt_cal=choose_calibration(truth,_array(adapted_raw,data,'dev'),lexical_dev)
        # Preserve the stronger held-out model instead of forcing a degraded adaptation into a ZIP.
        use_warm=settings.get('allow_warm_fallback',True) and warm_metrics['macro_f1']>adapt_metrics['macro_f1']+1e-9
        stage='warmup' if use_warm else 'hierarchy' if settings.get('two_stage') else 'adaptation'
        selected=root/'warm_selection/best.pt' if use_warm else adapted_checkpoint
        selected_features=feature_root if use_warm else adapted_features
        scores,calibration=(warm_metrics,warm_cal) if use_warm else (adapt_metrics,adapt_cal);chosen=warm_raw if use_warm else adapted_raw
        summary=read_json(selected.parent/'summary.json')
        dev_probability=_fuse(_array(chosen,data,'dev'),dev_ids,lexical,calibration)
        dev_prediction=[{'id':did,'pred':int(q>=calibration['threshold'])} for did,q in zip(dev_ids,dev_probability)]
        atomic_json(root/'dev_prediction.json',dev_prediction);atomic_json(root/'calibration.json',calibration)
        atomic_json(root/'dev_metrics.json',{'selected_stage':stage,'selected':scores,'warmup':warm_metrics,'adaptation':adapt_metrics})
        raw=_predict(project,selected,data_path,'test',root/'selected_test',settings,selected_features)
        p=_fuse(_array(raw,data,'test'),test_ids,lexical,calibration);preds=[int(q>=calibration['threshold']) for q in p]
        submission=project/'submissions'/f"{cfg['name']}.zip";prediction,sha=write_submission(test_ids,preds,submission)
        atomic_json(root/'prediction.json',prediction)
        item={'name':cfg['name'],'architecture':cfg['architecture'],'method':cfg['method'],'selected_stage':stage,
            'dev_metrics':scores,'warmup_metrics':warm_metrics,'adaptation_metrics':adapt_metrics,'calibration':calibration,
            'parameters':summary['parameters'],'selection_checkpoint':str(selected),'test_checkpoint':str(selected),
            'features':str(feature_root),'submission':str(submission),'prediction_sha256':sha,'dev_used_for_selection':True,
            'selected_features':str(selected_features) if selected_features else None,'hierarchy_epoch':hierarchy_epoch,
            'encoder_stage_parameters':read_json(root/'selection/summary.json')['parameters'],
            'warmup_epoch':read_json(root/'warm_selection/summary.json')['best_epoch'],'adaptation_epoch':read_json(root/'selection/summary.json')['best_epoch'],
            'teacher_name':cfg.get('teacher_name')}
        catalog.append(item);atomic_json(project/'catalog.json',catalog)
        print('READY:',submission,'development-tuned Macro-F1:',round(scores['macro_f1'],5),'selected:',stage,flush=True)
    if len(catalog)!=7 or len({r['architecture'] for r in catalog})!=4:raise ValueError('Experiment coverage incomplete.')
    distinct=len({r['prediction_sha256'] for r in catalog})
    print('Seven experiment ZIPs; distinct prediction sets:',distinct,flush=True)
    if distinct<7:print('Identical predictions do not count as different submissions. Inspect duplicate candidates; labels have not been altered to manufacture uniqueness.',flush=True)
    if any(r['selected_stage']=='warmup' for r in catalog):print('Some selected ZIPs use a frozen warmup fallback. Their adaptation experiments ran, but check method diversity among the actual selected submissions.',flush=True)
    return catalog

def bonus_ensemble(project,data_path,lexical=None):
    """Optional extra ZIP only for a development improvement; never a claimed eighth adaptation method."""
    project=Path(project);data=read_json(data_path);catalog=read_json(project/'catalog.json');ranked=sorted(catalog,key=lambda r:r['dev_metrics']['macro_f1'],reverse=True)
    dev_ids=[data['docs'][i]['id'] for i in data['splits']['dev']];test_ids=[data['docs'][i]['id'] for i in data['splits']['test']]
    truth=[data['docs'][i]['gold'] for i in data['splits']['dev']];groups=[ranked[:3]];families=[];seen=set()
    for item in ranked:
        if item['architecture'] not in seen:families.append(item);seen.add(item['architecture'])
    groups.append(families);best_score=ranked[0]['dev_metrics']['macro_f1'];winner=None
    for members in groups:
        dev_prob=[];test_prob=[]
        for item in members:
            root=project/'runs'/item['name']
            dev_dir='joint_dev' if item['selected_stage']=='joint' else 'warm_dev' if item['selected_stage']=='warmup' else 'adapted_dev'
            test_dir='refit_test' if item.get('all_label_refit') else 'joint_test' if item['selected_stage']=='joint' else 'selected_test'
            dev_raw=torch.load(root/dev_dir/'complete.pt',map_location='cpu',weights_only=False)
            test_raw=torch.load(root/test_dir/'complete.pt',map_location='cpu',weights_only=False)
            adjust=lambda p:1/(1+np.exp(-np.clip(np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))-math.log(item['calibration']['threshold']/(1-item['calibration']['threshold'])),-30,30)))
            dev_prob.append(adjust(_fuse(_array(dev_raw,data,'dev'),dev_ids,lexical,item['calibration'])))
            test_prob.append(adjust(_fuse(_array(test_raw,data,'test'),test_ids,lexical,item['calibration'])))
        scores,cal=choose_calibration(truth,np.mean(dev_prob,axis=0))
        if scores['macro_f1']>best_score+1e-9:
            best_score=scores['macro_f1'];winner=(members,scores,cal,np.mean(test_prob,axis=0))
    if winner is None:
        atomic_json(project/'recommended_submission.json',{'submission':ranked[0]['submission'],'name':ranked[0]['name'],'dev_metrics':ranked[0]['dev_metrics'],'kind':'single_selected_classifier'})
        print('No measured ensemble gain. Recommended existing ZIP:',ranked[0]['submission'],flush=True);return None
    members,scores,cal,p=winner;path=project/'bonus/score_ensemble.zip'
    prediction,sha=write_submission(test_ids,[int(q>=cal['threshold']) for q in p],path)
    atomic_json(project/'bonus/prediction.json',prediction)
    record={'submission':str(path),'kind':'development_selected_ensemble','members':[r['name'] for r in members],
        'weights':[1/len(members)]*len(members),'dev_metrics':scores,'calibration':cal,'prediction_sha256':sha,
        'counts_as_new_adaptation_method':False,'dev_used_for_selection':True}
    atomic_json(project/'recommended_submission.json',record);print('Optional score ensemble:',path,scores,flush=True);return record

def export_manifest(project):
    project=Path(project).resolve();catalog=read_json(project/'catalog.json');models=[]
    for item in catalog:
        checkpoint=Path(item['test_checkpoint']).resolve();saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        dependencies={}
        for key in ('parent','backbone_init'):
            if not saved['config'].get(key):continue
            path=Path(saved['config'][key]).resolve()
            if not path.is_relative_to(project):raise ValueError('Portable dependency must be inside PROJECT.')
            dependencies[key]={'relative_path':str(path.relative_to(project)),'sha256':file_hash(path)}
        models.append({'name':item['name'],'architecture':item['architecture'],'method':item['method'],
            'checkpoint':str(checkpoint.relative_to(project)),'checkpoint_sha256':file_hash(checkpoint),
            'dependencies':dependencies,'config':saved['config'],'provenance':saved['provenance']})
    atomic_json(project/'phase4_model_manifest.json',{'format_version':1,'tokenizer':'parents/tokenizer.json',
        'tokenizer_sha256':file_hash(project/'parents/tokenizer.json'),'models':models,
        'restoration':'Resolve checkpoint and dependency relative_path against the extracted capstone_phase4 root; override config paths before building.'})
    return project/'phase4_model_manifest.json'

def joint_refine_top(project,rows,data_path,settings,lexical,top_k=1):
    project=Path(project);catalog=read_json(project/'catalog.json');data=read_json(data_path);byname={r['name']:r for r in rows}
    truth=[data['docs'][i]['gold'] for i in data['splits']['dev']];dev_ids=[data['docs'][i]['id'] for i in data['splits']['dev']]
    test_ids=[data['docs'][i]['id'] for i in data['splits']['test']]
    for item in sorted(catalog,key=lambda r:r['dev_metrics']['macro_f1'],reverse=True)[:top_k]:
        if item.get('joint_refined'):continue
        root=project/'runs'/item['name'];selected=torch.load(item['selection_checkpoint'],map_location='cpu',weights_only=False)['config']
        cfg={**byname[item['name']],'view':'all','head_only':False,'head_init':item['selection_checkpoint'],
            'epochs':settings.get('joint_epochs',2),'patience':settings.get('joint_patience',1),
            'batch_docs':1,'accumulation':settings.get('joint_accumulation',4)}
        if selected.get('backbone_init'):cfg['backbone_init']=selected['backbone_init']
        path=root/'joint_config.json';atomic_json(path,cfg)
        args=['--config',path,'--data',data_path,'--split','train','--output',root/'joint_selection']
        if cfg['method']=='kd':args+=['--teacher',root/'teacher_train/teacher.pt']
        launch('train',args,project,settings['gpus'],root/'joint_selection.log');checkpoint=root/'joint_selection/best.pt'
        raw=_predict(project,checkpoint,data_path,'dev',root/'joint_dev',settings)
        scores,cal=choose_calibration(truth,_array(raw,data,'dev'),np.asarray([lexical['values'][did] for did in dev_ids]) if lexical else None)
        item['joint_refined']=True;item['joint_dev_metrics']=scores
        atomic_json(root/'joint_comparison.json',{'before':item['dev_metrics'],'joint':scores,'accepted':scores['macro_f1']>item['dev_metrics']['macro_f1']+1e-9})
        if scores['macro_f1']>item['dev_metrics']['macro_f1']+1e-9:
            raw_test=_predict(project,checkpoint,data_path,'test',root/'joint_test',settings)
            p=_fuse(_array(raw_test,data,'test'),test_ids,lexical,cal)
            backup=root/'before_joint_submission.zip'
            if not backup.exists():shutil.copy2(item['submission'],backup)
            prediction,sha=write_submission(test_ids,[int(q>=cal['threshold']) for q in p],item['submission'])
            atomic_json(root/'prediction.json',prediction)
            pdev=_fuse(_array(raw,data,'dev'),dev_ids,lexical,cal)
            atomic_json(root/'dev_prediction.json',[{'id':did,'pred':int(q>=cal['threshold'])} for did,q in zip(dev_ids,pdev)])
            item.update(pre_joint_checkpoint=item['selection_checkpoint'],selection_checkpoint=str(checkpoint),test_checkpoint=str(checkpoint),
                selected_stage='joint',selected_features=None,calibration=cal,dev_metrics=scores,prediction_sha256=sha,
                parameters=read_json(root/'joint_selection/summary.json')['parameters'],
                joint_epoch=read_json(root/'joint_selection/summary.json')['best_epoch'])
            atomic_json(root/'calibration.json',cal)
        atomic_json(project/'catalog.json',catalog)
        print('Joint refinement:',item['name'],'old-vs-joint comparison saved; selected stage:',item['selected_stage'],flush=True)
    return catalog

def refit_top(project,rows,data_path,settings,lexical,top_k=1):
    """Optional fixed-epoch all-label refit after all seven held-out submissions already exist."""
    if not top_k:return read_json(Path(project)/'catalog.json')
    project=Path(project);data=read_json(data_path);catalog=read_json(project/'catalog.json');original_byname={r['name']:r for r in rows}
    ids=[data['docs'][i]['id'] for i in data['splits']['test']];selected=sorted(catalog,key=lambda r:r['dev_metrics']['macro_f1'],reverse=True)[:top_k]
    for item in selected:
        if item.get('all_label_refit'):continue
        cfg=dict(original_byname[item['name']]);root=project/'runs'/cfg['name'];teacher=None
        if cfg['method']=='kd':
            candidates=[r for r in catalog if r['method']!='kd'];strongest=max(candidates,key=lambda r:r['dev_metrics']['macro_f1'])
            teacher=_teacher_cache(project,strongest,data_path,'all',root/'teacher_all',settings,lexical)
        warm=warm_config(cfg,settings);path=root/'warm_refit_config.json';atomic_json(path,warm)
        args=['--config',path,'--data',data_path,'--split','all','--output',root/'warm_refit','--features',item['features'],'--epochs',max(1,item['warmup_epoch'])]
        if teacher:args+=['--teacher',teacher]
        launch('train',args,project,1,root/'warm_refit.log');checkpoint=root/'warm_refit/best.pt'
        features=item['features']
        if item['selected_stage'] in ('adaptation','hierarchy','joint'):
            cfg['head_init']=str(checkpoint);path=root/'refit_config.json';atomic_json(path,cfg)
            args=['--config',path,'--data',data_path,'--split','all','--output',root/'refit','--epochs',max(1,item['adaptation_epoch'])]
            if teacher:args+=['--teacher',teacher]
            launch('train',args,project,settings['gpus'],root/'refit.log');checkpoint=root/'refit/best.pt';features=None
            if settings.get('two_stage'):
                hierarchy={**warm_config(cfg,settings),'backbone_init':str(checkpoint),'head_init':str(checkpoint)}
                path=root/'hierarchy_refit_config.json';atomic_json(path,hierarchy);features=root/'refit_features';features.mkdir(exist_ok=True)
                launch('features',['--config',path,'--data',data_path,'--output',features],project,settings['gpus'],root/'refit_features.log')
                args=['--config',path,'--data',data_path,'--split','all','--output',root/'hierarchy_refit','--features',features,
                    '--epochs',max(1,item.get('hierarchy_epoch') or item['warmup_epoch'])]
                if teacher:args+=['--teacher',teacher]
                launch('train',args,project,1,root/'hierarchy_refit.log');checkpoint=root/'hierarchy_refit/best.pt'
            if item['selected_stage']=='joint':
                joint={**cfg,'view':'all','head_init':str(checkpoint),'epochs':settings.get('joint_epochs',2),'batch_docs':1,
                    'accumulation':settings.get('joint_accumulation',4)}
                if settings.get('two_stage'):joint['backbone_init']=str(root/'refit/best.pt')
                path=root/'joint_refit_config.json';atomic_json(path,joint)
                args=['--config',path,'--data',data_path,'--split','all','--output',root/'joint_refit','--epochs',max(1,item['joint_epoch'])]
                if teacher:args+=['--teacher',teacher]
                launch('train',args,project,settings['gpus'],root/'joint_refit.log');checkpoint=root/'joint_refit/best.pt';features=None
        raw=_predict(project,checkpoint,data_path,'test',root/'refit_test',settings,features)
        p=_fuse(_array(raw,data,'test'),ids,lexical,item['calibration']);preds=[int(q>=item['calibration']['threshold']) for q in p]
        # Keep the held-out-model ZIP as a recovery artifact outside submissions/.
        recovery=root/'selection_submission.zip'
        if not recovery.exists():shutil.copy2(item['submission'],recovery)
        prediction,sha=write_submission(ids,preds,item['submission']);atomic_json(root/'prediction.json',prediction)
        item.update(test_checkpoint=str(checkpoint),prediction_sha256=sha,all_label_refit=True)
        atomic_json(project/'catalog.json',catalog)
    return catalog
def section_audit(project,data_path,settings):
    project=Path(project);catalog=read_json(project/'catalog.json');best=max(catalog,key=lambda r:r['dev_metrics']['macro_f1']);root=project/'section_audit';root.mkdir(exist_ok=True)
    data=read_json(data_path);y=[data['docs'][i]['gold'] for i in data['splits']['dev']];results={}
    for section in ('all','first','last'):
        raw=_predict(project,best['selection_checkpoint'],data_path,'dev',root/section,settings,
            best.get('selected_features'),section)
        results[section]=metric(y,_array(raw,data,'dev'))
    atomic_json(root/'metrics.json',{'model':best['name'],'scores':results,'interpretation':'Post-training occlusion sensitivity, not separately trained section-only models; attention is not causal explanation.'})
    return results
def bundle_working(working,destination):
    working=Path(working).resolve();destination=Path(destination).resolve();temporary=destination.with_suffix('.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for source in working.rglob('*'):
            if source.is_file() and source.resolve() not in (destination,temporary):archive.write(source,source.relative_to(working))
    temporary.replace(destination);return str(destination)
