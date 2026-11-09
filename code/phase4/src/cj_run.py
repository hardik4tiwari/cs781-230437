"""Distributed feature extraction, prediction, resumable selection and fixed-epoch refit."""
import argparse,contextlib,gc,json,math,os,random,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Sampler
from cj_data import Bags,collate,to_device,read_json,atomic_json,atomic_torch,file_hash,digest,metric,probabilities
from cj_model import build_model,parameter_report,classification_loss

def setup():
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    device=torch.device('cuda',local) if torch.cuda.is_available() else torch.device('cpu')
    if device.type=='cuda':torch.cuda.set_device(local)
    if world>1 and not dist.is_initialized():
        if device.type=='cuda':dist.init_process_group('nccl',device_id=device)
        else:dist.init_process_group('gloo')
    return rank,world,device
def barrier():
    if dist.is_initialized():dist.barrier()
def amp(device):return torch.autocast('cuda',dtype=torch.float16) if device.type=='cuda' else contextlib.nullcontext()
def signature(cfg):
    identity={'parent':file_hash(cfg['parent']),'student_layers':cfg.get('student_layers') if cfg['method']=='kd' else None,'pool':'mean_last','mask':'inherited'}
    if cfg.get('backbone_init'):identity['adapted_encoder']=file_hash(cfg['backbone_init'])
    return digest(identity)
def rng_state(device):return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
    'cuda':torch.cuda.get_rng_state(device) if device.type=='cuda' else None}
def restore_rng(state,device):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if device.type=='cuda':torch.cuda.set_rng_state(state['cuda'],device)
class LengthSampler(Sampler):
    def __init__(self,dataset,rank,world,seed):self.dataset,self.rank,self.world,self.seed,self.epoch=dataset,rank,world,seed,0
    def set_epoch(self,epoch):self.epoch=epoch
    def __len__(self):return math.ceil(len(self.dataset)/self.world)
    def __iter__(self):
        generator=torch.Generator().manual_seed(self.seed+self.epoch);order=torch.randperm(len(self.dataset),generator=generator).tolist();bucket=[]
        for first in range(0,len(order),256):
            bucket.extend(sorted(order[first:first+256],key=lambda i:len(self.dataset.data['docs'][self.dataset.rows[i]]['chunks'])))
        extra=len(self)*self.world-len(bucket)
        bucket.extend((bucket*math.ceil(extra/max(1,len(bucket))))[:extra])
        return iter(bucket[self.rank::self.world])

@torch.inference_mode()
def infer_local(model,dataset,rank,world,device,batch_size=1,sections='all'):
    model.eval();values={};attention={};indices=list(range(rank,len(dataset),world))
    # Cached feature batches avoid thousands of tiny GPU launches during head selection.
    for first in range(0,len(indices),batch_size):
        rows=[dataset[i] for i in indices[first:first+batch_size]]
        if sections!='all':
            for row in rows:
                n=len(row['position']);keep=list(range(n))
                if sections=='first':keep=keep[:max(1,math.ceil(n/4))]
                elif sections=='last':keep=keep[-max(1,math.ceil(n/4)):]
                else:raise ValueError(sections)
                row['position']=[row['position'][j] for j in keep]
                if row['features'] is not None:row['features']=row['features'][keep]
                else:row['chunks']=[row['chunks'][j] for j in keep]
        batch=to_device(collate(rows),device)
        with amp(device):out=model(batch)
        for i,row in enumerate(rows):
            values[row['id']]=out['logits'][i].float().cpu()
            if len(attention)<8:attention[row['id']]=out['attention'][i,:len(row['position'])].float().cpu().tolist()
    return values,attention
def gather_values(local,world):
    if world==1:return local
    parts=[None]*world;dist.all_gather_object(parts,local);result={}
    for part in parts:
        if set(part)&set(result):raise ValueError('Inference duplicated a case across ranks.')
        result.update(part)
    return result
def feature_cache(args,cfg,rank,world,device):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True);dataset=Bags(args.data,'every');data=dataset.data
    model,_=build_model(cfg);model.to(device).eval();width=model.pool_dim
    identity=digest({'data':data['fingerprint'],'encoder':signature(cfg),'shape':[data['total_chunks'],width]})
    manifest=root/'manifest.json';old=read_json(manifest) if manifest.exists() else None
    if old and old['identity']!=identity:raise ValueError('Feature cache source changed.')
    if old and old.get('complete'):
        if (root/'features.f16').stat().st_size!=2*data['total_chunks']*width:raise ValueError('Frozen feature file is incomplete.')
        if rank==0:print('Reusing frozen features:',root,flush=True)
        return
    if rank==0 and old is None:
        needed=2*data['total_chunks']*width
        if shutil_free(root)<needed+256*1024**2:raise RuntimeError(f'Need {needed/1024**3:.2f} GiB free for this complete feature cache. No cases are dropped.')
        with (root/'features.f16').open('wb') as handle:handle.truncate(needed)
        with (root/'done.u8').open('wb') as handle:handle.truncate(len(data['docs']))
        atomic_json(manifest,{'identity':identity,'complete':False,'shape':[data['total_chunks'],width],
            'encoder_signature':signature(cfg),'data_fingerprint':data['fingerprint']})
    barrier();matrix=np.memmap(root/'features.f16',dtype='<f2',mode='r+',shape=(data['total_chunks'],width))
    done=np.memmap(root/'done.u8',dtype='u1',mode='r+',shape=(len(data['docs']),));pending=[];started=time.monotonic();processed=0
    def flush():
        matrix.flush()
        if pending:done[pending]=1;done.flush();pending.clear()
    with torch.inference_mode():
        for index in range(rank,len(data['docs']),world):
            if done[index]:continue
            row=dataset.document(index)
            with amp(device):features=model.encode_chunks(row['chunks'],device)
            doc=data['docs'][index];a=doc['feature_start'];matrix[a:a+len(doc['chunks'])]=features.float().cpu().numpy().astype('<f2')
            pending.append(index);processed+=1
            if len(pending)>=64:flush()
            if processed%500==0 and rank==0:print(json.dumps({'feature_cases_rank0':processed,'elapsed_minutes':(time.monotonic()-started)/60}),flush=True)
    flush();barrier()
    if rank==0:
        if not np.all(done):raise ValueError('Feature extraction omitted cases.')
        info=read_json(manifest);info.update(complete=True,elapsed_seconds=time.monotonic()-started);atomic_json(manifest,info)
    barrier()
def shutil_free(path):
    import shutil
    return shutil.disk_usage(path).free
def predict(args,cfg,rank,world,device):
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False);cfg=saved['config'];root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    dataset=Bags(args.data,args.split,args.features,view=cfg.get('view','all'))
    if dataset.data['fingerprint']!=saved['data_fingerprint']:raise ValueError('Prediction dataset differs from the checkpoint.')
    if args.features:
        if not cfg.get('head_only'):raise ValueError('An adapted checkpoint must recompute its encoder features.')
        if read_json(Path(args.features)/'manifest.json')['encoder_signature']!=signature(cfg):raise ValueError('Wrong frozen feature encoder.')
    identity=digest({'checkpoint':file_hash(args.checkpoint),'data':dataset.data['fingerprint'],'split':args.split,'sections':args.sections})
    complete=root/'complete.pt'
    if complete.exists():
        old=torch.load(complete,map_location='cpu',weights_only=False)
        if old['identity']!=identity:raise ValueError('Prediction cache source changed.')
        if rank==0:print('Reusing logits:',root,flush=True)
        return
    model,_=build_model(cfg,args.checkpoint);model.to(device)
    local,attention=infer_local(model,dataset,rank,world,device,cfg.get('head_batch',16) if args.features else 1,args.sections)
    values=gather_values(local,world)
    if rank==0:
        expected=[dataset.data['docs'][i]['id'] for i in dataset.rows]
        if set(expected)!=set(values):raise ValueError('Prediction cache ID mismatch.')
        atomic_torch(complete,{'identity':identity,'values':values,'split':args.split,'data_fingerprint':dataset.data['fingerprint'],
            'checkpoint_sha256':file_hash(args.checkpoint),'config':cfg,'parameters':saved['parameters'],'attention_examples_rank0':attention})
    barrier()

def train(args,cfg,rank,world,device):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True);seed=cfg.get('seed',230437)
    random.seed(seed+rank);np.random.seed(seed+rank);torch.manual_seed(seed+rank)
    if bool(args.features)!=bool(cfg.get('head_only')):raise ValueError('Features are required exactly for frozen-backbone head training.')
    teacher=None;data=read_json(args.data)
    if args.features and read_json(Path(args.features)/'manifest.json')['encoder_signature']!=signature(cfg):raise ValueError('Frozen features came from another parent.')
    if args.teacher:
        cached=torch.load(args.teacher,map_location='cpu',weights_only=False)
        if cached.get('smoke_only') and not args.smoke:raise ValueError('Synthetic preflight teachers are forbidden in real training.')
        if cached['data_fingerprint']!=data['fingerprint'] or cached['split']!=args.split:raise ValueError('Teacher cache has the wrong dataset or training split.')
        teacher=cached['values']
    if cfg['method']=='kd' and teacher is None:raise ValueError('KD requires cached Phase-4 binary teacher logits.')
    dataset=Bags(args.data,args.split,args.features,teacher,view=cfg.get('view','all'))
    if teacher is not None and any(dataset.data['docs'][i]['id'] not in teacher for i in dataset.rows):raise ValueError('Teacher cache omits training cases.')
    dev=Bags(args.data,'dev',args.features,view=cfg.get('view','all'));sampler=LengthSampler(dataset,rank,world,seed)
    generator=torch.Generator();loader=DataLoader(dataset,batch_size=cfg.get('batch_docs',1),sampler=sampler,collate_fn=collate,
        num_workers=0,pin_memory=device.type=='cuda',generator=generator)
    model,provenance=build_model(cfg);model.to(device);report=parameter_report(model)
    if cfg['method']=='kd' and args.teacher:
        if cached['teacher_parameters']<=report['total']:raise ValueError('The teacher must have more parameters than this student.')
    # Gradual unfreezing keeps optimizer membership stable; DDP is rebuilt after each policy change.
    parameters=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    groups=[]
    for is_encoder in (True,False):
        for decay in (True,False):
            group=[p for n,p in parameters if n.startswith('backbone.')==is_encoder and (p.ndim>=2)==decay]
            if group:groups.append({'params':group,'lr':cfg['encoder_lr'] if is_encoder else cfg['head_lr'],
                'base_lr':cfg['encoder_lr'] if is_encoder else cfg['head_lr'],'weight_decay':cfg.get('weight_decay',.01) if decay else 0.})
    optimizer=torch.optim.AdamW(groups,fused=device.type=='cuda');scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    epochs=args.epochs or cfg['epochs'];accum=cfg.get('accumulation',1);planned=math.ceil(len(loader)/accum)*epochs
    support=np.bincount([data['docs'][i]['gold'] for i in dataset.rows],minlength=2).astype(float)
    weights=np.sqrt(support.sum()/np.maximum(support,1));weights=weights/weights.mean();weights=torch.tensor(weights,dtype=torch.float32,device=device)
    identity=digest({'config':cfg,'data':data['fingerprint'],'parent':file_hash(cfg['parent']),'epochs':epochs,'split':args.split,'world':world,
        'teacher':file_hash(args.teacher) if args.teacher else None,'features':signature(cfg) if args.features else None,
        'head_init':file_hash(cfg['head_init']) if cfg.get('head_init') else None,
        'backbone_init':file_hash(cfg['backbone_init']) if cfg.get('backbone_init') else None,'smoke':args.smoke})
    latest=root/'latest.pt';summary_path=root/'summary.json';best=root/'best.pt'
    if summary_path.exists():
        old=read_json(summary_path)
        if old['identity']!=identity:raise ValueError('Run configuration changed. Use a new PROJECT.')
        if old.get('complete') and best.exists():
            if rank==0:print('Already complete:',root,flush=True)
            return
    epoch=cursor=updates=skipped=bad=0;best_f1=-1.;best_epoch=0;history=[];elapsed_prior=0.
    if latest.exists():
        old=torch.load(latest,map_location='cpu',weights_only=False)
        if old['identity']!=identity:raise ValueError('Resume identity mismatch.')
        model.load_state_dict(old['model'],strict=not old['delta_only']);optimizer.load_state_dict(old['optimizer']);scaler.load_state_dict(old['scaler'])
        epoch,cursor,updates,skipped,bad=[old[k] for k in ('epoch','cursor','updates','skipped','bad')]
        best_f1,best_epoch,history,elapsed_prior=old['best_f1'],old['best_epoch'],old['history'],old['elapsed']
        restore_rng(old['rng'][rank],device)
    started=time.monotonic();trainable_snapshot={n for n,p in model.named_parameters() if p.requires_grad}
    delta_only=cfg.get('head_only') or cfg['method'] not in ('full','kd','gradual')
    def save(is_best=False,complete=False):
        states=[None]*world
        if world>1:dist.all_gather_object(states,rng_state(device))
        else:states[0]=rng_state(device)
        if rank==0:
            state=model.state_dict()
            if delta_only:state={k:v for k,v in state.items() if k in trainable_snapshot}
            light={'model':state,'delta_only':delta_only,'config':cfg,'provenance':provenance,'parameters':report,
                'data_fingerprint':data['fingerprint'],'training_origin':'own_phase1_phase2_phase3_phase4_cjpe',
                'parent_sha256':file_hash(cfg['parent']),
                'stage':args.split,'epoch':epoch,'updates':updates}
            if is_best:atomic_torch(best,light)
            checkpoint={**light,'identity':identity,'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'rng':states,
                'cursor':cursor,'skipped':skipped,'bad':bad,'best_f1':best_f1,'best_epoch':best_epoch,'history':history,'elapsed':elapsed_prior+time.monotonic()-started}
            atomic_torch(latest,checkpoint);atomic_json(summary_path,{'identity':identity,'complete':complete,'best_epoch':best_epoch,
                'best_f1':best_f1,'epoch':epoch,'updates':updates,'amp_skipped_updates':skipped,'parameters':report,
                'elapsed_seconds':checkpoint['elapsed'],'history':history,'stop_reason':'smoke' if args.smoke else 'epochs_or_patience'})
        barrier()
    if rank==0:print(json.dumps({'experiment':cfg['name'],'stage':args.split,'head_only':cfg.get('head_only',False),
        'cases':len(dataset),'parameters':report,'effective_batch':cfg.get('batch_docs',1)*world*accum}),flush=True)
    while epoch<epochs:
        model.set_epoch(epoch);sampler.set_epoch(epoch);generator.manual_seed(seed+epoch)
        wrapped=DDP(model,device_ids=[device.index] if device.type=='cuda' else None,broadcast_buffers=False) if world>1 else model
        iterator=iter(loader)
        for _ in range(cursor):next(iterator)
        model.train();loss_sum=loss_count=0
        while cursor<len(loader):
            micro=[next(iterator) for _ in range(min(accum,len(loader)-cursor))];sample_count=sum(len(b['rows']) for b in micro);optimizer.zero_grad(set_to_none=True)
            ratio=(updates+1)/max(1,planned);warm=cfg.get('warmup_ratio',.08)
            factor=min(1.,ratio/max(warm,1e-6)) if ratio<warm else .1+.9*.5*(1+math.cos(math.pi*min(1.,(ratio-warm)/max(1-warm,1e-6))))
            for group in optimizer.param_groups:group['lr']=group['base_lr']*factor
            for index,batch in enumerate(micro):
                batch=to_device(batch,device);sync=wrapped.no_sync() if world>1 and index<len(micro)-1 else contextlib.nullcontext()
                with sync:
                    with amp(device):result=wrapped(batch);loss,components=classification_loss(result['logits'],batch,cfg,weights)
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite loss; recovery states were retained.')
                    scaler.scale(loss*len(batch['rows'])/sample_count).backward()
                loss_sum+=float(loss.detach());loss_count+=1
            scaler.unscale_(optimizer);gradient=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            scale=scaler.get_scale();scaler.step(optimizer);scaler.update();skipped+=int(scaler.get_scale()<scale);updates+=int(scaler.get_scale()>=scale);cursor+=len(micro)
            if rank==0 and updates and updates%cfg.get('log_every',50)==0:
                print(json.dumps({'epoch':epoch+1,'batch':cursor,'updates':updates,'loss':loss_sum/max(1,loss_count),
                    'gradient_norm':float(gradient),'elapsed_hours':(elapsed_prior+time.monotonic()-started)/3600,'amp_skipped_updates':skipped}),flush=True)
                loss_sum=loss_count=0
            if cursor<len(loader) and updates and updates%cfg.get('checkpoint_every',100)==0:save()
            if args.smoke:
                epoch+=1;cursor=0;best_epoch=epoch;best_f1=0.;save(True,True)
                if rank==0:print('PASS actual distributed update:',cfg['name'],flush=True)
                return
        epoch+=1;cursor=0
        if args.split=='train':
            values,_=infer_local(model,dev,rank,world,device,cfg.get('head_batch',16) if args.features else 1)
            values=gather_values(values,world);ordered=[data['docs'][i] for i in data['splits']['dev']]
            result=metric([d['gold'] for d in ordered],probabilities(torch.stack([values[d['id']] for d in ordered]).numpy()))
            score=result['macro_f1'];history.append({'epoch':epoch,**result})
            improved=score>best_f1+cfg.get('min_delta',0.)
            if improved:best_f1=score;best_epoch=epoch;bad=0
            else:bad+=1
            if rank==0:print(json.dumps({'epoch':epoch,'dev_macro_f1_at_half':score,'best_epoch':best_epoch}),flush=True)
        else:improved=True;best_epoch=epoch;best_f1=0.
        save(improved,epoch>=epochs or (args.split=='train' and bad>=cfg.get('patience',2)))
        del wrapped;gc.collect()
        if args.split=='train' and bad>=cfg.get('patience',2):break
    save(False,True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=['features','train','predict'],required=True)
    p.add_argument('--config');p.add_argument('--checkpoint');p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--split',default='train',choices=['train','dev','test','all','every']);p.add_argument('--features');p.add_argument('--teacher')
    p.add_argument('--epochs',type=int);p.add_argument('--smoke',action='store_true');p.add_argument('--sections',default='all',choices=['all','first','last'])
    args=p.parse_args();rank,world,device=setup()
    cfg=read_json(args.config) if args.config else torch.load(args.checkpoint,map_location='cpu',weights_only=False)['config']
    try:
        if args.action=='features':feature_cache(args,cfg,rank,world,device)
        elif args.action=='predict':predict(args,cfg,rank,world,device)
        else:train(args,cfg,rank,world,device)
    finally:
        if dist.is_initialized():dist.destroy_process_group()
if __name__=='__main__':main()
