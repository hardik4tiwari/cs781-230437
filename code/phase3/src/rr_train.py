"""Resumable two-GPU head warmup and short full/PEFT/KD RR adaptation."""
import argparse,contextlib,json,math,random,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler
from rr_data import Blocks,collate,to_device,digest,atomic_json,atomic_torch,metric,file_hash
from rr_models import build_model,loss_function,parameter_report
from rr_predict import setup,autocast,logits_for_documents,decode
def rng_state(device):return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
    'cuda':torch.cuda.get_rng_state(device) if device.type=='cuda' else None}
def restore_rng(state,device):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if device.type=='cuda':torch.cuda.set_rng_state(state['cuda'],device)
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--cache',required=True)
    p.add_argument('--output',required=True);p.add_argument('--split',choices=['train','all'],default='train')
    p.add_argument('--features');p.add_argument('--teacher');p.add_argument('--epochs',type=int);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();rank,world,device=setup();cfg=json.loads(Path(args.config).read_text());root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    seed=cfg.get('seed',230437);random.seed(seed+rank);np.random.seed(seed+rank);torch.manual_seed(seed+rank)
    cache=torch.load(args.cache,map_location='cpu',weights_only=False)
    if bool(args.features)!=bool(cfg.get('freeze_backbone',False)):
        raise ValueError('Frozen feature caches are used only for frozen-backbone warmup.')
    def load_aux(path,kind):
        if not path:return None
        value=torch.load(path,map_location='cpu',weights_only=False)
        if not value['complete'] or value['cache_fingerprint']!=cache['fingerprint'] or value['kind']!=kind:raise ValueError('Auxiliary cache mismatch.')
        if kind=='features':
            expected=digest([file_hash(cfg['parent']),cfg.get('student_layers') if cfg['method']=='kd' else None,cfg.get('token_bidirectional',False)])
            if value['source_sha256']!=expected:raise ValueError('Frozen features came from another encoder or attention configuration.')
        if kind=='logits' and value['split']!=args.split:raise ValueError('KD teacher must match the current training split.')
        if any(did not in value['values'] for did in cache['splits'][args.split]):raise ValueError('Auxiliary cache omits training documents.')
        return value['values']
    features=load_aux(args.features,'features');teacher=load_aux(args.teacher,'logits')
    if cfg['method']=='kd' and teacher is None and not args.smoke:raise ValueError('The student requires a larger RR teacher logit cache.')
    dataset=Blocks(cache,args.split,features,teacher)
    if not len(dataset):raise ValueError('Empty training blocks.')
    sampler=DistributedSampler(dataset,world,rank,shuffle=True,seed=seed,drop_last=False);generator=torch.Generator()
    loader=DataLoader(dataset,batch_size=cfg.get('batch_blocks',1),sampler=sampler,collate_fn=collate,
        num_workers=0,pin_memory=device.type=='cuda',generator=generator)
    model,provenance=build_model(cfg);model.to(device);report=parameter_report(model)
    if cfg['method']=='kd' and cfg.get('teacher_parameters',float('inf'))<=report['total']:raise ValueError('RR teacher must be larger than the student.')
    backbone=[];head=[]
    for name,param in model.named_parameters():
        if param.requires_grad:(backbone if name.startswith('backbone.') else head).append(param)
    optimizer=torch.optim.AdamW([{'params':backbone,'lr':cfg['encoder_lr'],'base_lr':cfg['encoder_lr']},
        {'params':head,'lr':cfg['head_lr'],'base_lr':cfg['head_lr']}],weight_decay=cfg.get('weight_decay',.03),fused=device.type=='cuda')
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    support=np.array([sum(cache['docs'][did]['gold'].count(label) for did in cache['splits'][args.split])+1 for label in cache['labels']],dtype=np.float32)
    weights=np.sqrt(support.sum()/support);weights=np.clip(weights/weights.mean(),.25,3.)
    weights=torch.tensor(weights,device=device)
    epochs=args.epochs or cfg['epochs'];accum=cfg.get('accumulation',2)
    planned=math.ceil(len(loader)/accum)*cfg.get('schedule_epochs',cfg['epochs'])
    identity=digest({'cfg':cfg,'cache':cache['fingerprint'],'split':args.split,'epochs':epochs,'world':world,
        'features':file_hash(args.features) if args.features else None,'teacher':file_hash(args.teacher) if args.teacher else None,
        'parent':file_hash(cfg['parent']),'head_init':file_hash(cfg['head_init']) if cfg.get('head_init') else None,'smoke':args.smoke})
    latest=root/'latest.pt';summary_path=root/'summary.json';best=root/'best.pt'
    if summary_path.exists():
        summary=json.loads(summary_path.read_text())
        if summary['identity']!=identity:raise ValueError('Run configuration changed; use a fresh PROJECT directory.')
        if summary.get('complete') and best.exists():
            if rank==0:print('Already complete:',root,flush=True)
            if world>1:dist.destroy_process_group()
            return
    epoch=cursor=updates=skipped=bad=0;best_f1=-1.;best_epoch=0;history=[];elapsed_prior=0.
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=False)
        if state['identity']!=identity:raise ValueError('Resume identity changed.')
        model.load_state_dict(state['model'],strict=not state['delta_only']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        epoch=state['epoch'];cursor=state['cursor'];updates=state['updates'];skipped=state['skipped'];bad=state['bad']
        best_f1=state['best_f1'];best_epoch=state['best_epoch'];history=state['history'];elapsed_prior=state['elapsed']
        restore_rng(state['rng'][rank],device)
    wrapped=DDP(model,device_ids=[device.index] if device.type=='cuda' else None,broadcast_buffers=False) if world>1 else model
    started=time.monotonic();stopped=False
    trainable={name for name,param in model.named_parameters() if param.requires_grad}
    delta_only=cfg.get('freeze_backbone',False) or cfg['method'] not in ('full','kd')
    def save(is_best=False,complete=False):
        states=[None]*world
        if world>1:dist.all_gather_object(states,rng_state(device))
        else:states[0]=rng_state(device)
        if rank==0:
            weights_state=model.state_dict()
            if delta_only:weights_state={key:value for key,value in weights_state.items() if key in trainable}
            light={'model':weights_state,'delta_only':delta_only,'config':cfg,'provenance':provenance,'parameters':report,
                'cache_fingerprint':cache['fingerprint'],'labels':cache['labels'],'training_origin':'own_phase1_phase2_phase3_rr',
                'stage':'selection' if args.split=='train' else 'all_label_refit','epoch':epoch,'updates':updates}
            if is_best:atomic_torch(best,light)
            state={**light,'identity':identity,'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'rng':states,
                'cursor':cursor,'skipped':skipped,'bad':bad,'best_f1':best_f1,'best_epoch':best_epoch,'history':history,
                'elapsed':elapsed_prior+time.monotonic()-started}
            atomic_torch(latest,state)
            atomic_json(summary_path,{'identity':identity,'complete':complete,'parameters':report,'epoch':epoch,'updates':updates,
                'best_epoch':best_epoch,'best_f1':best_f1,'elapsed_seconds':state['elapsed'],'history':history,'amp_skipped_updates':skipped,
                'stop_reason':'time_cap' if stopped else 'epochs_or_patience' if complete else 'running'})
        if world>1:dist.barrier()
    if rank==0:print(json.dumps({'experiment':cfg['name'],'stage':args.split,'frozen_features':bool(features is not None),
        'blocks':len(dataset),'parameters':report,'world_size':world}),flush=True)
    while epoch<epochs:
        sampler.set_epoch(epoch);generator.manual_seed(seed+epoch);iterator=iter(loader)
        for _ in range(cursor):next(iterator)
        model.train();loss_sum=0.;loss_count=0
        while cursor<len(loader):
            micro=[next(iterator) for _ in range(min(accum,len(loader)-cursor))];optimizer.zero_grad(set_to_none=True)
            warmup=max(1,round(planned*.1));step=updates+1
            factor=step/warmup if step<=warmup else .1+.9*.5*(1+math.cos(math.pi*min(1.,(step-warmup)/max(1,planned-warmup))))
            for group in optimizer.param_groups:group['lr']=group['base_lr']*factor
            for i,batch in enumerate(micro):
                batch=to_device(batch,device)
                if args.smoke and cfg['method']=='kd' and batch['teacher'] is None:
                    batch['teacher']=torch.zeros(*batch['mask'].shape,len(cache['labels']),device=device)
                sync=wrapped.no_sync() if world>1 and i+1<len(micro) else contextlib.nullcontext()
                with sync:
                    with autocast(device):
                        result=wrapped(batch);loss=loss_function(model,result,batch,weights,cfg)
                        if not bool(torch.isfinite(loss)):raise FloatingPointError('Non-finite loss; inspect FP16, data and saved state.')
                    scaler.scale(loss/len(micro)).backward()
                loss_sum+=float(loss.detach());loss_count+=1
            scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            before=scaler.get_scale();scaler.step(optimizer);scaler.update()
            if scaler.get_scale()<before:skipped+=1
            else:updates+=1
            cursor+=len(micro)
            flag=torch.tensor(int(elapsed_prior+time.monotonic()-started>=cfg.get('max_stage_hours',.15)*3600 or (args.smoke and updates>=1)),device=device)
            if world>1:dist.all_reduce(flag,op=dist.ReduceOp.MAX)
            stopped=bool(flag)
            if updates and updates%cfg.get('checkpoint_every',100)==0:save()
            if rank==0 and updates and updates%cfg.get('log_every',20)==0:
                print(json.dumps({'epoch':epoch+1,'updates':updates,'loss':loss_sum/max(loss_count,1),'gradient_norm':float(norm),
                    'elapsed_hours':(elapsed_prior+time.monotonic()-started)/3600,'amp_skipped_updates':skipped}),flush=True)
            if stopped:break
        score=None
        if world>1:dist.barrier()
        if args.split=='train' and rank==0 and not args.smoke:
            values=logits_for_documents(model,cache,cache['splits']['dev'],device,features)
            pred=decode(values,model.crf.arrays(),cache['labels']);truth={did:cache['docs'][did]['gold'] for did in pred}
            score=metric(pred,truth,cache['labels']);atomic_json(root/'dev_prediction.json',pred)
            print(json.dumps({'epoch':epoch+1,'dev_macro_f1':score['macro_f1']}),flush=True)
        if world>1:
            box=[score];dist.broadcast_object_list(box,src=0);score=box[0]
        improved=args.split=='all' or args.smoke or score['macro_f1']>best_f1+1e-8
        if improved:best_f1=score['macro_f1'] if score else -1.;best_epoch=epoch+1;bad=0
        else:bad+=1
        history.append({'epoch':epoch+1,'updates':updates,'metrics':score,'partial_epoch':cursor<len(loader)})
        partial=cursor<len(loader)
        if not partial:epoch+=1;cursor=0
        complete=stopped or epoch>=epochs or bad>=cfg.get('patience',2)
        if complete and not updates:raise FloatingPointError('No successful optimizer updates. Inspect FP16 skipped-update logs before generating submissions.')
        save(improved,complete)
        if complete:break
    if args.smoke and rank==0:
        print(json.dumps({'smoke_pass':True,'experiment':cfg['name'],'world_size':world,'device':str(device),'fp16':device.type=='cuda','successful_updates':updates}),flush=True)
    if world>1:dist.destroy_process_group()
if __name__=='__main__':main()
