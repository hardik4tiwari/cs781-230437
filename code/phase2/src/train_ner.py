"""Two-GPU DDP/FP16 training with strict-span selection and paired all-label refits."""
import argparse, contextlib, functools, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from ner_data import NERWindows,collate_windows,class_weights,digest,atomic_json,strict_metrics
from ner_models import build_model,ner_loss,parameter_report
from ner_predict import device_setup,autocast,predict_documents

def atomic_torch(path,obj):
    path=Path(path);temp=path.with_name(path.name+'.tmp');torch.save(obj,temp);temp.replace(path)

def rng_state(device):
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state(device) if device.type=='cuda' else None}

def restore_rng(state,device):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if device.type=='cuda':torch.cuda.set_rng_state(state['cuda'],device)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--cache',required=True)
    p.add_argument('--output',required=True);p.add_argument('--split',choices=['train','all'],default='train')
    p.add_argument('--epochs',type=int);p.add_argument('--teacher-dir');args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text());rank,world,device=device_setup();root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    seed=cfg.get('seed',230437);random.seed(seed+rank);np.random.seed(seed+rank);torch.manual_seed(seed+rank)
    cache=torch.load(args.cache,map_location='cpu',weights_only=False)
    if args.teacher_dir:
        manifest=json.loads((Path(args.teacher_dir)/'manifest.json').read_text())
        if not manifest['complete'] or manifest['identity']['split']!=args.split or manifest['identity']['cache']!=cache['fingerprint']:
            raise ValueError('KD cache is incomplete or uses the wrong split.')
        if manifest['identity']['windows']!=digest(cache[args.split+'_windows']):raise ValueError('Teacher window geometry changed.')
    if cfg['method']=='kd' and not args.teacher_dir:raise ValueError('KD experiment requires its own larger teacher cache.')
    dataset=NERWindows(cache,args.split,cfg['head'],args.teacher_dir)
    if not len(dataset):raise ValueError('Empty training split.')
    sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,shuffle=True,seed=seed,drop_last=False)
    generator=torch.Generator()
    loader=DataLoader(dataset,batch_size=cfg['micro_batch'],sampler=sampler,num_workers=cfg.get('workers',0),
        pin_memory=device.type=='cuda',collate_fn=collate_windows,generator=generator)
    model,provenance=build_model(cfg,cache['labels'],cache['char_classes'],cache['tokenizer_sha256']);model.to(device)
    report=parameter_report(model)
    if cfg['method']=='kd':
        teacher_report=cfg['teacher_parameters']
        if teacher_report<=report['total_parameters']:raise ValueError('Teacher must be larger than the student.')
    backbone=[];head=[]
    for name,param in model.named_parameters():
        if param.requires_grad:(backbone if name.startswith('backbone.') else head).append(param)
    groups=[{'params':backbone,'lr':cfg['backbone_lr'],'base_lr':cfg['backbone_lr']},
            {'params':head,'lr':cfg['head_lr'],'base_lr':cfg['head_lr']}]
    optimizer=torch.optim.AdamW(groups,weight_decay=cfg.get('weight_decay',.01),fused=device.type=='cuda')
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda');weights=class_weights(cache,args.split).to(device)
    epochs=args.epochs or cfg.get('epochs',12);accum=cfg.get('accumulation',4)
    updates_per_epoch=math.ceil(len(loader)/accum);planned=updates_per_epoch*epochs
    identity=digest({'cfg':cfg,'phase1_sha256':provenance['phase1_sha256'],'cache':cache['fingerprint'],'split':args.split,'epochs':epochs,'world':world,
        'teacher_cache':manifest['identity'] if args.teacher_dir else None})
    epoch=cursor=updates=skipped=bad_epochs=0;best_f1=-1.;best_epoch=0;history=[];elapsed_prior=0.
    latest=root/'latest.pt';best=root/'best.pt';finished=root/'summary.json'
    if finished.exists():
        saved_summary=json.loads(finished.read_text())
        if saved_summary['identity']!=identity:raise ValueError('Completed run configuration changed; use a new run name.')
        if saved_summary.get('complete') and best.exists():
            if rank==0:print('Already complete:',root,flush=True)
            if world>1:dist.destroy_process_group()
            return
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=False)
        if state['identity']!=identity:raise ValueError('Resume requires identical data, config, teacher, epochs and GPU count.')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        epoch=state['epoch'];cursor=state['cursor'];updates=state['updates'];skipped=state['amp_skipped_updates']
        best_f1=state['best_f1'];best_epoch=state['best_epoch'];bad_epochs=state['bad_epochs'];history=state['history'];elapsed_prior=state['elapsed_seconds']
        restore_rng(state['rng'][rank],device)
    wrapped=DistributedDataParallel(model,device_ids=[device.index] if device.type=='cuda' else None,broadcast_buffers=False) if world>1 else model
    started=time.monotonic();stopped=False
    def checkpoint(next_epoch,next_cursor,metrics=None,is_best=False,complete=False):
        random_states=[None]*world
        if world>1:dist.all_gather_object(random_states,rng_state(device))
        else:random_states[0]=rng_state(device)
        if rank==0:
            light={'model':model.state_dict(),'config':cfg,'labels':cache['labels'],'char_classes':cache['char_classes'],
                'cache_fingerprint':cache['fingerprint'],'provenance':provenance,'parameters':report,'metrics':metrics,
                'stage':'selection' if args.split=='train' else 'all_label_refit','selected_epoch':best_epoch,
                'training_origin':'own_phase1_adapted','epoch':next_epoch,'updates':updates}
            if is_best:atomic_torch(best,light)
            state={**light,'identity':identity,'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                'cursor':next_cursor,'rng':random_states,'amp_skipped_updates':skipped,'best_f1':best_f1,'best_epoch':best_epoch,
                'bad_epochs':bad_epochs,'history':history,'elapsed_seconds':elapsed_prior+time.monotonic()-started}
            atomic_torch(latest,state)
            summary={k:state[k] for k in ('identity','epoch','updates','amp_skipped_updates','best_f1','best_epoch','history','elapsed_seconds')}
            summary.update({'parameters':report,'provenance':provenance,'complete':complete,'stop_reason':('time_cap' if stopped else 'epochs_or_patience') if complete else 'running','stage':light['stage']})
            atomic_json(finished,summary)
        if world>1:dist.barrier()
    if rank==0:print(json.dumps({'experiment':cfg['name'],'stage':args.split,'windows':len(dataset),'parameters':report,'effective_batch':world*cfg['micro_batch']*accum}),flush=True)
    while epoch<epochs:
        sampler.set_epoch(epoch);generator.manual_seed(seed+epoch);iterator=iter(loader)
        for _ in range(cursor):next(iterator)
        model.train();loss_total=0.;batches=0;epoch_start=time.monotonic()
        while cursor<len(loader):
            micro=[]
            for _ in range(min(accum,len(loader)-cursor)):micro.append(next(iterator))
            optimizer.zero_grad(set_to_none=True)
            warmup=max(1,round(planned*.1));step=updates+1
            factor=step/warmup if step<=warmup else .1+.9*.5*(1+math.cos(math.pi*min(1,(step-warmup)/max(planned-warmup,1))))
            for group in optimizer.param_groups:group['lr']=group['base_lr']*factor
            denominator=sum(len(batch['items']) for batch in micro)
            for mi,batch in enumerate(micro):
                for key in ('ids','valid','tags','left','right','teacher'):
                    if batch[key] is not None:batch[key]=batch[key].to(device,non_blocking=True)
                sync=wrapped.no_sync() if world>1 and mi+1<len(micro) else contextlib.nullcontext()
                with sync:
                    with autocast(device):
                        out=wrapped(batch['ids'],batch['valid']);loss=ner_loss(out,batch,cfg,weights)
                        if not bool(torch.isfinite(loss)):raise FloatingPointError('Non-finite forward loss; inspect the saved log and FP16/model stability before continuing.')
                        scaled_loss=loss*len(batch['items'])/denominator
                    scaler.scale(scaled_loss).backward()
                loss_total+=float(loss.detach());batches+=1
            scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.get('grad_clip',1.))
            scale_before=scaler.get_scale();scaler.step(optimizer);scaler.update()
            if scaler.get_scale()<scale_before:skipped+=1
            else:updates+=1
            cursor+=len(micro)
            flag=torch.tensor(int(time.monotonic()-started>=cfg.get('max_stage_hours',1.)*3600),device=device)
            if world>1:dist.all_reduce(flag,op=dist.ReduceOp.MAX)
            if bool(flag):stopped=True
            if updates and updates%cfg.get('checkpoint_every',50)==0 or stopped:checkpoint(epoch,cursor)
            if rank==0 and updates%cfg.get('log_every',20)==0:
                print(json.dumps({'epoch':epoch+1,'batch':cursor,'updates':updates,'loss':loss_total/max(batches,1),
                    'gradient_norm':float(norm),'elapsed_hours':(elapsed_prior+time.monotonic()-started)/3600,'amp_skipped_updates':skipped}),flush=True)
            if stopped:break
        metrics=None
        if world>1:dist.barrier()
        if args.split=='train' and rank==0:
            ids=cache['split']['val_ids'];pred=predict_documents(model,cache,ids,cfg.get('inference_batch',cfg['micro_batch']),device)
            metrics=strict_metrics(pred,{k:cache['gold'][k] for k in ids},cache['labels']);atomic_json(root/'validation_prediction.json',pred)
            print(json.dumps({'epoch':epoch+1,'strict_macro_f1':metrics['macro_f1']}),flush=True)
        if world>1:
            box=[metrics];dist.broadcast_object_list(box,src=0);metrics=box[0]
        improved=args.split=='all' or metrics['macro_f1']>best_f1+1e-7
        if improved:
            best_f1=metrics['macro_f1'] if metrics else -1.;best_epoch=epoch+1;bad_epochs=0
        else:bad_epochs+=1
        history.append({'epoch':epoch+1,'updates':updates,'metrics':metrics,'partial_epoch':cursor<len(loader),
                        'epoch_seconds':time.monotonic()-epoch_start})
        epoch+=1;cursor=0
        complete=stopped or epoch>=epochs or (args.split=='train' and bad_epochs>=cfg.get('patience',3))
        checkpoint(epoch,cursor,metrics,improved,complete)
        if complete:break
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
