"""One-model Kaggle runtime: full-document gradient caching and synchronous two-GPU SGD."""
import argparse, contextlib, gc, json, math, os, random, shutil, time, zipfile
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from s5_data import prepare, Documents
from s5_model import FullDocumentModel, amp, encode_all, cached_backward
from lsi_data import read_json, atomic_json, atomic_torch, digest, file_hash, probabilities, metric, write_submission

def setup():
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    device=torch.device('cuda',local) if torch.cuda.is_available() else torch.device('cpu')
    if device.type=='cuda':torch.cuda.set_device(device)
    if world>1:
        if device.type=='cuda':dist.init_process_group('nccl',device_id=device)
        else:dist.init_process_group('gloo')
    torch.set_num_threads(1)
    return rank,world,device

def barrier():
    if dist.is_initialized():dist.barrier()

def build(cfg,checkpoint=None):
    if checkpoint:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved.get('training_origin')!='own_phase1_phase5_full_document':raise ValueError('Wrong model provenance.')
        model=FullDocumentModel(saved['foundation_config'],cfg);model.load_state_dict(saved['model'],strict=True)
        return model,saved['provenance']
    parent=torch.load(cfg['phase1_checkpoint'],map_location='cpu',weights_only=False)
    if parent.get('training_origin')!='random_initialization_official_phase1_only':raise ValueError('Use original Phase-1 weights.')
    if parent['tokenizer_sha256']!=cfg['tokenizer_sha256']:raise ValueError('Tokenizer and foundation do not match.')
    if parent['model_config']['architecture']!='decoder':raise ValueError('Expected main_decoder/best_model.pt.')
    model=FullDocumentModel(parent['model_config'],cfg);model.backbone.load_state_dict(parent['model'],strict=True)
    count=sum(p.numel() for p in model.backbone.parameters())
    if not cfg.get('_synthetic_test',False) and not 50_000_000<=count<=300_000_000:raise ValueError('Invalid foundation parameter count.')
    return model,{'phase1_sha256':file_hash(cfg['phase1_checkpoint']),'tokenizer_sha256':cfg['tokenizer_sha256'],
        'foundation_parameters':count,'original_phase1_only':True}

def sync_gradients(model,world):
    if world==1:return
    # Explicit averaging is used because a logical document backward replays many encoder forwards.
    # Do not wrap this model in DDP as well: doing so would average gradients twice.
    bucket=[];size=0
    def flush():
        if not bucket:return
        flat=torch.cat([p.grad.reshape(-1) for p in bucket]);dist.all_reduce(flat);flat.div_(world)
        a=0
        for p in bucket:
            n=p.numel();p.grad.copy_(flat[a:a+n].view_as(p));a+=n
        bucket.clear()
    for p in model.parameters():
        if not p.requires_grad:continue
        if p.grad is None:p.grad=torch.zeros_like(p)
        if size+p.numel()>4_000_000 and bucket:flush();size=0
        bucket.append(p);size+=p.numel()
    flush()

def rng_state(device):
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
        'cuda':torch.cuda.get_rng_state(device) if device.type=='cuda' else None}
def restore_rng(state,device):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if device.type=='cuda':torch.cuda.set_rng_state(state['cuda'],device)

def save(path,model,cfg,provenance,optimizer=None,scaler=None,cursor=None,state=None,rank=0,world=1,device=None):
    random_states=None
    if optimizer is not None:
        current=rng_state(device);random_states=[None]*world
        if world>1:dist.all_gather_object(random_states,current)
        else:random_states=[current]
    if rank==0:
        result={'training_origin':'own_phase1_phase5_full_document','config':cfg,'foundation_config':vars(model.backbone.config),
            'provenance':provenance,'model':model.state_dict()}
        if optimizer is not None:result.update(optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),cursor=cursor,state=state,rng=random_states)
        atomic_torch(path,result)
    barrier()

def epoch_batches(data,cfg,epoch,rank,world):
    rows=list(data['splits']['train']);random.Random(cfg['seed']+epoch).shuffle(rows)
    # Local pools reduce padding in the document head. Chunks themselves are microbatched independently.
    for a in range(0,len(rows),128):rows[a:a+128]=sorted(rows[a:a+128],key=lambda i:len(data['docs'][i]['chunks']))
    quantum=world*cfg['batch_docs'];padding=(-len(rows))%quantum
    rows+=([rows[i%len(rows)] for i in range(padding)])
    local=rows[rank::world]
    return [local[a:a+cfg['batch_docs']] for a in range(0,len(local),cfg['batch_docs'])]

@torch.no_grad()
def predict(model,ds,split,cfg,rank,world,device,out):
    model.eval();indices=ds.data['splits'][split][rank::world];predictions=[]
    for a in range(0,len(indices),cfg['eval_batch_docs']):
        batch=ds.batch(indices[a:a+cfg['eval_batch_docs']],device)
        with amp(device):
            f=encode_all(model,batch,device,cfg['chunk_batch']);z,_=model.classify(f,batch)
        predictions.append(z.float().cpu().numpy())
    z=np.concatenate(predictions) if predictions else np.zeros((0,cfg['label_count']),np.float32)
    local=Path(out)/f'{split}_rank{rank}.npz';np.savez(local,indices=indices,logits=z);barrier()
    result=None
    if rank==0:
        byid={}
        for r in range(world):
            with np.load(Path(out)/f'{split}_rank{r}.npz') as saved:
                byid.update(zip(saved['indices'].tolist(),saved['logits']))
        expected=ds.data['splits'][split]
        if set(byid)!=set(expected):raise ValueError('Distributed inference missed/duplicated examples.')
        result=np.stack([byid[i] for i in expected]);np.save(Path(out)/f'{split}_logits.npy',result)
    barrier();return result

def targets(ds,split):
    values=[]
    for i in ds.data['splits'][split]:
        d=ds.data['docs'][i]
        if d['gold'] is None:raise ValueError('Cannot request test targets.')
        y=np.zeros(ds.data['label_count'],np.uint8);y[d['gold']]=1;values.append(y)
    return np.stack(values)

def thresholds(y,p,support,shrink=25.):
    """Exact per-label F1 cutoffs (tie-aware), shrunk in logit space toward one global cutoff."""
    C=y.shape[1];active=np.asarray(support)>0
    grid=np.r_[np.linspace(.01,.2,20),np.linspace(.225,.8,24)]
    t=float(max(grid,key=lambda value:metric(y,p,value)['macro_f1']))
    result=np.full(C,t,dtype=np.float64)
    logit=lambda x:np.log(np.clip(x,1e-5,1-1e-5)/np.clip(1-x,1e-5,1-1e-5))
    for j in range(C):
        positives=y[:,j].sum()
        if not active[j]:result[j]=1.;continue
        if positives<5:continue
        order=np.argsort(-p[:,j],kind='stable');sortedp=p[order,j]
        true=np.cumsum(y[order,j]);k=np.arange(1,len(order)+1)
        f=2*true/(k+positives);ends=np.r_[sortedp[:-1]!=sortedp[1:],True]
        candidates=np.flatnonzero(ends);best=candidates[np.argmax(f[candidates])]
        cut=float(np.clip(sortedp[best],1e-5,1-1e-5));strength=float(positives/(positives+shrink))
        result[j]=float(probabilities((1-strength)*logit(t)+strength*logit(cut)))
    return result

def monitor(y,p,support,seed):
    order=np.random.default_rng(seed).permutation(len(y));middle=len(order)//2
    calibration,selection=order[:middle],order[middle:]
    if not len(calibration) or not len(selection):raise ValueError('Need at least two development cases.')
    t=thresholds(y[calibration],p[calibration],support)
    return metric(y[selection],p[selection],t),t

def train(cfg,manifest,out,rank,world,device,smoke=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);ds=Documents(manifest)
    if ds.data['fingerprint']!=cfg['data_fingerprint']:raise ValueError('Data fingerprint changed.')
    torch.manual_seed(cfg['seed']);model,provenance=build(cfg);model.to(device)
    random.seed(cfg['seed']+rank);np.random.seed(cfg['seed']+rank);torch.manual_seed(cfg['seed']+rank)
    encoder=[];head=[];lexical=[]
    for name,p in model.named_parameters():
        (encoder if name.startswith('backbone.') else lexical if name.startswith('lexical.') else head).append(p)
    optimizer=torch.optim.AdamW([{'params':encoder,'lr':cfg['encoder_lr'],'base_lr':cfg['encoder_lr'],'weight_decay':.01},
        {'params':head,'lr':cfg['head_lr'],'base_lr':cfg['head_lr'],'weight_decay':.01},
        {'params':lexical,'lr':cfg['lexical_lr'],'base_lr':cfg['lexical_lr'],'weight_decay':.05}],
        fused=device.type=='cuda')
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda',init_scale=1024.)
    support=np.asarray(ds.data['support']);N=len(ds.data['splits']['train']);w=np.sqrt((N-support)/np.maximum(support,1)).clip(1,8)
    normalizer=1+(support/N)*(w-1)
    weights=torch.as_tensor(w,dtype=torch.float32,device=device);norm=torch.as_tensor(normalizer,dtype=torch.float32,device=device)
    identity=digest({'config':cfg,'world':world,'parent':provenance['phase1_sha256']})
    state={'identity':identity,'best_score':-1.,'best_epoch':0,'bad_epochs':0,'updates':0,'history':[]}
    epoch0=batch0=0;latest=out/'latest.pt'
    if latest.exists() and not smoke:
        old=torch.load(latest,map_location='cpu',weights_only=False)
        if old['state']['identity']!=identity:raise ValueError('Resume settings/world/parent changed. Use a new PROJECT.')
        model.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer']);scaler.load_state_dict(old['scaler'])
        epoch0,batch0=old['cursor'];state=old['state'];restore_rng(old['rng'][rank],device);del old
        if rank==0:print('Resuming at epoch, local batch:',epoch0+1,batch0,flush=True)
    if (out/'complete.json').exists() and not smoke:
        if rank==0:print('Training already complete.',flush=True)
        return
    batches_per_epoch=math.ceil(math.ceil(N/(world*cfg['batch_docs']))/cfg['accumulation'])
    total_updates=cfg['epochs']*batches_per_epoch;start=time.monotonic();optimizer.zero_grad(set_to_none=True)
    for epoch in range(epoch0,cfg['epochs']):
        model.train();model.backbone.requires_grad_(epoch>=cfg['freeze_epochs'] or smoke)
        batches=epoch_batches(ds.data,cfg,epoch,rank,world)
        stop=1 if smoke else len(batches)
        for b in range(batch0,stop):
            group_start=(b//cfg['accumulation'])*cfg['accumulation']
            divisor=min(cfg['accumulation'],stop-group_start)
            batch=ds.batch(batches[b],device)
            if smoke:
                # Stress the configured encoder microbatch using long real TRAIN cases.
                # Bounding this preflight only does not truncate actual training or inference.
                chosen=sorted(ds.data['splits']['train'],key=lambda i:len(ds.data['docs'][i]['chunks']),reverse=True)[:cfg['batch_docs']]
                batch=ds.batch(chosen,device);keep=[];counts=[];cursor=0
                for n in batch['counts']:
                    take=min(n,cfg['chunk_batch']);keep.extend(range(cursor,cursor+take));counts.append(take);cursor+=n
                batch['chunks']=[batch['chunks'][i] for i in keep]
                batch['position']=batch['position'][keep];batch['counts']=counts
            loss=cached_backward(model,batch,device,scaler,weights,norm,divisor)
            if (b+1)%cfg['accumulation']==0 or b+1==stop:
                warm=max(1,int(.06*total_updates));u=state['updates']
                factor=min(1.,(u+1)/warm)*(.15+.85*.5*(1+math.cos(math.pi*min(1,u/max(1,total_updates)))))
                for group in optimizer.param_groups:group['lr']=group['base_lr']*factor
                sync_gradients(model,world);scaler.unscale_(optimizer)
                g=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                previous_scale=scaler.get_scale();scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
                state['updates']+=1
                if rank==0 and (state['updates']%cfg['log_every']==0 or smoke):
                    print(json.dumps({'epoch':epoch+1,'batch':b+1,'updates':state['updates'],'loss':loss,
                        'gradient_norm':float(g),'amp_skipped':scaler.get_scale()<previous_scale,
                        'elapsed_hours':(time.monotonic()-start)/3600}),flush=True)
                if smoke:
                    probe=torch.stack([p.detach().float().sum() for p in model.parameters()])
                    if world>1:
                        lo=probe.clone();hi=probe.clone();dist.all_reduce(lo,op=dist.ReduceOp.MIN);dist.all_reduce(hi,op=dist.ReduceOp.MAX)
                        if not torch.allclose(lo,hi,atol=1e-5,rtol=1e-5):raise ValueError('GPU ranks diverged.')
                    if not torch.isfinite(probe).all() or not math.isfinite(float(g)):raise ValueError('Nonfinite smoke update.')
                    if rank==0:atomic_json(out/'smoke_pass.json',{'world_size':world,'device':str(device),'finite_update':True,
                        'encoder_microbatch':cfg['chunk_batch'],'preflight_chunks_per_rank':len(batch['chunks']),
                        'preflight_only_context_bound':True,
                        'peak_reserved_gib':torch.cuda.max_memory_reserved(device)/1024**3 if device.type=='cuda' else None})
                    barrier();return
                if state['updates']%cfg['checkpoint_every']==0:
                    save(latest,model,cfg,provenance,optimizer,scaler,(epoch,b+1),state,rank,world,device)
        batch0=0
        logits=predict(model,ds,'dev',cfg,rank,world,device,out)
        if rank==0:
            y=targets(ds,'dev');p=probabilities(logits);p[:,support==0]=0
            scores,_=monitor(y,p,support,cfg['seed']);score=scores['macro_f1']
            # A warm-up-only checkpoint is not eligible as this is a full-fine-tuning candidate.
            eligible=epoch>=cfg['freeze_epochs']
            improved=eligible and score>state['best_score']+1e-6
            state['history'].append({'epoch':epoch+1,'selection_half_metrics':scores})
            if improved:state.update(best_score=score,best_epoch=epoch+1,bad_epochs=0)
            elif eligible:state['bad_epochs']+=1
            atomic_json(out/'history.json',state['history'])
            print(json.dumps({'epoch':epoch+1,'held_out_dev_macro_f1':score,'best_epoch':state['best_epoch'],
                'elapsed_hours':(time.monotonic()-start)/3600}),flush=True)
        else:improved=None
        if world>1:
            objects=[state,improved];dist.broadcast_object_list(objects,src=0);state,improved=objects
        if improved:save(out/'best_model.pt',model,cfg,provenance,rank=rank,world=world)
        save(latest,model,cfg,provenance,optimizer,scaler,(epoch+1,0),state,rank,world,device)
        if epoch+1>=cfg['min_epochs'] and state['bad_epochs']>=cfg['patience']:break
    if rank==0:atomic_json(out/'complete.json',{'best_epoch':state['best_epoch'],'selection_half_macro_f1':state['best_score'],
        'wall_hours_this_invocation':(time.monotonic()-start)/3600,'world_size':world})
    barrier()

def export(cfg,manifest,out,rank,world,device):
    out=Path(out);ds=Documents(manifest);model,provenance=build(cfg,out/'best_model.pt');model.to(device)
    z=predict(model,ds,'dev',cfg,rank,world,device,out)
    if rank==0:
        y=targets(ds,'dev');p=probabilities(z);support=np.asarray(ds.data['support']);p[:,support==0]=0
        threshold=thresholds(y,p,support)
        atomic_json(out/'calibration.json',{'threshold':threshold.tolist(),'dev_metrics_after_calibration':metric(y,p,threshold),
            'note':'Final thresholds use all dev after checkpoint selection. Reported dev score is fitted, not a test estimate.',
            'trainable_label_ids':np.flatnonzero(support>0).tolist()})
        rows=[{'id':ds.data['docs'][i]['id'],'gold':ds.data['docs'][i]['gold']} for i in ds.data['splits']['dev']]
        atomic_json(out/'dev_truth.json',rows)
        n=max(j for r in rows for j in r['gold'])+1
        labels=[[int(j) for j in np.flatnonzero(row>=threshold) if support[j]>0 and j<n] for row in p]
        write_submission([r['id'] for r in rows],labels,out/'dev_submission.zip')
        with zipfile.ZipFile(out/'dev_submission.zip') as archive:(out/'dev_prediction.json').write_bytes(archive.read('prediction.json'))
    barrier()
    # All fitting decisions and weights are now frozen; test has target=None.
    z=predict(model,ds,'test',cfg,rank,world,device,out)
    if rank==0:
        cal=read_json(out/'calibration.json');p=probabilities(z);allowed=set(cal['trainable_label_ids'])
        labels=[[int(j) for j in np.flatnonzero(row>=np.asarray(cal['threshold'])) if j in allowed] for row in p]
        ids=[ds.data['docs'][i]['id'] for i in ds.data['splits']['test']]
        destination=Path(cfg['project']).parent/'phase5_one_model_submission.zip'
        rows,sha=write_submission(ids,labels,destination)
        atomic_json(Path(cfg['project'])/'prediction.json',rows)
        atomic_json(out/'submission_audit.json',{'test_count':len(ids),'prediction_sha256':sha,'submission':str(destination),
            'one_model':True,'test_labels_used':False,'full_document_at_train_and_test':True,'provenance':provenance})
        print('FINAL SUBMISSION:',destination,flush=True)
    barrier()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--action',choices=['prepare','smoke','train','export'],required=True)
    args=parser.parse_args();cfg=read_json(args.config);project=Path(cfg['project'])
    if args.action=='prepare':
        path=prepare(cfg['data_root'],cfg['tokenizer'],project/'cache',cfg);data=read_json(path)
        N=len(data['splits']['train']);support=np.asarray(data['support']);w=np.sqrt((N-support)/np.maximum(support,1)).clip(1,8)
        prevalence=(support+1)/(N+2);bias=np.log(prevalence*w/(1-prevalence))
        cfg.update(data_fingerprint=data['fingerprint'],label_count=data['label_count'],sparse_dim=data['sparse_dim'],
            initial_bias=bias.tolist(),tokenizer_sha256=data['tokenizer_sha256'])
        atomic_json(args.config,cfg);return
    rank,world,device=setup();out=project/'run';out.mkdir(parents=True,exist_ok=True)
    manifest=project/'cache/manifest.json'
    if args.action in ('train','smoke'):train(cfg,manifest,out if args.action=='train' else project/'preflight',rank,world,device,args.action=='smoke')
    else:export(cfg,manifest,out,rank,world,device)
    if dist.is_initialized():dist.destroy_process_group()
if __name__=='__main__':main()

