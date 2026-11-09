"""DDP worker for bounded training and validation by teacher-forced dev loss."""
import argparse,contextlib,json,math,random,time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from fast_model import build,collation,predict
from s6_data import atomic_json,digest,file_hash,read_json
from s6_run import all_finite,atomic_save,barrier,finish_update,initialize

def epoch_batches(rows,cfg,epoch,rank,world):
    """Shuffle globally, rotate through the pool, then bucket short ranges by length."""
    base=random.Random(cfg['seed']).sample(range(len(rows)),len(rows))
    stride=cfg['micro_batch']*world
    per_epoch=cfg['steps_per_epoch']*cfg['accumulation']*stride
    start=(epoch*per_epoch)%len(base)
    order=(base[start:]+base[:start])[:min(len(base),per_epoch)]
    random.Random(cfg['seed']+epoch+1).shuffle(order)
    bucket=16*stride
    local=[]
    for j in range(0,len(order),bucket):
        nearby=sorted(order[j:j+bucket],key=lambda i:len(rows[i]['source'])+len(rows[i]['target']))
        for z in range(0,len(nearby),stride):
            chosen=nearby[z:z+stride]
            n=len(chosen)
            chosen += [chosen[k%n] for k in range(stride-n)]
            off=rank*cfg['micro_batch']
            local.append([rows[i] for i in chosen[off:off+cfg['micro_batch']]])
    random.Random(cfg['seed']+epoch+191).shuffle(local)
    return local

def autocast(device):
    return torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda')

@torch.inference_mode()
def dev_loss(model,rows,cache,cfg,device,rank,world):
    model.eval()
    result=torch.zeros(2,device=device,dtype=torch.float64)
    rows=sorted(rows[rank::world],key=lambda r:len(r['source'])+len(r['target']))
    for i in range(0,len(rows),cfg['micro_batch']):
        batch=collation(rows[i:i+cfg['micro_batch']],cache,device,native=model.native)
        with autocast(device):value=model(batch)
        result[0]+=value.detach().double()*batch['token_count']
        result[1]+=batch['token_count']
    if world>1:dist.all_reduce(result)
    value=float(result[0]/result[1].clamp_min(1))
    if not math.isfinite(value):raise FloatingPointError('Nonfinite teacher-forced development loss.')
    return value

def lineage(cfg,cache,world):
    result={'configuration':cfg,'cache_fingerprint':cache['fingerprint'],
            'tokenizer_sha256':cache['tokenizer_sha256'],'parent_sha256':file_hash(cfg['parent']),
            'world_size':world,'code_sha256':{p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')}}
    if cfg.get('teacher_config'):
        tc=read_json(cfg['teacher_config'])
        result['teacher_config_sha256']=file_hash(cfg['teacher_config'])
        result['teacher_weights_sha256']=file_hash(Path(tc['run'])/'best.pt')
    if cfg.get('warm_checkpoint'):
        result['warm_weights_sha256']=file_hash(cfg['warm_checkpoint'])
        result['warm_cache_sha256']=file_hash(cfg['warm_cache'])
    return result

def warm_start(model,cfg,cache,own):
    path=cfg.get('warm_checkpoint')
    if not path:return
    old=torch.load(path,map_location='cpu',weights_only=False)
    oldmeta=old.get('lineage',{})
    oldcfg=oldmeta.get('config',{})
    if cfg['name']!='04_hybrid_lora' or any(oldcfg.get(k)!=cfg[k] for k in ('architecture','method','rank','alpha')):
        raise ValueError('Warm start requires the original Phase 6 04_hybrid_lora checkpoint (rank 16, alpha 32).')
    if oldmeta.get('parent_sha256')!=own['parent_sha256'] or oldmeta.get('tokenizer_sha256')!=cache['tokenizer_sha256']:
        raise ValueError('Warm checkpoint Phase 1 parent or tokenizer does not match.')
    oldcache=torch.load(cfg['warm_cache'],map_location='cpu',weights_only=False)
    if oldcache.get('fingerprint')!=oldmeta.get('data_fingerprint'):
        raise ValueError('Warm checkpoint and original v4 token cache do not match.')
    old_train={r['id'] for r in oldcache['rows'] if r['split']=='train'}
    old_train_text={r.get('document_sha256') for r in oldcache['rows'] if r['split']=='train'}
    if any(r['id'] in old_train or r['document_sha256'] in old_train_text for r in cache['rows'] if r['split']!='train'):
        raise ValueError('Warm checkpoint previously trained on a current dev or test document.')
    model.load_state_dict(old['model'],strict=True)

def rng_snapshot(device):
    return {'py':random.getstate(),'np':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state(device) if device.type=='cuda' else None}
def restore_rng(x,device):
    random.setstate(x['py']);np.random.set_state(x['np']);torch.set_rng_state(x['torch'])
    if device.type=='cuda':torch.cuda.set_rng_state(x['cuda'],device)

def train(model,cfg,cache,rank,world,device,sig,own):
    run=Path(cfg['run']);last=run/'last.pt';done=run/'train_complete.json'
    if done.exists():
        old=read_json(done)
        if old['signature']!=sig or old['checkpoint_sha256']!=file_hash(run/'best.pt'):raise ValueError('Completed training input changed. Use a new PROJECT.')
        if rank==0:print('Reusing verified trained checkpoint:',cfg['name'],flush=True)
        return
    train_rows=[r for r in cache['rows'] if r['split']=='train']
    dev=sorted([r for r in cache['rows'] if r['split']=='dev'],key=lambda r:digest([cfg['seed'],r['id']]))[:cfg['dev_loss_cases']]
    if not train_rows or not dev:raise ValueError('Missing training or held-out records.')
    teacher=None
    if cfg.get('teacher_config'):
        tc=read_json(cfg['teacher_config'])
        weights=torch.load(Path(tc['run'])/'best.pt',map_location='cpu',weights_only=False)
        if weights['signature']!=digest(lineage(tc,cache,world)):raise ValueError('Teacher configuration or lineage mismatch.')
        teacher=build(tc,cache,device);teacher.load_state_dict(weights['model'],strict=True);teacher.requires_grad_(False);teacher.eval()
        del weights
    if cfg.get('warm_checkpoint') and not last.exists():
        warm_start(model,cfg,cache,own)
        if rank==0:print('Loaded previous Model 04 weights; using a fresh optimizer and budget.',flush=True)
    params=[p for p in model.parameters() if p.requires_grad]
    wrapped=DDP(model,device_ids=[device.index] if device.type=='cuda' else None,broadcast_buffers=False,
                gradient_as_bucket_view=True) if world>1 else model
    optimizer=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=0.01,
                                **({'fused':True} if device.type=='cuda' else {}))
    scaler=torch.amp.GradScaler(device.type,init_scale=1024.,enabled=device.type=='cuda')
    updates=0;best=float('inf');best_epoch=0;begin=0;stale=0;seconds=0.;stop=False;skips=0
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        if saved['signature']!=sig:raise ValueError('Resume inputs differ. Use a new PROJECT.')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        best=saved['best'];best_epoch=saved['best_epoch'];begin=saved['epoch'];updates=saved['updates'];stale=saved['stale'];seconds=saved['train_seconds'];stop=saved['stop']
        restore_rng(saved['rng'][rank],device);del saved
    num_micro=len(epoch_batches(train_rows,cfg,0,rank,world))
    per_epoch=math.ceil(num_micro/cfg['accumulation']);total=per_epoch*cfg['epochs']
    if rank==0:print(json.dumps({'run':cfg['name'],'max_optimizer_updates':total,'training_pool':len(train_rows),
                                 'max_training_minutes':cfg['max_train_minutes'],'parameters':model.report()}),flush=True)
    for epoch in range(begin,cfg['epochs']):
        if stop:break
        model.train()
        batches=epoch_batches(train_rows,cfg,epoch,rank,world)
        start=time.perf_counter();running=torch.zeros((),device=device);attempts=0
        optimizer.zero_grad(set_to_none=True)
        for i,rows in enumerate(batches):
            batch=collation(rows,cache,device,native=model.native)
            sync=(i+1)%cfg['accumulation']==0 or i+1==len(batches)
            divisor=min(cfg['accumulation'],len(batches)-(i//cfg['accumulation'])*cfg['accumulation'])
            context=wrapped.no_sync() if world>1 and not sync else contextlib.nullcontext()
            with context:
                with autocast(device):
                    # Teacher uses the same gold-prefix positions, never generates long rollouts.
                    do_kd=teacher is not None and (epoch*len(batches)+i)%cfg['kd_every']==0
                    with torch.no_grad():
                        th=teacher.target_states(batch) if do_kd else None
                    loss=wrapped(batch,th,teacher.backbone.embedding.weight if do_kd else None)
                    scaled=loss/divisor
                scaler.scale(scaled).backward()
            running+=loss.detach()
            if not sync:continue
            attempts+=1
            warm=max(1,math.ceil(total*.06))
            progress=max(0,updates-warm)/max(1,total-warm)
            factor=(updates+1)/warm if updates<warm else .15+.85*.5*(1+math.cos(math.pi*progress))
            for group in optimizer.param_groups:group['lr']=cfg['lr']*factor
            _,stepped=finish_update(model,optimizer,scaler,params,device,f'epoch{epoch+1}-batch{i+1}')
            optimizer.zero_grad(set_to_none=True)
            updates+=int(stepped);skips=0 if stepped else skips+1
            if skips>=12:raise FloatingPointError('Repeated FP16 overflow. Inspect the logged parameters.')
            spent=seconds+time.perf_counter()-start
            if rank==0 and (attempts in (1,10) or attempts%25==0):
                eta=(time.perf_counter()-start)/attempts*max(0,total-updates)/60
                if cfg['max_train_minutes']:eta=min(eta,max(0,cfg['max_train_minutes']-spent/60))
                print(json.dumps({'epoch':epoch+1,'updates':updates,'running_training_loss':float(running/(i+1)),
                                  'train_minutes_elapsed':round(spent/60,2),'remaining_train_minutes_estimate':round(eta,2)}),flush=True)
            timed=torch.tensor([int(rank==0 and bool(updates) and bool(cfg['max_train_minutes']) and spent>=60*cfg['max_train_minutes'])],device=device)
            if world>1 and cfg['max_train_minutes']:dist.broadcast(timed,src=0)
            stop=bool(timed.item())
            if stop:break
        seconds+=time.perf_counter()-start
        if updates==0:raise FloatingPointError('No finite optimizer update completed; the model was not trained.')
        value=dev_loss(model,dev,cache,cfg,device,rank,world)
        improved=value<best-1e-5
        if improved:best=value;best_epoch=epoch+1;stale=0
        else:stale+=1
        stop=stop or stale>=cfg['patience']
        rng=rng_snapshot(device);rngs=[None]*world
        if world>1:dist.all_gather_object(rngs,rng)
        else:rngs=[rng]
        if rank==0:
            saved={'model':model.state_dict(),'signature':sig,'lineage':own,'epoch':epoch+1,'best_epoch':best_epoch,
                   'updates':updates,'best':best,'train_seconds':seconds,'stale':stale,'stop':stop,'rng':rngs}
            if improved:atomic_save(run/'best.pt',saved)
            atomic_save(last,{**saved,'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict()})
            print(json.dumps({'epoch':epoch+1,'dev_token_loss':value,'best_epoch':best_epoch,'minutes_training_only':round(seconds/60,2),
                              'training_budget_reached':bool(timed.item())}),flush=True)
        barrier(device)
    if rank==0:
        atomic_json(done,{'signature':sig,'best_epoch':best_epoch,'best_dev_loss':best,'training_seconds':seconds,
                          'optimizer_updates':updates,'checkpoint_sha256':file_hash(run/'best.pt'),'parameters':model.report()})
    barrier(device)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True);ap.add_argument('--action',choices=('train','generate'),required=True)
    ap.add_argument('--split',choices=('dev','test'),default='test');ap.add_argument('--output');ap.add_argument('--limit',type=int)
    args=ap.parse_args()
    torch.set_num_threads(2)
    cfg=read_json(args.config);cache=torch.load(cfg['cache'],map_location='cpu',weights_only=False)
    rank,world,device=initialize();random.seed(cfg['seed']+rank);np.random.seed(cfg['seed']+rank);torch.manual_seed(cfg['seed'])
    if device.type=='cuda':torch.backends.cudnn.benchmark=False
    run=Path(cfg['run']);run.mkdir(parents=True,exist_ok=True)
    own=lineage(cfg,cache,world);sig=digest(own);model=build(cfg,cache,device)
    if args.action=='train':
        train(model,cfg,cache,rank,world,device,sig,own)
    else:
        if not args.output:raise ValueError('--output is required for generation.')
        if args.split=='test' and args.limit is not None:raise ValueError('Test generation must cover every official ID.')
        done=read_json(run/'train_complete.json')
        if done['signature']!=sig or done['checkpoint_sha256']!=file_hash(run/'best.pt'):raise ValueError('Checkpoint provenance mismatch.')
        ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(ck['model'],strict=True);del ck
        rows=sorted([r for r in cache['rows'] if r['split']==args.split],key=lambda r:digest([cfg['seed'],r['id']]))
        if args.limit is not None:rows=rows[:args.limit]
        local=predict(model,rows[rank::world],cache,device,cfg['decode'],cfg['generation_batch'])
        atomic_json(args.output+f'.rank{rank}.json',local);barrier(device)
        if rank==0:
            combined=[item for i in range(world) for item in read_json(args.output+f'.rank{i}.json')]
            index={r['id']:r for r in combined}
            if len(rows)!=len(combined) or set(index)!={r['id'] for r in rows}:raise ValueError('Incomplete or duplicate generation IDs.')
            atomic_json(args.output,[index[r['id']] for r in rows])
            for i in range(world):Path(args.output+f'.rank{i}.json').unlink()
        barrier(device)
    if dist.is_initialized():dist.destroy_process_group()

if __name__=='__main__':main()
