"""torchrun worker: genuine two-GPU DDP, resume, BERTScore selection and generation."""
import argparse,contextlib,gc,json,math,os,random,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler
from tokenizers import Tokenizer
from s6_data import atomic_json,read_json,file_hash,digest
from s6_model import load_parent,Summarizer,collate,generate,merged_adapters

def atomic_save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');torch.save(x,t);t.replace(p)
def initialize():
    world=int(os.environ.get('WORLD_SIZE',1));rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0))
    device=torch.device(f'cuda:{local}' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.cuda.set_device(device)
    if world>1:
        opts={'device_id':device} if device.type=='cuda' else {}
        dist.init_process_group('nccl' if device.type=='cuda' else 'gloo',**opts)
    return rank,world,device
def barrier(device):
    if dist.is_initialized():dist.barrier(device_ids=[device.index] if device.type=='cuda' else None)
def all_finite(value,device):
    flag=torch.isfinite(value.detach()).all().to(device=device,dtype=torch.int32)
    if dist.is_initialized():dist.all_reduce(flag,op=dist.ReduceOp.MIN)
    return bool(flag.item())
def finish_update(model,optimizer,scaler,params,device,stage):
    """Unscale once at the effective-batch boundary; all ranks step or skip together."""
    scaler.unscale_(optimizer)
    gradients=[p.grad for p in params if p.grad is not None]
    if not gradients:raise RuntimeError('No trainable gradients were produced.')
    norm=torch.nn.utils.get_total_norm(gradients,error_if_nonfinite=False)
    if not bool(torch.isfinite(norm)) and all(bool(torch.isfinite(g).all()) for g in gradients):
        # A float32 norm reduction itself can overflow even when every gradient element is finite.
        norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.detach().double()) for g in gradients]))
    if not all_finite(norm,device):
        bad=[n for n,p in model.named_parameters() if p.grad is not None and not bool(torch.isfinite(p.grad).all())][:8]
        previous=float(scaler.get_scale());new_scale=previous*.5
        print(json.dumps({'event':'amp_gradient_overflow','stage':stage,'rank':dist.get_rank() if dist.is_initialized() else 0,
                          'scale_before':previous,'scale_after':new_scale,'nonfinite_gradient_parameters':bad,
                          'optimizer_step':'skipped on all ranks; parameters unchanged'}),flush=True)
        if not scaler.is_enabled():raise FloatingPointError(f'{stage}: nonfinite unscaled gradients with AMP disabled: {bad}')
        # Reset per-optimizer scaler state without updating weights, even if only a peer rank overflowed.
        scaler.update(new_scale=new_scale);optimizer.zero_grad(set_to_none=True)
        return float('inf'),False
    torch.nn.utils.clip_grads_with_norm_(params,1.,norm)
    scaler.step(optimizer);scaler.update()
    return float(norm),True
def preflight_update(wrapped,model,optimizer,scaler,params,batch,device,amp,stage,max_attempts=12):
    for attempt in range(max_attempts):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):loss=wrapped(batch)
        if not all_finite(loss,device):
            print(json.dumps({'event':'nonfinite_forward_loss','stage':stage,'rank':dist.get_rank() if dist.is_initialized() else 0,
                              'local_loss':str(float(loss.detach())),'amp_scale':float(scaler.get_scale())}),flush=True)
            raise FloatingPointError(f'{stage}: forward loss is nonfinite; this is not a loss-scaler-only overflow.')
        scaler.scale(loss).backward()
        # Watch a matrix with a nonzero gradient, not an arbitrary softmax bias whose gradient can be exactly zero.
        watch=next((p for p in params if p.grad is not None and p.ndim>1 and bool(torch.isfinite(p.grad).all()) and bool(p.grad.ne(0).any())),None)
        before=watch.detach().clone() if watch is not None else None
        norm,stepped=finish_update(model,optimizer,scaler,params,device,stage)
        if stepped:
            if watch is None or torch.equal(before,watch.detach()):raise ValueError('Preflight optimizer produced no detectable trainable matrix update.')
            return {'loss':float(loss.detach()),'gradient_norm':norm,'overflow_retries':attempt,'amp_scale':float(scaler.get_scale())}
    raise FloatingPointError(f'{stage}: gradients stayed nonfinite after {max_attempts} attempts. Last AMP scale={scaler.get_scale()}; inspect the printed parameter diagnostics.')
def build(cfg,cache,device):
    parent=load_parent(cfg['parent'],cache['tokenizer_sha256'],cfg.get('allow_tiny',False))
    if parent.config.architecture!=cfg['architecture']:raise ValueError('Wrong original architecture family.')
    total=cache['source_tokens']+cache['target_tokens']+cfg.get('prompt_tokens',0)+1
    if not cfg['architecture']=='encoder_decoder' and total>parent.config.max_positions:raise ValueError(f'Source+target+prompt ({total}) exceeds original RoPE limit {parent.config.max_positions}.')
    if cfg['architecture']=='encoder_decoder' and max(cache['source_tokens'],cache['target_tokens'])>parent.config.max_positions:raise ValueError('Native encoder/decoder context exceeds Phase-1 position limit.')
    return Summarizer(parent,cfg).to(device)
def predict(model,rows,cache,device,decode,batch_size):
    with merged_adapters(model):return _predict(model,rows,cache,device,decode,batch_size)
def _predict(model,rows,cache,device,decode,batch_size):
    tok=Tokenizer.from_file(cache['tokenizer_path']);out=[];start=time.perf_counter()
    for a in range(0,len(rows),batch_size):
        local=rows[a:a+batch_size];tokens=generate(model,local,cache,device,decode)
        for r,t in zip(local,tokens):
            text=tok.decode(t,skip_special_tokens=True).strip()
            if not text and r['split']!='dev':raise ValueError(f'{r["id"]}: empty neural summary; inspect training and special tokens before submission.')
            out.append({'id':r['id'],'summary':text})
        if (a//batch_size+1)%25==0:print(json.dumps({'generated':min(len(rows),a+batch_size),'total':len(rows),'seconds':round(time.perf_counter()-start,1)}),flush=True)
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--action',choices=['train','preflight','generate'],default='train');ap.add_argument('--split',default='test');ap.add_argument('--output');ap.add_argument('--limit',type=int);ap.add_argument('--checkpoint',default='best.pt');ap.add_argument('--decode-file');args=ap.parse_args()
    cfg=read_json(args.config);cache=torch.load(cfg['cache'],map_location='cpu',weights_only=False);rank,world,device=initialize()
    random.seed(cfg['seed']+rank);np.random.seed(cfg['seed']+rank);torch.manual_seed(cfg['seed'])
    if device.type=='cuda':torch.backends.cudnn.benchmark=True
    run=Path(cfg['run']);run.mkdir(parents=True,exist_ok=True)
    lineage={'parent_sha256':file_hash(cfg['parent']),'tokenizer_sha256':cache['tokenizer_sha256'],'data_fingerprint':cache['fingerprint'],'config':cfg,'world_size':world,
             'code_sha256':{p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')}}
    if cfg['method']=='kd':lineage['kd_cache_sha256']=file_hash(cfg['kd_cache']) if Path(cfg['kd_cache']).is_file() else None
    signature=digest(lineage);model=build(cfg,cache,device);report=model.report()
    if args.action=='generate':
        saved=torch.load(run/args.checkpoint,map_location='cpu',weights_only=False)
        if saved['signature']!=signature:raise ValueError('Checkpoint provenance/config mismatch.')
        model.load_state_dict(saved['model'],strict=True)
        rows=sorted([r for r in cache['rows'] if r['split']==args.split],key=lambda r:digest([cfg['seed'],r['id']]) if args.split=='train' else r['id'])
        if args.limit:rows=rows[:args.limit]
        local=rows[rank::world];decode=read_json(args.decode_file) if args.decode_file else cfg['decode'];pred=predict(model,local,cache,device,decode,cfg['generation_batch'])
        output=Path(args.output);atomic_json(str(output)+f'.rank{rank}.json',pred);barrier(device)
        if rank==0:
            pred=sum([read_json(str(output)+f'.rank{i}.json') for i in range(world)],[]);byid={r['id']:r['summary'] for r in pred}
            if len(byid)!=len(rows) or set(byid)!={r['id'] for r in rows}:raise ValueError('Distributed generation IDs missing or duplicated.')
            atomic_json(output,[{'id':r['id'],'summary':byid[r['id']]} for r in rows])
        barrier(device)
        if dist.is_initialized():dist.destroy_process_group()
        return
    train=[r for r in cache['rows'] if r['split']=='train'];dev=sorted([r for r in cache['rows'] if r['split']=='dev'],key=lambda r:digest([cfg['seed'],r['id']]))[:cfg['dev_probe']]
    teacher={}
    if cfg['method']=='kd' and args.action=='train':
        kd=read_json(cfg['kd_cache']);allowed={r['id'] for r in train}
        if kd['data_fingerprint']!=cache['fingerprint'] or not set(kd['targets'])<=allowed:raise ValueError('KD cache must contain training IDs from this exact split.')
        if kd['teacher_checkpoint_sha256']!=file_hash(kd['teacher_checkpoint']):raise ValueError('KD teacher changed; rebuild KD cache.')
        teacher=kd['targets']
    wrapped=DDP(model,device_ids=[device.index] if device.type=='cuda' else None,broadcast_buffers=False) if world>1 else model
    params=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=cfg['weight_decay'],**({'fused':True} if device.type=='cuda' else {}))
    amp=device.type=='cuda';initial_scale=1024.
    if args.action=='train' and (run/'preflight.json').is_file():
        initial_scale=min(initial_scale,float(read_json(run/'preflight.json').get('amp_scale',initial_scale)))
    scaler=torch.amp.GradScaler(device.type,init_scale=initial_scale,enabled=amp)
    last=run/'last.pt';best=-float('inf');best_epoch=0;start_epoch=0;updates=0;stale=0;elapsed_base=0.;start=time.perf_counter()
    if args.action=='train' and last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        if saved['signature']!=signature:raise ValueError('Resume config/input differs; use a new PROJECT or experiment name.')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        start_epoch=saved['epoch'];updates=saved['updates'];best=saved['best'];best_epoch=saved['best_epoch'];stale=saved['stale'];elapsed_base=saved['elapsed_hours']
        state=saved['rng'][rank];random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
        if device.type=='cuda':torch.cuda.set_rng_state(state['cuda'],device)
    if args.action=='preflight':
        model.train();batch=collate([max(train,key=lambda r:len(r['target']))],cache,device)
        # Measure the full configured context and a real long target, not a reduced smoke-test sequence.
        timings=[];retry_count=0
        for i in range(3):
            if amp:torch.cuda.synchronize()
            t=time.perf_counter()
            update=preflight_update(wrapped,model,optimizer,scaler,params,batch,device,amp,f'full_context_{i+1}')
            retry_count+=update['overflow_retries']
            if amp:torch.cuda.synchronize()
            timings.append(time.perf_counter()-t)
        checksum=torch.stack([p.detach().float().sum() for p in params]).sum()
        if world>1:
            lo=checksum.clone();hi=checksum.clone();dist.all_reduce(lo,op=dist.ReduceOp.MIN);dist.all_reduce(hi,op=dist.ReduceOp.MAX)
            if not torch.allclose(lo,hi,rtol=1e-6,atol=1e-5):raise ValueError('DDP ranks diverged.')
        result={'world_size':world,'finite_update':True,'full_source_tokens':cache['source_tokens'],'target_tokens':int((batch['target']!=-100).sum()),'seconds_per_microbatch_per_rank':sum(timings[1:])/2,'peak_reserved_gib_per_rank':torch.cuda.max_memory_reserved()/1024**3 if amp else None,'parameters':report,'amp_scale':float(scaler.get_scale()),'overflow_retries':retry_count}
        if rank==0:atomic_json(run/'preflight.json',result);print(json.dumps(result),flush=True)
        barrier(device)
        if dist.is_initialized():dist.destroy_process_group()
        return
    sampler=DistributedSampler(train,num_replicas=world,rank=rank,shuffle=True,seed=cfg['seed'])
    loader=DataLoader(train,batch_size=cfg['micro_batch'],sampler=sampler,num_workers=0,collate_fn=lambda rows:rows)
    total_updates=max(1,math.ceil(len(loader)/cfg['accumulation'])*cfg['epochs']);skips=0;consecutive_skips=0
    if rank==0:print(json.dumps({'experiment':cfg['name'],'documents':len(train),'effective_batch':world*cfg['micro_batch']*cfg['accumulation'],'parameters':report}),flush=True)
    if stale>=cfg['patience']:start_epoch=cfg['epochs']
    for epoch in range(start_epoch,cfg['epochs']):
        sampler.set_epoch(epoch);model.train();optimizer.zero_grad(set_to_none=True);running=0
        for i,rows in enumerate(loader):
            # A deterministic Bernoulli choice gives E[L] = .5 gold NLL + .5 sequence-KD NLL on cached training cases.
            override={r['id']:teacher[r['id']] for r in rows if r['id'] in teacher and int(digest([cfg['seed'],epoch,r['id']])[:8],16)%2==0}
            batch=collate(rows,cache,device,override);group_start=(i//cfg['accumulation'])*cfg['accumulation'];divisor=min(cfg['accumulation'],len(loader)-group_start)
            sync=(i+1)%cfg['accumulation']==0 or i+1==len(loader);ctx=wrapped.no_sync() if world>1 and not sync else contextlib.nullcontext()
            with ctx:
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):loss=wrapped(batch)
                if not all_finite(loss,device):raise FloatingPointError(f'Nonfinite forward loss at epoch={epoch+1} batch={i+1}.')
                scaler.scale(loss/divisor).backward()
            running+=float(loss.detach())
            if sync:
                progress=updates/total_updates;warm=cfg['warmup_fraction'];factor=min(1.,(updates+1)/max(1,total_updates*warm)) if progress<warm else .1+.9*.5*(1+math.cos(math.pi*(progress-warm)/max(1e-8,1-warm)))
                for group in optimizer.param_groups:group['lr']=cfg['lr']*factor
                norm,stepped=finish_update(model,optimizer,scaler,params,device,f'epoch_{epoch+1}_batch_{i+1}')
                skips+=not stepped;consecutive_skips=0 if stepped else consecutive_skips+1
                optimizer.zero_grad(set_to_none=True);updates+=int(stepped)
                if consecutive_skips>=12:raise FloatingPointError('Persistent gradient overflow; inspect the logged parameters.')
                if rank==0 and stepped and updates%cfg['log_every']==0:print(json.dumps({'epoch':epoch+1,'batch':i+1,'updates':updates,'nll_smoothed':running/(i+1),'gradient_norm':float(norm),'lr':optimizer.param_groups[0]['lr'],'amp_skips':skips,'elapsed_hours':elapsed_base+(time.perf_counter()-start)/3600}),flush=True)
        # No references on test ever enter this branch. All ranks generate disjoint validation IDs.
        model.eval();local=predict(model,dev[rank::world],cache,device,cfg['decode'],cfg['generation_batch'])
        atomic_json(run/f'dev_epoch_{epoch+1}.rank{rank}.json',local);barrier(device)
        decision=[None]
        if rank==0:
            pred=sum([read_json(run/f'dev_epoch_{epoch+1}.rank{r}.json') for r in range(world)],[]);byid={r['id']:r['summary'] for r in pred}
            if len(byid)!=len(dev):raise ValueError('Incomplete validation predictions.')
            from s6_metrics import score,release
            metric=score([byid[r['id']] for r in dev],[r['summary'] for r in dev],device)
            release()
            value=metric['bertscore_f1'];improved=value>best+cfg['min_delta']
            if improved:best=value;best_epoch=epoch+1;stale=0
            else:stale+=1
            decision[0]={'best':best,'best_epoch':best_epoch,'stale':stale,'improved':improved,'metrics':metric}
            atomic_json(run/f'metrics_epoch_{epoch+1}.json',decision[0]);print(json.dumps({'epoch':epoch+1,**decision[0],'elapsed_hours':elapsed_base+(time.perf_counter()-start)/3600}),flush=True)
        if world>1:dist.broadcast_object_list(decision,src=0,device=device)
        d=decision[0];best=d['best'];best_epoch=d['best_epoch'];stale=d['stale']
        rng={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(device) if amp else None};rngs=[None]*world
        if world>1:dist.all_gather_object(rngs,rng)
        else:rngs=[rng]
        if rank==0:
            elapsed=elapsed_base+(time.perf_counter()-start)/3600
            saved={'model':model.state_dict(),'signature':signature,'lineage':lineage,'epoch':epoch+1,'updates':updates,'best':best,'best_epoch':best_epoch,'stale':stale,'elapsed_hours':elapsed,'parameters':report,'rng':rngs}
            if d['improved']:
                atomic_save(run/'best.pt',saved);atomic_json(run/'best_metrics.json',d)
            atomic_save(last,{**saved,'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict()})
        barrier(device)
        if stale>=cfg['patience']:break
    if dist.is_initialized():dist.destroy_process_group()
if __name__=='__main__':main()
