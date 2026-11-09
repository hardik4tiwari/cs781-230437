"""Two-GPU DDP training and split-specific retrieval workers."""
import argparse,contextlib,json,math,os,random,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler,DataLoader
from p7_data import Store,read_json,atomic_json,file_hash,digest
from p7_model import Retriever,load_parent,encode_batch,pairs_batch
from p7_eval import rank,select_fusion,query_ids,teacher_scores

def save(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp');torch.save(x,tmp);tmp.replace(path)
def initialize():
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0));device=torch.device(f'cuda:{local}' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.cuda.set_device(device)
    if world>1:dist.init_process_group('nccl' if device.type=='cuda' else 'gloo',**({'device_id':device} if device.type=='cuda' else {}))
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
def build(cfg,store,device):
    parent=load_parent(cfg['parent'],store.manifest['tokenizer_sha256'],cfg.get('allow_tiny',False))
    if parent.config.architecture!=cfg['architecture']:raise ValueError('Parent architecture does not match experiment.')
    if max(cfg['chunk_tokens']+cfg.get('prompt_tokens',0),cfg['pair_tokens'])>parent.config.max_positions:raise ValueError('Retrieval sequence exceeds original position limit.')
    return Retriever(parent,cfg).to(device)
def base_rankings(cfg,split):
    if not cfg.get('base_run'):return None
    path=Path(cfg['base_run'])/f'{split}_rankings.json'
    if not path.is_file():raise FileNotFoundError(f'Complete run 01 before cross-encoder ranking: {path}')
    return read_json(path)
def training_queries(store):
    pool=store.splits['train']['candidates'];return [q for q,r in store.splits['train']['queries'].items() if r['relevant'] and len(set(r['relevant']))<len(pool)]
def samples(store,qids,cfg,epoch,hard=None,kd=None):
    result=[];pool=store.train_candidate_ids;n=cfg['negatives']
    for qid in qids:
        row=store.splits['train']['queries'][qid];gold=set(row['relevant']);rng=random.Random(int(digest([cfg['seed'],epoch,qid])[:16],16));teacher=kd.get(qid) if kd else None
        positives=sorted(gold & set(teacher['ids'])) if teacher else sorted(gold)
        if not positives:raise ValueError(f'{qid}: no teacher-supervised positive.')
        positive=positives[epoch%len(positives)];neg_pool=([c for c in teacher['ids'] if c not in gold] if teacher else [c for c in (hard or {}).get(qid,store.lexical['train'][qid]['ids']) if c not in gold])
        if not neg_pool:neg_pool=[c for c in pool if c not in gold]
        negatives=rng.sample(neg_pool,min(n,len(neg_pool)))
        # Mix one random negative to reduce overfitting to a single static hard-negative list.
        if not teacher and len(negatives)==n and n>1:
            blocked=gold|set(negatives[:-1]);draw=None
            for _ in range(32):
                candidate=rng.choice(pool)
                if candidate not in blocked:draw=candidate;break
            if draw is not None:negatives[-1]=draw
        while len(negatives)<n:negatives.append(rng.choice(neg_pool))
        ids=[positive]+negatives;lex=store.lexical['train'][qid];lexscores=dict(zip(lex['ids'],lex['scores']));lexscores.update(lex.get('extra_scores',{}))
        result.append({'qid':qid,'cids':ids,'gold':gold,'lexical':[lexscores.get(c,0.) for c in ids],
                       'teacher_scores':[dict(zip(teacher['ids'],teacher['scores']))[c] for c in ids] if teacher else None})
    return result
def training_batch(store,qids,cfg,epoch,device,hard=None,kd=None):
    rows=samples(store,qids,cfg,epoch,hard,kd);qkeys=[store.splits['train']['queries'][r['qid']]['key'] for r in rows]
    cids=[c for r in rows for c in r['cids']];dkeys=[store.splits['train']['candidates'][c]['key'] for c in cids]
    if cfg['kind']=='cross':
        batch=pairs_batch(store,[q for q,r in zip(qkeys,rows) for _ in r['cids']],dkeys,[x for r in rows for x in r['lexical']],cfg,device)
        batch.update(mode='loss',query_count=len(rows),positive=torch.tensor([[c in r['gold'] for c in r['cids']] for r in rows],device=device))
        return batch
    batch=encode_batch(store,qkeys+dkeys,[0]*len(qkeys)+[1]*len(dkeys),cfg,device);global_ids=cids;rank_id=0
    if dist.is_initialized():
        rank_id=dist.get_rank();gathered=[None]*dist.get_world_size();dist.all_gather_object(gathered,cids);global_ids=sum(gathered,[])
    batch.update(mode='loss',query_count=len(rows),positive=torch.tensor([[c in r['gold'] for c in global_ids] for r in rows],device=device))
    if kd:
        batch['teacher_scores']=torch.tensor([r['teacher_scores'] for r in rows],device=device,dtype=torch.float32)
        width=cfg['negatives']+1;batch['teacher_columns']=torch.arange(len(cids),device=device).reshape(len(rows),width)+rank_id*len(cids)
    return batch
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--action',choices=['train','preflight','rank','teach'],default='train');ap.add_argument('--split',default='test');ap.add_argument('--output');args=ap.parse_args()
    cfg=read_json(args.config);store=Store(cfg['manifest']);rank_id,world,device=initialize();amp=device.type=='cuda'
    random.seed(cfg['seed']+rank_id);np.random.seed(cfg['seed']+rank_id);torch.manual_seed(cfg['seed'])
    if amp:torch.backends.cudnn.benchmark=True
    run=Path(cfg['run']);run.mkdir(parents=True,exist_ok=True)
    lineage={'config':cfg,'world_size':world,'parent_sha256':file_hash(cfg['parent']),'data_fingerprint':store.manifest['fingerprint'],
             'code_sha256':{p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')}}
    if cfg.get('hard_file'):lineage['hard_cache_sha256']=file_hash(cfg['hard_file']) if Path(cfg['hard_file']).is_file() else None
    if cfg.get('base_run'):lineage['base_rankings_sha256']={s:file_hash(Path(cfg['base_run'])/f'{s}_rankings.json') for s in ('train','dev','test') if (Path(cfg['base_run'])/f'{s}_rankings.json').is_file()}
    if cfg['method']=='kd':lineage['kd_cache_sha256']=file_hash(cfg['kd_file']) if Path(cfg['kd_file']).is_file() else None
    signature=digest(lineage);model=build(cfg,store,device);report=model.report();bestpath=run/'best.pt'
    if args.action in ('rank','teach'):
        saved=torch.load(bestpath,map_location='cpu',weights_only=False)
        if saved['signature']!=signature:raise ValueError('Checkpoint/input/code provenance mismatch.')
        model.load_state_dict(saved['model'],strict=True);del saved
        ids=query_ids(store,'train' if args.action=='teach' else args.split,seed=cfg['seed'])
        if args.action=='teach':raw=teacher_scores(model,store,ids[rank_id::world],cfg,device,cfg['teacher_candidates'])
        else:raw=rank(model,store,args.split,ids[rank_id::world],cfg,device,base_rankings(cfg,args.split))
        output=Path(args.output);atomic_json(str(output)+f'.rank{rank_id}.json',raw);barrier(device)
        if rank_id==0:
            merged={}
            for r in range(world):
                part=read_json(str(output)+f'.rank{r}.json')
                if set(part)&set(merged):raise ValueError('Duplicate distributed query output.')
                merged.update(part)
            if args.action=='rank' and set(merged)!=set(ids):raise ValueError('Missing query rankings.')
            atomic_json(output,merged)
        barrier(device)
        if dist.is_initialized():dist.destroy_process_group()
        return
    train=training_queries(store)
    if not train:raise ValueError('No training query has both a positive and a possible negative.')
    hard=None;kd=None
    if cfg.get('hard_file') and Path(cfg['hard_file']).is_file():
        h=read_json(cfg['hard_file'])
        if h['data_fingerprint']!=store.manifest['fingerprint']:raise ValueError('Hard negatives belong to different data.')
        hard=h['rankings']
    if cfg['method']=='kd' and args.action=='train':
        k=read_json(cfg['kd_file'])
        if k['data_fingerprint']!=store.manifest['fingerprint'] or not set(k['scores'])<=set(train):raise ValueError('Unsafe teacher cache; only eligible TRAIN queries may be distilled.')
        if k['teacher_checkpoint_sha256']!=file_hash(k['teacher_checkpoint']):raise ValueError('Teacher checkpoint changed.')
        kd=k['scores'];train=[q for q in train if q in kd]
        if not train:raise ValueError('KD cache contains no trainable queries.')
    wrapped=DDP(model,device_ids=[device.index] if amp else None,broadcast_buffers=False) if world>1 else model
    params=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=.01,**({'fused':True} if amp else {}))
    initial_scale=float(cfg.get('amp_init_scale',1024.))
    if args.action=='train' and (run/'preflight.json').is_file():
        initial_scale=min(initial_scale,float(read_json(run/'preflight.json').get('amp_scale',initial_scale)))
    scaler=torch.amp.GradScaler(device.type,init_scale=initial_scale,enabled=amp)
    if args.action=='preflight':
        qids=[train[(rank_id*cfg['micro_batch']+i)%len(train)] for i in range(cfg['micro_batch'])];model.train();times=[];retry_count=0
        for i in range(3):
            optimizer.zero_grad(set_to_none=True);batch=training_batch(store,qids,cfg,i,device,hard)
            if amp:torch.cuda.synchronize()
            start=time.perf_counter()
            update=preflight_update(wrapped,model,optimizer,scaler,params,batch,device,amp,f'real_microbatch_{i+1}')
            retry_count+=update['overflow_retries']
            if amp:torch.cuda.synchronize()
            times.append(time.perf_counter()-start)
        # One discarded synthetic worst-shape step catches longer-document memory failures before training.
        optimizer.zero_grad(set_to_none=True)
        if cfg['kind']=='cross':
            n=batch['ids'].shape[0];T=cfg['pair_tokens'];batch['ids']=torch.randint(0,model.backbone.config.vocab_size,(n,T),device=device);batch['valid']=torch.ones(n,T,device=device,dtype=torch.bool)
            batch['segments']=torch.zeros(n,T,device=device,dtype=torch.long);batch['segments'][:,T//2:]=1
        else:
            B=batch['documents'];M=cfg['max_chunks'];T=cfg['chunk_tokens'];batch['ids']=torch.randint(0,model.backbone.config.vocab_size,(B*M,T),device=device);batch['valid']=torch.ones(B*M,T,device=device,dtype=torch.bool)
            batch['owner']=torch.arange(B,device=device).repeat_interleave(M);batch['slot']=torch.arange(M,device=device).repeat(B)
        stress_update=preflight_update(wrapped,model,optimizer,scaler,params,batch,device,amp,'worst_shape')
        retry_count+=stress_update['overflow_retries']
        checksum=torch.stack([p.detach().float().sum() for p in params]).sum()
        if world>1:
            lo=checksum.clone();hi=checksum.clone();dist.all_reduce(lo,op=dist.ReduceOp.MIN);dist.all_reduce(hi,op=dist.ReduceOp.MAX)
            if not torch.allclose(lo,hi,rtol=1e-6,atol=1e-5):raise ValueError('DDP rank parameters diverged.')
        result={'world_size':world,'finite_update':True,'micro_batch':cfg['micro_batch'],'chunk_tokens':cfg['chunk_tokens'],'max_chunks':cfg['max_chunks'],'seconds_per_microbatch':sum(times[1:])/2,
                'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30 if amp else None,'parameters':report,'worst_shape_update_passed':True,
                'amp_scale':float(scaler.get_scale()),'overflow_retries':retry_count,'last_finite_loss':stress_update['loss'],
                'source':'Timing uses real training microbatches. Peak includes one discarded synthetic step at configured maximum chunk/pair shapes.'}
        if rank_id==0:atomic_json(run/'preflight.json',result);print(json.dumps(result),flush=True)
        barrier(device)
        if dist.is_initialized():dist.destroy_process_group()
        return
    sampler=DistributedSampler(train,num_replicas=world,rank=rank_id,shuffle=True,seed=cfg['seed']);loader=DataLoader(train,batch_size=cfg['micro_batch'],sampler=sampler,num_workers=0,collate_fn=lambda x:x)
    total=max(1,math.ceil(len(loader)/cfg['accumulation'])*cfg['epochs']);updates=0;best=-1.;stale=0;start_epoch=0;best_epoch=0;elapsed_base=0;started=time.perf_counter();last=run/'last.pt';skipped=0;consecutive_skips=0
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        if saved['signature']!=signature:raise ValueError('Resume inputs/code changed. Use a new PROJECT.')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        start_epoch=saved['epoch'];updates=saved['updates'];best=saved['best'];stale=saved['stale'];best_epoch=saved['best_epoch'];elapsed_base=saved['elapsed_hours']
        r=saved['rng'][rank_id];random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'])
        if amp:torch.cuda.set_rng_state(r['cuda'],device)
    if stale>=cfg['patience']:start_epoch=cfg['epochs']
    if rank_id==0:print(json.dumps({'experiment':cfg['name'],'train_queries':len(train),'effective_query_batch':world*cfg['micro_batch']*cfg['accumulation'],'cross_rank_in_batch_negatives':world>1 and not model.cross,'parameters':report}),flush=True)
    probe=query_ids(store,'dev',cfg['dev_probe'],cfg['seed'])
    for epoch in range(start_epoch,cfg['epochs']):
        model.train();sampler.set_epoch(epoch);optimizer.zero_grad(set_to_none=True);running=0
        for i,qids in enumerate(loader):
            sync=(i+1)%cfg['accumulation']==0 or i+1==len(loader);group=(i//cfg['accumulation'])*cfg['accumulation'];divisor=min(cfg['accumulation'],len(loader)-group)
            # All ranks participate in candidate gathering even in no_sync microbatches.
            batch=training_batch(store,qids,cfg,epoch,device,hard,kd);ctx=wrapped.no_sync() if world>1 and not sync else contextlib.nullcontext()
            with ctx:
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):loss=wrapped(batch)
                if not all_finite(loss,device):raise FloatingPointError(f'Nonfinite forward loss at epoch {epoch+1}, batch {i+1}; not a loss-scaler-only overflow.')
                scaler.scale(loss/divisor).backward()
            running+=float(loss.detach())
            if sync:
                progress=updates/total;warm=.06;factor=min(1.,(updates+1)/max(1,total*warm)) if progress<warm else .1+.9*.5*(1+math.cos(math.pi*(progress-warm)/(1-warm)))
                for g in optimizer.param_groups:g['lr']=cfg['lr']*factor
                norm,stepped=finish_update(model,optimizer,scaler,params,device,f'epoch_{epoch+1}_batch_{i+1}')
                skipped+=not stepped;consecutive_skips=0 if stepped else consecutive_skips+1
                optimizer.zero_grad(set_to_none=True);updates+=int(stepped)
                if consecutive_skips>=12:raise FloatingPointError('Persistent gradient overflow during training; inspect the printed diagnostics.')
                if rank_id==0 and stepped and updates%cfg['log_every']==0:print(json.dumps({'epoch':epoch+1,'updates':updates,'loss':running/(i+1),'gradient_norm':float(norm),'amp_scale':float(scaler.get_scale()),'amp_skipped_updates':skipped,'elapsed_hours':elapsed_base+(time.perf_counter()-started)/3600}),flush=True)
        raw=rank(model,store,'dev',probe[rank_id::world],cfg,device,base_rankings(cfg,'dev'))
        atomic_json(run/f'dev_epoch_{epoch+1}.rank{rank_id}.json',raw);barrier(device);decision=[None]
        if rank_id==0:
            merged={}
            for r in range(world):merged.update(read_json(run/f'dev_epoch_{epoch+1}.rank{r}.json'))
            if set(merged)!=set(probe):raise ValueError('Incomplete validation probe rankings.')
            selection=select_fusion(store,merged);value=selection['metrics']['micro_f1_at7'];improved=value>best+cfg['min_delta']
            if improved:best=value;best_epoch=epoch+1;stale=0
            else:stale+=1
            decision[0]={'best':best,'best_epoch':best_epoch,'stale':stale,'improved':improved,'selection':selection};atomic_json(run/f'metrics_epoch_{epoch+1}.json',decision[0]);print(json.dumps({'epoch':epoch+1,**decision[0]}),flush=True)
        if world>1:dist.broadcast_object_list(decision,src=0,device=device)
        dec=decision[0];best=dec['best'];stale=dec['stale'];best_epoch=dec['best_epoch']
        state={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(device) if amp else None};rng=[None]*world
        if world>1:dist.all_gather_object(rng,state)
        else:rng=[state]
        if rank_id==0:
            saved={'model':model.state_dict(),'signature':signature,'lineage':lineage,'epoch':epoch+1,'best':best,'best_epoch':best_epoch,'stale':stale,'updates':updates,'rng':rng,'parameters':report,'elapsed_hours':elapsed_base+(time.perf_counter()-started)/3600}
            if dec['improved']:save(bestpath,saved);atomic_json(run/'best_metrics.json',dec)
            save(last,{**saved,'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict()})
        barrier(device)
        if stale>=cfg['patience']:break
    if dist.is_initialized():dist.destroy_process_group()
if __name__=='__main__':main()
