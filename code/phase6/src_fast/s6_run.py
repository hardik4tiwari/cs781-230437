"""Distributed setup and synchronized FP16 updates, inherited from Phase 6."""
import json,os
from pathlib import Path
import torch
import torch.distributed as dist

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
