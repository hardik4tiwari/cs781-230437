"""Disposable two-update DDP check for every method on the actual Kaggle GPUs."""
import argparse,gc,json
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from ner_predict import device_setup,autocast
from ner_models import build_model,ner_loss
from ner_data import NERWindows,collate_windows,class_weights,atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();rank,world,device=device_setup()
    if world!=2 or device.type!='cuda':raise RuntimeError('This smoke test requires the actual two Kaggle GPUs.')
    cache=torch.load(args.cache,map_location='cpu',weights_only=False);rows=json.loads(Path(args.plan).read_text());reports=[]
    for cfg in rows:
        model,_=build_model(cfg,cache['labels'],cache['char_classes'],cache['tokenizer_sha256']);model.to(device).train()
        wrapped=DistributedDataParallel(model,device_ids=[device.index],broadcast_buffers=False)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,fused=True)
        scaler=torch.amp.GradScaler('cuda');weights=class_weights(cache,'train').to(device);ds=NERWindows(cache,'train',cfg['head'])
        indices=sorted(range(len(ds)),key=lambda i:len(ds[i]['ids']),reverse=True)
        batch=collate_windows([ds[indices[(rank*cfg['micro_batch']+i)%len(indices)]] for i in range(cfg['micro_batch'])])
        for key in ('ids','valid','tags','left','right'):batch[key]=batch[key].to(device)
        successful=attempts=0
        while successful<2 and attempts<16:
            attempts+=1
            optimizer.zero_grad(set_to_none=True)
            with autocast(device):out=wrapped(batch['ids'],batch['valid']);loss=ner_loss(out,batch,cfg,weights)
            if not bool(torch.isfinite(loss)):raise ValueError('Non-finite loss during distributed preflight.')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
            if scaler.get_scale()>=old_scale:successful+=1
        if successful<2:raise ValueError('FP16 preflight could not complete two finite updates after scale adjustment; inspect model and loss before long training.')
        dist.barrier();reports.append({'name':cfg['name'],'two_ddp_updates_passed':True,'attempted_updates':attempts})
        if rank==0:print('PASS actual GPU DDP:',cfg['name'],flush=True)
        del wrapped,model,optimizer,scaler,weights,batch,out,loss;gc.collect();torch.cuda.empty_cache()
    if rank==0:atomic_json(args.output,reports)
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
