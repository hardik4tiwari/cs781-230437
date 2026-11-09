"""Own Phase-1 retrieval backbones, multi-positive contrastive loss and pair ranking."""
import contextlib,math,types
from dataclasses import fields
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from legal_models import LegalLM,ModelConfig,Attention

class LoRMA(nn.Module):
    def __init__(self,base,rank=16,alpha=32):
        super().__init__();self.base=base;base.requires_grad_(False);self.scale=alpha/rank
        self.A=nn.Linear(base.out_features,rank,bias=False);self.B=nn.Linear(rank,base.out_features,bias=False)
        nn.init.normal_(self.A.weight,std=.02);nn.init.zeros_(self.B.weight)
    def forward(self,x):
        y=self.base(x);return y+self.scale*self.B(self.A(y))
class IA3(nn.Module):
    def __init__(self,base,on_input=False):
        super().__init__();self.base=base;base.requires_grad_(False);self.on_input=on_input
        self.gain=nn.Parameter(torch.ones(base.in_features if on_input else base.out_features))
    def forward(self,x):return self.base(x*self.gain) if self.on_input else self.base(x)*self.gain
@contextlib.contextmanager
def merged_adapters(model):
    replaced=[]
    with torch.no_grad():
        for name,m in list(model.named_modules()):
            if not isinstance(m,(LoRMA,IA3)):continue
            parent,attr=name.rsplit('.',1);parent=model.get_submodule(parent);base=m.base
            layer=nn.Linear(base.in_features,base.out_features,bias=base.bias is not None,device=base.weight.device,dtype=base.weight.dtype)
            if isinstance(m,LoRMA):weight=base.weight+m.scale*m.B.weight@(m.A.weight@base.weight)
            else:weight=base.weight*(m.gain[None,:] if m.on_input else m.gain[:,None])
            layer.weight.copy_(weight)
            if base.bias is not None:layer.bias.copy_(base.bias if isinstance(m,IA3) and m.on_input else (base.bias*m.gain if isinstance(m,IA3) else base.bias+m.scale*m.B.weight@(m.A.weight@base.bias)))
            layer.requires_grad_(False);setattr(parent,attr,layer);replaced.append((parent,attr,m))
    try:yield
    finally:
        for parent,attr,m in replaced:setattr(parent,attr,m)
def masked_attention(self,x,memory=None,past=None,position=0,use_cache=False):
    if past is not None or use_cache:raise ValueError('Retrieval encodes complete independent chunks; generation caches are not used.')
    b,t,_=x.shape;source=x if memory is None else memory;s=source.size(1)
    q=self.q(x).view(b,t,self.heads,self.hd).transpose(1,2);k=self.k(source).view(b,s,self.kv_heads,self.hd).transpose(1,2);v=self.v(source).view(b,s,self.kv_heads,self.hd).transpose(1,2)
    if self.rotary is not None:q=self.rotary(q);k=self.rotary(k)
    if self.kv_heads!=self.heads:k=k.repeat_interleave(self.heads//self.kv_heads,1);v=v.repeat_interleave(self.heads//self.kv_heads,1)
    mask=self._valid[:,None,None,:s]
    if self.causal:
        mask=mask&(torch.arange(s,device=x.device)[None,:]<=torch.arange(t,device=x.device)[:,None])[None,None]
    # Encoder-style bidirectional context for retrieval. GQA projections and all Phase-1 tensors are preserved.
    out=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=self.dropout if self.training else 0.)
    return self.out(out.transpose(1,2).contiguous().view(b,t,-1)),None
def load_parent(path,tokenizer_sha,allow_tiny=False):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('training_origin')!='random_initialization_official_phase1_only':raise ValueError('An original own Phase-1 checkpoint is required.')
    if saved.get('tokenizer_sha256')!=tokenizer_sha:raise ValueError('Checkpoint/tokenizer SHA256 mismatch.')
    c=ModelConfig(**{k:v for k,v in saved['model_config'].items() if k in {f.name for f in fields(ModelConfig)}});c.gradient_checkpointing=False
    model=LegalLM(c);model.load_state_dict(saved['model'],strict=True);n=sum(p.numel() for p in model.parameters())
    if not allow_tiny and not 50_000_000<=n<=300_000_000:raise ValueError(f'Original foundation has {n:,} parameters, outside 50M–300M.')
    return model
class Retriever(nn.Module):
    def __init__(self,backbone,cfg):
        super().__init__();self.backbone=backbone;self.cfg=cfg;self.cross=cfg['kind']=='cross';self.native=backbone.config.architecture=='encoder_decoder';d=backbone.config.dim
        method=cfg['method'];backbone.requires_grad_(method in ('full','kd'))
        if method=='kd':
            indices=np.linspace(0,len(backbone.blocks)-1,max(1,len(backbone.blocks)//2)).round().astype(int).tolist()
            backbone.blocks=nn.ModuleList([backbone.blocks[i] for i in indices]);backbone.config.layers=len(indices);self.student_layers=indices
        elif method=='partial':
            for block in list(backbone.blocks)[-cfg.get('unfrozen_layers',2):]:block.requires_grad_(True)
            backbone.final_norm.requires_grad_(True)
        elif method in ('lorma','ia3'):
            for name,module in list(backbone.named_modules()):
                if not isinstance(module,nn.Linear):continue
                leaf=name.rsplit('.',1)[-1]
                use=leaf in ({'q','k','v','out','gate','up','down'} if method=='lorma' else {'k','v','down'})
                if use:
                    parent,attr=name.rsplit('.',1);replacement=LoRMA(module,cfg.get('rank',16),cfg.get('alpha',32)) if method=='lorma' else IA3(module,leaf=='down')
                    setattr(backbone.get_submodule(parent),attr,replacement)
        elif method=='prompt':self.prompt=nn.Parameter(backbone.embedding.weight.detach()[:cfg.get('prompt_tokens',32)].clone())
        elif method!='full':
            if method!='kd':raise ValueError(f'Unknown method {method}')
        for m in backbone.modules():
            if isinstance(m,Attention):m.forward=types.MethodType(masked_attention,m);m.causal=False;m.window=0;m.dropout=cfg.get('dropout',.05)
            if isinstance(m,nn.Dropout):m.p=cfg.get('dropout',.05)
        self.checkpointing=cfg.get('checkpointing',True)
        if self.native:
            self.readouts=nn.Parameter(backbone.embedding.weight.detach()[cfg['bos']][None].repeat(4,1)+.01*torch.randn(4,d))
        if self.cross:
            self.segment=nn.Embedding(2,d);nn.init.zeros_(self.segment.weight)
            self.rank_head=nn.Sequential(nn.LayerNorm(4*d+3),nn.Linear(4*d+3,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,1))
        else:
            h=cfg.get('embedding_dim',256);self.token_gate=nn.Linear(d,1,bias=False)
            self.project=nn.Sequential(nn.LayerNorm(2*d),nn.Linear(2*d,h),nn.GELU(),nn.Linear(h,h))
            self.roles=nn.Parameter(torch.zeros(2,h));self.chunk_gate=nn.Linear(h+2,1)
    def _block(self,block,x,valid,memory=None,memory_valid=None):
        if memory is None:block.attn._valid=valid;return block(x)
        block.self_attn._valid=valid;block.cross_attn._valid=memory_valid;return block(x,memory)
    def hidden(self,ids,valid,segments=None):
        x=self.backbone.embedding(ids)
        if segments is not None:x=x+self.segment(segments)
        if hasattr(self,'prompt'):
            x=torch.cat([self.prompt[None].expand(len(ids),-1,-1),x],1)
            valid=torch.cat([torch.ones(len(ids),len(self.prompt),device=ids.device,dtype=torch.bool),valid],1)
        if hasattr(self.backbone,'rnn'):
            # Pack right padding; keep cuDNN LSTM outside activation-checkpoint recomputation.
            packed=nn.utils.rnn.pack_padded_sequence(x,valid.sum(1).cpu(),batch_first=True,enforce_sorted=False)
            y,_=self.backbone.rnn(packed);y,_=nn.utils.rnn.pad_packed_sequence(y,batch_first=True,total_length=x.shape[1]);x=x+self.backbone.rnn_norm(y)
        memory=None;memory_valid=None
        if self.native:
            for block in self.backbone.encoder:
                def f(x,valid,block=block):return self._block(block,x,valid)
                x=checkpoint(f,x,valid,use_reentrant=False) if self.training and self.checkpointing else f(x,valid)
            memory=self.backbone.encoder_norm(x);memory_valid=valid;x=self.readouts[None].expand(len(ids),-1,-1);valid=torch.ones(x.shape[:2],device=x.device,dtype=torch.bool)
        for block in self.backbone.blocks:
            def f(x,valid,block=block,memory=memory,memory_valid=memory_valid):return self._block(block,x,valid,memory,memory_valid)
            x=checkpoint(f,x,valid,use_reentrant=False) if self.training and self.checkpointing else f(x,valid)
        return self.backbone.final_norm(x),valid
    def encode(self,batch):
        ids=batch['ids'];valid=batch['valid'];pooled=[];step=self.cfg.get('chunk_microbatch',4)
        for a in range(0,len(ids),step):
            h,m=self.hidden(ids[a:a+step],valid[a:a+step]);gate=self.token_gate(h)
            with torch.autocast(device_type=h.device.type,enabled=False):
                weights=torch.softmax(gate.squeeze(-1).float().masked_fill(~m,float('-inf')),dim=-1)
                attention=(h.float()*weights[:,:,None]).sum(1);mean=(h.float()*m[:,:,None]).sum(1)/m.sum(1)[:,None]
            pooled.append(self.project(torch.cat([attention,mean],-1)))
        vectors=torch.cat(pooled);B=batch['documents'];M=batch['max_chunks'];owner=batch['owner'];slot=batch['slot'];index=owner*M+slot
        padded=vectors.new_zeros(B*M,vectors.shape[-1]).index_copy(0,index,vectors).reshape(B,M,-1)
        mask=torch.zeros(B*M,device=ids.device,dtype=torch.bool).index_fill(0,index,True).reshape(B,M)
        padded=F.normalize(padded.float()+self.roles[batch['roles']][:,None,:],dim=-1)*mask[:,:,None]
        position=batch['position'];gate=self.chunk_gate(torch.cat([padded,position],-1))
        with torch.autocast(device_type=ids.device.type,enabled=False):
            weight=torch.softmax(gate.squeeze(-1).float().masked_fill(~mask,float('-inf')),dim=-1)
            global_vector=F.normalize((padded.float()*weight[:,:,None]).sum(1),dim=-1)
        return global_vector,padded,mask
    def rank_pairs(self,batch):
        scores=[];step=self.cfg.get('pair_microbatch',4)
        for a in range(0,len(batch['ids']),step):
            ids=batch['ids'][a:a+step];valid=batch['valid'][a:a+step];segment=batch['segments'][a:a+step]
            h,_=self.hidden(ids,valid,segment);q=valid&(segment==0);d=valid&(segment==1)
            with torch.autocast(device_type=h.device.type,enabled=False):
                qm=(h.float()*q[:,:,None]).sum(1)/q.sum(1).clamp_min(1)[:,None];dm=(h.float()*d[:,:,None]).sum(1)/d.sum(1).clamp_min(1)[:,None]
                features=torch.cat([qm,dm,qm*dm,(qm-dm).abs(),batch['lexical'][a:a+step]],-1)
            scores.append(self.rank_head(features).squeeze(-1))
        return torch.cat(scores).float()
    def forward(self,batch):
        if batch['mode']=='encode':return self.encode(batch)
        if batch['mode']=='pairs':return self.rank_pairs(batch)
        if self.cross:
            logits=self.rank_pairs(batch).reshape(batch['query_count'],-1);gold=batch['positive'].bool()
            with torch.autocast(device_type=logits.device.type,enabled=False):
                rank=multi_positive_loss(logits.float(),gold)
                binary=F.binary_cross_entropy_with_logits(logits.float(),gold.float())
            return .8*rank+.2*binary
        global_v,chunks,mask=self.encode(batch);nq=batch['query_count'];q=(global_v[:nq],chunks[:nq],mask[:nq]);d=(global_v[nq:],chunks[nq:],mask[nq:])
        # Cross-rank in-batch negatives retain gradient paths through gathered candidate embeddings.
        if torch.distributed.is_initialized():
            from torch.distributed.nn.functional import all_gather
            # One differentiable collective keeps global/chunk backward ordering identical across ranks.
            packed=torch.cat([d[0][:,None,:],d[1]],dim=1);packed=torch.cat(all_gather(packed),0)
            d=(packed[:,0],packed[:,1:],torch.cat(all_gather(d[2].float()),0).bool())
        with torch.autocast(device_type=global_v.device.type,enabled=False):
            scores=similarity(q,d,self.cfg['kind'])/self.cfg.get('temperature',.05)
            loss=multi_positive_loss(scores,batch['positive'])
            if 'teacher_scores' in batch:
                indices=batch['teacher_columns'];student=scores.gather(1,indices);teacher=batch['teacher_scores'].float();T=self.cfg.get('kd_temperature',2.)
                kd=F.kl_div(F.log_softmax(student/T,-1),F.softmax(teacher/T,-1),reduction='batchmean')*T*T
                loss=(1-self.cfg.get('kd_weight',.5))*loss+self.cfg.get('kd_weight',.5)*kd
        return loss
    def report(self):return {'architecture':self.backbone.config.architecture,'kind':self.cfg['kind'],'method':self.cfg['method'],'total_parameters':sum(p.numel() for p in self.parameters()),'trainable_parameters':sum(p.numel() for p in self.parameters() if p.requires_grad)}
def multi_positive_loss(scores,positive):
    if not positive.any(1).all():raise ValueError('Each training query needs at least one positive column.')
    # Sum probability mass over all known relevant IDs, including other queries' positives.
    return (torch.logsumexp(scores,-1)-torch.logsumexp(scores.masked_fill(~positive,float('-inf')),-1)).mean()
def similarity(q,d,kind='dense'):
    if kind!='late':return q[0].float()@d[0].float().T
    # Chunk-level MaxSim, not token-level ColBERT or its compression/index implementation.
    interaction=torch.einsum('qih,djh->qdij',q[1].float(),d[1].float())
    maximum=interaction.masked_fill(~d[2][None,:,None,:],float('-inf')).max(-1).values
    late=(maximum*q[2][:,None,:]).sum(-1)/q[2].sum(-1)[:,None]
    # Include the learned global path so the complete document pooling head is trained too.
    return .8*late+.2*(q[0].float()@d[0].float().T)

def encode_batch(store,keys,roles,cfg,device):
    chunks=[];owners=[];slots=[];positions=[]
    M=cfg['max_chunks'];position=np.zeros((len(keys),M,2),dtype=np.float32)
    for i,key in enumerate(keys):
        for j,(ids,p) in enumerate(store.chunks(key,M)):
            chunks.append(ids);owners.append(i);slots.append(j);position[i,j]=[p,min(1.,len(ids)/cfg['chunk_tokens'])]
    T=max(map(len,chunks));pad=store.manifest['special']['pad'];ids=torch.full((len(chunks),T),pad,device=device,dtype=torch.long);valid=torch.zeros_like(ids,dtype=torch.bool)
    for i,tokens in enumerate(chunks):ids[i,:len(tokens)]=torch.as_tensor(tokens.astype(np.int64),device=device);valid[i,:len(tokens)]=True
    return {'mode':'encode','ids':ids,'valid':valid,'owner':torch.tensor(owners,device=device),'slot':torch.tensor(slots,device=device),
            'roles':torch.tensor(roles,device=device),'position':torch.tensor(position,device=device),'documents':len(keys),'max_chunks':M}
def pairs_batch(store,qkeys,dkeys,lexical,cfg,device):
    pairs=[store.pair(q,d,cfg['pair_tokens']) for q,d in zip(qkeys,dkeys)];T=max(len(ids) for ids,_ in pairs)
    ids=torch.full((len(pairs),T),store.manifest['special']['pad'],device=device,dtype=torch.long);valid=torch.zeros_like(ids,dtype=torch.bool);segments=torch.zeros_like(ids)
    features=[]
    for i,((tokens,segment),score) in enumerate(zip(pairs,lexical)):
        ids[i,:len(tokens)]=torch.tensor(tokens,device=device);valid[i,:len(tokens)]=True;segments[i,:len(tokens)]=torch.tensor(segment,device=device)
        q=tokens[segment==0];d=tokens[segment==1];overlap=len(set(q)&set(d))/max(1,len(set(q)))
        features.append([math.log1p(max(0.,score))/5,overlap,min(len(tokens)/cfg['pair_tokens'],1.)])
    return {'mode':'pairs','ids':ids,'valid':valid,'segments':segments,'lexical':torch.tensor(features,device=device,dtype=torch.float32)}
