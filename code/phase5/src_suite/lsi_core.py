"""Reusable own-foundation encoder components; no task-adapted parent restoration."""
import gc,math,shutil,types

from pathlib import Path

import numpy as np

import torch

from torch import nn

from torch.nn import functional as F

from torch.utils.checkpoint import checkpoint

from legal_models import LegalLM,ModelConfig,Attention

from lsi_data import atomic_json,atomic_torch,read_json,file_hash,digest

class LoRMA(nn.Module):
    def __init__(self,base,rank,alpha):
        super().__init__();self.base=base;self.scale=alpha/rank;base.requires_grad_(False)
        self.A=nn.Linear(base.out_features,rank,bias=False);self.B=nn.Linear(rank,base.out_features,bias=False)
        nn.init.normal_(self.A.weight,std=.02);nn.init.zeros_(self.B.weight)
    def forward(self,x):
        y=self.base(x);return y+self.scale*self.B(self.A(y))

class Bottleneck(nn.Module):
    def __init__(self,width,bottleneck):
        super().__init__();self.norm=nn.LayerNorm(width);self.down=nn.Linear(width,bottleneck);self.up=nn.Linear(bottleneck,width)
        nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
    def forward(self,x):return self.up(F.gelu(self.down(self.norm(x))))

def adapter_hook(module,args,value):
    if not isinstance(value,torch.Tensor):raise ValueError('Adapters require classification mode without generation caches.')
    return value+module.task_adapter(value)

def masked_attention(self,x,memory=None,past=None,position=0,use_cache=False):
    if past is not None or use_cache:raise ValueError('CJPE uses complete classification chunks without generation caches.')
    b,t,_=x.shape;source=x if memory is None else memory;s=source.shape[1]
    q=self.q(x).view(b,t,self.heads,self.hd).transpose(1,2)
    k=self.k(source).view(b,s,self.kv_heads,self.hd).transpose(1,2)
    v=self.v(source).view(b,s,self.kv_heads,self.hd).transpose(1,2)
    if self.rotary is not None:q=self.rotary(q);k=self.rotary(k)
    if self.kv_heads!=self.heads:k=k.repeat_interleave(self.heads//self.kv_heads,1);v=v.repeat_interleave(self.heads//self.kv_heads,1)
    valid=self._cj_valid;causal=self.causal;mask=None
    if self.window or not self._cj_all_valid:
        key=(t,s,self.window,causal,x.device)
        if getattr(self,'_cj_allowed_key',None)!=key:
            qi=torch.arange(t,device=x.device)[:,None];ki=torch.arange(s,device=x.device)[None,:]
            allowed=ki<=qi if causal else torch.ones(t,s,device=x.device,dtype=torch.bool)
            if self.window:allowed&=(qi-ki)<self.window
            self._cj_allowed_key=key;self._cj_allowed=allowed
        allowed=self._cj_allowed
        mask=allowed[None,None]&valid[:,None,None,:s];causal=False
    out=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,is_causal=causal,dropout_p=self.dropout if self.training else 0.)
    return self.out(out.transpose(1,2).contiguous().view(b,t,-1)),None

class JudgmentModel(nn.Module):
    def __init__(self,backbone,cfg):
        super().__init__();self.backbone=backbone;self.cfg=cfg;d=backbone.config.dim;h=cfg.get('head_dim',192)
        self.pool_dim=2*d;method=cfg['method'];backbone.requires_grad_(False)
        if method in ('full','kd','gradual'):backbone.requires_grad_(True)
        elif method=='partial':
            for block in list(backbone.blocks)[-cfg.get('unfrozen_layers',2):]:block.requires_grad_(True)
            backbone.final_norm.requires_grad_(True)
        elif method=='lorma':
            targets={'q','k','v','out','gate','up','down'}
            replacements=[(name,m) for name,m in backbone.named_modules() if isinstance(m,nn.Linear) and name.rsplit('.',1)[-1] in targets]
            for name,m in replacements:
                parent,attr=name.rsplit('.',1);setattr(backbone.get_submodule(parent),attr,LoRMA(m,cfg.get('rank',8),cfg.get('alpha',16.)))
        elif method=='adapter':
            blocks=list(backbone.blocks)+list(backbone.encoder) if hasattr(backbone,'encoder') else list(backbone.blocks)
            for block in blocks:
                block.task_adapter=Bottleneck(d,cfg.get('adapter_dim',48));block.register_forward_hook(adapter_hook)
        elif method=='bitfit':
            biases=[p for name,p in backbone.named_parameters() if 'bias' in name.rsplit('.',1)[-1]]
            if not biases:raise ValueError('This parent has no biases. BitFit uses the hybrid backbone with existing LSTM biases.')
            for p in biases:p.requires_grad_(True)
        else:raise ValueError(f'Unknown method {method}')
        self.attention_modules=[m for m in backbone.modules() if isinstance(m,Attention)]
        for m in self.attention_modules:m.forward=types.MethodType(masked_attention,m);m.dropout=cfg.get('encoder_dropout',.05)
        backbone.config.gradient_checkpointing=False
        for m in backbone.modules():
            if isinstance(m,nn.Dropout):m.p=cfg.get('encoder_dropout',.05)
        if cfg.get('head_only'):backbone.requires_grad_(False)
        self.project=nn.Sequential(nn.LayerNorm(2*d+6),nn.Linear(2*d+6,h),nn.GELU(),nn.Dropout(cfg.get('dropout',.2)))
        self.head_type=cfg.get('head_type','bigru')
        if self.head_type=='transformer':
            layer=nn.TransformerEncoderLayer(h,4,3*h,dropout=cfg.get('dropout',.2),batch_first=True,norm_first=True,activation='gelu')
            self.context=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        else:self.context=nn.GRU(h,h//2,num_layers=2,bidirectional=True,batch_first=True,dropout=cfg.get('dropout',.2))
        self.norm=nn.LayerNorm(h);self.attention=nn.Sequential(nn.Linear(h,h//2),nn.Tanh(),nn.Linear(h//2,1,bias=False))
        self.classifier=nn.Sequential(nn.LayerNorm(4*h),nn.Dropout(cfg.get('dropout',.2)),nn.Linear(4*h,2))
    def set_epoch(self,epoch):
        if self.cfg['method']!='gradual' or self.cfg.get('head_only'):return
        count=min(len(self.backbone.blocks),self.cfg.get('unfrozen_layers',2)*2**epoch)
        self.backbone.requires_grad_(False)
        for block in list(self.backbone.blocks)[-count:]:block.requires_grad_(True)
        self.backbone.final_norm.requires_grad_(True)
        if count==len(self.backbone.blocks):self.backbone.requires_grad_(True)
    def _hybrid_block(self,x,valid,block):
        # Re-establish this chunk's mask during backward recomputation as well as forward.
        all_valid=bool(valid.all())
        for module in self.attention_modules:module._cj_valid=valid;module._cj_all_valid=all_valid
        return block(x)
    def _hybrid_tokens(self,ids,valid):
        # Keep the fused cuDNN LSTM out of activation-checkpoint recomputation.
        # This is the same residual LSTM -> Transformer computation as LegalLM.forward.
        x=self.backbone.embedding(ids)
        y,_=self.backbone.rnn(x)
        x=x+self.backbone.rnn_norm(y)
        for block in self.backbone.blocks:
            x=checkpoint(self._hybrid_block,x,valid,block,use_reentrant=False)
        return self.backbone.final_norm(x)
    def pool_tokens(self,ids,valid):
        all_valid=bool(valid.all())
        for module in self.attention_modules:module._cj_valid=valid;module._cj_all_valid=all_valid
        hybrid_checkpoint=(self.backbone.config.architecture=='hybrid' and self.training
            and self.cfg.get('encoder_checkpointing',True)
            and any(p.requires_grad for p in self.backbone.parameters()))
        if hybrid_checkpoint:hidden=self._hybrid_tokens(ids,valid)
        else:hidden=self.backbone(ids,source_ids=ids) if self.backbone.config.architecture=='encoder_decoder' else self.backbone(ids)
        mean=(hidden.float()*valid[:,:,None]).sum(1)/valid.sum(1)[:,None]
        last=hidden[torch.arange(len(ids),device=ids.device),valid.sum(1)-1].float()
        return torch.cat((mean,last),-1)
    def encode_chunks(self,chunks,device):
        values=[];step=self.cfg.get('chunk_batch',8)
        for a in range(0,len(chunks),step):
            local=chunks[a:a+step];T=max(map(len,local));ids=torch.zeros(len(local),T,device=device,dtype=torch.long);valid=torch.zeros_like(ids,dtype=torch.bool)
            for i,row in enumerate(local):ids[i,:len(row)]=torch.as_tensor(row.copy(),device=device);valid[i,:len(row)]=True
            checkpoint_encoder=(self.training and self.cfg.get('encoder_checkpointing',True)
                and any(p.requires_grad for p in self.backbone.parameters()))
            if checkpoint_encoder and self.backbone.config.architecture!='hybrid':
                value=checkpoint(self.pool_tokens,ids,valid,use_reentrant=False)
            else:value=self.pool_tokens(ids,valid)
            values.append(value)
        return torch.cat(values)
    def chunk_features(self,batch):
        if batch['features'] is not None:
            if not self.cfg.get('head_only'):raise ValueError('Frozen feature caches are forbidden after backbone adaptation.')
            return batch['features']
        B,S=batch['mask'].shape;vectors=self.encode_chunks(batch['chunks'],batch['mask'].device)
        indices=torch.tensor(batch['owners'],device=vectors.device);pooled=vectors.new_zeros(B*S,self.pool_dim)
        return pooled.index_copy(0,indices,vectors).reshape(B,S,-1)
    def forward(self,batch):
        x=self.project(torch.cat((self.chunk_features(batch),batch['position']),-1));mask=batch['mask'];B,S,_=x.shape
        if self.head_type=='transformer':x=self.context(x,src_key_padding_mask=~mask)
        else:
            packed=nn.utils.rnn.pack_padded_sequence(x,mask.sum(1).cpu(),batch_first=True,enforce_sorted=False)
            x,_=self.context(packed);x,_=nn.utils.rnn.pad_packed_sequence(x,batch_first=True,total_length=S)
        x=self.norm(x);score=self.attention(x).squeeze(-1).float().masked_fill(~mask,-1e9)
        attention=torch.softmax(score,dim=1);global_pool=(x*attention[:,:,None]).sum(1)
        maximum=x.float().masked_fill(~mask[:,:,None],-1e9).max(1).values
        # Complementary section summaries keep beginning and ending evidence available.
        ordinal=torch.arange(S,device=x.device)[None,:];length=mask.sum(1)[:,None]
        first=mask&(ordinal<torch.ceil(length.float()/4));last=mask&(ordinal>=torch.floor(3*length.float()/4))
        section=lambda m:(x*m[:,:,None]).sum(1)/m.sum(1).clamp_min(1)[:,None]
        vector=torch.cat((global_pool,maximum,section(first),section(last)),-1)
        return {'logits':self.classifier(vector),'attention':attention}

def parameter_report(model):return {'total':sum(p.numel() for p in model.parameters()),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
    'backbone':sum(p.numel() for p in model.backbone.parameters())}
