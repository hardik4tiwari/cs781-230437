"""Original Phase-1 weights with conditional generation, PEFT and KV caches."""
import math,types,contextlib
from dataclasses import fields
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from legal_models import LegalLM,ModelConfig,Attention
from s6_data import file_hash

class LowRank(nn.Module):
    def __init__(self,base,rank,alpha,multiplicative=False):
        super().__init__();self.base=base;self.scale=alpha/rank;self.multiplicative=multiplicative;base.requires_grad_(False)
        self.A=nn.Linear(base.out_features if multiplicative else base.in_features,rank,bias=False)
        self.B=nn.Linear(rank,base.out_features,bias=False)
        nn.init.normal_(self.A.weight,std=.02);nn.init.zeros_(self.B.weight)
    def forward(self,x):
        y=self.base(x)
        # LoRMA+ Eq.8: (I + alpha/r BA) W x. LoRA: W x + alpha/r BA x.
        return y+self.scale*self.B(self.A(y if self.multiplicative else x))
@contextlib.contextmanager
def merged_adapters(model):
    """Exact FP32 weight algebra for LoRA/LoRMA+: remove adapter matmul overhead while decoding."""
    replacements=[]
    with torch.no_grad():
        for name,m in list(model.named_modules()):
            if not isinstance(m,LowRank):continue
            parent_name,attr=name.rsplit('.',1);parent=model.get_submodule(parent_name)
            layer=nn.Linear(m.base.in_features,m.base.out_features,bias=m.base.bias is not None,device=m.base.weight.device,dtype=m.base.weight.dtype)
            update=m.B.weight@(m.A.weight@m.base.weight if m.multiplicative else m.A.weight)
            layer.weight.copy_(m.base.weight+m.scale*update)
            if m.base.bias is not None:
                bias=m.base.bias
                layer.bias.copy_(bias+m.scale*m.B.weight@(m.A.weight@bias) if m.multiplicative else bias)
            layer.requires_grad_(False);setattr(parent,attr,layer);replacements.append((parent,attr,m))
    try:yield
    finally:
        for parent,attr,original in replacements:setattr(parent,attr,original)
class Adapter(nn.Module):
    def __init__(self,d,b):
        super().__init__();self.norm=nn.LayerNorm(d);self.down=nn.Linear(d,b);self.up=nn.Linear(b,d)
        nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
    def forward(self,x):return x+self.up(F.gelu(self.down(self.norm(x))))
def attention(self,x,memory=None,past=None,position=0,use_cache=False):
    b,t,_=x.shape;q=self.q(x).view(b,t,self.heads,self.hd).transpose(1,2)
    if self.cross and past is not None:k,v=past
    else:
        src=x if memory is None else memory;s=src.size(1)
        k=self.k(src).view(b,s,self.kv_heads,self.hd).transpose(1,2);v=self.v(src).view(b,s,self.kv_heads,self.hd).transpose(1,2)
        if self.rotary is not None:k=self.rotary(k,position)
        if past is not None:k=torch.cat([past[0],k],-2);v=torch.cat([past[1],v],-2)
    if self.rotary is not None:q=self.rotary(q,position)
    cache=(k,v) if use_cache else None
    if self.heads!=self.kv_heads:k=k.repeat_interleave(self.heads//self.kv_heads,1);v=v.repeat_interleave(self.heads//self.kv_heads,1)
    if self._valid is None:
        if past is not None or use_cache or self.window:
            raise ValueError('Implicit causal mask requires dense right-padded training input.')
        y=F.scaled_dot_product_attention(q,k,v,is_causal=self.causal,dropout_p=self.dropout if self.training else 0.)
        return self.out(y.transpose(1,2).contiguous().view(b,t,-1)),cache
    s=k.shape[-2];valid=self._valid[:,:s];allowed=valid[:,None,None,:].expand(b,1,t,s)
    if self.causal:
        qi=torch.arange(position,position+t,device=x.device)[:,None];ki=torch.arange(s,device=x.device)[None,:]
        relation=ki<=qi
        if self.window:relation=relation&((qi-ki)<self.window)
        allowed=allowed&relation[None,None]
    # Every real query sees at least BOS/source; padded queries may be fully masked.
    # SDPA handles empty query rows as zero. No FP16 -1e9 masked-fill path.
    y=F.scaled_dot_product_attention(q,k,v,attn_mask=allowed,dropout_p=self.dropout if self.training else 0.)
    return self.out(y.transpose(1,2).contiguous().view(b,t,-1)),cache
def load_parent(path,tokenizer_sha,allow_tiny=False):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('training_origin')!='random_initialization_official_phase1_only':raise ValueError('Parent must be an original own Phase-1 checkpoint, not an adapted task checkpoint.')
    if saved.get('tokenizer_sha256')!=tokenizer_sha:raise ValueError('Parent/tokenizer SHA256 mismatch.')
    c=ModelConfig(**{k:v for k,v in saved['model_config'].items() if k in {f.name for f in fields(ModelConfig)}})
    c.gradient_checkpointing=False;model=LegalLM(c);model.load_state_dict(saved['model'],strict=True)
    n=sum(p.numel() for p in model.parameters())
    if not allow_tiny and not 50_000_000<=n<=300_000_000:raise ValueError(f'Original parent has {n:,} parameters, outside 50M–300M.')
    return model
class Summarizer(nn.Module):
    def __init__(self,backbone,cfg):
        super().__init__();self.backbone=backbone;self.cfg=cfg;self.native=backbone.config.architecture=='encoder_decoder'
        method=cfg['method'];backbone.requires_grad_(method in ('full','kd'))
        if method=='kd':
            count=max(1,len(backbone.blocks)//2);indices=np.linspace(0,len(backbone.blocks)-1,count).round().astype(int).tolist()
            backbone.blocks=nn.ModuleList([backbone.blocks[i] for i in indices]);backbone.config.layers=count;self.student_layers=indices
        if method in ('lora','lorma'):
            targets={'q','k','v','out','gate','up','down'}
            replacements=[(name,m) for name,m in backbone.named_modules() if isinstance(m,nn.Linear) and name.rsplit('.',1)[-1] in targets]
            for name,m in replacements:
                parent,attr=name.rsplit('.',1);setattr(backbone.get_submodule(parent),attr,LowRank(m,cfg.get('rank',16),cfg.get('alpha',32),method=='lorma'))
        if method=='adapter':
            for block in backbone.blocks:
                block.after_attention_adapter=Adapter(backbone.config.dim,cfg.get('adapter_dim',64))
                block.after_ff_adapter=Adapter(backbone.config.dim,cfg.get('adapter_dim',64))
        if method=='cross_only':
            if not self.native:raise ValueError('Cross-attention tuning needs a native encoder-decoder.')
            for block in backbone.blocks:
                block.cross_attn.requires_grad_(True)
                block.cross_norm.requires_grad_(True)
            backbone.final_norm.requires_grad_(True)
        if method=='prompt':
            n=cfg.get('prompt_tokens',32)
            self.prompt=nn.Parameter(backbone.embedding.weight.detach()[:n].clone())
        for m in backbone.modules():
            if isinstance(m,Attention):
                m.forward=types.MethodType(attention,m);m.dropout=cfg.get('dropout',.05)
                # Keep GQA projections; allow long-document information instead of losing source outside a 256-token window.
                if cfg.get('globalize_local',False):m.window=0
            if isinstance(m,nn.Dropout):m.p=cfg.get('dropout',.05)
        self.checkpointing=cfg.get('checkpointing',True)
    def block(self,block,x,valid,memory=None,source_valid=None,position=0,past=None,cross_past=None,cache=False):
        if self.native and memory is not None:
            block.self_attn._valid=valid;block.cross_attn._valid=source_valid
            return block(x,memory,past,cross_past,position,cache)
        block.attn._valid=valid
        if hasattr(block,'after_attention_adapter'):
            y,c=block.attn(block.attn_norm(x),past=past,position=position,use_cache=cache)
            x=block.after_attention_adapter(x+block.drop(y));x=block.after_ff_adapter(x+block.drop(block.ff(block.ff_norm(x))))
            return (x,c) if cache else x
        return block(x,past,position,cache)
    def encode(self,ids,valid):
        x=self.backbone.embedding(ids)
        for block in self.backbone.encoder:
            def f(x,valid,block=block):return self.block(block,x,valid)
            x=checkpoint(f,x,valid,use_reentrant=False) if self.training and self.checkpointing else f(x,valid)
        return self.backbone.encoder_norm(x)
    def hidden(self,ids,valid,source=None,source_valid=None,state=None,use_cache=False,source_width=None,source_lengths=None,compact=False):
        state={} if state is None else state;pos=state.get('position',0);x=self.backbone.embedding(ids)
        if hasattr(self,'prompt') and pos==0:
            x=torch.cat([self.prompt[None].expand(len(ids),-1,-1),x],1)
            if valid is not None:valid=torch.cat([torch.ones(len(ids),self.prompt.shape[0],device=ids.device,dtype=torch.bool),valid],1)
        if pos and 'valid' in state:valid=torch.cat([state['valid'],valid],1)
        new={'position':pos+x.size(1),'valid':valid,'self':[]}
        if hasattr(self.backbone,'rnn'):
            # Fused LSTM stays outside checkpoint recomputation (cuDNN metadata safety).
            if compact:
                y,rnn=self.backbone.rnn(x)
            elif pos==0:
                # Skip masked source PADs in the recurrent state too, not only in attention.
                width=source_width or self.cfg['source_tokens'];lengths=source_lengths if source_lengths is not None else valid[:,:width].sum(1).cpu()
                packed=nn.utils.rnn.pack_padded_sequence(x[:,:width],lengths,batch_first=True,enforce_sorted=False)
                py,rnn=self.backbone.rnn(packed);sy,_=nn.utils.rnn.pad_packed_sequence(py,batch_first=True,total_length=width)
                ty,rnn=self.backbone.rnn(x[:,width:],rnn);y=torch.cat([sy,ty],1)
            else:y,rnn=self.backbone.rnn(x,state.get('rnn'))
            x=x+self.backbone.rnn_norm(y);new['rnn']=rnn
        memory=None
        if self.native:
            memory=state.get('memory')
            if memory is None:memory=self.encode(source,source_valid)
            else:source_valid=state['source_valid']
            new.update(memory=memory,source_valid=source_valid,cross=[])
        for i,block in enumerate(self.backbone.blocks):
            if use_cache:
                y=self.block(block,x,valid,memory,source_valid,pos,state.get('self',[None]*len(self.backbone.blocks))[i],state.get('cross',[None]*len(self.backbone.blocks))[i] if self.native else None,True)
                x=y[0];new['self'].append(y[1])
                if self.native:new['cross'].append(y[2])
            else:
                def f(x,valid,memory=memory,source_valid=source_valid,block=block):return self.block(block,x,valid,memory,source_valid)
                x=checkpoint(f,x,valid,use_reentrant=False) if self.training and self.checkpointing else f(x,valid)
        return self.backbone.final_norm(x),new
    def report(self):return {'total':sum(p.numel() for p in self.parameters()),'trainable':sum(p.numel() for p in self.parameters() if p.requires_grad),'architecture':self.backbone.config.architecture,'method':self.cfg['method']}

