"""Own Phase-1 weights adapted by full/partial tuning, LoRA, LoRMA+, adapters, KD."""
import math
import types
from dataclasses import asdict
import torch
from torch import nn
from torch.nn import functional as F
from legal_models import LegalLM,ModelConfig,Attention,RMSNorm,Rotary
from ner_data import file_hash

class LowRankLinear(nn.Module):
    def __init__(self,base,rank=8,alpha=16.,multiplicative=False):
        super().__init__();self.base=base;self.scale=alpha/rank;self.multiplicative=multiplicative
        self.A=nn.Linear(base.out_features if multiplicative else base.in_features,rank,bias=False)
        self.B=nn.Linear(rank,base.out_features,bias=False)
        nn.init.normal_(self.A.weight,std=.02);nn.init.zeros_(self.B.weight)
        self.base.requires_grad_(False)
    def forward(self,x):
        y=self.base(x)
        # LoRMA+ Eq. 8: (I + alpha/r BA) W0 x. Projection targets are bias-free.
        return y+self.scale*self.B(self.A(y if self.multiplicative else x))
    def merged_weight(self):
        change=self.B.weight@self.A.weight
        return self.base.weight+self.scale*(change@self.base.weight if self.multiplicative else change)

class AdapterBlock(nn.Module):
    """Post-block residual bottleneck variant; not an exact Houlsby reproduction."""
    def __init__(self,base,dim,bottleneck):
        super().__init__();self.base=base;self.norm=RMSNorm(dim)
        self.down=nn.Linear(dim,bottleneck);self.up=nn.Linear(bottleneck,dim)
        nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
    def forward(self,*args,**kwargs):
        x=self.base(*args,**kwargs)
        if isinstance(x,tuple):raise ValueError('NER adapters use full-sequence inference.')
        return x+self.up(F.gelu(self.down(self.norm(x))))

def ner_attention(self,x,memory=None,past=None,position=0,use_cache=False):
    if past is not None or use_cache:raise ValueError('NER attention is full-sequence, not LM cached generation.')
    b,t,_=x.shape;source=x if memory is None else memory;s=source.size(1)
    q=self.q(x).view(b,t,self.heads,self.hd).transpose(1,2)
    k=self.k(source).view(b,s,self.kv_heads,self.hd).transpose(1,2)
    v=self.v(source).view(b,s,self.kv_heads,self.hd).transpose(1,2)
    if self.rotary is not None:q=self.rotary(q);k=self.rotary(k)
    if self.kv_heads!=self.heads:
        k=k.repeat_interleave(self.heads//self.kv_heads,1);v=v.repeat_interleave(self.heads//self.kv_heads,1)
    valid=getattr(self,'_ner_valid',None);mask=None;causal=self.causal
    if self.window or valid is not None:
        mask=torch.ones(t,s,device=x.device,dtype=torch.bool)
        if self.causal:
            qi=torch.arange(t,device=x.device)[:,None];ki=torch.arange(s,device=x.device)[None,:]
            mask=ki<=qi
            if self.window:mask&=(qi-ki)<self.window
        if valid is not None:mask=mask[None,None]&valid[:,None,None,:s]
        causal=False
    y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,is_causal=causal,
        dropout_p=self.dropout if self.training else 0.)
    return self.out(y.transpose(1,2).contiguous().view(b,t,-1)),None

def load_foundation(path,tokenizer_hash,architecture,student_layers=None,allow_tiny=False):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('training_origin')!='random_initialization_official_phase1_only':raise ValueError('Use your own Phase-1 checkpoint.')
    if saved['tokenizer_sha256']!=tokenizer_hash:raise ValueError('Checkpoint/tokenizer mismatch.')
    c=ModelConfig(**saved['model_config'])
    if c.architecture!=architecture:raise ValueError('Foundation architecture mismatch.')
    parent_layers=c.layers;source=saved['model'];retained=None
    if student_layers is not None:
        if architecture!='decoder' or not 1<=student_layers<parent_layers:raise ValueError('Student must retain fewer decoder blocks.')
        retained=[round(i*(parent_layers-1)/(student_layers-1)) for i in range(student_layers)] if student_layers>1 else [parent_layers-1]
        c.layers=student_layers;state={k:v for k,v in source.items() if not k.startswith('blocks.')}
        for new,old in enumerate(retained):
            prefix=f'blocks.{old}.'
            state.update({f'blocks.{new}.'+k[len(prefix):]:v for k,v in source.items() if k.startswith(prefix)})
    else:state=source
    backbone=LegalLM(c);backbone.load_state_dict(state,strict=True)
    parent_parameters=sum(p.numel() for p in source.values())
    if not allow_tiny and not 50_000_000<=parent_parameters<=300_000_000:raise ValueError('Phase-1 parent violates parameter bounds.')
    return backbone,{'phase1_sha256':file_hash(path),'parent_parameters':parent_parameters,
                     'retained_parent_blocks':retained,'phase1_model_config':saved['model_config']}

class NERModel(nn.Module):
    def __init__(self,backbone,cfg,labels,char_classes):
        super().__init__();self.backbone=backbone;self.cfg=cfg;self.labels=labels;self.char_classes=char_classes
        method=cfg['method'];dim=backbone.config.dim
        backbone.requires_grad_(method in ('full','kd'))
        if method=='partial':
            for block in list(backbone.blocks)[-cfg.get('partial_layers',3):]:block.requires_grad_(True)
            backbone.final_norm.requires_grad_(True)
        elif method in ('lora','lorma'):
            targets=cfg.get('projection_targets',['q','v','out','gate','up','down'])
            replacements=[(name,module) for name,module in backbone.named_modules() if isinstance(module,nn.Linear) and name.rsplit('.',1)[-1] in targets]
            for name,module in replacements:
                parent_name,attribute=name.rsplit('.',1);parent=backbone.get_submodule(parent_name)
                setattr(parent,attribute,LowRankLinear(module,cfg.get('rank',8),cfg.get('alpha',16),method=='lorma'))
        elif method=='adapter':
            for index,block in enumerate(backbone.blocks):backbone.blocks[index]=AdapterBlock(block,dim,cfg.get('adapter_dim',64))
            for name,p in backbone.named_parameters():
                if 'attn_norm' in name or 'ff_norm' in name or name.startswith('final_norm.'):p.requires_grad_(True)
        elif method not in ('full','kd','partial'):raise ValueError(method)
        self.attention_modules=[module for module in backbone.modules() if isinstance(module,Attention)]
        for module in self.attention_modules:module.forward=types.MethodType(ner_attention,module)
        self.two_directions=cfg.get('two_directions',True) and backbone.config.architecture!='encoder_decoder'
        width=dim*(2 if self.two_directions else 1)
        self.dropout=nn.Dropout(cfg.get('head_dropout',.1))
        self.feature_norm=nn.LayerNorm(width)
        self.left=nn.Linear(width,len(labels)*char_classes);self.right=nn.Linear(width,len(labels)*char_classes)
        if cfg['head']=='bio':self.classifier=nn.Linear(width,1+2*len(labels))
        else:
            inner=cfg.get('span_dim',32)
            self.span_projection=nn.Linear(width,len(labels)*2*inner)
            self.span_rotary=Rotary(inner,backbone.config.max_positions)
        # Heads are new task parameters; the backbone was loaded before they were created.
    def features(self,ids,valid):
        mask=None if bool(valid.all()) else valid
        for module in self.attention_modules:module._ner_valid=mask
        if self.backbone.config.architecture=='encoder_decoder':hidden=self.backbone(ids,source_ids=ids)
        else:
            hidden=self.backbone(ids)
            if self.two_directions:
                positions=torch.arange(ids.size(1),device=ids.device)[None,:].expand_as(ids)
                lengths=valid.sum(1,keepdim=True)
                reverse=torch.where(positions<lengths,lengths-1-positions,positions)
                reverse_ids=ids.gather(1,reverse)
                backward=self.backbone(reverse_ids).gather(1,reverse[:,:,None].expand(-1,-1,hidden.size(-1)))
                hidden=torch.cat((hidden,backward),-1)
        return self.dropout(self.feature_norm(hidden))
    def forward(self,ids,valid):
        hidden=self.features(ids,valid);b,t,_=hidden.shape;L=len(self.labels);C=self.char_classes
        result={'left':self.left(hidden).view(b,t,L,C),'right':self.right(hidden).view(b,t,L,C)}
        if self.cfg['head']=='bio':result['logits']=self.classifier(hidden)
        else:
            d=self.cfg.get('span_dim',32);pairs=self.span_projection(hidden).view(b,t,L,2,d)
            q=pairs[:,:,:,0].transpose(1,2);k=pairs[:,:,:,1].transpose(1,2)
            q=self.span_rotary(q);k=self.span_rotary(k)
            scores=torch.einsum('blid,bljd->blij',q,k)/math.sqrt(d)
            start=torch.arange(t,device=ids.device)[:,None];end=torch.arange(t,device=ids.device)[None,:]
            allowed=(end>=start)&(end-start<self.cfg.get('max_span_tokens',t))
            allowed=allowed[None,None]&valid[:,None,:,None]&valid[:,None,None,:]
            result['logits']=scores.masked_fill(~allowed,-1e4);result['span_valid']=allowed
        return result

def build_model(cfg,labels,char_classes,tokenizer_hash,restore=None):
    backbone,provenance=load_foundation(cfg['phase1_checkpoint'],tokenizer_hash,cfg['architecture'],
        cfg.get('student_layers') if cfg['method']=='kd' else None,cfg.get('allow_tiny',False))
    backbone.config.gradient_checkpointing=cfg.get('gradient_checkpointing',False)
    model=NERModel(backbone,cfg,labels,char_classes)
    if restore:model.load_state_dict(torch.load(restore,map_location='cpu',weights_only=False)['model'],strict=True)
    return model,provenance

def safe_ce(scores,targets):
    if bool((targets!=-100).any()):return F.cross_entropy(scores.float().reshape(-1,scores.size(-1)),targets.reshape(-1),ignore_index=-100)
    return scores.float().sum()*0

def ner_loss(result,batch,cfg,weights):
    if cfg['head']=='bio':
        hard=F.cross_entropy(result['logits'].float().reshape(-1,result['logits'].size(-1)),
            batch['tags'].reshape(-1),weight=weights,ignore_index=-100)
    else:
        scores=result['logits'].float();target=torch.zeros_like(scores,dtype=torch.bool)
        for row,item in enumerate(batch['items']):
            for start,end,label in item['entities']:target[row,label,start,end]=True
        # GlobalPointer Eq.13. Separate positive/negative logsumexp terms per label.
        valid=result['span_valid'].expand_as(scores)
        negative=scores.masked_fill(target|~valid,-1e4).flatten(2)
        positive=(-scores).masked_fill(~target|~valid,-1e4).flatten(2)
        zero=torch.zeros(*negative.shape[:2],1,device=scores.device)
        hard=(torch.logsumexp(torch.cat((negative,zero),-1),-1)+torch.logsumexp(torch.cat((positive,zero),-1),-1)).mean()
    boundary=safe_ce(result['left'],batch['left'])+safe_ce(result['right'],batch['right'])
    loss=hard+cfg.get('boundary_weight',.25)*boundary
    if batch.get('teacher') is not None:
        T=cfg.get('kd_temperature',2.);mask=batch['valid']
        student=result['logits'].float()[mask];teacher=batch['teacher'][mask]
        kl=F.kl_div(F.log_softmax(student/T,-1),F.softmax(teacher/T,-1),reduction='batchmean')*T*T
        alpha=cfg.get('kd_alpha',.5);loss=(1-alpha)*hard+alpha*kl+cfg.get('boundary_weight',.25)*boundary
    return loss

def parameter_report(model):
    return {'total_parameters':sum(p.numel() for p in model.parameters()),
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'backbone_parameters':sum(p.numel() for p in model.backbone.parameters())}
