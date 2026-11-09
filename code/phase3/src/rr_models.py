"""Own Phase-2 backbones, hierarchical RR heads, learned CRF and five adaptation routes."""
import math,types
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from legal_models import LegalLM,ModelConfig,Attention

class LowRank(nn.Module):
    def __init__(self,base,rank,alpha,kind):
        super().__init__();self.base=base;self.kind=kind
        self.scale=alpha/(math.sqrt(rank) if kind=='rslora' else rank)
        self.A=nn.Linear(base.out_features if kind=='lorma' else base.in_features,rank,bias=False)
        self.B=nn.Linear(rank,base.out_features,bias=False)
        nn.init.normal_(self.A.weight,std=.02);nn.init.zeros_(self.B.weight);base.requires_grad_(False)
    def forward(self,x):
        y=self.base(x);return y+self.scale*self.B(self.A(y if self.kind=='lorma' else x))
    def merged(self):
        delta=self.B.weight@(self.A.weight@self.base.weight) if self.kind=='lorma' else self.B.weight@self.A.weight
        return self.base.weight+self.scale*delta
class IA3(nn.Module):
    def __init__(self,base,on_input=False):
        super().__init__();self.base=base;self.on_input=on_input;base.requires_grad_(False)
        self.scale=nn.Parameter(torch.ones(base.in_features if on_input else base.out_features))
    def forward(self,x):return self.base(x*self.scale) if self.on_input else self.base(x)*self.scale

def attention_forward(self,x,memory=None,past=None,position=0,use_cache=False):
    if past is not None or use_cache:raise ValueError('RR is full-sequence classification.')
    b,t,_=x.shape;source=x if memory is None else memory;s=source.size(1)
    q=self.q(x).view(b,t,self.heads,self.hd).transpose(1,2)
    k=self.k(source).view(b,s,self.kv_heads,self.hd).transpose(1,2)
    v=self.v(source).view(b,s,self.kv_heads,self.hd).transpose(1,2)
    if self.rotary is not None:q=self.rotary(q);k=self.rotary(k)
    if self.kv_heads!=self.heads:
        k=k.repeat_interleave(self.heads//self.kv_heads,1);v=v.repeat_interleave(self.heads//self.kv_heads,1)
    valid=self._rr_valid;causal=self.causal and not self._rr_bidirectional
    window=self.window if causal else 0;mask=None
    if window or not bool(valid.all()):
        qi=torch.arange(t,device=x.device)[:,None];ki=torch.arange(s,device=x.device)[None,:]
        allowed=ki<=qi if causal else torch.ones(t,s,dtype=torch.bool,device=x.device)
        if window:allowed=allowed&((qi-ki)<window)
        mask=allowed[None,None]&valid[:,None,None,:s];causal=False
    output=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,is_causal=causal,dropout_p=self.dropout if self.training else 0.)
    return self.out(output.transpose(1,2).contiguous().view(b,t,-1)),None

class CRF(nn.Module):
    def __init__(self,count):
        super().__init__();self.start=nn.Parameter(torch.zeros(count));self.end=nn.Parameter(torch.zeros(count))
        self.transitions=nn.Parameter(torch.zeros(count,count))
    def nll(self,emissions,target,mask):
        e=emissions.float();tags=target.clamp_min(0);B,S,C=e.shape;row=torch.arange(B,device=e.device)
        score=self.start[tags[:,0]]+e[row,0,tags[:,0]];forward=self.start+e[:,0]
        for i in range(1,S):
            step=self.transitions[tags[:,i-1],tags[:,i]]+e[row,i,tags[:,i]]
            score=score+step*mask[:,i]
            update=torch.logsumexp(forward[:,:,None]+self.transitions[None]+e[:,i,None,:],dim=1)
            forward=torch.where(mask[:,i,None],update,forward)
        last=tags.gather(1,(mask.sum(1)-1)[:,None]).squeeze(1)
        score=score+self.end[last];partition=torch.logsumexp(forward+self.end,1)
        return ((partition-score)/mask.sum(1)).mean()
    def arrays(self):return [self.start.detach().float().cpu().numpy(),self.transitions.detach().float().cpu().numpy(),self.end.detach().float().cpu().numpy()]
    @staticmethod
    def decode(emissions,arrays,transition_scale=1.,bias=None):
        n,c=emissions.shape
        if not n:return []
        start,transitions,end=arrays;values=emissions.astype(np.float32)
        if bias is not None:values=values+np.asarray(bias,dtype=np.float32)
        score=values[0]+start*transition_scale;back=[]
        for row in values[1:]:
            options=score[:,None]+transitions*transition_scale;previous=options.argmax(0)
            score=options[previous,np.arange(c)]+row;back.append(previous)
        tag=int((score+end*transition_scale).argmax());tags=[tag]
        for previous in reversed(back):tag=int(previous[tag]);tags.append(tag)
        return tags[::-1]

class RRModel(nn.Module):
    def __init__(self,backbone,cfg,label_count):
        super().__init__();self.backbone=backbone;self.cfg=cfg;d=backbone.config.dim
        method=cfg['method'];backbone.requires_grad_(method in ('full','kd'))
        if method in ('lora','rslora','lorma'):
            targets={'q','v','out','gate','up','down'}
            replacements=[(name,module) for name,module in backbone.named_modules() if isinstance(module,nn.Linear) and name.rsplit('.',1)[-1] in targets]
            for name,module in replacements:
                parent,attribute=name.rsplit('.',1)
                setattr(backbone.get_submodule(parent),attribute,LowRank(module,cfg.get('rank',8),cfg.get('alpha',16),method))
        elif method=='ia3':
            replacements=[(name,module) for name,module in backbone.named_modules() if isinstance(module,nn.Linear) and name.rsplit('.',1)[-1] in ('k','v','down')]
            for name,module in replacements:
                parent,attribute=name.rsplit('.',1);setattr(backbone.get_submodule(parent),attribute,IA3(module,attribute=='down'))
        elif method not in ('full','kd'):raise ValueError(method)
        self.attention_modules=[m for m in backbone.modules() if isinstance(m,Attention)]
        for module in self.attention_modules:
            module.forward=types.MethodType(attention_forward,module);module.dropout=cfg.get('encoder_dropout',.05)
            module._rr_bidirectional=cfg.get('token_bidirectional',False)
        for module in backbone.modules():
            if isinstance(module,nn.Dropout):module.p=cfg.get('encoder_dropout',.05)
        backbone.config.gradient_checkpointing=False
        if cfg.get('freeze_backbone'):backbone.requires_grad_(False)
        h=cfg.get('head_dim',192);self.pool_dim=3*d
        self.domain=nn.Embedding(2,8);self.projection=nn.Sequential(nn.LayerNorm(3*d+14),nn.Linear(3*d+14,h),nn.GELU(),nn.Dropout(cfg.get('dropout',.25)))
        self.head_type=cfg.get('head_type','bilstm')
        if self.head_type=='transformer':
            layer=nn.TransformerEncoderLayer(h,4,4*h,dropout=cfg.get('dropout',.25),activation='gelu',batch_first=True,norm_first=True)
            self.context=nn.TransformerEncoder(layer,2,enable_nested_tensor=False);width=h
        else:
            self.context=nn.LSTM(h,h//2,num_layers=2,batch_first=True,bidirectional=True,dropout=cfg.get('dropout',.25));width=h
        self.output_norm=nn.LayerNorm(width);self.drop=nn.Dropout(cfg.get('dropout',.25))
        self.classifier=nn.Linear(width,label_count);self.shift=nn.Linear(width*3,1);self.crf=CRF(label_count)
    def pool_chunk(self,ids,valid):
        for module in self.attention_modules:module._rr_valid=valid
        hidden=self.backbone(ids,source_ids=ids) if self.backbone.config.architecture=='encoder_decoder' else self.backbone(ids)
        lengths=valid.sum(1);mask=valid[:,:,None]
        mean=(hidden.float()*mask).sum(1)/lengths[:,None]
        maximum=hidden.float().masked_fill(~mask,-1e9).max(1).values
        last=hidden[torch.arange(len(ids),device=ids.device),lengths-1].float()
        return torch.cat((mean,maximum,last),-1)
    def encode(self,batch):
        if batch['features'] is not None:return batch['features']
        pieces=[];step=self.cfg.get('chunk_batch',8)
        for first in range(0,len(batch['ids']),step):
            ids=batch['ids'][first:first+step];valid=batch['valid'][first:first+step]
            if self.training and self.cfg.get('encoder_checkpointing',True) and any(p.requires_grad for p in self.backbone.parameters()):
                value=checkpoint(self.pool_chunk,ids,valid,use_reentrant=False)
            else:value=self.pool_chunk(ids,valid)
            pieces.append(value)
        vectors=torch.cat(pieces);B,S=batch['mask'].shape
        pooled=vectors.new_zeros(B*S,self.pool_dim);counts=vectors.new_zeros(B*S,1)
        weights=batch['chunk_lengths'][:,None].to(vectors.dtype)
        pooled=pooled.index_add(0,batch['owners'],vectors*weights);counts=counts.index_add(0,batch['owners'],weights)
        return (pooled/counts.clamp_min(1)).reshape(B,S,-1)
    def forward(self,batch):
        pooled=self.encode(batch);B,S,_=pooled.shape
        domain=self.domain(batch['domain'])[:,None].expand(-1,S,-1)
        hidden=self.projection(torch.cat((pooled,batch['position'],domain),-1))
        if self.head_type=='transformer':hidden=self.context(hidden,src_key_padding_mask=~batch['mask'])
        else:
            packed=nn.utils.rnn.pack_padded_sequence(hidden,batch['mask'].sum(1).cpu(),batch_first=True,enforce_sorted=False)
            hidden,_=self.context(packed);hidden,_=nn.utils.rnn.pad_packed_sequence(hidden,batch_first=True,total_length=S)
        hidden=self.drop(self.output_norm(hidden));emissions=self.classifier(hidden)
        pairs=torch.cat((hidden[:,1:],hidden[:,:-1],(hidden[:,1:]-hidden[:,:-1]).abs()),-1)
        return {'logits':emissions,'shift':self.shift(pairs).squeeze(-1)}

def build_model(cfg,restore=None):
    parent=torch.load(cfg['parent'],map_location='cpu',weights_only=False);config=ModelConfig(**parent['config'])
    state=parent['model'];retained=None
    if cfg['method']=='kd':
        count=cfg.get('student_layers',4);original=config.layers
        if config.architecture!='decoder' or not 1<=count<original:raise ValueError('Student must retain fewer original decoder blocks.')
        retained=[round(i*(original-1)/(count-1)) for i in range(count)] if count>1 else [original-1]
        state={key:value for key,value in state.items() if not key.startswith('blocks.')}
        for new,old in enumerate(retained):
            prefix=f'blocks.{old}.'
            state.update({f'blocks.{new}.'+k[len(prefix):]:v for k,v in parent['model'].items() if k.startswith(prefix)})
        config.layers=count
    backbone=LegalLM(config);backbone.load_state_dict(state,strict=True);model=RRModel(backbone,cfg,cfg['label_count'])
    if restore:
        saved=torch.load(restore,map_location='cpu',weights_only=False)
        missing,unexpected=model.load_state_dict(saved['model'],strict=not saved.get('delta_only',False))
        trainable={name for name,p in model.named_parameters() if p.requires_grad}
        if unexpected or (set(missing)&trainable):raise ValueError('Incomplete RR checkpoint.')
    elif cfg.get('head_init'):
        trained=torch.load(cfg['head_init'],map_location='cpu',weights_only=False)['model']
        head={name:value for name,value in trained.items() if not name.startswith('backbone.')}
        missing,unexpected=model.load_state_dict(head,strict=False)
        if unexpected or any(not name.startswith('backbone.') for name in missing):raise ValueError('Head warmup checkpoint is incompatible.')
    return model,{**parent['provenance'],'retained_parent_blocks':retained,'backbone_config':vars(config)}
def parameter_report(model):return {'total':sum(p.numel() for p in model.parameters()),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
    'backbone':sum(p.numel() for p in model.backbone.parameters())}
def loss_function(model,result,batch,weights,cfg):
    e=result['logits'].float();targets=batch['targets'];mask=batch['mask']
    crf=model.crf.nll(e,targets,mask)
    ce=F.cross_entropy(e.flatten(0,1),targets.flatten(),weight=weights,ignore_index=-100,label_smoothing=.03)
    shift_mask=mask[:,1:]&mask[:,:-1]
    if bool(shift_mask.any()):
        y=(targets[:,1:]!=targets[:,:-1]).float()
        shift=F.binary_cross_entropy_with_logits(result['shift'].float()[shift_mask],y[shift_mask])
    else:shift=result['shift'].float().sum()*0
    supervised=crf+cfg.get('ce_weight',.5)*ce+cfg.get('shift_weight',.15)*shift
    if batch['teacher'] is not None:
        T=cfg.get('temperature',2.);student=e[mask];teacher=batch['teacher'][mask]
        kd=F.kl_div(F.log_softmax(student/T,-1),F.softmax(teacher.float()/T,-1),reduction='batchmean')*T*T
        return (1-cfg.get('kd_alpha',.5))*supervised+cfg.get('kd_alpha',.5)*kd
    return supervised
