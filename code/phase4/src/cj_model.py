"""Own Phase-3 checkpoint restoration and full-document CJPE classifiers."""
import gc,math,shutil,types
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from legal_models import LegalLM,ModelConfig,Attention
from cj_data import atomic_json,atomic_torch,read_json,file_hash,digest

SOURCES={'decoder':'01_decoder_full','encoder_decoder':'02_encoder_decoder_full','local_gqa':'03_local_gqa_lora','hybrid':'04_hybrid_rslora'}
def fold_phase3(saved,parent):
    cfg=saved['config'];kind=cfg['method']
    if kind not in ('full','lora','rslora','lorma','ia3'):raise ValueError('Choose a full-sized, foldable Phase-3 source, not its KD student.')
    state={k[len('backbone.'):]:v.float() for k,v in saved['model'].items() if k.startswith('backbone.')}
    if not state:raise ValueError('Phase-3 checkpoint has no learned backbone weights.')
    dense={k:v.float().clone() for k,v in parent['model'].items()};covered=set()
    for key,value in state.items():
        if key in dense:dense[key]=value;covered.add(key)
        elif key.endswith('.base.weight'):
            dest=key[:-len('.base.weight')]+'.weight'
            if dest not in dense:raise ValueError(f'Unexpected wrapped parent weight: {key}')
            dense[dest]=value;covered.add(dest)
        elif not key.endswith(('.A.weight','.B.weight','.scale')):raise ValueError(f'Unrecognized Phase-3 backbone field: {key}')
    if kind in ('lora','rslora','lorma'):
        rank=cfg.get('rank',8);scale=cfg.get('alpha',16.)/(math.sqrt(rank) if kind=='rslora' else rank)
        prefixes={key[:-len('.A.weight')] for key in state if key.endswith('.A.weight')}
        if not prefixes:raise ValueError('Phase-3 low-rank checkpoint contains no adaptation factors.')
        for name in prefixes:
            A=state[name+'.A.weight'];B=state[name+'.B.weight'];W=dense[name+'.weight']
            dense[name+'.weight']=W+scale*(B@(A@W) if kind=='lorma' else B@A)
    elif kind=='ia3':
        for key,scale in state.items():
            if not key.endswith('.scale'):continue
            name=key[:-len('.scale')];W=dense[name+'.weight']
            dense[name+'.weight']=W*(scale[None,:] if name.rsplit('.',1)[-1]=='down' else scale[:,None])
    if not saved.get('delta_only',False) and covered!=set(dense):raise ValueError('Dense Phase-3 checkpoint omits foundation weights.')
    return dense
def restore_phase3(root,output,allow_selection=True,overrides=None,allow_tiny=False):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True);overrides=overrides or {}
    if not root.is_dir():raise FileNotFoundError(f'Attach phase-3-outputs and set PHASE3_ROOT to its capstone_phase3 folder: {root}')
    if set(overrides)-set(SOURCES):raise ValueError('Unknown architecture in PARENT_OVERRIDES.')
    source_tokenizer=root/'parents/tokenizer.json'
    if not source_tokenizer.is_file():raise FileNotFoundError(f'Keep the complete Phase-3 outputs, including {source_tokenizer}')
    tokenizer_hash=file_hash(source_tokenizer);tokenizer=output/'tokenizer.json'
    if tokenizer.exists() and file_hash(tokenizer)!=tokenizer_hash:raise ValueError('Tokenizer changed; use a new PROJECT.')
    if not tokenizer.exists():shutil.copy2(source_tokenizer,tokenizer)
    parents={};audit={}
    for architecture,name in SOURCES.items():
        refit=root/'runs'/name/'refit/best.pt';selection=root/'runs'/name/'selection/best.pt'
        source=Path(overrides[architecture]) if architecture in overrides else refit if refit.is_file() else selection if allow_selection else refit
        base=root/'parents'/f'{architecture}.pt'
        if not source.resolve().is_relative_to(root.resolve()):raise ValueError('Parent overrides must stay inside PHASE3_ROOT.')
        if not source.is_file() or not base.is_file():raise FileNotFoundError(f'Need both {source} and {base}. Keep parents/ with compact checkpoints.')
        sha=file_hash(source);base_sha=file_hash(base);target=output/f'{architecture}.pt'
        if target.exists():
            previous=torch.load(target,map_location='cpu',weights_only=False)
            if previous['provenance']['phase3_sha256']!=sha or previous['provenance']['phase3_base_sha256']!=base_sha:
                raise ValueError('A Phase-3 parent changed; use a new PROJECT.')
            del previous
        else:
            saved=torch.load(source,map_location='cpu',weights_only=False);parent=torch.load(base,map_location='cpu',weights_only=False)
            if saved.get('training_origin')!='own_phase1_phase2_phase3_rr' or saved['config']['architecture']!=architecture:
                raise ValueError(f'{source}: expected your own matching Phase-3 RR model.')
            provenance=saved['provenance'];origin=parent['provenance']
            for key in ('phase1_sha256','phase2_sha256','tokenizer_sha256'):
                if provenance.get(key)!=origin.get(key):raise ValueError(f'Phase-3/parent lineage mismatch: {key}')
            if origin['tokenizer_sha256']!=tokenizer_hash:raise ValueError('Tokenizer does not match the saved lineage.')
            config=parent['config'];state=fold_phase3(saved,parent);model=LegalLM(ModelConfig(**config));model.load_state_dict(state,strict=True)
            count=sum(p.numel() for p in model.parameters())
            if not allow_tiny and not 50_000_000<=count<=300_000_000:raise ValueError('Foundation parent must have 50M–300M parameters.')
            atomic_torch(target,{'model':state,'config':config,'provenance':{**origin,'phase3_sha256':sha,
                'phase3_base_sha256':base_sha,'phase3_checkpoint':str(source),'phase3_method':saved['config']['method'],
                'source_stage':saved['stage'],'source_parameters':count}})
            del saved,parent,state,model;gc.collect()
        parents[architecture]=str(target)
        audit[architecture]={'source':str(source),'sha256':sha,'base_sha256':base_sha,'selection_fallback':source==selection}
    atomic_json(output/'input_manifest.json',{'parents':audit,'tokenizer_sha256':tokenizer_hash,'input_root':str(root)})
    return parents,str(tokenizer)

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

def build_model(cfg,checkpoint_path=None):
    parent=torch.load(cfg['parent'],map_location='cpu',weights_only=False);config=ModelConfig(**parent['config']);state=parent['model'];retained=None
    if cfg['method']=='kd':
        n=cfg.get('student_layers',4);original=config.layers
        if config.architecture!='decoder' or not 1<=n<original:raise ValueError('Student must retain fewer decoder blocks than its own parent.')
        retained=[round(i*(original-1)/(n-1)) for i in range(n)] if n>1 else [original-1]
        state={k:v for k,v in state.items() if not k.startswith('blocks.')}
        for new,old in enumerate(retained):
            prefix=f'blocks.{old}.';state.update({f'blocks.{new}.'+k[len(prefix):]:v for k,v in parent['model'].items() if k.startswith(prefix)})
        config.layers=n
    backbone=LegalLM(config);backbone.load_state_dict(state,strict=True);model=JudgmentModel(backbone,cfg)
    if cfg.get('backbone_init'):
        source=torch.load(cfg['backbone_init'],map_location='cpu',weights_only=False)
        if source['config']['method']!=cfg['method'] or source['config']['architecture']!=cfg['architecture']:
            raise ValueError('Encoder-stage initialization has an incompatible adaptation method/family.')
        if source.get('training_origin')!='own_phase1_phase2_phase3_phase4_cjpe':raise ValueError('Expected your own Phase-4 encoder-stage checkpoint.')
        for key in ('phase3_sha256','phase3_base_sha256','tokenizer_sha256'):
            if source['provenance'].get(key)!=parent['provenance'].get(key):raise ValueError('Encoder-stage lineage changed.')
        if source.get('parent_sha256') and source['parent_sha256']!=file_hash(cfg['parent']):raise ValueError('Encoder-stage parent changed.')
        learned={k:v for k,v in source['model'].items() if k.startswith('backbone.')}
        if not learned:raise ValueError('Encoder-stage checkpoint has no adapted backbone weights.')
        missing,unexpected=model.load_state_dict(learned,strict=False)
        if unexpected:raise ValueError('Encoder-stage weights do not match the classifier skeleton.')
    path=checkpoint_path or cfg.get('head_init')
    if path:
        saved=torch.load(path,map_location='cpu',weights_only=False);weights=saved['model']
        if not checkpoint_path:weights={k:v for k,v in weights.items() if not k.startswith('backbone.')}
        missing,unexpected=model.load_state_dict(weights,strict=False)
        required={k for k,p in model.named_parameters() if p.requires_grad} if checkpoint_path else {k for k in model.state_dict() if not k.startswith('backbone.')}
        if unexpected or set(missing)&required:raise ValueError('Incomplete or incompatible CJPE checkpoint.')
    return model,{**parent['provenance'],'retained_parent_blocks':retained,'backbone_config':vars(config)}
def parameter_report(model):return {'total':sum(p.numel() for p in model.parameters()),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
    'backbone':sum(p.numel() for p in model.backbone.parameters())}
def classification_loss(logits,batch,cfg,weights):
    z=logits.float();ce=F.cross_entropy(z,batch['target'],weight=weights,label_smoothing=cfg.get('label_smoothing',.02))
    if cfg['method']!='kd':return ce,{'ce':float(ce.detach())}
    if batch['teacher'] is None:raise ValueError('KD needs real Phase-4 two-class teacher logits, never Phase-3 RR logits.')
    teacher=batch['teacher'].detach().float();tau=cfg.get('temperature',2.);alpha=cfg.get('kd_alpha',.5)
    kl=F.kl_div(F.log_softmax(z/tau,-1),F.softmax(teacher/tau,-1),reduction='batchmean')*tau*tau
    return (1-alpha)*ce+alpha*kl,{'ce':float(ce.detach()),'kd':float(kl.detach())}
