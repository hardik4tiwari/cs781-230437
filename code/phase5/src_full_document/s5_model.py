"""One jointly trained own-decoder model: token LWAN, document BiGRU, lexical residual."""
import math, contextlib
import torch
from torch import nn
from torch.nn import functional as F
from legal_models import LegalLM,ModelConfig
from s5_data import chunk_tensors

class FullDocumentModel(nn.Module):
    def __init__(self,foundation_config,cfg):
        super().__init__();self.cfg=cfg;self.backbone=LegalLM(ModelConfig(**foundation_config))
        if self.backbone.config.architecture!='decoder':raise ValueError('This candidate uses only main_decoder.')
        self.backbone.config.gradient_checkpointing=False
        # Deterministic chunk encoder is required for exact recomputation. Dropout remains in the document head.
        for m in self.backbone.modules():
            if isinstance(m,nn.Dropout):m.p=0.
            if hasattr(m,'dropout') and isinstance(m.dropout,(int,float)):m.dropout=0.
        d=self.backbone.config.dim;h=cfg['token_dim'];g=cfg['document_dim'];C=cfg['label_count']
        self.C,self.h,self.d=C,h,d
        self.token_project=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,h),nn.Tanh())
        self.token_query=nn.Parameter(torch.empty(C,h));nn.init.normal_(self.token_query,std=.1)
        self.token_output=nn.Parameter(torch.empty(C,h));nn.init.normal_(self.token_output,std=.02)
        self.chunk_project=nn.Sequential(nn.LayerNorm(2*d+3),nn.Linear(2*d+3,g),nn.GELU())
        self.context=nn.GRU(g,g//2,bidirectional=True,batch_first=True)
        self.context_norm=nn.LayerNorm(g);self.dropout=nn.Dropout(cfg['dropout'])
        self.document_query=nn.Parameter(torch.empty(C,g));nn.init.normal_(self.document_query,std=.1)
        self.document_output=nn.Parameter(torch.empty(C,g));nn.init.normal_(self.document_output,std=.02)
        self.bias=nn.Parameter(torch.as_tensor(cfg['initial_bias'],dtype=torch.float32))
        self.lexical=nn.EmbeddingBag(cfg['sparse_dim'],C,mode='sum',include_last_offset=True)
        nn.init.zeros_(self.lexical.weight)

    def encode(self,ids,valid):
        # Right-padded causal decoder: valid tokens never attend to padding that follows them.
        hidden=self.backbone(ids)
        value=self.token_project(hidden).float()
        # Explicit FP32 scope: autocast can downcast einsum even when its inputs are float().
        with torch.autocast(device_type=ids.device.type,enabled=False):
            score=torch.einsum('nth,lh->ntl',value.float(),self.token_query.float())/math.sqrt(self.h)
            score=score.masked_fill(~valid[:,:,None],float('-inf'))
            logz=score.logsumexp(1)
            weights=score.softmax(1);pooled=torch.einsum('ntl,nth->nlh',weights,value.float())
        mean=(hidden.float()*valid[:,:,None]).sum(1)/valid.sum(1)[:,None]
        last=hidden[torch.arange(len(ids),device=ids.device),valid.sum(1)-1].float()
        return torch.cat((pooled.flatten(1),logz,mean,last),1)

    def classify(self,features,batch):
        C,h=self.C,self.h
        pooled=features[:,:C*h].reshape(-1,C,h);logz=features[:,C*h:C*h+C]
        global_chunks=features[:,C*h+C:]
        counts=batch['counts'];S=max(counts);B=len(counts)
        global_pad=features.new_zeros(B,S,2*self.d+3);mask=torch.zeros(B,S,dtype=torch.bool,device=features.device)
        token_vectors=[];cursor=0
        for b,n in enumerate(counts):
            w=logz[cursor:cursor+n].softmax(0)
            token_vectors.append((pooled[cursor:cursor+n]*w[:,:,None]).sum(0))
            global_pad[b,:n]=torch.cat((global_chunks[cursor:cursor+n],batch['position'][cursor:cursor+n]),1)
            mask[b,:n]=True;cursor+=n
        token_vectors=self.dropout(torch.stack(token_vectors))
        token_logits=(token_vectors*self.token_output[None]).sum(-1)
        x=self.chunk_project(global_pad)
        packed=nn.utils.rnn.pack_padded_sequence(x,torch.as_tensor(counts),batch_first=True,enforce_sorted=False)
        packed,_=self.context(packed)
        x,_=nn.utils.rnn.pad_packed_sequence(packed,batch_first=True,total_length=S)
        x=self.dropout(self.context_norm(x)).float()
        with torch.autocast(device_type=features.device.type,enabled=False):
            a=torch.einsum('bsh,lh->bsl',x.float(),self.document_query.float())/math.sqrt(x.shape[-1])
            a=a.masked_fill(~mask[:,:,None],float('-inf')).softmax(1)
            representation=torch.einsum('bsl,bsh->blh',a,x.float())
            neural=token_logits.float()+(representation*self.document_output[None].float()).sum(-1)+self.bias.float()
        # EmbeddingBag implements a sparse linear residual, learned in the SAME loss/checkpoint.
        # FP32 here avoids unsupported FP16 per_sample_weights combinations on older CUDA builds.
        with torch.autocast(device_type=features.device.type,enabled=False):
            values=F.dropout(batch['sparse_values'],p=self.cfg.get('lexical_dropout',.15),training=self.training)
            lexical=self.lexical(batch['sparse_ids'],batch['sparse_offsets'],per_sample_weights=values)
        return neural+lexical,neural

def amp(device):
    return torch.autocast('cuda',dtype=torch.float16) if device.type=='cuda' else contextlib.nullcontext()

def encode_all(model,batch,device,chunk_batch):
    return torch.cat([model.encode(*chunk_tensors(batch['chunks'][a:a+chunk_batch],device))
        for a in range(0,len(batch['chunks']),chunk_batch)],0)

def loss_fn(logits,neural,target,pos_weight,normalizer):
    if target is None:raise ValueError('Test data cannot enter a fitting loss.')
    main=(F.binary_cross_entropy_with_logits(logits.float(),target,pos_weight=pos_weight,reduction='none')/normalizer).mean()
    auxiliary=(F.binary_cross_entropy_with_logits(neural.float(),target,pos_weight=pos_weight,reduction='none')/normalizer).mean()
    return main+.2*auxiliary

def cached_backward(model,batch,device,scaler,pos_weight,normalizer,divisor=1):
    """Chain-rule VJP: exact document gradient, one chunk microbatch activation graph at a time."""
    step=model.cfg['chunk_batch']
    with torch.no_grad(),amp(device):cached=encode_all(model,batch,device,step)
    cached=cached.detach().requires_grad_(True)
    with amp(device):
        logits,neural=model.classify(cached,batch)
        loss=loss_fn(logits,neural,batch['target'],pos_weight,normalizer)
    scaler.scale(loss/divisor).backward()
    representation_gradient=cached.grad.detach()
    # The representation gradient already contains GradScaler's scale. Do not scale it twice.
    for a in range(0,len(batch['chunks']),step):
        with amp(device):value=model.encode(*chunk_tensors(batch['chunks'][a:a+step],device))
        torch.autograd.backward(value,representation_gradient[a:a+len(value)])
    return float(loss.detach())

