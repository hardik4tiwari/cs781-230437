"""Phase-1-only LSI: label-wise attention, genuine adaptation, Bernoulli distillation."""
import gc,math,shutil
from pathlib import Path
from dataclasses import asdict
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from legal_models import LegalLM,ModelConfig
from lsi_core import JudgmentModel,LoRMA,Bottleneck,parameter_report
from lsi_data import read_json,atomic_json,atomic_torch,digest,file_hash

ARCHITECTURES=('decoder','encoder_decoder','local_gqa','hybrid')

def restore_phase1(root,output,overrides=None,allow_tiny=False):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True);overrides=overrides or {}
    tokenizer=root/'tokenizer.json'
    if not tokenizer.is_file():raise FileNotFoundError(f'Missing {tokenizer}. Attach capstone_phase1 directly.')
    token_hash=file_hash(tokenizer);destination=output/'tokenizer.json'
    if destination.exists() and file_hash(destination)!=token_hash:raise ValueError('Tokenizer changed. Use a new PROJECT.')
    if tokenizer.resolve()!=destination.resolve():shutil.copy2(tokenizer,destination)
    parents={};audit={}
    for architecture in ARCHITECTURES:
        source=Path(overrides.get(architecture,root/'runs'/f'main_{architecture}'/'best_model.pt'))
        if not source.is_file():raise FileNotFoundError(f'Missing Phase-1 parent: {source}')
        sha=file_hash(source);target=output/f'{architecture}.pt'
        if target.exists():
            saved=torch.load(target,map_location='cpu',weights_only=False)
            if saved['provenance']['phase1_sha256']!=sha or saved['provenance']['tokenizer_sha256']!=token_hash:raise ValueError('Parent changed. Use a new PROJECT.')
            count=saved['provenance']['parent_parameters'];del saved
        else:
            saved=torch.load(source,map_location='cpu',weights_only=False)
            if saved.get('training_origin')!='random_initialization_official_phase1_only':raise ValueError('Every experiment must start from your original Phase-1 checkpoint; Phase-2/3/4 task weights are rejected.')
            if saved['tokenizer_sha256']!=token_hash:raise ValueError('Phase-1 checkpoint/tokenizer hash mismatch.')
            config=ModelConfig(**saved['model_config'])
            if config.architecture!=architecture:raise ValueError('Phase-1 architecture mismatch.')
            model=LegalLM(config);model.load_state_dict(saved['model'],strict=True)
            count=sum(p.numel() for p in model.parameters())
            if not allow_tiny and not 50_000_000<=count<=300_000_000:raise ValueError('Phase-1 parent must have 50M–300M parameters.')
            provenance={'phase1_sha256':sha,'tokenizer_sha256':token_hash,'parent_parameters':count,
                'source':str(source),'training_origin':saved['training_origin']}
            atomic_torch(target,{'config':asdict(config),'model':model.state_dict(),'provenance':provenance})
            del model,saved;gc.collect()
        parents[architecture]=str(target);audit[architecture]={'source':str(source),'sha256':sha,'parameters':count}
    atomic_json(output/'input_manifest.json',{'phase1_parents':audit,'tokenizer_sha256':token_hash})
    return parents,str(destination)

def soft_prompt_hook(module,args,value):
    count=module.soft_prompt.shape[0]
    return torch.cat((module.soft_prompt[None].expand(value.shape[0],-1,-1),value[:,count:]),dim=1)

class StatuteModel(JudgmentModel):
    def __init__(self,backbone,cfg):
        setup_cfg={**cfg,'method':'partial','unfrozen_layers':0} if cfg['method']=='prompt' else cfg
        super().__init__(backbone,setup_cfg);self.cfg=cfg
        if cfg['method']=='prompt':
            backbone.requires_grad_(False);count=cfg.get('prompt_tokens',16)
            initial=backbone.embedding.weight[:count].detach().clone()
            backbone.embedding.soft_prompt=nn.Parameter(initial,requires_grad=not cfg.get('head_only',False))
            backbone.embedding.register_forward_hook(soft_prompt_hook)
        h=cfg.get('head_dim',192);labels=cfg['label_count']
        del self.attention,self.classifier
        self.label_queries=nn.Parameter(torch.empty(labels,h));self.label_output=nn.Parameter(torch.empty(labels,h))
        nn.init.normal_(self.label_queries,std=.02);nn.init.normal_(self.label_output,std=.02)
        self.label_bias=nn.Parameter(torch.zeros(labels))
        self.global_classifier=nn.Linear(2*h,labels,bias=False)
        self.chunk_classifier=nn.Linear(h,labels,bias=False)
        self.global_gate=nn.Parameter(torch.tensor(-2.));self.mil_gate=nn.Parameter(torch.tensor(-2.))
        if cfg.get('label_prior') is not None:
            prior=torch.as_tensor(cfg['label_prior']).clamp(.001,.999)
            with torch.no_grad():self.label_bias.copy_(torch.logit(prior))
    def pool_tokens(self,ids,valid):
        if self.cfg['method']!='prompt':return super().pool_tokens(ids,valid)
        count=self.cfg.get('prompt_tokens',16)
        if ids.shape[1]+count>self.backbone.config.max_positions:raise ValueError('Soft prompt plus chunk exceeds the Phase-1 position limit. Reduce chunk_tokens.')
        extended_ids=torch.cat((ids.new_zeros(ids.shape[0],count),ids),1)
        extended_valid=torch.cat((valid.new_ones(valid.shape[0],count),valid),1)
        all_valid=bool(extended_valid.all())
        for module in self.attention_modules:module._cj_valid=extended_valid;module._cj_all_valid=all_valid
        hidden=self.backbone(extended_ids)[:,count:]
        mean=(hidden.float()*valid[:,:,None]).sum(1)/valid.sum(1)[:,None]
        last=hidden[torch.arange(len(ids),device=ids.device),valid.sum(1)-1].float()
        return torch.cat((mean,last),-1)
    def forward(self,batch):
        x=self.project(torch.cat((self.chunk_features(batch),batch['position']),-1));mask=batch['mask'];B,S,_=x.shape
        if self.head_type=='transformer':x=self.context(x,src_key_padding_mask=~mask)
        else:
            packed=nn.utils.rnn.pack_padded_sequence(x,mask.sum(1).cpu(),batch_first=True,enforce_sorted=False)
            x,_=self.context(packed);x,_=nn.utils.rnn.pad_packed_sequence(x,batch_first=True,total_length=S)
        x=self.norm(x)
        scores=torch.einsum('bsh,lh->bsl',x,self.label_queries).float().masked_fill(~mask[:,:,None],-1e9)
        attention=torch.softmax(scores,dim=1)
        pooled=torch.einsum('bsl,bsh->blh',attention,x.float())
        logits=(pooled*self.label_output[None]).sum(-1)+self.label_bias
        mean=(x*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
        maximum=x.float().masked_fill(~mask[:,:,None],-1e9).max(1).values
        mil=self.chunk_classifier(x).float().masked_fill(~mask[:,:,None],-1e9).max(1).values
        logits=logits+self.global_gate.sigmoid()*self.global_classifier(torch.cat((mean,maximum),-1))+self.mil_gate.sigmoid()*mil
        return {'logits':logits,'attention':attention.mean(-1)}

def build_model(cfg,checkpoint_path=None):
    parent=torch.load(cfg['parent'],map_location='cpu',weights_only=False)
    if parent['provenance'].get('training_origin')!='random_initialization_official_phase1_only':raise ValueError('Invalid Phase-1 lineage.')
    config=ModelConfig(**parent['config']);state=parent['model'];retained=None
    if cfg['method']=='kd':
        n=cfg.get('student_layers',4);original=config.layers
        if config.architecture!='decoder' or not 1<=n<original:raise ValueError('KD student must retain fewer decoder blocks.')
        retained=[round(i*(original-1)/(n-1)) for i in range(n)] if n>1 else [original-1]
        state={k:v for k,v in state.items() if not k.startswith('blocks.')}
        for new,old in enumerate(retained):
            prefix=f'blocks.{old}.';state.update({f'blocks.{new}.'+k[len(prefix):]:v for k,v in parent['model'].items() if k.startswith(prefix)})
        config.layers=n
    backbone=LegalLM(config);backbone.load_state_dict(state,strict=True);model=StatuteModel(backbone,cfg)
    if cfg.get('backbone_init'):
        saved=torch.load(cfg['backbone_init'],map_location='cpu',weights_only=False)
        if saved.get('training_origin')!='own_phase1_phase5_lsi' or saved['config']['architecture']!=cfg['architecture'] or saved['config']['method']!=cfg['method']:
            raise ValueError('Encoder-stage initialization has wrong task/family/method.')
        if saved['provenance']['phase1_sha256']!=parent['provenance']['phase1_sha256']:raise ValueError('Phase-1 lineage changed.')
        learned={k:v for k,v in saved['model'].items() if k.startswith('backbone.')}
        if not learned:raise ValueError('Adapted checkpoint has no encoder updates.')
        _,unexpected=model.load_state_dict(learned,strict=False)
        if unexpected:raise ValueError('Adapted encoder does not match model skeleton.')
    path=checkpoint_path or cfg.get('head_init')
    if path:
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved.get('training_origin')!='own_phase1_phase5_lsi' or saved['config']['label_count']!=cfg['label_count']:raise ValueError('Head initialization has wrong task/label dimension.')
        if saved['provenance']['phase1_sha256']!=parent['provenance']['phase1_sha256']:raise ValueError('Head/checkpoint belongs to a different Phase-1 parent.')
        weights=saved['model'] if checkpoint_path else {k:v for k,v in saved['model'].items() if not k.startswith('backbone.')}
        missing,unexpected=model.load_state_dict(weights,strict=False)
        required={k for k,p in model.named_parameters() if p.requires_grad} if checkpoint_path else {k for k in model.state_dict() if not k.startswith('backbone.')}
        if unexpected or set(missing)&required:raise ValueError('Incomplete LSI checkpoint.')
    return model,{**parent['provenance'],'retained_parent_blocks':retained,'backbone_config':asdict(config)}

def classification_loss(logits,batch,cfg,weights):
    if batch['target'] is None:raise ValueError('Test labels cannot enter the training loss.')
    z=logits.float();y=batch['target'].float()
    if cfg.get('loss','asl')=='bce':hard=F.binary_cross_entropy_with_logits(z,y,pos_weight=weights)
    else:
        positive=z.sigmoid();negative=(1-positive+cfg.get('asl_clip',.05)).clamp(max=1)
        log_positive=F.logsigmoid(z);log_negative=negative.clamp_min(1e-8).log()
        focus_positive=(1-positive).pow(cfg.get('gamma_positive',0.))
        focus_negative=(1-negative).pow(cfg.get('gamma_negative',4.))
        hard=-(y*focus_positive*log_positive+(1-y)*focus_negative*log_negative).mean()
    if cfg['method']!='kd':return hard,{'hard':float(hard.detach())}
    if batch['teacher'] is None or batch['teacher'].shape!=z.shape:raise ValueError('KD needs real Phase-5 per-label teacher logits.')
    tau=cfg.get('temperature',2.);alpha=cfg.get('kd_alpha',.5)
    teacher=(batch['teacher'].detach().float()/tau).sigmoid().clamp(1e-6,1-1e-6)
    # Independent Bernoulli KL, including both positive and negative outcomes. Never label softmax.
    kl=(teacher*(teacher.log()-F.logsigmoid(z/tau))+(1-teacher)*((1-teacher).log()-F.logsigmoid(-z/tau))).mean()*tau*tau
    return (1-alpha)*hard+alpha*kl,{'hard':float(hard.detach()),'kd':float(kl.detach())}
