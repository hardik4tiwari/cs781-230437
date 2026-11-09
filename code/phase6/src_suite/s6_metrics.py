"""Official-equivalent metrics; pretrained BERT is used for detached text evaluation only."""
import torch
_scorer=None
def release():
    global _scorer
    import gc
    _scorer=None;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
def score(predictions,references,device='cpu'):
    global _scorer
    from rouge_score import rouge_scorer,tokenizers
    from bert_score import BERTScorer
    from evaluate_phase6_official import lcs
    if len(predictions)!=len(references) or not predictions:raise ValueError('Metric input lengths differ/empty.')
    if _scorer is None or str(_scorer.device)!=str(device):
        _scorer=BERTScorer(lang='en',batch_size=8,device=str(device),rescale_with_baseline=False,idf=False)
    # The provided official evaluator uses bert_score.score(..., lang='en'), without rescaling or IDF.
    with torch.inference_mode():_,_,f=_scorer.score(predictions,references,batch_size=8)
    rs=rouge_scorer.RougeScorer(['rouge1'],use_stemmer=True);tok=tokenizers.DefaultTokenizer(use_stemmer=True)
    r1=[];rl=[]
    for p,r in zip(predictions,references):
        r1.append(rs.score(r,p)['rouge1'].fmeasure);pp=tok.tokenize(p);rr=tok.tokenize(r)
        rl.append(2*lcs(pp,rr)/max(1,len(pp)+len(rr)) if pp or rr else 1.)
    return {'bertscore_f1':float(f.mean()),'rouge1_f1':sum(r1)/len(r1),'rougeL_f1':sum(rl)/len(rl),
            'documents':len(predictions),'bertscore_hash':_scorer.hash,'metric_model':'evaluation only; never part of generator/optimizer'}
