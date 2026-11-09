"""Official input discovery and submission validation from Phase 6."""
import os,shutil,zipfile
from pathlib import Path
import torch
from s6_data import FILES,atomic_json,read_json,file_hash,digest

def find_phase1(explicit=None):
    if explicit:
        p=Path(explicit)
        if not (p/'tokenizer.json').is_file():raise FileNotFoundError(f'No tokenizer.json under {p}')
        return p
    preferred=Path('/kaggle/input/datasets/hardik4tiwari/phase-1-outputs/capstone_phase1')
    candidates=[preferred] if preferred.is_dir() else sorted({p.parent for p in Path('/kaggle/input').rglob('tokenizer.json') if p.parent.name=='capstone_phase1'})
    if len(candidates)!=1:raise FileNotFoundError('Set PHASE1_ROOT to the unique original capstone_phase1 directory. Attach the Phase-1 outputs dataset.')
    return candidates[0]

def obtain_data(project,local_root=None,revision=None):
    if not local_root:
        candidates=sorted({p for p in Path('/kaggle/input').rglob('Phase_6_Summ') if (p/FILES[0]).is_file() and (p/FILES[1]).is_file()})
        if len(candidates)==1:local_root=candidates[0]
        elif len(candidates)>1:raise ValueError('Multiple Phase-6 datasets: set DATA_ROOT explicitly.')
    if local_root:
        source=Path(local_root)
        if (source/'Phase_6_Summ').is_dir():source/='Phase_6_Summ'
        if any(not (source/f).is_file() for f in FILES):raise FileNotFoundError('Need Phase_6_Summ/Train/train.json and Test/test.json.')
        atomic_json(Path(project)/'data_source.json',{'kind':'attached_official_phase6','root':str(source),'sha256':{n:file_hash(source/n) for n in FILES}})
        return str(source)
    from huggingface_hub import HfApi,hf_hub_download
    token=os.environ.get('HF_TOKEN')
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            token=UserSecretsClient().get_secret('HF_TOKEN')
        except Exception:token=None
    root=Path(project)/'data';manifest=root/'source.json';old=read_json(manifest) if manifest.exists() else None
    repo='Exploration-Lab/CS-781-Capstone'
    try:
        commit=old['revision'] if old else HfApi(token=token).dataset_info(repo,revision=revision).sha
        if old and revision and revision!=commit:raise ValueError('Pinned revision differs. Choose a new PROJECT.')
        for name in FILES:
            target=root/name
            if target.exists() and old and old.get('sha256',{}).get(name)==file_hash(target):continue
            source=hf_hub_download(repo,'Phase_6_Summ/'+name,repo_type='dataset',revision=commit,token=token)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        atomic_json(manifest,{'repo':repo,'revision':commit,'sha256':{n:file_hash(root/n) for n in FILES}})
    except Exception as e:raise RuntimeError('Attach official Phase_6_Summ data, or accept its Hugging Face terms, enable Kaggle Internet and add an HF_TOKEN secret.') from e
    return str(root)

def pack_prediction(project,name,pred_path,cache_path):
    cache=torch.load(cache_path,map_location='cpu',weights_only=False);expected={r['id'] for r in cache['rows'] if r['split']=='test'}
    pred=read_json(pred_path)
    if not isinstance(pred,list) or len(pred)!=len(expected):raise ValueError('Wrong number of test predictions.')
    if len({r['id'] for r in pred})!=len(pred) or {r['id'] for r in pred}!=expected:raise ValueError('Submission IDs mismatch.')
    if any(set(r)!={'id','summary'} or not isinstance(r['summary'],str) or not r['summary'].strip() for r in pred):raise ValueError('Invalid summary schema.')
    folder=Path(project)/'submissions'/name;atomic_json(folder/'prediction.json',pred);archive=folder.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:z.write(folder/'prediction.json','prediction.json')
    with zipfile.ZipFile(archive) as z:
        if z.namelist()!=['prediction.json']:raise ValueError('Incorrect ZIP root.')
    return {'name':name,'submission_zip':str(archive),'prediction_sha256':digest(sorted(pred,key=lambda r:r['id'])),'test_documents':len(pred)}
