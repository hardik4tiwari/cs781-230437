"""Official Phase-5 input acquisition; no external pretrained weights."""
import os,shutil
from pathlib import Path
from lsi_data import read_json,atomic_json,file_hash,DATA_FILES
def obtain_data(project,local_root=None,revision=None):
    destination=Path(project)/'data';destination.mkdir(parents=True,exist_ok=True);provenance=destination/'source.json'
    if local_root:
        source=Path(local_root)
        if (source/'Phase_5_LSI').is_dir():source/='Phase_5_LSI'
        missing=[name for name in DATA_FILES if not (source/name).is_file()]
        if missing:raise FileNotFoundError('Missing official Phase-5 files: '+', '.join(missing))
        origin={'kind':'attached_official_phase5','source':str(source)}
        for name in DATA_FILES:
            target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
            if target.resolve()!=(source/name).resolve():shutil.copy2(source/name,target)
    else:
        old=read_json(provenance) if provenance.exists() else None
        if old and old.get('kind')=='official_huggingface' and old.get('sha256') and (revision is None or revision==old.get('revision')):
            if all((destination/name).is_file() for name in DATA_FILES):
                hashes={name:file_hash(destination/name) for name in DATA_FILES}
                if hashes!=old['sha256']:raise ValueError('Cached official data changed; restore it or use a new PROJECT.')
                return str(destination)
        from huggingface_hub import HfApi,hf_hub_download
        token=os.environ.get('HF_TOKEN')
        if not token:
            try:
                from kaggle_secrets import UserSecretsClient
                token=UserSecretsClient().get_secret('HF_TOKEN')
            except Exception:token=None
        repo='Exploration-Lab/CS-781-Capstone';commit=old['revision'] if old and old.get('kind')=='official_huggingface' else revision
        try:
            commit=HfApi(token=token).dataset_info(repo,revision=commit).sha
            origin={'kind':'official_huggingface','repo':repo,'revision':commit};atomic_json(provenance,origin)
            for name in DATA_FILES:
                target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():
                    source=hf_hub_download(repo,'Phase_5_LSI/'+name,repo_type='dataset',revision=commit,token=token)
                    shutil.copy2(source,target)
        except Exception as error:raise RuntimeError('Attach Phase_5_LSI, or accept the official dataset terms and enable the HF_TOKEN Kaggle secret. Never paste API tokens into code.') from error
    origin['sha256']={name:file_hash(destination/name) for name in DATA_FILES};atomic_json(provenance,origin)
    return str(destination)
