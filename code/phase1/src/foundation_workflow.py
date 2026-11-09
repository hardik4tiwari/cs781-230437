"""Run planning, compatible checkpoint recovery, and distinct candidate bookkeeping."""
import hashlib
import json
from pathlib import Path
import shutil
import stat
import zipfile
import zlib

ARCHITECTURES = ('decoder', 'encoder_decoder', 'local_gqa', 'hybrid')
OBJECTIVES = {'decoder':'full', 'encoder_decoder':'continuation',
              'local_gqa':'full', 'hybrid':'full'}
IDENTITY_KEYS = ('architecture', 'target_tokens', 'seq_len', 'objective', 'prefix_fraction',
                 'seed', 'learning_rate', 'min_learning_rate', 'warmup_tokens', 'weight_decay',
                 'dropout', 'gradient_checkpointing', 'checkpoint_loss_chunks',
                 'tokenizer_sha256', 'cache_fingerprint')


def prepare_foundation_run(expected):
    """Reuse completed runs, or preserve saved batch geometry for exact recovery."""
    folder = Path(expected['output_dir'])
    config_path = folder/'run_config.json'
    summary_path = folder/'training_summary.json'
    best_path, latest_path = folder/'best_model.pt', folder/'latest.pt'
    if not config_path.exists():
        if latest_path.exists() or best_path.exists():
            raise ValueError(f'{folder}: retain run_config.json beside existing checkpoints.')
        return dict(expected), None, False
    previous = json.loads(config_path.read_text())
    for key in IDENTITY_KEYS:
        if previous.get(key) != expected.get(key):
            raise ValueError(f'{folder}: changed {key}; restore its original configuration or use a new run directory.')
    if previous.get('model_config') != expected.get('model_config'):
        raise ValueError(f'{folder}: model configuration changed.')
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary['tokens_seen'] >= expected['target_tokens']:
            if not best_path.exists():
                raise ValueError(f'{folder}: completed run is missing best_model.pt.')
            actual = dict(previous)
            for key in ('output_dir', 'cache_dir', 'tokenizer_path', 'max_hours', 'num_workers', 'cpu_threads'):
                actual[key] = expected[key]
            return actual, None, True
    if latest_path.exists():
        actual = dict(previous)
        for key in ('output_dir', 'cache_dir', 'tokenizer_path', 'max_hours', 'num_workers', 'cpu_threads'):
            actual[key] = expected[key]
        return actual, latest_path, False
    if best_path.exists():
        raise ValueError(f'{folder}: retain latest.pt for exact recovery; best_model.pt alone is insufficient.')
    return dict(expected), None, False


def select_saved_cache(project, tokenizer_hash, settings, explicit=None):
    root = Path(project)
    candidates = [Path(explicit)/'manifest.json'] if explicit else sorted((root/'cache').glob('*/manifest.json'))
    matches = []
    for path in candidates:
        if not path.exists():
            if explicit: raise FileNotFoundError(path)
            continue
        manifest = json.loads(path.read_text())
        actual = manifest.get('settings', {})
        valid = manifest.get('complete') and actual.get('tokenizer_sha256') == tokenizer_hash
        valid = valid and all(actual.get(k) == v for k,v in settings.items())
        valid = valid and all((path.parent/f'{split}.bin').is_file() and
            (path.parent/f'{split}.bin').stat().st_size == manifest[split]['tokens']*2 for split in ('train','val'))
        if valid: matches.append((path.parent,manifest))
        elif explicit: raise ValueError(f'{path}: cache is incomplete or does not match tokenizer/data settings.')
    if not matches: return None, None
    revisions = {m['settings']['revision'] for _,m in matches}
    if len(revisions)>1:
        raise ValueError('Several saved source revisions match. Set EXISTING_CACHE_DIR to the intended cache.')
    return max(matches,key=lambda item:item[1]['train']['tokens'])


def restore_outputs(archive_path, destination):
    """Extract a prior output archive; reject traversal, symlinks, and conflicting files."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len({m.filename for m in members}) != len(members):
            raise ValueError('Duplicate paths in restore archive.')
        needed = 0
        for member in members:
            path = (destination/member.filename).resolve()
            if not path.is_relative_to(destination) or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f'Invalid restore path: {member.filename}')
            if member.is_dir(): continue
            if path.exists():
                if not path.is_file() or path.stat().st_size != member.file_size:
                    raise ValueError(f'Conflicting existing restore file: {member.filename}')
                crc=0
                with path.open('rb') as handle:
                    for block in iter(lambda:handle.read(1024*1024),b''): crc=zlib.crc32(block,crc)
                if (crc & 0xffffffff) != member.CRC:
                    raise ValueError(f'Conflicting existing restore file: {member.filename}')
            else: needed += member.file_size
        if shutil.disk_usage(destination).free < needed + 64*1024**2:
            raise RuntimeError('Not enough free disk space to restore this output archive.')
        for member in members:
            path=destination/member.filename
            if member.is_dir(): path.mkdir(parents=True,exist_ok=True)
            elif not path.exists():
                path.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(member) as src,path.open('wb') as dst:
                    shutil.copyfileobj(src,dst,1024*1024)
    return destination


def submission_plan():
    primary=[('decoder','balanced',0),('decoder','focused',0),
             ('encoder_decoder','balanced',0),('local_gqa','balanced',0),('hybrid','balanced',0)]
    reserves=[('decoder','diverse',0),('encoder_decoder','focused',0),
              ('local_gqa','focused',0),('hybrid','focused',0),('decoder','balanced',104729)]
    return [{'label':f'{arch}_{variant}'+(f'_seedplus{offset}' if offset else ''),
             'architecture':arch,'variant':variant,'seed_offset':offset,'reserve':index>=5}
            for index,(arch,variant,offset) in enumerate(primary+reserves)]


def prediction_fingerprint(predictions):
    raw=json.dumps(predictions,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()
    return hashlib.sha256(raw).hexdigest()


def add_distinct_candidate(records, candidate, predictions):
    fingerprint=prediction_fingerprint(predictions)
    duplicate=next((r['label'] for r in records if r['prediction_fingerprint']==fingerprint),None)
    if duplicate: return False, duplicate
    records.append({**candidate,'prediction_fingerprint':fingerprint,'examples':len(predictions)})
    return True,None
