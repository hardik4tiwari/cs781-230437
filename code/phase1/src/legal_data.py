"""Stream only official Phase-1 data, split by document, train BPE, write token caches."""
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
from array import array
import ijson
import requests
from huggingface_hub import HfApi, hf_hub_url
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

REPO = "Exploration-Lab/CS-781-Capstone"
SOURCE_FILES = {
    "acts": "Phase_1_Pretraining/Train/train_acts.json",
    "high_courts": "Phase_1_Pretraining/Train/train_high_courts.json",
    "supreme_court": "Phase_1_Pretraining/Train/train_supreme_court.json",
    "tribunals": "Phase_1_Pretraining/Train/train_tribunals.json",
}
TEXT_KEYS = ("text", "content", "judgment", "judgement", "document", "raw_text")
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def normalize_text(text):
    # Preserve case, citations, punctuation, and paragraph boundaries.
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(re.sub(r"[\t ]+", " ", line).strip() for line in text.split("\n")).strip()


def document_hash(text):
    # Whitespace/case variants receive the same train/validation assignment.
    key = " ".join(text.split()).casefold()
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def is_validation(digest, validation_per_mille=20):
    return int(digest[:16], 16) % 1000 < validation_per_mille


def extract_text(record):
    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        for key in TEXT_KEYS:
            value = record.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, (dict, list)):
                found = extract_text(value)
                if found:
                    return found
        for key in ("paragraphs", "sentences"):
            if key in record:
                return extract_text(record[key])
    if isinstance(record, list):
        return "\n".join(x for x in (extract_text(v) for v in record) if x)
    return ""


@contextlib.contextmanager
def open_source(path, token=None, revision="main", local_root=None):
    if local_root:
        target = Path(local_root) / path
        if not target.exists():
            matches = list(Path(local_root).rglob(Path(path).name))
            if len(matches) != 1:
                raise FileNotFoundError(f"Expected one local file for {path}; found {len(matches)}.")
            target = matches[0]
        with open(target, "rb") as f:
            yield f
    else:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        url = hf_hub_url(REPO, path, repo_type="dataset", revision=revision)
        with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as r:
            if r.status_code in (401, 403):
                raise RuntimeError(
                    "Dataset access denied. Accept the dataset access conditions using your own "
                    "Hugging Face account, then add a Kaggle Secret named HF_TOKEN with read access."
                )
            r.raise_for_status()
            r.raw.decode_content = True
            # Pull-based parsing stops the network stream as soon as quotas are met.
            yield io.BufferedReader(r.raw, buffer_size=1 << 20)


def iter_records(stream, layout="auto"):
    if not hasattr(stream, "peek"):
        stream = io.BufferedReader(stream)
    head = stream.peek(4096).lstrip()
    if head.startswith(b"\xef\xbb\xbf"):
        stream.read(3)
        head = stream.peek(4096).lstrip()
    if layout == "auto":
        if head.startswith(b"["):
            layout = "array"
        elif head.startswith(b"{"):
            m = re.match(rb'\{\s*"([^"\\]+)"\s*:', head)
            if not m:
                raise ValueError("Unable to detect JSON layout. Set SOURCE_LAYOUT explicitly.")
            key = m.group(1).decode("utf-8")
            if key in TEXT_KEYS or key in ("id", "document_id", "doc_id", "case_id", "uid"):
                layout = "records"  # Single object or JSONL/multiple objects.
            elif key in ("data", "records", "documents", "train"):
                layout = f"wrapper:{key}"
            else:
                layout = "mapping"  # Official ID -> text / {text: ...} form.
        else:
            raise ValueError("Expected a JSON array, ID mapping, or JSONL records.")
    if layout == "array":
        yield from ijson.items(stream, "item")
    elif layout == "records":
        yield from ijson.items(stream, "", multiple_values=True)
    elif layout == "mapping":
        for _, value in ijson.kvitems(stream, ""):
            yield value
    elif layout.startswith("wrapper:"):
        key = layout.split(":", 1)[1]
        yield from ijson.items(stream, f"{key}.item")
    else:
        raise ValueError(f"Unknown JSON layout {layout}")


def iter_documents(source, token=None, revision="main", local_root=None, layout="auto"):
    bad, good = 0, 0
    with open_source(SOURCE_FILES[source], token, revision, local_root) as f:
        for record in iter_records(f, layout):
            text = normalize_text(extract_text(record))
            if not text:
                bad += 1
                if bad >= 20 and good == 0:
                    keys = list(record)[:12] if isinstance(record, dict) else type(record).__name__
                    raise ValueError(f"No legal text found in {source}; record keys/type: {keys}. Inspect the schema.")
                continue
            good += 1
            yield text, document_hash(text)
    if good == 0:
        raise ValueError(f"No documents read from {source}; verify SOURCE_LAYOUT and TEXT_KEYS.")


def resolve_revision(token=None):
    return HfApi(token=token).dataset_info(REPO).sha


def discover_test_path(token=None, revision="main", local_root=None):
    if local_root:
        paths = [str(p.relative_to(local_root)) for p in Path(local_root).rglob("*.json")
                 if "/Test/" in "/" + str(p.relative_to(local_root)) and "Phase_1" in str(p)]
    else:
        paths = [p for p in HfApi(token=token).list_repo_files(REPO, repo_type="dataset", revision=revision)
                 if p.startswith("Phase_1_Pretraining/Test/") and p.endswith(".json")]
    if len(paths) != 1:
        raise ValueError(f"Expected one Phase-1 test JSON; found {paths}. Set TEST_PATH_OVERRIDE.")
    return paths[0]


def read_test_prompts(token=None, revision="main", local_root=None, path_override=None):
    path = path_override or discover_test_path(token, revision, local_root)
    with open_source(path, token, revision, local_root) as f:
        data = json.load(f)  # Official test file is small; training files are streamed.
    if isinstance(data, dict) and all(isinstance(v, dict) and isinstance(v.get("text"), str) for v in data.values()):
        prompts = {str(k): {"text": v["text"]} for k, v in data.items()}
    elif isinstance(data, list) and all(isinstance(v, dict) and "id" in v and isinstance(v.get("text"), str) for v in data):
        ids = [str(v["id"]) for v in data]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate official test IDs.")
        prompts = {str(v["id"]): {"text": v["text"]} for v in data}
    else:
        raise ValueError("Unknown test schema. Preserve official IDs; do not invent row-number IDs.")
    if not prompts:
        raise ValueError("Test prompts are empty.")
    return prompts, path


def train_tokenizer(output_path, token=None, revision="main", local_root=None,
                    chars_per_source=5_000_000, validation_per_mille=20, layouts=None):
    output_path = Path(output_path)
    sample_path = output_path.with_name("tokenizer_training_sample.jsonl")
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    layouts = layouts or {}
    source_stats, seen = {}, set()
    with open(sample_path, "w", encoding="utf-8") as out:
        for source in SOURCE_FILES:
            chars, docs = 0, 0
            for text, digest in iter_documents(source, token, revision, local_root, layouts.get(source, "auto")):
                if is_validation(digest, validation_per_mille) or digest in seen or len(text) < 200:
                    continue
                seen.add(digest)
                portion = text[:min(chars_per_source - chars, 100_000)]
                out.write(json.dumps({"text": portion}, ensure_ascii=False) + "\n")
                chars += len(portion)
                docs += 1
                if chars >= chars_per_source:
                    break
            source_stats[source] = {"characters": chars, "documents": docs}
    def texts():
        with open(sample_path, encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)["text"]
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=32768, min_frequency=2, special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True,
    )
    tok.train_from_iterator(texts(), trainer=trainer)
    if tok.get_vocab_size() != 32768:
        raise ValueError("Tokenizer did not reach 32,768 entries. Increase chars_per_source and retrain.")
    tok.save(str(output_path))
    atomic_json(output_path.with_suffix(".provenance.json"), {
        "repository": REPO, "revision": revision, "train_only": True,
        "chars_per_source": chars_per_source,
        "validation_per_mille": validation_per_mille, "sources": source_stats,
        "sha256": file_hash(output_path), "vocab_size": tok.get_vocab_size(),
    })
    return tok


def chunks_at_whitespace(text, max_chars=32_000):
    # Bound single tokenization calls. Whitespace cuts may slightly alter BPE
    # segmentation; the same procedure is fixed for all four training caches.
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            cut = max(text.rfind(" ", start, end), text.rfind("\n", start, end))
            if cut > start + max_chars // 2:
                end = cut + 1
        yield text[start:end]
        start = end


class TokenWriter:
    def __init__(self, path):
        if sys.byteorder != "little":
            raise RuntimeError("The token cache expects a little-endian machine.")
        self.file = open(path, "wb")
        self.count, self.hash = 0, hashlib.sha256()

    def write(self, ids):
        if ids and (min(ids) < 0 or max(ids) >= 32768):
            raise ValueError("Token ID does not fit the signed-int16 cache.")
        data = array("H", ids).tobytes()
        self.file.write(data)
        self.hash.update(data)
        self.count += len(ids)

    def close(self):
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()


def build_cache(cache_root, tokenizer_path, token=None, revision="main", local_root=None,
                train_tokens=240_000_000, val_tokens=2_000_000,
                mix=None, validation_per_mille=20, layouts=None, generation_pairs=200):
    mix = mix or {"acts": 0.10, "high_courts": 0.45, "supreme_court": 0.25, "tribunals": 0.20}
    assert set(mix) == set(SOURCE_FILES) and abs(sum(mix.values()) - 1.0) < 1e-8
    assert all(v > 0 for v in mix.values()) and 0 < validation_per_mille < 1000
    tok = Tokenizer.from_file(str(tokenizer_path))
    assert tok.get_vocab_size() == 32768
    settings = dict(repository=REPO, revision=revision, tokenizer_sha256=file_hash(tokenizer_path),
                    train_tokens=train_tokens, val_tokens=val_tokens, mix=mix,
                    validation_per_mille=validation_per_mille, layouts=layouts or {},
                    generation_pairs=generation_pairs, tokenization_chunk_chars=32_000, version=1)
    fingerprint = canonical_hash(settings)
    root = Path(cache_root) / fingerprint[:12]
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("complete") and previous.get("fingerprint") == fingerprint:
            for split in ("train", "val"):
                assert (root / f"{split}.bin").stat().st_size == previous[split]["tokens"] * 2
            return root, previous
    # Partial caches never get marked complete and are rebuilt on the next run.
    writers = {s: TokenWriter(root / f"{s}.bin.partial") for s in ("train", "val")}
    stats, seen, pairs, audit = {}, set(), [], []
    bos, eos = tok.token_to_id("<bos>"), tok.token_to_id("<eos>")
    try:
        for source, weight in mix.items():
            quotas = {"train": math.ceil(train_tokens * weight), "val": math.ceil(val_tokens * weight)}
            counts = {"train": 0, "val": 0, "train_docs": 0, "val_docs": 0, "duplicates": 0}
            pair_quota = math.ceil(generation_pairs * weight)
            source_pairs = 0
            for text, digest in iter_documents(source, token, revision, local_root, (layouts or {}).get(source, "auto")):
                if len(text) < 200:
                    continue
                if digest in seen:
                    counts["duplicates"] += 1
                    continue
                seen.add(digest)
                split = "val" if is_validation(digest, validation_per_mille) else "train"
                if split == "val" and source_pairs < pair_quota:
                    words = text.split()
                    if len(words) >= 640:
                        pairs.append({"id": digest, "source": source,
                                      "prompt": " ".join(words[:512]), "reference": " ".join(words[512:640])})
                        source_pairs += 1
                if counts[split] >= quotas[split]:
                    if all(counts[s] >= quotas[s] for s in ("train", "val")):
                        break
                    continue
                # Hash assignment uses the whole document even when a quota cuts
                # its last cached portion. No part is moved to the other split.
                remaining = quotas[split] - counts[split]
                writers[split].write([bos])
                counts[split] += 1
                remaining -= 1
                for piece in chunks_at_whitespace(text):
                    if remaining <= 1:
                        break
                    ids = tok.encode(piece).ids[:remaining - 1]
                    writers[split].write(ids)
                    counts[split] += len(ids)
                    remaining -= len(ids)
                if remaining > 0:
                    writers[split].write([eos])
                    counts[split] += 1
                counts[f"{split}_docs"] += 1
                audit.append({"hash": digest, "source": source, "split": split})
                if all(counts[s] >= quotas[s] for s in ("train", "val")):
                    break
            if counts["train"] == 0 or counts["val"] == 0:
                raise RuntimeError(f"Insufficient train/validation documents for {source}. Increase sampling or check schema.")
            stats[source] = counts
            print(source, counts, flush=True)
    finally:
        for writer in writers.values():
            writer.close()
    for split in writers:
        (root / f"{split}.bin.partial").replace(root / f"{split}.bin")
    atomic_json(root / "validation_generation.json", pairs[:generation_pairs])
    with open(root / "document_audit.jsonl", "w") as f:
        for row in audit:
            f.write(json.dumps(row) + "\n")
    manifest = {"complete": True, "fingerprint": fingerprint, "settings": settings,
                "sources": stats, "dtype": "int16-little-endian", "vocab_size": 32768,
                "generation_pairs": min(len(pairs), generation_pairs),
                **{s: {"tokens": w.count, "sha256": w.hash.hexdigest()} for s, w in writers.items()}}
    atomic_json(manifest_path, manifest)
    return root, manifest
