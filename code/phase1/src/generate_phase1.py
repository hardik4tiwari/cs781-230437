"""Cached 128-word continuations, one inference worker/GPU, strict Codabench archive."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile
import torch
import torch.distributed as dist
from tokenizers import Tokenizer
from legal_models import LegalLM, ModelConfig, parameter_count
from legal_data import atomic_json, canonical_hash, file_hash
from train_phase1 import setup_distributed, amp_context


def sample_token(logits, generated, cfg, tokenizer, generator):
    logits = logits.float().clone()
    # The continuation must reach 128 complete words; EOS is disabled while
    # generating it. Other control tokens must never appear in submitted text.
    for spelling in ("<pad>", "<bos>", "<eos>", "<unk>"):
        idx = tokenizer.token_to_id(spelling)
        if idx is not None:
            logits[idx] = -float("inf")
    penalty = cfg.get("repetition_penalty", 1.0)
    if generated and penalty != 1.0:
        ids = torch.tensor(sorted(set(generated[-256:])), device=logits.device)
        values = logits[ids]
        logits[ids] = torch.where(values < 0, values * penalty, values / penalty)
    temperature = cfg.get("temperature", 0.8)
    if temperature == 0:
        return int(logits.argmax().item())
    logits = logits / temperature
    k = cfg.get("top_k", 50)
    if 0 < k < logits.numel():
        threshold = torch.topk(logits, k).values[-1]
        logits.masked_fill_(logits < threshold, -float("inf"))
    p = cfg.get("top_p", 0.95)
    if 0 < p < 1:
        values, indices = logits.sort(descending=True)
        remove = values.softmax(-1).cumsum(-1) > p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        values.masked_fill_(remove, -float("inf"))
        logits.scatter_(0, indices, values)
    return int(torch.multinomial(logits.softmax(-1), 1, generator=generator).item())


@torch.inference_mode()
def generate_continuation(model, tokenizer, prompt, cfg, device, example_id="0"):
    target_words = cfg.get("target_words", 128)
    # A trailing space puts generation at a word boundary, rather than allowing
    # continuation of the prompt's final word to count as a new word.
    tokens = tokenizer.encode(prompt.rstrip() + " ").ids
    if not tokens:
        raise ValueError(f"Empty prompt: {example_id}")
    if len(tokens) > cfg["prompt_max_tokens"]:
        raise ValueError(f"Prompt {example_id} has {len(tokens)} tokens; raise prompt_max_tokens. Full prompts are preserved.")
    if len(tokens) + cfg["max_new_tokens"] > model.config.max_positions and model.config.architecture != "encoder_decoder":
        raise ValueError("Prompt plus generation token cap exceeds the RoPE table. Increase max_positions before loading.")
    seed = cfg["seed"] + int(hashlib.sha256(str(example_id).encode()).hexdigest()[:8], 16)
    generator = torch.Generator(device=device).manual_seed(seed % (2**63 - 1))
    prompt_ids = torch.tensor([tokens], device=device, dtype=torch.long)
    if model.config.architecture == "encoder_decoder":
        current, source = prompt_ids[:, -1:], prompt_ids
    else:
        current, source = prompt_ids, None
    state, generated = None, []
    for index in range(cfg["max_new_tokens"]):
        with amp_context(device, cfg.get("fp16", True)):
            hidden, state = model(current, source_ids=source, state=state, use_cache=True)
            logits = model.project(hidden[:, -1:])[0, 0]
        next_id = sample_token(logits, generated, cfg, tokenizer, generator)
        generated.append(next_id)
        current = torch.tensor([[next_id]], device=device)
        source = None
        # Seeing a 129th word proves that the 128th word is complete, even if
        # the tokenizer emits several pieces per word. Tokens != words.
        if (index + 1) % 8 == 0 or index + 1 == cfg["max_new_tokens"]:
            decoded = tokenizer.decode(generated, skip_special_tokens=True)
            words = decoded.split()
            if len(words) >= target_words + 1:
                text = " ".join(words[:target_words])
                assert len(text.split()) == target_words
                return text, {"prompt_tokens": len(tokens), "generated_tokens": len(generated)}
    raise RuntimeError(f"{example_id}: token cap reached before {target_words} complete words. Increase max_new_tokens; no filler text is inserted.")


def validate_predictions(predictions, prompts, target_words=128):
    if not isinstance(predictions, dict) or set(predictions) != set(prompts):
        missing, extra = set(prompts) - set(predictions), set(predictions) - set(prompts)
        raise ValueError(f"ID mismatch: {len(missing)} missing, {len(extra)} extra.")
    for key, value in predictions.items():
        if not isinstance(value, dict) or set(value) != {"text"} or not isinstance(value["text"], str):
            raise ValueError(f"Invalid prediction schema for {key}.")
        if len(value["text"].split()) != target_words:
            raise ValueError(f"{key} must have exactly {target_words} whitespace-separated words.")
        if any(control in value["text"] for control in ("<pad>", "<bos>", "<eos>", "<unk>")):
            raise ValueError(f"Control token in {key}.")
    return True


def write_submission(predictions, prompts, output_dir, target_words=128):
    validate_predictions(predictions, prompts, target_words)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "prediction.json", predictions)
    archive = root / "submission.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # Stable zip metadata prevents timestamps from changing identical archives.
        info = zipfile.ZipInfo("prediction.json", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, (root / "prediction.json").read_bytes())
    with zipfile.ZipFile(archive) as z:
        assert z.namelist() == ["prediction.json"]
        recovered = json.loads(z.read("prediction.json").decode("utf-8"))
        validate_predictions(recovered, prompts, target_words)
    return archive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    device, world, rank = setup_distributed()
    torch.set_num_threads(2)
    cfg = json.loads(Path(args.config).read_text())
    if cfg.get("target_words", 128) != 128 and not cfg.get("allow_short_test", False):
        raise ValueError("Official Phase-1 submissions require 128 words.")
    prompts = json.loads(Path(args.prompts).read_text())
    assert prompts and all(isinstance(v, dict) and isinstance(v.get("text"), str) for v in prompts.values())
    tokenizer = Tokenizer.from_file(args.tokenizer)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert saved["training_origin"] == "random_initialization_official_phase1_only"
    assert saved["tokenizer_sha256"] == file_hash(args.tokenizer)
    model = LegalLM(ModelConfig(**saved["model_config"]))
    model.load_state_dict(saved["model"])
    del saved
    model = model.to(device).eval()
    if device.type == "cuda" and cfg.get("fp16", True):
        model.half()
    if not cfg.get("allow_short_test", False):
        assert 50_000_000 <= parameter_count(model) <= 300_000_000
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_id = canonical_hash({"checkpoint": file_hash(args.checkpoint), "tokenizer": file_hash(args.tokenizer),
                             "prompts": file_hash(args.prompts), "decode": cfg})
    shard_path = root / f"partial_{rank:02}.json"
    metadata_path = root / f"partial_{rank:02}.metadata.json"
    predictions, audit = {}, {}
    if metadata_path.exists() and shard_path.exists():
        existing = json.loads(metadata_path.read_text())
        if existing["run_id"] == run_id:
            predictions = json.loads(shard_path.read_text())
            audit = existing.get("audit", {})
    ids = list(prompts)[rank::world]
    # When process count changes, retain only this worker's current ID subset.
    predictions = {k: v for k, v in predictions.items() if k in set(ids)}
    start = time.perf_counter()
    for position, example_id in enumerate(ids):
        if example_id in predictions:
            continue
        try:
            text, details = generate_continuation(model, tokenizer, prompts[example_id]["text"], cfg, device, example_id)
        except Exception as error:
            atomic_json(shard_path, predictions)
            atomic_json(metadata_path, {"run_id": run_id, "audit": audit})
            atomic_json(root / f"failure_{rank:02}.json", {"example_id": example_id, "error": str(error)})
            raise
        predictions[example_id] = {"text": text}
        audit[example_id] = details
        if (position + 1) % cfg.get("save_every_examples", 10) == 0 or position + 1 == len(ids):
            atomic_json(shard_path, predictions)
            atomic_json(metadata_path, {"run_id": run_id, "audit": audit})
            print(f"rank {rank}: {position + 1}/{len(ids)} prompts", flush=True)
    # Also writes empty shards if a rank has no examples.
    atomic_json(shard_path, predictions)
    atomic_json(metadata_path, {"run_id": run_id, "audit": audit})
    if world > 1:
        dist.barrier()
    if rank == 0:
        merged, complete_audit = {}, {}
        for worker in range(world):
            shard = json.loads((root / f"partial_{worker:02}.json").read_text())
            meta = json.loads((root / f"partial_{worker:02}.metadata.json").read_text())
            assert meta["run_id"] == run_id
            if set(merged) & set(shard):
                raise ValueError("Duplicate IDs across generation shards.")
            merged.update(shard)
            complete_audit.update(meta["audit"])
        archive = write_submission(merged, prompts, root, cfg.get("target_words", 128))
        atomic_json(root / "generation_metadata.json", {
            "run_id": run_id, "decode": cfg, "examples": len(prompts),
            "seconds_this_session": time.perf_counter() - start, "world_size": world,
            "parameters": parameter_count(model), "archive_sha256": file_hash(archive),
            "audit": complete_audit,
        })
        print(f"Validated archive: {archive}", flush=True)
    if world > 1:
        dist.destroy_process_group()




# PHASE1_LENGTH_RECOVERY_V1
_original_sample_token = sample_token
_original_generate_continuation = generate_continuation
_word_start_cache = {}


def _word_start_ids(tokenizer):
    key = id(tokenizer)
    if key not in _word_start_cache:
        starts = set()
        for idx in tokenizer.get_vocab().values():
            text = tokenizer.decode([idx], skip_special_tokens=True)
            if text and text[0].isspace() and text.strip():
                starts.add(idx)
        _word_start_cache[key] = starts
    return _word_start_cache[key]


def sample_token(logits, generated, cfg, tokenizer, generator):
    n = int(cfg.get("no_repeat_ngram_size", 0))
    guard = int(cfg.get("word_progress_guard_tokens", 0))
    if not n and not guard:
        return _original_sample_token(logits, generated, cfg, tokenizer, generator)
    scores = logits.float().clone()
    if n > 1 and len(generated) >= n - 1:
        prefix = tuple(generated[-(n - 1):])
        banned = {generated[i + n - 1] for i in range(len(generated) - n + 1)
                  if tuple(generated[i:i + n - 1]) == prefix}
        if banned:
            scores[list(banned)] = -float("inf")
    if guard and len(generated) >= guard:
        current_words = len(tokenizer.decode(generated, skip_special_tokens=True).split())
        earlier_words = len(tokenizer.decode(generated[:-guard], skip_special_tokens=True).split())
        if current_words <= earlier_words:
            starts = _word_start_ids(tokenizer)
            if not starts:
                raise RuntimeError("Tokenizer has no whitespace-prefixed lexical tokens for length recovery.")
            allowed = torch.zeros_like(scores, dtype=torch.bool)
            allowed[list(starts)] = True
            scores.masked_fill_(~allowed, -float("inf"))
    if not torch.isfinite(scores).any():
        raise RuntimeError("No finite token scores remain under the recovery constraints.")
    return _original_sample_token(scores, generated, cfg, tokenizer, generator)


def generate_continuation(model, tokenizer, prompt, cfg, device, example_id="0"):
    prompt_tokens = len(tokenizer.encode(prompt.rstrip() + " ").ids)
    room = model.config.max_positions - (1 if model.config.architecture == "encoder_decoder" else prompt_tokens)
    extended_cap = min(room, max(int(cfg["max_new_tokens"]), 2048))
    attempts = [dict(cfg)]
    if extended_cap > int(cfg["max_new_tokens"]):
        attempts.append({**cfg, "max_new_tokens": extended_cap})
    attempts.append({**cfg, "max_new_tokens": extended_cap,
        "repetition_penalty": max(1.15, float(cfg.get("repetition_penalty", 1.0))),
        "no_repeat_ngram_size": 4, "word_progress_guard_tokens": 12,
        "seed": int(cfg["seed"]) + 1_000_003})
    failures = []
    for attempt, actual_cfg in enumerate(attempts):
        try:
            text, audit = _original_generate_continuation(
                model, tokenizer, prompt, actual_cfg, device, example_id)
        except RuntimeError as error:
            if "token cap reached before" not in str(error):
                raise
            failures.append({"attempt": attempt, "max_new_tokens": actual_cfg["max_new_tokens"],
                             "error": str(error)})
            print(f"rank {os.environ.get('RANK', '0')}: length recovery for ID {example_id}, "
                  f"attempt {attempt + 1}", flush=True)
            continue
        audit.update(recovery_policy="length-recovery-v1", recovery_attempt=attempt,
                     effective_decode=actual_cfg, previous_length_failures=failures)
        return text, audit
    raise RuntimeError(f"{example_id}: could not produce {cfg.get('target_words', 128)} complete words "
                       f"after {len(attempts)} bounded attempts. No filler text was inserted.")

if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
