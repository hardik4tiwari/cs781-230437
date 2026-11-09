"""Run under torchrun: one process/GPU, FP16, correct gradient accumulation, recovery."""
import argparse
import contextlib
import csv
from datetime import timedelta
import inspect
import json
import math
import os
from pathlib import Path
import random
import signal
import shutil
import time
from dataclasses import asdict
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from legal_models import LegalLM, ModelConfig, architecture_config, parameter_count, make_batch, language_model_loss
from legal_data import atomic_json, canonical_hash, file_hash


class TokenBlocks(Dataset):
    def __init__(self, path, seq_len):
        self.path, self.seq_len = str(path), seq_len
        self.tokens = Path(path).stat().st_size // 2
        self.mapped = None

    def __len__(self):
        return (self.tokens - 1) // self.seq_len

    def __getitem__(self, index):
        if self.mapped is None:
            self.mapped = torch.from_file(self.path, shared=False, size=self.tokens, dtype=torch.int16)
        start = int(index) * self.seq_len
        return self.mapped[start:start + self.seq_len + 1].to(torch.long)


def setup_distributed():
    world, rank, local_rank = (int(os.environ.get(k, default)) for k, default in
                              (("WORLD_SIZE", 1), ("RANK", 0), ("LOCAL_RANK", 0)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(minutes=30))
    return device, world, rank


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def rng_state(device):
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(state, device):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def crop_loss_view(hidden, targets, architecture, prefix_fraction, objective, seq_len):
    if objective == "continuation" and architecture != "encoder_decoder":
        start = max(1, min(seq_len, int(seq_len * prefix_fraction))) - 1
        return hidden[:, start:], targets[:, start:]
    return hidden, targets


def optimizer_for(model, lr, weight_decay, device):
    groups = [{"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": weight_decay},
              {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0}]
    kwargs = {"lr": lr, "betas": (0.9, 0.95), "eps": 1e-8}
    if "fused" in inspect.signature(torch.optim.AdamW).parameters:
        kwargs["fused"] = device.type == "cuda"
    return torch.optim.AdamW(groups, **kwargs)


def amp_context(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


@torch.no_grad()
def validate(raw, dataset, cfg, device, world, rank):
    was_training = raw.training
    raw.eval()
    total = torch.zeros(2, device=device, dtype=torch.float64)
    micro = cfg["micro_batch"]
    batches = min(cfg["validation_batches"], len(dataset) // (world * micro))
    if batches == 0:
        raise ValueError("Validation cache is too small for this batch/world size.")
    for batch_idx in range(batches):
        first = (batch_idx * world + rank) * micro
        blocks = torch.stack([dataset[first + j] for j in range(micro)]).to(device, non_blocking=True)
        inputs, targets, source = make_batch(blocks, cfg["architecture"], cfg["prefix_fraction"], "continuation")
        with amp_context(device, cfg["fp16"]):
            hidden = raw(inputs, source_ids=source)
            hidden, targets = crop_loss_view(hidden, targets, cfg["architecture"], cfg["prefix_fraction"], "continuation", cfg["seq_len"])
            _, loss_sum, count = language_model_loss(hidden, targets, raw.embedding.weight, cfg["loss_chunk_tokens"])
        total[0] += loss_sum.double()
        total[1] += count.double()
    if world > 1:
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    raw.train(was_training)
    nll = (total[0] / total[1]).item()
    return {"validation_nll": nll, "validation_perplexity": math.exp(min(nll, 50)),
            "validation_target_tokens": int(total[1].item())}


def checkpoint_state(raw, optimizer, scaler, cfg, progress, best_nll, device, world, rank):
    state = rng_state(device)
    if world > 1:
        states = [None] * world
        dist.all_gather_object(states, state)
    else:
        states = [state]
    if rank == 0:
        destination = Path(cfg["output_dir"])
        payload = {"model": raw.state_dict(), "model_config": asdict(raw.config),
                   "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                   "run_config": cfg, "progress": progress, "best_nll": best_nll,
                   "rng_states": states, "world_size": world,
                   "cache_fingerprint": cfg["cache_fingerprint"],
                   "tokenizer_sha256": cfg["tokenizer_sha256"],
                   "training_origin": "random_initialization_official_phase1_only"}
        temporary = destination / "latest.pt.tmp"
        torch.save(payload, temporary)
        temporary.replace(destination / "latest.pt")
    if world > 1:
        dist.barrier()


def save_best(raw, cfg, progress, metrics, rank):
    if rank != 0:
        return
    payload = {"model": {k: v.detach().cpu() for k, v in raw.state_dict().items()},
               "model_config": asdict(raw.config), "progress": progress,
               "metrics": metrics, "cache_fingerprint": cfg["cache_fingerprint"],
               "tokenizer_sha256": cfg["tokenizer_sha256"],
               "training_origin": "random_initialization_official_phase1_only"}
    destination = Path(cfg["output_dir"])
    torch.save(payload, destination / "best_model.pt.tmp")
    (destination / "best_model.pt.tmp").replace(destination / "best_model.pt")
    atomic_json(destination / "best_metrics.json", {**metrics, **progress, "parameters": parameter_count(raw)})


def learning_rate(tokens_seen, cfg):
    warm = min(cfg["warmup_tokens"], max(1, int(cfg["target_tokens"] * 0.05)))
    if tokens_seen < warm:
        return cfg["learning_rate"] * max(1, tokens_seen) / warm
    fraction = min(1.0, (tokens_seen - warm) / max(1, cfg["target_tokens"] - warm))
    return cfg["min_learning_rate"] + 0.5 * (cfg["learning_rate"] - cfg["min_learning_rate"]) * (1 + math.cos(math.pi * fraction))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--benchmark-steps", type=int, default=30)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    device, world, rank = setup_distributed()
    torch.set_num_threads(max(1, cfg.get("cpu_threads", 2)))
    random.seed(cfg["seed"] + rank)
    torch.manual_seed(cfg["seed"])  # Identical initialization on every rank.
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg["seed"])
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(cfg["cache_dir"])
    manifest = json.loads((cache / "manifest.json").read_text())
    assert manifest["complete"] and manifest["fingerprint"] == cfg["cache_fingerprint"]
    assert cfg["tokenizer_sha256"] == file_hash(cfg["tokenizer_path"])
    assert manifest["settings"]["tokenizer_sha256"] == cfg["tokenizer_sha256"]
    for split in ("train", "val"):
        assert (cache / f"{split}.bin").stat().st_size == manifest[split]["tokens"] * 2
    c = ModelConfig(**cfg["model_config"]) if "model_config" in cfg else architecture_config(
        cfg["architecture"], dropout=cfg["dropout"], gradient_checkpointing=cfg["gradient_checkpointing"])
    assert c.architecture == cfg["architecture"]
    raw = LegalLM(c).to(device)
    count = parameter_count(raw)
    if not cfg.get("allow_tiny_test_model", False):
        assert 50_000_000 <= count <= 300_000_000, f"Invalid parameter count: {count:,}"
    optimizer = optimizer_for(raw, cfg["learning_rate"], cfg["weight_decay"], device)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=cfg["fp16"] and device.type == "cuda")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=cfg["fp16"] and device.type == "cuda")
    progress = {"step": 0, "tokens_seen": 0, "input_tokens_seen": 0, "epoch": 0, "batch_offset": 0}
    best_nll = float("inf")
    if args.resume or args.init_from:
        previous = torch.load(args.resume or args.init_from, map_location="cpu", weights_only=False)
        assert previous["training_origin"] == "random_initialization_official_phase1_only"
        assert previous["tokenizer_sha256"] == cfg["tokenizer_sha256"]
        assert previous["model_config"] == asdict(c)
        raw.load_state_dict(previous["model"])
        if args.resume:
            assert previous["cache_fingerprint"] == cfg["cache_fingerprint"]
            assert previous["world_size"] == world, "Exact resume requires the same GPU/process count."
            for key in ("seq_len", "micro_batch", "accumulation_steps", "objective", "prefix_fraction", "seed",
                        "target_tokens", "learning_rate", "min_learning_rate", "warmup_tokens", "weight_decay", "grad_clip"):
                assert previous["run_config"][key] == cfg[key], f"Resume mismatch: {key}. Use --init-from for a new schedule."
            optimizer.load_state_dict(previous["optimizer"])
            scaler.load_state_dict(previous["scaler"])
            progress, best_nll = previous["progress"], previous["best_nll"]
            saved_rng = previous["rng_states"][rank]
            if rank == 0 and not (out / "best_model.pt").exists() and math.isfinite(best_nll):
                previous_dir = Path(args.resume).parent
                if not (previous_dir / "best_model.pt").exists():
                    raise ValueError("Resume also requires the saved best_model.pt next to latest.pt; retain both files.")
                shutil.copyfile(previous_dir / "best_model.pt", out / "best_model.pt")
                if (previous_dir / "best_metrics.json").exists():
                    shutil.copyfile(previous_dir / "best_metrics.json", out / "best_metrics.json")
        else:
            saved_rng = None
        del previous
    else:
        saved_rng = None
    if world > 1:
        model = DDP(raw, device_ids=[device.index] if device.type == "cuda" else None,
                    broadcast_buffers=False, gradient_as_bucket_view=True, find_unused_parameters=False)
    else:
        model = raw
    # Same initialized model, independent dropout RNG stream per rank.
    # Exact resume restores the saved rank-specific state below.
    torch.manual_seed(cfg["seed"] + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed(cfg["seed"] + rank)
    train = TokenBlocks(cache / "train.bin", cfg["seq_len"])
    val = TokenBlocks(cache / "val.bin", cfg["seq_len"])
    sampler = DistributedSampler(train, num_replicas=world, rank=rank, shuffle=True,
                                 seed=cfg["seed"], drop_last=True)
    loader_args = dict(batch_size=cfg["micro_batch"], sampler=sampler, drop_last=True,
                       num_workers=cfg["num_workers"], pin_memory=device.type == "cuda")
    if cfg["num_workers"]:
        # NCCL is not fork-safe. Workers read CPU token mappings only, using spawn.
        loader_args.update(persistent_workers=True, prefetch_factor=2, multiprocessing_context="spawn")
    loader = DataLoader(train, **loader_args)
    if len(loader) == 0:
        raise ValueError("Training cache too small for this batch/world size.")
    sampler.set_epoch(progress["epoch"])
    iterator = iter(loader)
    for _ in range(progress["batch_offset"]):
        next(iterator)
    if saved_rng is not None:
        restore_rng(saved_rng, device)
    stop_requested = [False]
    def request_stop(signum, frame):
        stop_requested[0] = True
    signal.signal(signal.SIGTERM, request_stop)
    # torchrun/kernel interruption can terminate workers before a new save;
    # latest.pt is guaranteed only at completed save points.
    if rank == 0:
        atomic_json(out / "run_config.json", cfg)
        atomic_json(out / "environment.json", {"torch": torch.__version__, "world_size": world,
                    "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                    "cuda": torch.version.cuda, "parameters": count,
                    "capability": torch.cuda.get_device_capability(device) if device.type == "cuda" else None})
        print(f"{cfg['architecture']}: {count:,} parameters; {world} process(es); device={device}", flush=True)
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start_time = time.perf_counter()
    benchmark_start, benchmark_start_tokens = None, 0
    initial_tokens = progress["tokens_seen"]
    log_time, log_tokens = start_time, initial_tokens
    last_metrics, skips = {}, 0
    while progress["tokens_seen"] < cfg["target_tokens"] or args.benchmark:
        optimizer.zero_grad(set_to_none=True)
        step_stats = torch.zeros(2, device=device, dtype=torch.float64)
        step_target_tokens, step_input_tokens = 0, 0
        lr = learning_rate(progress["tokens_seen"] + 1, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        for micro_idx in range(cfg["accumulation_steps"]):
            try:
                blocks = next(iterator)
            except StopIteration:
                progress["epoch"] += 1
                progress["batch_offset"] = 0
                sampler.set_epoch(progress["epoch"])
                iterator = iter(loader)
                blocks = next(iterator)
            progress["batch_offset"] += 1
            blocks = blocks.to(device, non_blocking=True)
            inputs, targets, source = make_batch(blocks, cfg["architecture"], cfg["prefix_fraction"], cfg["objective"])
            sync_context = model.no_sync() if world > 1 and micro_idx < cfg["accumulation_steps"] - 1 else contextlib.nullcontext()
            with sync_context:  # Both forward AND backward belong inside no_sync.
                with amp_context(device, cfg["fp16"]):
                    hidden = model(inputs, source_ids=source)
                    hidden, targets = crop_loss_view(hidden, targets, cfg["architecture"], cfg["prefix_fraction"], cfg["objective"], cfg["seq_len"])
                    loss, loss_sum, supervised = language_model_loss(
                        hidden, targets, raw.embedding.weight, cfg["loss_chunk_tokens"], cfg["checkpoint_loss_chunks"])
                scaler.scale(loss / cfg["accumulation_steps"]).backward()
            step_stats[0] += loss_sum.double()
            step_stats[1] += supervised.double()
            step_target_tokens += targets.numel()
            step_input_tokens += inputs.numel() + (source.numel() if source is not None else 0)
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), cfg["grad_clip"])
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.get_scale() < old_scale
        progress["input_tokens_seen"] += step_input_tokens * world
        if skipped:
            skips += 1
            if skips > 50:
                raise RuntimeError("More than 50 AMP-overflow updates. Reduce LR; inspect data and gradients.")
            continue
        progress["step"] += 1
        progress["tokens_seen"] += step_target_tokens * world
        step = progress["step"]
        if args.benchmark:
            warmup = min(10, max(1, args.benchmark_steps // 3))
            if step == warmup:
                synchronize(device)
                benchmark_start, benchmark_start_tokens = time.perf_counter(), progress["tokens_seen"]
            if step >= args.benchmark_steps:
                synchronize(device)
                elapsed = time.perf_counter() - benchmark_start
                peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
                reserved = torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0.0
                summary = torch.tensor([elapsed, peak, reserved], device=device, dtype=torch.float64)
                if world > 1:
                    dist.all_reduce(summary, op=dist.ReduceOp.MAX)
                measured_tokens = progress["tokens_seen"] - benchmark_start_tokens
                result = {"loss_target_tokens_per_second": measured_tokens / summary[0].item(),
                          "peak_allocated_gib_per_gpu": summary[1].item(), "warmup_steps": warmup,
                          "peak_reserved_gib_per_gpu": summary[2].item(),
                          "measured_steps": step - warmup, "parameters": count, "world_size": world,
                          "seq_len": cfg["seq_len"], "micro_batch": cfg["micro_batch"],
                          "accumulation_steps": cfg["accumulation_steps"], "architecture": cfg["architecture"]}
                if rank == 0:
                    atomic_json(out / "benchmark.json", result)
                    print(json.dumps(result, indent=2), flush=True)
                break
            continue
        if step % cfg["log_every"] == 0 or step == 1:
            if world > 1:
                dist.all_reduce(step_stats, op=dist.ReduceOp.SUM)
            synchronize(device)
            now = time.perf_counter()
            if rank == 0:
                row = {"step": step, "tokens_seen": progress["tokens_seen"], "train_nll": (step_stats[0] / step_stats[1]).item(),
                       "learning_rate": lr, "gradient_norm": float(norm),
                       "tokens_per_second": (progress["tokens_seen"] - log_tokens) / max(1e-6, now - log_time),
                       "elapsed_hours": (now - start_time) / 3600, "amp_skipped_updates": skips}
                log_path = out / "train_log.csv"
                fresh = not log_path.exists()
                with open(log_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    if fresh:
                        writer.writeheader()
                    writer.writerow(row)
                print(json.dumps(row), flush=True)
            log_time, log_tokens = now, progress["tokens_seen"]
        should_stop = False
        if step % cfg.get("time_check_every", 10) == 0 or progress["tokens_seen"] >= cfg["target_tokens"]:
            flag = torch.tensor(int(stop_requested[0] or time.perf_counter() - start_time >= cfg["max_hours"] * 3600), device=device)
            if world > 1:
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            should_stop = bool(flag.item())
        final = should_stop or progress["tokens_seen"] >= cfg["target_tokens"]
        evaluate = step % cfg["eval_every"] == 0 or final
        if evaluate:
            last_metrics = validate(raw, val, cfg, device, world, rank)
            if not math.isfinite(last_metrics["validation_nll"]):
                raise RuntimeError("Validation loss is not finite.")
            if last_metrics["validation_nll"] < best_nll:
                best_nll = last_metrics["validation_nll"]
                save_best(raw, cfg, progress, last_metrics, rank)
            if rank == 0:
                with open(out / "validation_log.jsonl", "a") as f:
                    f.write(json.dumps({**progress, **last_metrics}) + "\n")
                print("VALIDATION", json.dumps(last_metrics), flush=True)
        if evaluate or final or step % cfg["save_every"] == 0:
            checkpoint_state(raw, optimizer, scaler, cfg, progress, best_nll, device, world, rank)
        if final:
            break
    if not args.benchmark and rank == 0:
        atomic_json(out / "training_summary.json", {**progress, **last_metrics, "best_validation_nll": best_nll,
                    "parameters": count, "wall_hours_this_session": (time.perf_counter() - start_time) / 3600,
                    "stopped_before_target": progress["tokens_seen"] < cfg["target_tokens"],
                    "amp_skipped_updates": skips})
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
