"""Four randomly initialized legal language models; no downloaded model weights."""
import math
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    architecture: str = "decoder"
    vocab_size: int = 32768
    dim: int = 512
    heads: int = 8
    kv_heads: int = 8
    layers: int = 8
    ff_dim: int = 2048
    encoder_layers: int = 0
    rnn_layers: int = 0
    local_window: int = 0
    max_positions: int = 4096
    dropout: float = 0.0
    gradient_checkpointing: bool = False


def architecture_config(name, **overrides):
    c = ModelConfig(architecture=name)
    if name == "decoder":
        pass
    elif name == "local_gqa":
        c.kv_heads, c.layers, c.local_window = 2, 9, 256
    elif name == "hybrid":
        c.layers, c.rnn_layers = 7, 2
    elif name == "encoder_decoder":
        c.layers, c.encoder_layers = 4, 3
    else:
        raise ValueError(f"Unknown architecture: {name}")
    for k, v in overrides.items():
        if not hasattr(c, k):
            raise ValueError(k)
        setattr(c, k, v)
    assert c.dim % c.heads == 0 and c.heads % c.kv_heads == 0
    assert (c.dim // c.heads) % 2 == 0
    return c


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # Reduction in FP32 is useful with T4 FP16 training.
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight.to(x.dtype)


class Rotary(nn.Module):
    def __init__(self, head_dim, max_positions):
        super().__init__()
        inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        angles = torch.outer(torch.arange(max_positions).float(), inv)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x, offset=0):
        n = x.size(-2)
        if offset + n > self.cos.size(0):
            raise ValueError("Position limit exceeded; increase max_positions in the model config.")
        cos = self.cos[offset:offset + n].to(x.dtype)[None, None]
        sin = self.sin[offset:offset + n].to(x.dtype)[None, None]
        a, b = x[..., 0::2], x[..., 1::2]
        return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, c, causal=True, cross=False):
        super().__init__()
        self.heads, self.kv_heads = c.heads, c.kv_heads
        self.hd = c.dim // c.heads
        self.q = nn.Linear(c.dim, c.dim, bias=False)
        self.k = nn.Linear(c.dim, self.kv_heads * self.hd, bias=False)
        self.v = nn.Linear(c.dim, self.kv_heads * self.hd, bias=False)
        self.out = nn.Linear(c.dim, c.dim, bias=False)
        self.rotary = None if cross else Rotary(self.hd, c.max_positions)
        self.causal, self.cross = causal, cross
        self.window = c.local_window if causal and not cross else 0
        self.dropout = c.dropout

    def forward(self, x, memory=None, past=None, position=0, use_cache=False):
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.heads, self.hd).transpose(1, 2)
        if self.cross and past is not None:
            k, v = past  # Encoder K/V projections are also cached during decoding.
        else:
            source = x if memory is None else memory
            s = source.size(1)
            k = self.k(source).view(b, s, self.kv_heads, self.hd).transpose(1, 2)
            v = self.v(source).view(b, s, self.kv_heads, self.hd).transpose(1, 2)
            if self.rotary is not None:
                k = self.rotary(k, position)
            if past is not None:
                k, v = torch.cat((past[0], k), -2), torch.cat((past[1], v), -2)
        if self.rotary is not None:
            q = self.rotary(q, position)
        new_cache = (k, v) if use_cache else None
        # Portable GQA: project/cache fewer KV heads, repeat only for the SDPA call.
        # This is not a block-sparse implementation or a guaranteed training speedup.
        if self.kv_heads != self.heads:
            repeats = self.heads // self.kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        mask, is_causal = None, False
        if self.causal:
            s = k.size(-2)
            if self.window or (past is not None and t > 1):
                qi = torch.arange(s - t, s, device=x.device)[:, None]
                ki = torch.arange(s, device=x.device)[None, :]
                mask = ki <= qi
                if self.window:
                    mask = mask & ((qi - ki) < self.window)
            elif past is None:
                is_causal = True
            # Cached single-token query must see all prior keys. is_causal=True
            # here would incorrectly use the upper-left non-square mask.
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = self.out(y.transpose(1, 2).contiguous().view(b, t, -1))
        return y, new_cache


class SwiGLU(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate = nn.Linear(c.dim, c.ff_dim, bias=False)
        self.up = nn.Linear(c.dim, c.ff_dim, bias=False)
        self.down = nn.Linear(c.ff_dim, c.dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, c, encoder=False):
        super().__init__()
        self.attn_norm, self.ff_norm = RMSNorm(c.dim), RMSNorm(c.dim)
        self.attn = Attention(c, causal=not encoder)
        self.ff = SwiGLU(c)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, past=None, position=0, use_cache=False):
        y, cache = self.attn(self.attn_norm(x), past=past, position=position, use_cache=use_cache)
        x = x + self.drop(y)
        x = x + self.drop(self.ff(self.ff_norm(x)))
        return (x, cache) if use_cache else x


class CrossBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_norm, self.cross_norm, self.ff_norm = (RMSNorm(c.dim) for _ in range(3))
        self.self_attn = Attention(c)
        self.cross_attn = Attention(c, causal=False, cross=True)
        self.ff, self.drop = SwiGLU(c), nn.Dropout(c.dropout)

    def forward(self, x, memory, past=None, cross_past=None, position=0, use_cache=False):
        y, cache = self.self_attn(self.self_norm(x), past=past, position=position, use_cache=use_cache)
        x = x + self.drop(y)
        y, cross_cache = self.cross_attn(self.cross_norm(x), memory=memory, past=cross_past, use_cache=use_cache)
        x = x + self.drop(y)
        x = x + self.drop(self.ff(self.ff_norm(x)))
        return (x, cache, cross_cache) if use_cache else x


class LegalLM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.embedding = nn.Embedding(c.vocab_size, c.dim)
        # Output projection uses embedding.weight; there is no second vocabulary matrix.
        if c.architecture == "encoder_decoder":
            self.encoder = nn.ModuleList(Block(c, encoder=True) for _ in range(c.encoder_layers))
            self.encoder_norm = RMSNorm(c.dim)
            self.blocks = nn.ModuleList(CrossBlock(c) for _ in range(c.layers))
        else:
            self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        if c.rnn_layers:
            self.rnn = nn.LSTM(c.dim, c.dim, num_layers=c.rnn_layers, batch_first=True)
            self.rnn_norm = RMSNorm(c.dim)
        self.final_norm = RMSNorm(c.dim)
        self.apply(self._init)
        # Scale residual projections to keep initial residual variance controlled.
        std = 0.02 / math.sqrt(2 * c.layers)
        for name, p in self.named_parameters():
            if name.endswith(("attn.out.weight", "self_attn.out.weight", "cross_attn.out.weight", "ff.down.weight")):
                nn.init.normal_(p, mean=0.0, std=std)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LSTM):
            for name, p in m.named_parameters():
                if "weight" in name:
                    nn.init.normal_(p, mean=0.0, std=0.02)
                else:
                    nn.init.zeros_(p)

    def encode(self, source_ids):
        x = self.embedding(source_ids)
        for block in self.encoder:
            if self.config.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.encoder_norm(x)

    def forward(self, input_ids, source_ids=None, state=None, use_cache=False):
        state = state or {}
        position = state.get("position", 0)
        x = self.embedding(input_ids)
        new_state = {"position": position + input_ids.size(1), "self": []}
        if self.config.rnn_layers:
            y, recurrent_state = self.rnn(x, state.get("rnn"))
            x = x + self.rnn_norm(y)
            new_state["rnn"] = recurrent_state
        if self.config.architecture == "encoder_decoder":
            memory = state.get("memory")
            if memory is None:
                if source_ids is None:
                    raise ValueError("encoder_decoder requires source_ids on the first call.")
                memory = self.encode(source_ids)
            new_state.update(memory=memory, cross=[])
            for i, block in enumerate(self.blocks):
                if use_cache:
                    p = state.get("self", [None] * len(self.blocks))[i]
                    cp = state.get("cross", [None] * len(self.blocks))[i]
                    x, cache, cc = block(x, memory, p, cp, position, True)
                    new_state["self"].append(cache)
                    new_state["cross"].append(cc)
                elif self.config.gradient_checkpointing and self.training:
                    x = checkpoint(block, x, memory, use_reentrant=False)
                else:
                    x = block(x, memory)
        else:
            for i, block in enumerate(self.blocks):
                if use_cache:
                    p = state.get("self", [None] * len(self.blocks))[i]
                    x, cache = block(x, p, position, True)
                    new_state["self"].append(cache)
                elif self.config.gradient_checkpointing and self.training:
                    x = checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)
        x = self.final_norm(x)
        return (x, new_state) if use_cache else x

    def project(self, hidden):
        return F.linear(hidden, self.embedding.weight)


def parameter_count(model):
    # PyTorch de-duplicates shared Parameter objects here.
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_formula(c):
    d, h, kv = c.dim, c.dim // c.heads, c.kv_heads
    attention = 2 * d * d + 2 * d * kv * h
    ff = 3 * d * c.ff_dim
    total = c.vocab_size * d + d
    if c.architecture == "encoder_decoder":
        total += c.encoder_layers * (attention + ff + 2 * d) + d
        total += c.layers * (2 * attention + ff + 3 * d)
    else:
        total += c.layers * (attention + ff + 2 * d)
    if c.rnn_layers:
        total += c.rnn_layers * (8 * d * d + 8 * d) + d
    return total


def make_batch(blocks, architecture, prefix_fraction=0.80, objective="full"):
    # blocks has T+1 tokens. The target suffix is identical across architectures.
    t = blocks.size(1) - 1
    split = max(1, min(t, int(t * prefix_fraction)))
    if architecture == "encoder_decoder":
        source = blocks[:, :split]
        inputs, targets = blocks[:, split - 1:-1], blocks[:, split:]
    else:
        source = None
        inputs, targets = blocks[:, :-1], blocks[:, 1:].clone()
        if objective == "continuation":
            targets[:, :split - 1] = -100
    return inputs, targets, source


def language_model_loss(hidden, targets, weight, chunk_tokens=256, checkpoint_chunks=False):
    """Sum/token count lets validation aggregate the true token-weighted NLL."""
    hidden, targets = hidden.reshape(-1, hidden.size(-1)), targets.reshape(-1)
    total = hidden.new_zeros((), dtype=torch.float32)
    count = (targets != -100).sum()
    def loss_chunk(h, y, w):
        # Keep logits in autocast precision until the bounded FP32 CE conversion.
        return F.cross_entropy(F.linear(h, w).float(), y, ignore_index=-100, reduction="sum")
    for i in range(0, targets.numel(), chunk_tokens):
        h, y = hidden[i:i + chunk_tokens], targets[i:i + chunk_tokens]
        if checkpoint_chunks and torch.is_grad_enabled():
            total = total + checkpoint(loss_chunk, h, y, weight, use_reentrant=False)
        else:
            total = total + loss_chunk(h, y, weight)
    return total / count.clamp_min(1), total.detach(), count.detach()


if __name__ == "__main__":
    for name in ("decoder", "encoder_decoder", "local_gqa", "hybrid"):
        c = architecture_config(name)
        with torch.device("meta"):
            model = LegalLM(c)
        n = parameter_count(model)
        assert n == parameter_formula(c) and 50_000_000 <= n <= 300_000_000
        print(name, f"{n:,}", asdict(c))
