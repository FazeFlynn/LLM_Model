"""
═══════════════════════════════════════════════════════════════════════════════
LLM 350M — OPTIMIZED TRAINING SCRIPT (A100 80GB)
═══════════════════════════════════════════════════════════════════════════════
Key optimizations techniques used:
1. Chunked cross-entropy (custom autograd): never materializes full (B*T, V) logits
   → saves ~18-20 GB peak memory during backward
2. Larger micro_batch_size=32, fewer grad_accum_steps=17 (was 24/22)
   → fewer loop iterations (less overhead), better GPU utilization
3. Background data prefetching thread with proper state tracking
   → hides variable 10-100ms data loading latency
4. Data kept as uint16 in RAM (4x smaller than int64)
5. MFU (Model FLOPs Utilization) tracking for performance analysis
═══════════════════════════════════════════════════════════════════════════════
Expected: ~75-90K tok/s on A100 80GB
Peak memory: ~68-72 GB (vs ~76.5 GB before, enabling larger batch)
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import math
import time
import json
import signal
import queue
import threading
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    vocab_size: int = 50304
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 1024
    block_size: int = 2048
    dropout: float = 0.0
    bias: bool = False
    # Chunk size for cross-entropy: controls peak memory of loss computation.
    # 4096 tokens × 50304 vocab × 4 bytes (FP32) ≈ 0.8 GB per chunk.
    loss_chunk_size: int = 4096


@dataclass
class TrainConfig:
    data_dir: str = "/workspace/tokenized"
    out_dir: str = "/workspace/checkpoints"

    # OPTIMIZED: larger micro batch + fewer accum steps = fewer iterations of overhead
    # Effective batch: 32 × 17 = 544 seqs × 2048 tok = 1,114,112 tok/step
    micro_batch_size: int = 32
    gradient_accum_steps: int = 17

    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 613      # ~10% of max_steps
    max_steps: int = 6131        # ~6.83B tokens / 1,114,112 tok/step

    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    dtype: str = "bfloat16"
    compile_model: bool = True

    log_interval: int = 10
    eval_interval: int = 250
    eval_steps: int = 50
    checkpoint_interval: int = 500
    keep_last_n_checkpoints: int = 5

    wandb_log: bool = True
    wandb_project: str = "llm-350m"
    wandb_run_name: str = ""

    header_ints: int = 256
    prefetch_batches: int = 3  # Number of batches to prefetch in background


# ═══════════════════════════════════════════════════════════════════════════════
# CHUNKED CROSS-ENTROPY (Custom Autograd Function)
# ═══════════════════════════════════════════════════════════════════════════════
# This is the key memory optimization. Instead of computing:
#   logits = hidden @ weight.T   → shape (B*T, V), ~10 GB in FP32
#   loss = cross_entropy(logits, targets)
#
# We process in chunks of `chunk_size` tokens at a time:
#   - Forward: compute loss chunk-by-chunk, only keep scalar loss
#   - Backward: recompute logits per chunk, compute gradients online
#
# Memory savings: ~18-20 GB (eliminates the full logits + gradients tensor)
# Compute overhead: <3% (one extra matmul per chunk in backward for recomputation)
# ═══════════════════════════════════════════════════════════════════════════════

class ChunkedCrossEntropyLoss(torch.autograd.Function):
    """
    Fused linear projection + cross-entropy loss that never materializes
    the full (N, V) logits tensor. Processes tokens in chunks.

    Forward: For each chunk, compute logits → CE loss → accumulate scalar.
    Backward: For each chunk, recompute logits → softmax → grad.

    This function is opaque to torch.compile (treated as a single op),
    which is fine since the transformer layers still get fully compiled.
    """

    @staticmethod
    def forward(ctx, hidden, weight, targets, chunk_size):
        """
        Args:
            hidden: (N, D) bfloat16 — output of last layer norm
            weight: (V, D) bfloat16 — lm_head weight (tied with wte)
            targets: (N,) int64
            chunk_size: int — tokens per chunk
        Returns:
            scalar loss (float32)
        """
        N, D = hidden.shape
        V = weight.shape[0]

        loss_sum = 0.0
        n_valid = 0

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            h_chunk = hidden[start:end]          # (cs, D) bf16
            t_chunk = targets[start:end]         # (cs,)

            # Compute logits in float32 for numerical stability
            logits = h_chunk.float() @ weight.float().T  # (cs, V)

            # Cross-entropy (reduction='sum' for proper averaging later)
            chunk_loss = F.cross_entropy(
                logits, t_chunk, ignore_index=-1, reduction='sum'
            )

            loss_sum += chunk_loss.item()
            n_valid += (t_chunk != -1).sum().item()

        # Save for backward (only hidden states + metadata, NOT logits)
        ctx.save_for_backward(hidden, weight, targets)
        ctx.chunk_size = chunk_size
        ctx.n_valid = n_valid

        # Return mean loss
        loss_val = loss_sum / max(n_valid, 1)
        return hidden.new_tensor(loss_val, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Recompute logits per chunk and compute gradients online.
        Peak additional memory: ~2 × chunk_size × V × 4 bytes ≈ 1.6 GB
        """
        hidden, weight, targets = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        n_valid = ctx.n_valid
        N, D = hidden.shape

        grad_hidden = torch.zeros_like(hidden)  # (N, D) bf16
        grad_weight_acc = torch.zeros(
            weight.shape, device=weight.device, dtype=torch.float32
        )

        # Scale factor: grad_output / n_valid (chain rule through mean)
        scale = grad_output.float() / max(n_valid, 1)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            h_chunk = hidden[start:end].float()  # (cs, D) fp32
            t_chunk = targets[start:end]          # (cs,)

            # Recompute logits (this is the "extra" compute, ~2% overhead)
            logits = h_chunk @ weight.float().T   # (cs, V) fp32

            # Softmax → probabilities
            probs = F.softmax(logits, dim=-1)     # (cs, V) fp32

            # CE gradient w.r.t. logits: d_logits = probs - one_hot(targets)
            d_logits = probs
            valid_mask = (t_chunk != -1)
            if valid_mask.any():
                valid_rows = torch.arange(
                    end - start, device=hidden.device
                )[valid_mask]
                d_logits[valid_rows, t_chunk[valid_mask]] -= 1.0
            # Zero out padded/ignored positions
            d_logits[~valid_mask] = 0.0
            d_logits *= scale  # Apply chain rule scaling

            # Gradient w.r.t. hidden: (cs, V) @ (V, D) → (cs, D)
            grad_hidden[start:end] = (d_logits @ weight.float()).to(hidden.dtype)

            # Gradient w.r.t. weight: (V, cs) @ (cs, D) → (V, D)
            grad_weight_acc += d_logits.T @ h_chunk

        return grad_hidden, grad_weight_acc.to(weight.dtype), None, None


def chunked_cross_entropy(hidden, weight, targets, chunk_size):
    """Wrapper for the custom autograd function."""
    return ChunkedCrossEntropyLoss.apply(hidden, weight, targets, chunk_size)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.act = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f=RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # Scale residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n = sum(p.numel() for p in self.parameters()) - \
            self.transformer.wpe.weight.numel()
        print(f"  Model params     : {n/1e6:.1f}M  (excl. position emb)")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # ── CHUNKED CROSS-ENTROPY ────────────────────────────────────
            # Instead of: logits = lm_head(x); loss = CE(logits, targets)
            # which materializes a (B*T, V) = (65536, 50304) FP32 tensor (~13 GB!),
            # we compute loss in chunks, never holding full logits in memory.
            hidden_flat = x.view(-1, self.config.n_embd)  # (B*T, D)
            targets_flat = targets.view(-1)                # (B*T,)
            loss = chunked_cross_entropy(
                hidden_flat, self.lm_head.weight, targets_flat,
                self.config.loss_chunk_size
            )
            return None, loss  # Don't return logits during training (saves memory)
        else:
            # Inference: only compute logits for last position
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    def configure_optimizers(self, cfg: TrainConfig):
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() < 2 or 'wpe' in name or 'wte' in name:
                no_decay.append(param)
            else:
                decay.append(param)
        print(f"  Decay params     : {sum(p.numel() for p in decay):,}")
        print(f"  No-decay params  : {sum(p.numel() for p in no_decay):,}")
        use_fused = torch.cuda.is_available()
        print(f"  Fused AdamW      : {use_fused}")
        return torch.optim.AdamW(
            [
                {"params": decay,    "weight_decay": cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=cfg.max_lr,
            betas=(cfg.beta1, cfg.beta2),
            eps=1e-8,
            fused=use_fused,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING WITH PREFETCHING
# ═══════════════════════════════════════════════════════════════════════════════

class ShardedDataLoader:
    """
    Memory-efficient data loader with background prefetching.

    Improvements over original:
    - Data stored as uint16 in RAM (4x less than int64)
    - Background thread pre-loads batches → hides 10-100ms loading latency
    - Proper state tracking for checkpointing (even with prefetch queue)
    """

    def __init__(self, data_dir: str, split: str, batch_size: int,
                 seq_len: int, header_ints: int = 256, prefetch_batches: int = 3):
        self.B = batch_size
        self.T = seq_len
        self.header_ints = header_ints
        self.shards = sorted(Path(data_dir).glob(f"{split}_*.bin"))
        if not self.shards:
            raise FileNotFoundError(f"No {split} shards in {data_dir}")

        total = 0
        for s in self.shards:
            header = np.fromfile(s, dtype=np.int32, count=header_ints)
            total += int(header[2])
        print(f"  [{split}] {len(self.shards)} shards, {total:,} tokens")

        self.shard_idx = 0
        self.pos = 0
        self.data = None  # uint16 numpy array (memory efficient)
        self._load_shard(0)

        # State tracking: the position the MAIN THREAD has consumed up to
        self._consumed_shard_idx = 0
        self._consumed_pos = 0

        # Prefetching
        self._prefetch_n = prefetch_batches
        self._queue = None
        self._stop_event = None
        self._thread = None
        if prefetch_batches > 0:
            self._queue = queue.Queue(maxsize=prefetch_batches)
            self._stop_event = threading.Event()
            self._start_worker()

    def _load_shard(self, idx: int):
        """Load shard as uint16 array (compact representation)."""
        self.shard_idx = idx % len(self.shards)
        raw = np.fromfile(self.shards[self.shard_idx], dtype=np.uint16)
        # Skip header: header_ints int32 values = header_ints * 2 uint16 values
        self.data = raw[self.header_ints * 2:]
        self.pos = 0

    def _produce_batch(self):
        """
        Produce one batch from current position.
        Returns: (x_np, y_np, shard_idx_before, pos_before)
        Called by the worker thread (or main thread if no prefetch).
        """
        B, T = self.B, self.T
        needed = B * T + 1
        if self.pos + needed > len(self.data):
            self._load_shard(self.shard_idx + 1)

        # Record state BEFORE consuming (for checkpoint tracking)
        s_idx = self.shard_idx
        s_pos = self.pos

        buf = self.data[self.pos:self.pos + needed].astype(np.int64)
        x = buf[:-1].reshape(B, T).copy()
        y = buf[1:].reshape(B, T).copy()
        self.pos += B * T

        return x, y, s_idx, s_pos

    def _start_worker(self):
        """Start background prefetch thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def _worker_loop(self):
        """Background thread: continuously produce batches into queue."""
        while not self._stop_event.is_set():
            try:
                item = self._produce_batch()
                self._queue.put(item, timeout=0.5)
            except queue.Full:
                continue
            except Exception as e:
                if not self._stop_event.is_set():
                    print(f"  ⚠ Prefetch worker error: {e}")
                break

    def _stop_worker(self):
        """Stop background thread and drain queue."""
        if self._thread is not None and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=3.0)
            # Drain queue
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

    def next_batch(self):
        """
        Get next batch as pinned tensors. Thread-safe for main thread.
        Returns: (x, y) — both LongTensor, pinned memory
        """
        if self._queue is not None:
            x_np, y_np, s_idx, s_pos = self._queue.get()
        else:
            x_np, y_np, s_idx, s_pos = self._produce_batch()

        # Track consumed position (for checkpointing)
        self._consumed_shard_idx = s_idx
        self._consumed_pos = s_pos + self.B * self.T

        x = torch.from_numpy(x_np).pin_memory()
        y = torch.from_numpy(y_np).pin_memory()
        return x, y

    def get_state(self) -> dict:
        """Get state for checkpoint. Returns position of next unconsumed batch."""
        return {
            "shard_idx": self._consumed_shard_idx,
            "pos": self._consumed_pos,
        }

    def set_state(self, state: dict):
        """Restore state from checkpoint."""
        # Stop worker if running
        if self._queue is not None:
            self._stop_worker()

        # Restore position
        self._load_shard(state.get("shard_idx", 0))
        self.pos = state.get("pos", 0)
        self._consumed_shard_idx = self.shard_idx
        self._consumed_pos = self.pos

        # Restart worker
        if self._queue is not None:
            self._stop_event = threading.Event()
            self._start_worker()

    def stop(self):
        """Clean shutdown."""
        if self._queue is not None:
            self._stop_worker()


# ═══════════════════════════════════════════════════════════════════════════════
# LR SCHEDULE
# ═══════════════════════════════════════════════════════════════════════════════

def get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.max_lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    t = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * t)) * (cfg.max_lr - cfg.min_lr)


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINTING
# ═══════════════════════════════════════════════════════════════════════════════

def get_raw(model: nn.Module) -> nn.Module:
    m = model
    if hasattr(m, '_orig_mod'):
        m = m._orig_mod
    if hasattr(m, 'module'):
        m = m.module
    return m


def save_checkpoint(model, optimizer, step, best_val_loss,
                    loader_state, cfg, mcfg, losses):
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(cfg.out_dir) / f"ckpt_step_{step:06d}.pt"
    tmp = path.with_suffix('.tmp')
    torch.save({
        "model_state_dict": get_raw(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "train_loader_state": loader_state,
        "model_config": vars(mcfg),
        "train_config": vars(cfg),
        "losses": losses,
        "rng_state": {
            "python": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
            "numpy": np.random.get_state(),
        },
    }, tmp)
    tmp.rename(path)
    latest = Path(cfg.out_dir) / "ckpt_latest.pt"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(path.name)
    all_ckpts = sorted(Path(cfg.out_dir).glob("ckpt_step_*.pt"))
    for old in all_ckpts[:-cfg.keep_last_n_checkpoints]:
        old.unlink()
    print(f"  💾 Checkpoint saved: {path.name}")


def load_checkpoint(path, model, optimizer, device):
    print(f"  Loading: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    get_raw(model).load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # Restore RNG safely
    if "rng_state" in ckpt:
        try:
            py_rng = ckpt["rng_state"]["python"]
            if not isinstance(py_rng, torch.ByteTensor):
                py_rng = torch.ByteTensor(list(py_rng)) if not isinstance(
                    py_rng, torch.Tensor) else py_rng.byte()
            torch.random.set_rng_state(py_rng.cpu())
            if "cuda" in ckpt["rng_state"]:
                cuda_rng = ckpt["rng_state"]["cuda"]
                if isinstance(cuda_rng, torch.Tensor):
                    torch.cuda.set_rng_state(cuda_rng.cpu())
            if "numpy" in ckpt["rng_state"]:
                np.random.set_state(ckpt["rng_state"]["numpy"])
            print("  ✓ RNG states restored")
        except Exception as e:
            print(f"  ⚠ RNG restore skipped: {e}")

    print(f"  ✓ Resumed from step {ckpt['step']}")
    return ckpt


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, val_loader, eval_steps, device, ctx):
    raw = get_raw(model)
    raw.eval()
    losses = []
    for _ in range(eval_steps):
        x, y = val_loader.next_batch()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with ctx:
            _, loss = raw(x, y)
        losses.append(loss.item())
    raw.train()
    return float(np.mean(losses))


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mfu(tok_per_sec: float, n_params: int, gpu_bf16_tflops: float = 312.0) -> float:
    """
    Compute Model FLOPs Utilization.
    Standard approximation: 6 * N_params FLOPs per token for training
    (2 for forward, 4 for backward ≈ 2x forward for grads + weight updates).
    A100 SXM4 80GB peak: 312 TFLOPS BF16.
    """
    flops_per_token = 6 * n_params
    achieved_tflops = tok_per_sec * flops_per_token / 1e12
    return achieved_tflops / gpu_bf16_tflops


def log_mem(tag: str):
    """Log current and peak GPU memory."""
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"    [MEM {tag}] allocated={alloc:.2f}GB  reserved={reserved:.2f}GB  peak={peak:.2f}GB")


def log_time(tag: str, t_ref: float) -> float:
    """Log elapsed ms since t_ref. Returns current time."""
    torch.cuda.synchronize()
    now = time.perf_counter()
    print(f"    [TIME {tag}] {(now - t_ref)*1000:.2f} ms")
    return now


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--debug-steps", type=int, default=5,
                        help="Detailed per-microstep logging for first N steps")
    parser.add_argument("--micro-batch", type=int, default=None,
                        help="Override micro_batch_size (for memory testing)")
    parser.add_argument("--accum-steps", type=int, default=None,
                        help="Override gradient_accum_steps")
    args = parser.parse_args()

    mcfg = ModelConfig()
    cfg = TrainConfig()
    if args.no_compile:
        cfg.compile_model = False
    if args.no_wandb:
        cfg.wandb_log = False
    if args.micro_batch:
        cfg.micro_batch_size = args.micro_batch
    if args.accum_steps:
        cfg.gradient_accum_steps = args.accum_steps
    if not cfg.wandb_run_name:
        cfg.wandb_run_name = f"350m-opt-{time.strftime('%m%d-%H%M')}"

    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')  # Enable TF32 for FP32 matmuls in chunked CE
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n{'═'*70}")
    print(f"  GPU : {gpu_name}  ({gpu_mem_gb:.0f} GB)")
    print(f"{'═'*70}")

    # ── System info ──────────────────────────────────────────────────────────
    print(f"\n{'═'*70}\n  SYSTEM INFO\n{'═'*70}")
    print(f"  PyTorch version    : {torch.__version__}")
    print(f"  CUDA version       : {torch.version.cuda}")
    print(f"  cuDNN version      : {torch.backends.cudnn.version()}")
    print(f"  Flash Attention    : {torch.backends.cuda.flash_sdp_enabled()}")
    print(f"  Mem-Efficient SDPA : {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    print(f"  TF32 matmul        : {torch.backends.cuda.matmul.allow_tf32}")
    print(f"  Float32 precision  : {torch.get_float32_matmul_precision()}")

    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    # ── Model ────────────────────────────────────────────────────────────────
    print(f"\n{'═'*70}\n  INITIALIZING MODEL\n{'═'*70}")
    model = GPT(mcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) - \
        model.transformer.wpe.weight.numel()
    optimizer = model.configure_optimizers(cfg)

    torch.cuda.synchronize()
    log_mem("after model init + optimizer")

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"\n  Parameter memory on GPU   : {param_bytes / 1e9:.3f} GB")
    print(f"  Optimizer states (est.)   : {param_bytes * 3 / 1e9:.3f} GB "
          f"(fp32 params + momentum + variance)")
    print(f"  Chunked CE chunk_size     : {mcfg.loss_chunk_size} tokens")
    print(f"  Chunk logits memory (est.): {mcfg.loss_chunk_size * mcfg.vocab_size * 4 / 1e9:.3f} GB")

    # ── Compile ──────────────────────────────────────────────────────────────
    if cfg.compile_model:
        print(f"\n  Compiling with torch.compile (mode='default')…")
        t_compile_start = time.perf_counter()
        model = torch.compile(model, mode="default")
        print(f"  ✓ Compiled (setup took {time.perf_counter()-t_compile_start:.1f}s, "
              f"actual compilation happens on first forward)")

    # ── Data ─────────────────────────────────────────────────────────────────
    print(f"\n{'═'*70}\n  DATA\n{'═'*70}")
    train_loader = ShardedDataLoader(
        cfg.data_dir, "train", cfg.micro_batch_size, mcfg.block_size,
        cfg.header_ints, prefetch_batches=cfg.prefetch_batches,
    )
    val_loader = ShardedDataLoader(
        cfg.data_dir, "val", cfg.micro_batch_size, mcfg.block_size,
        cfg.header_ints, prefetch_batches=1,
    )

    # ── Data loading speed test ──────────────────────────────────────────────
    print(f"\n  Data loading speed test (20 batches with prefetch):")
    time.sleep(0.5)  # Let prefetch fill queue
    t_dl = time.perf_counter()
    for _ in range(20):
        _x, _y = train_loader.next_batch()
    t_dl = time.perf_counter() - t_dl
    print(f"    20 batches in {t_dl*1000:.1f} ms (avg {t_dl*50:.2f} ms/batch)")
    print(f"    Shape: x={list(_x.shape)}, y={list(_y.shape)}, dtype={_x.dtype}")
    print(f"    Pinned: x={_x.is_pinned()}, y={_y.is_pinned()}")
    # Reset loader
    train_loader.set_state({"shard_idx": 0, "pos": 0})

    # ── Resume ───────────────────────────────────────────────────────────────
    start_step = 0
    best_val_loss = float('inf')
    loss_history = {"train": [], "val": [], "lr": []}

    if args.resume:
        ckpt_path = (Path(args.ckpt) if args.ckpt
                     else Path(cfg.out_dir) / "ckpt_latest.pt")
        if ckpt_path.exists():
            if ckpt_path.is_symlink():
                ckpt_path = ckpt_path.parent / os.readlink(ckpt_path)
            ckpt = load_checkpoint(ckpt_path, model, optimizer, device)
            start_step = ckpt["step"] + 1
            best_val_loss = ckpt.get("best_val_loss", float('inf'))
            loss_history = ckpt.get("losses", loss_history)
            if ckpt.get("train_loader_state"):
                train_loader.set_state(ckpt["train_loader_state"])
            # Note: if resuming from a checkpoint with different batch config,
            # the token position in the data is still correct.
            print(f"  ℹ️  Note: If batch config changed, LR schedule may differ slightly")
        else:
            print(f"  ⚠  No checkpoint at {ckpt_path}, starting fresh")

    # ── W&B ──────────────────────────────────────────────────────────────────
    if cfg.wandb_log:
        import wandb
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config={**vars(mcfg), **vars(cfg)},
            resume="allow" if args.resume else None,
        )

    # ── Training info ─────────────────────────────────────────────────────────
    tps_tok = cfg.micro_batch_size * cfg.gradient_accum_steps * mcfg.block_size
    total_tokens_target = cfg.max_steps * tps_tok

    print(f"\n{'═'*70}\n  TRAINING CONFIG\n{'═'*70}")
    print(f"  Micro batch       : {cfg.micro_batch_size}")
    print(f"  Grad accum steps  : {cfg.gradient_accum_steps}")
    print(f"  Effective batch   : {cfg.micro_batch_size * cfg.gradient_accum_steps} seqs")
    print(f"  Tokens/step       : {tps_tok:,}")
    print(f"  Total steps       : {cfg.max_steps:,}")
    print(f"  Total tokens      : {total_tokens_target:,}  ({total_tokens_target/1e9:.2f}B)")
    print(f"  Warmup steps      : {cfg.warmup_steps}")
    print(f"  Max LR / Min LR   : {cfg.max_lr} / {cfg.min_lr}")
    print(f"  Compile           : {cfg.compile_model}")
    print(f"  Precision         : {cfg.dtype}")
    print(f"  Data prefetch     : {cfg.prefetch_batches} batches")
    if start_step > 0:
        print(f"  Resuming from     : step {start_step}")

    # ── Memory budget estimate ───────────────────────────────────────────────
    B = cfg.micro_batch_size
    T = mcfg.block_size
    V = mcfg.vocab_size
    D = mcfg.n_embd
    L = mcfg.n_layer
    CS = mcfg.loss_chunk_size

    print(f"\n  {'─'*60}")
    print(f"  MEMORY BUDGET (ESTIMATED WITH CHUNKED CE)")
    print(f"  {'─'*60}")
    print(f"    Batch config: B={B}, T={T}, V={V}, D={D}, L={L}, chunk={CS}")
    print(f"    Model + optimizer (fixed)    : ~12 GB")
    # Rough activation estimate: each layer saves input + attention output ≈ 2*B*T*D*2 bytes
    act_est = L * 2 * B * T * D * 2 / 1e9
    print(f"    Activations (est., bf16)     : ~{act_est:.1f} GB")
    chunk_mem = CS * V * 4 / 1e9
    print(f"    Chunked CE peak (per chunk)  : ~{chunk_mem:.2f} GB  (vs ~{B*T*V*4/1e9:.1f} GB full logits)")
    est_peak = 12 + act_est + chunk_mem * 2 + 5  # +5 for backward overhead
    print(f"    Estimated peak               : ~{est_peak:.0f} GB")
    print(f"    GPU available                : {gpu_mem_gb:.0f} GB")
    print(f"    Estimated headroom           : ~{gpu_mem_gb - est_peak:.0f} GB")
    if est_peak > gpu_mem_gb * 0.95:
        print(f"    ⚠ WARNING: Tight on memory! Consider reducing micro_batch_size")
    print(f"  {'─'*60}")

    # ── MFU reference ────────────────────────────────────────────────────────
    print(f"\n  MFU REFERENCE:")
    print(f"    Model params (for MFU): {n_params/1e6:.1f}M")
    print(f"    FLOPs per token (6N)  : {6*n_params/1e9:.2f} GFLOPS")
    print(f"    A100 peak BF16        : 312 TFLOPS")
    print(f"    100% MFU at           : {312e12 / (6*n_params) / 1e3:.1f}K tok/s")
    print(f"    Target 50% MFU        : {0.5 * 312e12 / (6*n_params) / 1e3:.1f}K tok/s")
    print()

    # ── Signal handlers ──────────────────────────────────────────────────────
    shutdown = [False]

    def _handler(s, f):
        if shutdown[0]:
            sys.exit(1)
        print("\n\n  ⚠  Ctrl+C — saving checkpoint after this step…")
        shutdown[0] = True
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    # ── Save initial checkpoint ──────────────────────────────────────────────
    if start_step == 0:
        save_checkpoint(model, optimizer, 0, best_val_loss,
                        train_loader.get_state(), cfg, mcfg, loss_history)

    # ═══════════════════════════════════════════════════════════════════════
    # TRAINING LOOP
    # ═══════════════════════════════════════════════════════════════════════
    model.train()
    t0 = time.perf_counter()
    t_start = time.perf_counter()
    tokens_processed = start_step * tps_tok
    running_loss = 0.0
    step_times = []

    DEBUG_STEPS = args.debug_steps

    print(f"\n{'═'*70}")
    print(f"  TRAINING LOOP START (debug logging for first {DEBUG_STEPS} steps)")
    print(f"{'═'*70}\n")

    for step in range(start_step, cfg.max_steps):

        debug = (step < start_step + DEBUG_STEPS)

        if debug:
            print(f"\n  {'='*60}")
            print(f"  DEBUG STEP {step}")
            print(f"  {'='*60}")
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t_step_start = time.perf_counter()
            log_mem("step-start")

        # ── LR schedule ─────────────────────────────────────────────────
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # ── Gradient accumulation ────────────────────────────────────────
        if debug:
            t_mark = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)

        if debug:
            torch.cuda.synchronize()
            t_mark = log_time("zero_grad", t_mark)

        loss_accum = 0.0
        t_data_total = 0.0
        t_fwd_total = 0.0
        t_bwd_total = 0.0

        for micro_step in range(cfg.gradient_accum_steps):

            if debug:
                torch.cuda.synchronize()
                t_ms = time.perf_counter()

            # ── Data ─────────────────────────────────────────────────────
            t_d0 = time.perf_counter()
            x, y = train_loader.next_batch()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            t_d1 = time.perf_counter()
            t_data_total += (t_d1 - t_d0)

            # ── Forward ──────────────────────────────────────────────────
            if debug:
                torch.cuda.synchronize()
                t_f0 = time.perf_counter()

            with ctx:
                _, loss = model(x, y)
                loss = loss / cfg.gradient_accum_steps

            if debug:
                torch.cuda.synchronize()
                t_f1 = time.perf_counter()
                t_fwd_total += (t_f1 - t_f0)
            else:
                t_fwd_total += 0  # Not tracked outside debug

            # ── Backward ─────────────────────────────────────────────────
            if debug:
                torch.cuda.synchronize()
                t_b0 = time.perf_counter()

            loss.backward()

            if debug:
                torch.cuda.synchronize()
                t_b1 = time.perf_counter()
                t_bwd_total += (t_b1 - t_b0)

            loss_accum += loss.detach().item()

            # ── Per-microstep debug ──────────────────────────────────────
            if debug and (micro_step < 3 or micro_step == cfg.gradient_accum_steps - 1):
                torch.cuda.synchronize()
                t_ms_end = time.perf_counter()
                data_ms = (t_d1 - t_d0) * 1000
                fwd_ms = (t_f1 - t_f0) * 1000
                bwd_ms = (t_b1 - t_b0) * 1000
                total_ms = (t_ms_end - t_ms) * 1000
                print(f"    [micro {micro_step:2d}] data={data_ms:.1f}ms  fwd={fwd_ms:.1f}ms  "
                      f"bwd={bwd_ms:.1f}ms  total={total_ms:.1f}ms  "
                      f"loss={loss.item()*cfg.gradient_accum_steps:.4f}")
                log_mem(f"micro {micro_step}")
                if micro_step == 2 and cfg.gradient_accum_steps > 4:
                    print(f"    ... (micro steps 3–{cfg.gradient_accum_steps-2} omitted) ...")

        # ── Grad clip ────────────────────────────────────────────────────
        if debug:
            torch.cuda.synchronize()
            t_clip = time.perf_counter()

        if cfg.grad_clip > 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        else:
            grad_norm = torch.tensor(0.0)

        if debug:
            torch.cuda.synchronize()
            t_clip_end = time.perf_counter()
            print(f"    [CLIP] {(t_clip_end-t_clip)*1000:.2f}ms  grad_norm={grad_norm.item():.4f}")

        # ── Optimizer step ───────────────────────────────────────────────
        if debug:
            torch.cuda.synchronize()
            t_opt = time.perf_counter()

        optimizer.step()

        if debug:
            torch.cuda.synchronize()
            t_opt_end = time.perf_counter()
            print(f"    [OPTIM] {(t_opt_end-t_opt)*1000:.2f}ms")

        # ── Step timing ──────────────────────────────────────────────────
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        dt = t1 - t0
        t0 = t1
        step_times.append(dt)
        if len(step_times) > 100:
            step_times.pop(0)
        avg_dt = sum(step_times) / len(step_times)

        tokens_processed += tps_tok
        tps = tps_tok / dt
        mfu = compute_mfu(tps, n_params)
        running_loss = 0.95 * running_loss + 0.05 * loss_accum if running_loss else loss_accum

        # ── Debug step summary ───────────────────────────────────────────
        if debug:
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            t_total = t1 - t_step_start
            print(f"\n    {'─'*50}")
            print(f"    STEP {step} SUMMARY:")
            print(f"    {'─'*50}")
            print(f"      Total step time      : {t_total*1000:.1f} ms")
            print(f"      Data loading (sum)   : {t_data_total*1000:.1f} ms "
                  f"({t_data_total/t_total*100:.1f}% of step)")
            print(f"      Forward (sum)        : {t_fwd_total*1000:.1f} ms "
                  f"({t_fwd_total/t_total*100:.1f}% of step)")
            print(f"      Backward (sum)       : {t_bwd_total*1000:.1f} ms "
                  f"({t_bwd_total/t_total*100:.1f}% of step)")
            print(f"      Tokens               : {tps_tok:,}")
            print(f"      Throughput            : {tps/1e3:.1f}K tok/s")
            print(f"      MFU                   : {mfu*100:.1f}%")
            print(f"      Peak memory           : {peak_gb:.2f} GB / {gpu_mem_gb:.0f} GB "
                  f"({peak_gb/gpu_mem_gb*100:.0f}%)")
            print(f"      Loss                  : {loss_accum:.4f}")
            print(f"      Grad norm             : {grad_norm.item():.4f}")
            print(f"      LR                    : {lr:.2e}")
            if step == start_step:
                print(f"\n      ℹ️  First step includes torch.compile warmup (~30s).")
                print(f"      ℹ️  Subsequent steps show true training speed.")
            print()

        # ── Periodic logging ─────────────────────────────────────────────
        if step % cfg.log_interval == 0:
            loss_history["train"].append({"step": step, "loss": loss_accum})
            loss_history["lr"].append({"step": step, "lr": lr})

            gpu_used = torch.cuda.memory_allocated() / 1e9
            gpu_peak = torch.cuda.max_memory_allocated() / 1e9
            gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
            progress = (step + 1) / cfg.max_steps * 100
            eta_h = avg_dt * (cfg.max_steps - step - 1) / 3600

            # Distinguish compile warmup from steady state
            if step < start_step + 3:
                note = "  [compile warmup]"
            else:
                note = ""

            print(
                f"  step {step:5d}/{cfg.max_steps} │ "
                f"loss {loss_accum:.4f} │ "
                f"lr {lr:.2e} │ "
                f"‖g‖ {gn:.3f} │ "
                f"{tps/1e3:.1f}K tok/s │ "
                f"MFU {mfu*100:.1f}% │ "
                f"mem {gpu_used:.1f}/{gpu_peak:.1f}GB │ "
                f"{progress:.1f}% │ "
                f"ETA {eta_h:.1f}h"
                f"{note}"
            )

            if cfg.wandb_log:
                import wandb
                wandb.log({
                    "train/loss": loss_accum,
                    "train/loss_smooth": running_loss,
                    "train/lr": lr,
                    "train/grad_norm": gn,
                    "train/tokens_total": tokens_processed,
                    "perf/tok_per_sec": tps,
                    "perf/mfu": mfu,
                    "perf/step_ms": dt * 1000,
                    "perf/gpu_mem_gb": gpu_used,
                    "perf/gpu_peak_gb": gpu_peak,
                    "perf/eta_hours": eta_h,
                }, step=step)

        # ── Evaluation ──────────────────────────────────────────────────
        if step > 0 and step % cfg.eval_interval == 0:
            t_eval_start = time.perf_counter()
            val_loss = evaluate(model, val_loader, cfg.eval_steps, device, ctx)
            t_eval_end = time.perf_counter()
            loss_history["val"].append({"step": step, "loss": val_loss})
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
            ppl = math.exp(min(val_loss, 20))
            print(
                f"\n  {'─'*60}\n"
                f"  📊 EVAL  step {step:5d} │ "
                f"val_loss {val_loss:.4f} │ "
                f"ppl {ppl:.1f} │ "
                f"best {best_val_loss:.4f} │ "
                f"eval_time {t_eval_end-t_eval_start:.1f}s"
                f"{'  ✨ NEW BEST' if improved else ''}\n"
                f"  {'─'*60}\n"
            )
            if cfg.wandb_log:
                import wandb
                wandb.log({
                    "val/loss": val_loss,
                    "val/perplexity": ppl,
                    "val/best_loss": best_val_loss,
                }, step=step)
            if improved:
                best_path = Path(cfg.out_dir) / "ckpt_best.pt"
                torch.save({
                    "model_state_dict": get_raw(model).state_dict(),
                    "step": step,
                    "val_loss": val_loss,
                    "model_config": vars(mcfg),
                }, best_path)
                print(f"  💾 Best model → {best_path.name}")
            torch.cuda.reset_peak_memory_stats()

        # ── Checkpoint ──────────────────────────────────────────────────
        if step > 0 and step % cfg.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, step, best_val_loss,
                            train_loader.get_state(), cfg, mcfg, loss_history)

        # ── Graceful shutdown ────────────────────────────────────────────
        if shutdown[0]:
            save_checkpoint(model, optimizer, step, best_val_loss,
                            train_loader.get_state(), cfg, mcfg, loss_history)
            train_loader.stop()
            val_loader.stop()
            print("  ✓ Checkpoint saved. Exiting cleanly.")
            sys.exit(0)

    # ═══════════════════════════════════════════════════════════════════════
    # TRAINING COMPLETE
    # ═══════════════════════════════════════════════════════════════════════
    total_time = time.perf_counter() - t_start
    print(f"\n{'═'*70}\n  TRAINING COMPLETE\n{'═'*70}")

    save_checkpoint(model, optimizer, cfg.max_steps - 1, best_val_loss,
                    train_loader.get_state(), cfg, mcfg, loss_history)
    final_val = evaluate(model, val_loader, cfg.eval_steps * 2, device, ctx)
    final_ppl = math.exp(min(final_val, 20))
    avg_tps = tokens_processed / total_time
    final_mfu = compute_mfu(avg_tps, n_params)

    print(f"  Total time         : {total_time/3600:.2f} hours")
    print(f"  Tokens processed   : {tokens_processed:,}")
    print(f"  Avg throughput     : {avg_tps/1e3:.1f}K tok/s")
    print(f"  Avg MFU            : {final_mfu*100:.1f}%")
    print(f"  Final val loss     : {final_val:.4f}")
    print(f"  Final perplexity   : {final_ppl:.2f}")
    print(f"  Best val loss      : {best_val_loss:.4f}")
    print(f"  Best perplexity    : {math.exp(min(best_val_loss, 20)):.2f}")

    summary = {
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "total_time_hours": total_time / 3600,
        "tokens_processed": tokens_processed,
        "avg_tok_per_sec": avg_tps,
        "avg_mfu": final_mfu,
        "final_val_loss": final_val,
        "final_perplexity": final_ppl,
        "best_val_loss": best_val_loss,
        "model_config": vars(mcfg),
        "train_config": vars(cfg),
    }
    (Path(cfg.out_dir) / "training_summary.json").write_text(
        json.dumps(summary, indent=2))

    if cfg.wandb_log:
        import wandb
        wandb.log({
            "final/val_loss": final_val,
            "final/perplexity": final_ppl,
            "final/best_loss": best_val_loss,
            "final/total_hours": total_time / 3600,
            "final/avg_tps": avg_tps,
            "final/avg_mfu": final_mfu,
        })
        wandb.finish()

    # Cleanup
    train_loader.stop()
    val_loader.stop()


if __name__ == "__main__":
    main()