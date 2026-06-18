
"""
============================================================
  PHASE 2 INSTRUCTION TUNING — Full Fine-tune
  Run on RunPod A100 SXM 80GB

  SETUP (once in terminal):
    pip install torch tiktoken numpy tqdm wandb

  Then:
    python instruction_tune.py
============================================================
"""

import os, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# ── Paths ─────────────────────────────────────────────────────────────────────
PRETRAINED_CKPT = "/workspace/checkpoints/ckpt_step_005000.pt"  # your checkpoint
DATA_DIR        = "/workspace/data"
OUT_DIR         = "/workspace/checkpoints/instruct"
TRAIN_BIN       = "openhermes_train.bin"
VAL_BIN         = "openhermes_val.bin"

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE     = 8      # micro-batch; A100 80GB handles this easily
GRAD_ACCUM     = 8      # effective batch = 64 sequences
MAX_ITERS      = 25_000 # ~6-7 hrs on A100 SXM
EVAL_INTERVAL  = 500
EVAL_ITERS     = 50
SAVE_INTERVAL  = 2_000
LOG_INTERVAL   = 25

# ── Optimizer ─────────────────────────────────────────────────────────────────
LR             = 2e-5   # low LR — we're fine-tuning, not training from scratch
MIN_LR         = 2e-6   # cosine decay floor
WEIGHT_DECAY   = 0.1
BETA1, BETA2   = 0.9, 0.95
GRAD_CLIP      = 1.0

# ── LR schedule ───────────────────────────────────────────────────────────────
WARMUP_ITERS   = 300
LR_DECAY_ITERS = 25_000

# ── System ────────────────────────────────────────────────────────────────────
DEVICE         = "cuda"
COMPILE        = True   # torch.compile — free ~20% speedup on A100
SEED           = 42
BLOCK_SIZE     = 2048   # must match your model's block_size / x[:, -2048:]

# ── Wandb (set False to disable) ──────────────────────────────────────────────
WANDB_LOG      = True
WANDB_PROJECT  = "LLM_350M_instruct"
WANDB_RUN      = "openhermes_full_ft"

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL — exact copy of your architecture
#  (RMSNorm, dict-based config, blocks.X.ln1/ln2 naming)
# ─────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head  = config["n_head"]
        self.n_embd  = config["n_embd"]
        self.head_dim = self.n_embd // self.n_head
        self.c_attn  = nn.Linear(self.n_embd, 3 * self.n_embd, bias=config["bias"])
        self.c_proj  = nn.Linear(self.n_embd, self.n_embd,     bias=config["bias"])

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config["n_embd"], 4 * config["n_embd"], bias=config["bias"])
        self.c_proj = nn.Linear(4 * config["n_embd"], config["n_embd"], bias=config["bias"])
        self.act    = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1  = RMSNorm(config["n_embd"])
        self.attn = CausalSelfAttention(config)
        self.ln2  = RMSNorm(config["n_embd"])
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte    = nn.Embedding(config["vocab_size"], config["n_embd"])
        self.wpe    = nn.Embedding(config["block_size"], config["n_embd"])
        self.blocks = nn.ModuleList([Block(config) for _ in range(config["n_layer"])])
        self.ln_f   = RMSNorm(config["n_embd"])
        self.lm_head = nn.Linear(config["n_embd"], config["vocab_size"], bias=False)
        self.wte.weight = self.lm_head.weight   # weight tying

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)
        x    = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.75, top_k=50, rep_penalty=1.15):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # repetition penalty (mirrors your inference script)
            recent = idx[0][-50:].tolist()
            for tok in set(recent):
                logits[:, tok] /= rep_penalty

            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
            probs     = F.softmax(logits, dim=-1)
            idx_next  = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

            if idx_next.item() == enc.eot_token:
                break
        return idx


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

enc = tiktoken.get_encoding("gpt2")

def get_batch(split: str):
    fname = TRAIN_BIN if split == "train" else VAL_BIN
    data  = np.memmap(os.path.join(DATA_DIR, fname), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x  = torch.stack([torch.from_numpy(data[i     : i + BLOCK_SIZE].astype(np.int64)) for i in ix])
    y  = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + BLOCK_SIZE].astype(np.int64)) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

def get_lr(it: int) -> float:
    if it < WARMUP_ITERS:
        return LR * it / WARMUP_ITERS
    if it > LR_DECAY_ITERS:
        return MIN_LR
    ratio = (it - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return MIN_LR + coeff * (LR - MIN_LR)

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def sample_model(model, prompt: str, max_new: int = 120) -> str:
    model.eval()
    ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    x   = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.generate(x, max_new)
    text = enc.decode(out[0].tolist())
    model.train()
    return text


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {PRETRAINED_CKPT}")
    ckpt   = torch.load(PRETRAINED_CKPT, map_location=DEVICE, weights_only=False)
    config = ckpt["model_config"]

    print(f"Model config: {config}")

    model = GPT(config).to(DEVICE)

    # ── Key remapping — same logic as your inference script ──────────────────
    state_dict = ckpt["model_state_dict"]
    renamed = {}
    for k, v in state_dict.items():
        if k.startswith("transformer."):
            new_k = k[len("transformer."):]
            if new_k.startswith("h."):
                parts  = new_k.split(".")
                mapped = f"blocks.{parts[1]}"
                for p in parts[2:]:
                    if   p == "ln_1": mapped += ".ln1"
                    elif p == "ln_2": mapped += ".ln2"
                    else:             mapped += f".{p}"
                renamed[mapped] = v
            elif new_k in ("wte.weight", "wpe.weight", "ln_f.weight", "lm_head.weight"):
                renamed[new_k] = v
            else:
                renamed[new_k] = v
        else:
            renamed[k] = v

    missing, unexpected = model.load_state_dict(renamed, strict=False)
    if missing:
        print(f"WARNING — missing keys  : {missing}")
    if unexpected:
        print(f"WARNING — unexpected keys: {unexpected}")
    print("Checkpoint loaded successfully.")

    # ── Compile ───────────────────────────────────────────────────────────────
    if COMPILE:
        print("Compiling with torch.compile … (takes ~3-5 min, speeds up training ~20%)")
        model = torch.compile(model)
        print("Compile done.")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    decay_params   = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() <  2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,   "weight_decay": WEIGHT_DECAY},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=LR, betas=(BETA1, BETA2), fused=True
    )

    # ── Wandb ─────────────────────────────────────────────────────────────────
    use_wandb = False
    if WANDB_LOG:
        try:
            import wandb
            wandb.init(project=WANDB_PROJECT, name=WANDB_RUN,
                       config={"lr": LR, "batch": BATCH_SIZE * GRAD_ACCUM,
                               "max_iters": MAX_ITERS, "model": config})
            use_wandb = True
            print("Wandb initialized.")
        except Exception as e:
            print(f"Wandb skipped ({e})")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Instruction tuning started")
    print(f"  Max iters  : {MAX_ITERS:,}")
    print(f"  Eff batch  : {BATCH_SIZE * GRAD_ACCUM} sequences")
    print(f"  LR         : {LR}  →  {MIN_LR} (cosine)")
    print(f"  Block size : {BLOCK_SIZE}")
    print(f"{'='*60}\n")

    X, Y      = get_batch("train")
    t0        = time.time()
    best_val  = float("inf")
    last_loss = 0.0

    for iter_num in range(MAX_ITERS + 1):

        # ── Eval + checkpoint ──────────────────────────────────────────────
        if iter_num % EVAL_INTERVAL == 0:
            losses = estimate_loss(model)
            elapsed = time.time() - t0
            print(f"[{iter_num:6d}/{MAX_ITERS}]  "
                  f"train={losses['train']:.4f}  val={losses['val']:.4f}  "
                  f"lr={get_lr(iter_num):.2e}  elapsed={elapsed/3600:.2f}h")

            # Sample output every 4 evals so you can watch it improve
            if iter_num % (EVAL_INTERVAL * 4) == 0:
                test_prompts = [
                    "### Human: What is the capital of France?\n### Assistant:",
                    "### Human: What caused World War 2?\n### Assistant:",
                ]
                for p in test_prompts:
                    out = sample_model(model, p, max_new=80)
                    answer = out[len(p):].split("<|endoftext|>")[0].strip()
                    q = p.split("### Human:")[1].split("\n")[0].strip()
                    print(f"  Q: {q}")
                    print(f"  A: {answer[:200]}")

            if use_wandb:
                import wandb
                wandb.log({"train/loss": losses["train"], "val/loss": losses["val"],
                           "lr": get_lr(iter_num), "iter": iter_num})

            # Save best checkpoint
            if losses["val"] < best_val:
                best_val = losses["val"]
                raw = model._orig_mod if hasattr(model, "_orig_mod") else model
                torch.save({
                    "model_state_dict": raw.state_dict(),
                    "model_config":     config,
                    "iter_num":         iter_num,
                    "best_val":         best_val,
                }, os.path.join(OUT_DIR, "ckpt_instruct_best.pt"))
                print(f"  ✓ Best val {best_val:.4f} — saved ckpt_instruct_best.pt")

        # Periodic saves (safety net — never lose more than 2k iters)
        if iter_num > 0 and iter_num % SAVE_INTERVAL == 0:
            raw = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save({
                "model_state_dict": raw.state_dict(),
                "model_config":     config,
                "iter_num":         iter_num,
            }, os.path.join(OUT_DIR, f"ckpt_instruct_{iter_num:06d}.pt"))
            print(f"  Periodic save → ckpt_instruct_{iter_num:06d}.pt")

        if iter_num == MAX_ITERS:
            break

        # ── Forward + backward ────────────────────────────────────────────
        lr = get_lr(iter_num)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(GRAD_ACCUM):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(X, Y)
                loss    = loss / GRAD_ACCUM
            X, Y = get_batch("train")
            loss.backward()
            accum_loss += loss.item()

        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        last_loss = accum_loss

        if iter_num % LOG_INTERVAL == 0:
            print(f"  iter {iter_num:6d} | loss {last_loss:.4f} | lr {lr:.2e}")

    # ── Final save ────────────────────────────────────────────────────────────
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        "model_state_dict": raw.state_dict(),
        "model_config":     config,
        "iter_num":         MAX_ITERS,
    }, os.path.join(OUT_DIR, "ckpt_instruct_final.pt"))

    print(f"\nTraining complete.")
    print(f"Best val loss : {best_val:.4f}")
    print(f"Best ckpt     : {OUT_DIR}/ckpt_instruct_best.pt")

    # ── Final inference test ───────────────────────────────────────────────────
    print("\n── Final model outputs ──────────────────────────────────────")
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw.eval()
    tests = [
        "What is the capital of France?",
        "What caused World War 2?",
        "How does photosynthesis work?",
        "What is the population of India approximately?",
        "Explain what DNA is.",
    ]
    for q in tests:
        prompt = f"### Human: {q}\n### Assistant:"
        out    = sample_model(raw, prompt, max_new=150)
        answer = out[len(prompt):].split("<|endoftext|>")[0].strip()
        print(f"\nQ: {q}")
        print(f"A: {answer}")
    print("─────────────────────────────────────────────────────────────")
    print("\nBackup command:")
    print("  rclone copy /workspace/checkpoints/instruct/ckpt_instruct_best.pt gdrive:LLM_350M/instruct/ -P")


if __name__ == "__main__":
    main()
