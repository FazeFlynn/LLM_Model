<div align="center">

# 🧠 350M Parameter LLM — Trained From Scratch

**A GPT-style language model built entirely from the ground up**

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-FazeFlynn%2Fmy--350M--LLM-yellow)](https://huggingface.co/FazeFlynn/my-350M-LLM)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-orange)](https://pytorch.org)

*By [Islam Kathat](https://github.com/FazeFlynn) — AI/ML Engineer & Researcher + Full Stack Developer*

</div>

---

## 📦 Model on HuggingFace

> **[🤗 FazeFlynn/my-350M-LLM](https://huggingface.co/FazeFlynn/my-350M-LLM)**

> **[Read W&B Report and Graphs](https://api.wandb.ai/links/faiz-14a-self/wlfu98yb)**

Download the model, read the full model card, and run inference directly from HuggingFace.

---

## What This Is

This is a **350 million parameter GPT-style language model** that I built completely from scratch — no pre-existing model weights, no HuggingFace Trainer, no shortcuts.

Every part of the pipeline was designed and implemented independently:

```
Raw Web Data  →  Data Pipeline  →  Pretraining  →  Instruction Tuning  →  Published Model
   (7B tokens)    (Colab, free)    (A100 80GB)      (OpenHermes 2.5)     (HuggingFace)
```

The goal was to deeply understand how modern LLMs actually work by building one — not fine-tuning an existing model, but training from random weights.

---

## Model Specs

| Property | Value |
|----------|-------|
| Parameters | **353.6M** |
| Architecture | GPT decoder-only transformer |
| Layers | 24 |
| Attention heads | 16 |
| Hidden size | 1024 |
| Context length | 2048 tokens |
| Tokenizer | GPT-2 (tiktoken) |
| Normalization | RMSNorm |
| Attention | Flash Attention (SDPA) |
| Final val loss | ~3.04 |
| Final perplexity | ~20.9 |

---

## Training Pipeline

### Phase 1 — Pretraining

Trained on **6.83 billion tokens** of high-quality web text using Chinchilla-optimal data scaling.

- **Dataset**: FineWeb-Edu (filtered educational web text)
- **Tokens**: 6.83B (Chinchilla optimal for 350M params = 20× parameters)
- **Hardware**: NVIDIA A100 SXM 80GB on RunPod
- **Cost**: ~$35 total
- **Data pipeline**: Streamed and tokenized on Google Colab free tier (cost: $0)

Key engineering decisions:
- Streamed dataset instead of bulk downloading — saved ~20GB of disk
- Tokenized offline on Colab CPU to keep A100 time purely for training
- Used BF16 precision with fused AdamW for maximum throughput
- Cosine LR schedule with linear warmup

### Phase 2 — Instruction Tuning

Fine-tuned on **OpenHermes 2.5** to teach the model to answer questions instead of just continuing text.

- **Dataset**: 746,250 instruction-response pairs
- **Format**: `### Human: {question}\n### Assistant: {answer}`
- **Steps**: 8,000
- **LR**: 1e-5 (10× lower than pretraining — gentle fine-tune)

---

## Key Engineering Challenges

This project involved solving several real ML infrastructure problems:

**1. Memory-efficient data pipeline**
Streaming 7B tokens from HuggingFace without downloading the full 28GB dataset. Built a crash-safe downloader with per-shard checksums and resume support.

**2. Resolving training OOM errors**
`torch.compile` with BF16 caused the AOT Autograd compiler to materialize a 12.28GB FP32 gradient buffer for the logit tensor. Fixed by disabling compilation for the `lm_head + cross_entropy` path using `@torch.compiler.disable`.

**3. Throughput optimization**
Diagnosed and fixed 3 bottlenecks using custom debug logging: FP32 matmuls in chunked loss (2× slowdown), CPU dataloader stalling GPU (100ms/step), and GradScaler misuse with BF16. Final throughput: ~65K tokens/second.

**4. Instruction tuning without catastrophic forgetting**
Used 10× lower LR, conservative iteration count, and mixed datasets to preserve pretrained knowledge while teaching instruction-following behavior.

---

## Quick Start

**Requirements:**
```bash
pip install torch tiktoken
```

**Download and run:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from huggingface_hub import hf_hub_download


# ── Download checkpoint ────────────────────────────────────────────────────────
path   = hf_hub_download(repo_id="FazeFlynn/my-350M-LLM", filename="llm-350m.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt   = torch.load(path, map_location=device, weights_only=False)
config = ckpt["model_config"]


# ── Model definition ───────────────────────────────────────────────────────────

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
        self.config  = config
        self.wte     = nn.Embedding(config["vocab_size"], config["n_embd"])
        self.wpe     = nn.Embedding(config["block_size"], config["n_embd"])
        self.blocks  = nn.ModuleList([Block(config) for _ in range(config["n_layer"])])
        self.ln_f    = RMSNorm(config["n_embd"])
        self.lm_head = nn.Linear(config["n_embd"], config["vocab_size"], bias=False)
        self.wte.weight = self.lm_head.weight  # weight tying

    def forward(self, idx):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)
        x    = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)  # (B, T, vocab_size)


# ── Load weights ───────────────────────────────────────────────────────────────
model = GPT(config).to(device)

state_dict = ckpt["model_state_dict"]
renamed    = {}
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
        else:
            renamed[new_k] = v
    else:
        renamed[k] = v

model.load_state_dict(renamed, strict=False)
model.eval()
print(f"Model loaded on {device}  |  config: {config}")


# ── Tokenizer ─────────────────────────────────────────────────────────────────
enc = tiktoken.get_encoding("gpt2")


# ── Inference ─────────────────────────────────────────────────────────────────
def ask(question, max_new=150, temperature=0.8, top_k=50):
    prompt = f"### Human: {question}\n### Assistant:"
    ids    = enc.encode(prompt)
    x      = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new):        
            logits = model(x[:, -2048:])          # (1, T, vocab_size)
            logits = logits[:, -1, :] / temperature  # (1, vocab_size)

            v, _   = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")

            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            x   = torch.cat([x, nxt], dim=1)

            if nxt.item() == enc.eot_token:
                break

    full = enc.decode(x[0].tolist())
    return full.split("### Assistant:")[-1].split("<|endoftext|>")[0].strip()


print(ask("What is machine learning?"))
print(ask("How does backpropagation work?"))
print(ask("Who are you?"))
```
---

## Repository Structure

```
├── train/
│   ├── 01_download_data.py       # Stream FineWeb-Edu, save as Parquet shards
│   ├── 02_tokenize.py            # GPT-2 tokenize → uint16 .bin shards
│   ├── 03_verify_on_runpod.py    # Validate data integrity + benchmark I/O
│   └── train.py                  # Main pretraining script
│
├── instruct/
│   ├── prep_openhermes.py        # Download + tokenize OpenHermes 2.5
│   └── instruction_tune.py       # Phase 2 fine-tuning script
│
├── inference/
│   └── generate.py               # Interactive CLI inference
│
└── README.md
```

---

## Results

**Pretraining:**
```
Final val loss   : 3.04
Final perplexity : 20.9
Training tokens  : 6.83B
Training time    : ~15 hours
Total cost       : ~$35
```

**Sample outputs after instruction tuning:**

```
Q: What is backpropagation in neural networks?
A: Backpropagation is an algorithm for training neural networks by computing
   gradients of the loss function with respect to each weight...

Q: Who are you?
A: I am a large language model trained by Islam Kathat...

Q: How does photosynthesis work?
A: Photosynthesis is the process by which plants convert sunlight, water,
   and carbon dioxide into glucose and oxygen...
```

---

## What I Learned

Building this from scratch gave me a much deeper understanding of:

- **Why Chinchilla scaling matters** — more data beats bigger model at same compute budget
- **How torch.compile actually works** — AOT Autograd, inductor, and when it causes OOM
- **Data pipeline engineering** — streaming, sharding, checksums, crash recovery
- **The difference pretraining vs fine-tuning** — what each phase teaches and what it can't
- **Real GPU memory management** — activation storage, FP32 vs BF16, gradient buffers

---

## About

**Islam Kathat** — AI/ML Engineer & Full Stack Developer based in Jaipur, India.

I build AI-powered products and intelligent systems. This project was built to gain deep, hands-on understanding of the full LLM training pipeline.

- 🤗 HuggingFace: [FazeFlynn](https://huggingface.co/FazeFlynn)
- 💼 LinkedIn: [islam-khan](https://www.linkedin.com/in/islam-khan-4644211b2/)
- 🐙 GitHub: [FazeFlynn](https://github.com/FazeFlynn)
- 🐦 Twitter/X: [@fazeflynn](https://x.com/fazeflynn)
- 📧 Email: faiz.14a@gmail.com

---

## License

MIT — free to use, modify, and build on with attribution. This project was an independent research project to understand how LLMs are really trained and tuned.

---

<div align="center">

**[🤗 Download Model on HuggingFace](https://huggingface.co/FazeFlynn/my-350M-LLM)**

*If this project was useful or interesting, consider giving it a ⭐*

</div>