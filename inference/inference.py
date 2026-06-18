# ===============================================================
# run this command; pip install torch tiktoken
# ===============================================================

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



# Run these in another cell is you are using colab or notebooks
print(ask("What is machine learning?"))
print(ask("How does backpropagation work?"))
print(ask("Who are you?"))