
"""
============================================================
  PHASE 2 DATA PREP — OpenHermes 2.5
  Run this on Google Colab T4

  SETUP CELL (run first):
    !pip install datasets tiktoken tqdm

  OUTPUT: openhermes_train.bin + openhermes_val.bin
  Then download both and upload to RunPod /workspace/data/
============================================================
"""

# ── Cell 1: Imports ──────────────────────────────────────────────────────────
import os, random
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

print("Imports OK")

# ── Cell 2: Config ───────────────────────────────────────────────────────────
SEED        = 42
VAL_SPLIT   = 0.005          # 0.5% val
MAX_SAMPLES = 750_000
OUTPUT_DIR  = "/content/drive/MyDrive/tuning_dataset/openhermes"
MIN_RESP_CHARS = 30          # skip trivially short assistant replies

# GPT-2 tokenizer — MUST match your pretraining
enc     = tiktoken.get_encoding("gpt2")
EOT_ID  = enc.eot_token          # 50256

random.seed(SEED)
np.random.seed(SEED)
print(f"GPT-2 tokenizer loaded | vocab={enc.n_vocab} | EOT={EOT_ID}")

# ── Cell 3: Chat format ──────────────────────────────────────────────────────
# We use simple text markers — no special token IDs needed.
# The model learns the pattern from seeing it thousands of times.
#
# Format per turn:
#   ### Human: {question}\n### Assistant: {answer}<|endoftext|>
#
# Why this format?
#   - Simple, no new tokens needed
#   - "<|endoftext|>" is already in GPT-2 vocab (token 50256)
#   - Model learns to stop generating at <|endoftext|>

def format_conversation(sample) -> str | None:
    """
    OpenHermes 2.5 schema:
      sample['conversations'] = [
          {"from": "human",  "value": "..."},
          {"from": "gpt",    "value": "..."},
          ...  (may be multi-turn)
      ]
    """
    convs = sample.get("conversations", [])
    if not convs or len(convs) < 2:
        return None

    parts = []
    i = 0
    while i < len(convs):
        role  = convs[i].get("from", "").strip()
        value = convs[i].get("value", "").strip()
        i += 1

        if not value:
            continue

        if role in ("human", "user"):
            parts.append(f"### Human: {value}\n")

        elif role in ("gpt", "assistant"):
            if len(value) < MIN_RESP_CHARS:
                return None        # skip low-quality sample entirely
            parts.append(f"### Assistant: {value}<|endoftext|>\n")

        # skip system prompts — too confusing for a 350M model

    if len(parts) < 2:
        return None

    return "".join(parts)

# ── Cell 4: Load & filter ────────────────────────────────────────────────────
print("Loading OpenHermes 2.5 …")
ds = load_dataset("teknium/OpenHermes-2.5", split="train")
print(f"Raw size: {len(ds):,}")

formatted, skipped = [], 0
for sample in tqdm(ds, desc="Formatting"):
    text = format_conversation(sample)
    if text is None:
        skipped += 1
        continue
    formatted.append(text)
    if len(formatted) >= MAX_SAMPLES:
        break

print(f"Kept: {len(formatted):,}  |  Skipped: {skipped:,}")

# ── Cell 5: Shuffle & split ──────────────────────────────────────────────────
random.shuffle(formatted)
n_val   = max(500, int(len(formatted) * VAL_SPLIT))
val_set = formatted[:n_val]
trn_set = formatted[n_val:]
print(f"Train: {len(trn_set):,}  |  Val: {len(val_set):,}")

# ── Cell 6: Tokenize & write .bin ────────────────────────────────────────────
def tokenize_and_write(texts, path):
    all_ids = []
    for text in tqdm(texts, desc=f"Tokenizing → {os.path.basename(path)}"):
        # allow_special so <|endoftext|> encodes to token 50256, not chars
        ids = enc.encode(text, allowed_special={"<|endoftext|>"})
        all_ids.extend(ids)

    arr = np.array(all_ids, dtype=np.uint16)
    arr.tofile(path)
    print(f"Saved {path}  |  {len(arr):,} tokens  |  {arr.nbytes/1e6:.1f} MB")
    return len(arr)

os.makedirs(OUTPUT_DIR, exist_ok=True)
n_train = tokenize_and_write(trn_set, os.path.join(OUTPUT_DIR, "openhermes_train.bin"))
n_val   = tokenize_and_write(val_set, os.path.join(OUTPUT_DIR, "openhermes_val.bin"))

print(f"\n{'='*55}")
print(f"  Train tokens : {n_train:,}")
print(f"  Val tokens   : {n_val:,}")
print(f"  Files ready  : openhermes_train.bin  openhermes_val.bin")
print(f"{'='*55}")
print("Download both files, then upload to RunPod /workspace/data/")

# ── Cell 7: Sanity check ─────────────────────────────────────────────────────

arr = np.fromfile(
    os.path.join(
        OUTPUT_DIR,
        "openhermes_train.bin"
    ),
    dtype=np.uint16
)
snippet = arr[:300].tolist()
decoded = enc.decode(snippet)
print("\nFirst ~300 tokens decoded:")
print(repr(decoded))

# ============================================================
# Size report
# ============================================================

train_size_mb = os.path.getsize(
    os.path.join(OUTPUT_DIR, "openhermes_train.bin")
) / (1024**2)

val_size_mb = os.path.getsize(
    os.path.join(OUTPUT_DIR, "openhermes_val.bin")
) / (1024**2)

total_size_mb = train_size_mb + val_size_mb

print("\n" + "="*55)
print(f"Train tokens : {n_train:,}")
print(f"Val tokens   : {n_val:,}")
print()
print(f"Train size   : {train_size_mb:.2f} MB")
print(f"Val size     : {val_size_mb:.2f} MB")
print(f"Total size   : {total_size_mb:.2f} MB")
print("="*55)
