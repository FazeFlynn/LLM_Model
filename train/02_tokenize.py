# ── 0. Install ───────────────────────────────────────────────────────────────
import subprocess, sys

def pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

pip("tiktoken", "pyarrow", "numpy", "tqdm", "psutil")

# ── 1. Imports ───────────────────────────────────────────────────────────────
import os, json, time, struct, hashlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tiktoken
from tqdm.auto import tqdm
import psutil

# ── 2. Config ────────────────────────────────────────────────────────────────
GDRIVE_ROOT   = "/content/drive/MyDrive/LLM_350M"
RAW_DIR       = Path(GDRIVE_ROOT) / "raw_shards"
TOK_DIR       = Path(GDRIVE_ROOT) / "tokenized"
META_DIR      = Path(GDRIVE_ROOT) / "metadata"
TOK_DIR.mkdir(parents=True, exist_ok=True)

VAL_FRACTION  = 0.02        # 2% of shards → validation
SHARD_TOKENS  = 100_000_000 # tokens per output .bin shard (100M = ~200MB on disk)
DTYPE         = np.uint16   # GPT-2 vocab is 50257, fits in uint16
HEADER_INTS   = 256         # nanoGPT-compatible header size
MAGIC         = 20240520    # magic number identifies our format

# ── 3. Load tokenizer ────────────────────────────────────────────────────────
print("Loading GPT-2 tokenizer (tiktoken)…")
enc = tiktoken.get_encoding("gpt2")
EOT = enc.encode_single_token("<|endoftext|>")   # = 50256
VOCAB_SIZE = enc.n_vocab                          # = 50257
print(f"  Vocab size : {VOCAB_SIZE}")
print(f"  EOT token  : {EOT}")

# ── 4. Load manifest ─────────────────────────────────────────────────────────
manifest_path = META_DIR / "download_manifest.json"
if not manifest_path.exists():
    raise FileNotFoundError(
        f"No manifest found at {manifest_path}. Run 01_inspect_and_download.py first."
    )
manifest = json.loads(manifest_path.read_text())
print(f"\nFound {len(manifest)} raw shards in manifest")

# Sort shards by index for deterministic train/val split
manifest.sort(key=lambda x: x["shard"])

# Train/val split: last N shards are val
n_val_shards = max(1, int(len(manifest) * VAL_FRACTION))
train_shards = manifest[:-n_val_shards]
val_shards   = manifest[-n_val_shards:]
print(f"Train shards : {len(train_shards)}")
print(f"Val shards   : {n_val_shards}")

# ── 5. Header writer (nanoGPT-compatible) ────────────────────────────────────
def write_header(f, num_tokens: int):
    """Write 256-int32 header. Magic, version, token count, vocab size."""
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0] = MAGIC
    header[1] = 1          # version
    header[2] = num_tokens
    header[3] = VOCAB_SIZE
    f.write(header.tobytes())

# ── 6. Tokenize a single Parquet shard → yield token arrays ─────────────────
def tokenize_shard(parquet_path: str):
    """
    Read all rows from a Parquet shard and yield one numpy uint16 array
    per document (with EOT appended).
    """
    table = pq.read_table(parquet_path, columns=["text"])
    texts = table.column("text").to_pylist()
    for text in texts:
        if not text or len(text) < 10:
            continue
        ids = enc.encode_ordinary(text)   # no special tokens mid-doc
        ids.append(EOT)                   # end-of-text between docs
        yield np.array(ids, dtype=DTYPE)

# ── 7. Main writer: streams token arrays → output .bin shards ────────────────
def write_split(
    shards_info: list,
    split_name: str,           # "train" or "val"
    max_tokens: int = None,    # optional hard cap
):
    """
    Processes a list of raw Parquet shards and writes them as
    contiguous uint16 .bin files of SHARD_TOKENS tokens each.
    Returns a dict of stats.
    """
    out_shard_idx    = 0
    total_tokens     = 0
    total_docs       = 0
    out_buf          = []   # accumulate token arrays until SHARD_TOKENS
    out_buf_tokens   = 0
    out_shard_paths  = []

    def flush_out_shard():
        nonlocal out_shard_idx, out_buf, out_buf_tokens
        if not out_buf:
            return
        tokens_arr = np.concatenate(out_buf)
        n          = len(tokens_arr)
        path       = TOK_DIR / f"{split_name}_{out_shard_idx:05d}.bin"
        with open(path, "wb") as f:
            write_header(f, n)
            f.write(tokens_arr.tobytes())
        sha = hashlib.md5(path.read_bytes()).hexdigest()
        out_shard_paths.append({
            "path"      : str(path),
            "tokens"    : int(n),
            "md5"       : sha,
        })
        out_shard_idx  += 1
        out_buf         = []
        out_buf_tokens  = 0

    with tqdm(total=len(shards_info), desc=f"[{split_name}] raw shards", unit="shard") as pbar:
        for shard_info in shards_info:
            raw_path = shard_info["path"]
            if not Path(raw_path).exists():
                print(f"  ⚠  Missing shard: {raw_path} — skipping")
                pbar.update(1)
                continue

            for tok_arr in tokenize_shard(raw_path):
                out_buf.append(tok_arr)
                out_buf_tokens  += len(tok_arr)
                total_tokens    += len(tok_arr)
                total_docs      += 1

                if out_buf_tokens >= SHARD_TOKENS:
                    flush_out_shard()

                if max_tokens and total_tokens >= max_tokens:
                    flush_out_shard()
                    return {
                        "total_tokens"   : total_tokens,
                        "total_docs"     : total_docs,
                        "output_shards"  : out_shard_paths,
                        "capped"         : True,
                    }

            pbar.update(1)
            pbar.set_postfix({"tokens": f"{total_tokens/1e9:.3f}B", "docs": f"{total_docs:,}"})

    flush_out_shard()  # flush remainder
    return {
        "total_tokens"  : total_tokens,
        "total_docs"    : total_docs,
        "output_shards" : out_shard_paths,
        "capped"        : False,
    }

# ── 8. Run tokenization ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("TOKENIZING TRAIN SPLIT")
print("="*60)
t0 = time.time()
train_stats = write_split(train_shards, "train")
print(f"\n  Train tokens : {train_stats['total_tokens']:,}  ({train_stats['total_tokens']/1e9:.2f}B)")
print(f"  Train docs   : {train_stats['total_docs']:,}")
print(f"  Output shards: {len(train_stats['output_shards'])}")

print("\n" + "="*60)
print("TOKENIZING VAL SPLIT")
print("="*60)
val_stats = write_split(val_shards, "val")
print(f"\n  Val tokens   : {val_stats['total_tokens']:,}  ({val_stats['total_tokens']/1e9:.3f}B)")
print(f"  Val docs     : {val_stats['total_docs']:,}")
print(f"  Output shards: {len(val_stats['output_shards'])}")

elapsed = time.time() - t0

# ── 9. Validate output ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("VALIDATING OUTPUT FILES")
print("="*60)

errors   = []
warnings = []

def validate_bin(path: str, expected_tokens: int = None):
    """Read header and spot-check token values in a .bin shard."""
    p = Path(path)
    if not p.exists():
        errors.append(f"Missing: {path}")
        return False

    with open(p, "rb") as f:
        raw_header = f.read(HEADER_INTS * 4)
        header = np.frombuffer(raw_header, dtype=np.int32)

        if header[0] != MAGIC:
            errors.append(f"Bad magic in {p.name}: {header[0]} != {MAGIC}")
            return False
        if header[3] != VOCAB_SIZE:
            warnings.append(f"Vocab size mismatch in {p.name}: {header[3]} != {VOCAB_SIZE}")

        n_tokens = int(header[2])
        if expected_tokens and abs(n_tokens - expected_tokens) > 100:
            warnings.append(f"Token count mismatch in {p.name}: header says {n_tokens}, expected {expected_tokens}")

        # Spot-check: read first and last 100 tokens
        data_bytes = p.stat().st_size - HEADER_INTS * 4
        if data_bytes != n_tokens * 2:
            errors.append(f"File size mismatch in {p.name}: {data_bytes} bytes != {n_tokens} * 2")
            return False

        # Read first 100 tokens
        first_toks = np.frombuffer(f.read(200), dtype=np.uint16)
        # Seek to last 100 tokens
        f.seek(-200, 2)
        last_toks = np.frombuffer(f.read(200), dtype=np.uint16)

        # All token IDs must be < VOCAB_SIZE
        if first_toks.max() >= VOCAB_SIZE or last_toks.max() >= VOCAB_SIZE:
            errors.append(f"Invalid token ID in {p.name} (> vocab size)")
            return False

        # First token must not be EOT (document boundary at start = bad)
        if first_toks[0] == EOT:
            warnings.append(f"Shard {p.name} starts with EOT token")

    return True

all_shards = train_stats["output_shards"] + val_stats["output_shards"]
for shard_info in tqdm(all_shards, desc="Validating shards"):
    validate_bin(shard_info["path"], shard_info["tokens"])

if errors:
    print(f"\n❌ {len(errors)} ERROR(S):")
    for e in errors:
        print(f"   {e}")
else:
    print("\n✓ All shards passed validation")

if warnings:
    print(f"\n⚠  {len(warnings)} WARNING(S):")
    for w in warnings:
        print(f"   {w}")

# ── 10. Write metadata file (read by training scripts on RunPod) ──────────────
meta = {
    "created_at"        : __import__("datetime").datetime.utcnow().isoformat(),
    "dataset"           : "HuggingFaceFW/fineweb-edu",
    "tokenizer"         : "gpt2",
    "vocab_size"        : VOCAB_SIZE,
    "eot_token"         : EOT,
    "dtype"             : "uint16",
    "header_ints"       : HEADER_INTS,
    "magic"             : MAGIC,
    "format"            : "nanoGPT-compatible",
    "train": {
        "total_tokens"  : train_stats["total_tokens"],
        "total_docs"    : train_stats["total_docs"],
        "shards"        : train_stats["output_shards"],
        "n_shards"      : len(train_stats["output_shards"]),
    },
    "val": {
        "total_tokens"  : val_stats["total_tokens"],
        "total_docs"    : val_stats["total_docs"],
        "shards"        : val_stats["output_shards"],
        "n_shards"      : len(val_stats["output_shards"]),
    },
    "total_tokens"      : train_stats["total_tokens"] + val_stats["total_tokens"],
    "errors"            : errors,
    "warnings"          : warnings,
    "elapsed_seconds"   : int(elapsed),
}

meta_path = TOK_DIR / "meta.json"
meta_path.write_text(json.dumps(meta, indent=2))
print(f"\nMetadata written → {meta_path}")

# ── 11. Summary ───────────────────────────────────────────────────────────────
total_tok_size = sum(
    Path(s["path"]).stat().st_size
    for s in all_shards
    if Path(s["path"]).exists()
)

print("\n" + "="*60)
print("TOKENIZATION COMPLETE")
print("="*60)
print(f"  Total tokens    : {meta['total_tokens']:,}  ({meta['total_tokens']/1e9:.2f}B)")
print(f"  Train tokens    : {train_stats['total_tokens']:,}  ({train_stats['total_tokens']/1e9:.2f}B)")
print(f"  Val tokens      : {val_stats['total_tokens']:,}  ({val_stats['total_tokens']/1e9:.3f}B)")
print(f"  Output size     : {total_tok_size/1e9:.2f} GB")
print(f"  Output shards   : {len(all_shards)} files in {TOK_DIR}")
print(f"  Time elapsed    : {elapsed/60:.1f} min")
print(f"\n  Format          : nanoGPT-compatible uint16 binary")
print(f"  Compatible with : nanoGPT, LitGPT, GPT-NeoX, custom scripts")
print(f"\n  Next step → copy tokenized/ to RunPod, run 03_verify_on_runpod.py")

# Quick sanity: decode first 20 tokens of first train shard
first_shard = train_stats["output_shards"][0]["path"]
if Path(first_shard).exists():
    with open(first_shard, "rb") as f:
        f.seek(HEADER_INTS * 4)   # skip header
        sample_ids = np.frombuffer(f.read(40), dtype=np.uint16).tolist()
    sample_text = enc.decode(sample_ids)
    print(f"\n  Sanity check (first 20 tokens):\n  {sample_text!r}")