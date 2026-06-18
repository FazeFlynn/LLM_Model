"""
=============================================================================
STEP 1: INSPECT & DOWNLOAD — FineWeb-Edu sample-10BT → Google Drive
=============================================================================
Run this in Google Colab (CPU runtime is fine for downloading).

Target: ~7 Billion tokens (Chinchilla optimal for 350M param model)
Source: HuggingFaceFW/fineweb-edu  (sample-10BT split)

Strategy:
  - Stream the dataset row-by-row (never loads all 10BT into RAM)
  - Estimate tokens via whitespace-split word count × 1.35 (GPT-2 ratio)
  - Stop once we hit TARGET_TOKENS
  - Save as compressed Parquet shards (~500MB each) to Google Drive
  - Each shard is self-contained and can be processed independently

Run order:
  01_inspect_and_download.py   ← this file  (Colab, CPU)
  02_tokenize_and_validate.py              (Colab, GPU/CPU)
  03_verify_on_runpod.py                   (RunPod, before training)
=============================================================================
"""

# ── 0. Install deps ──────────────────────────────────────────────────────────
import subprocess, sys

def pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

pip("datasets", "huggingface_hub", "pyarrow", "tqdm", "psutil")

# ── 1. Imports ───────────────────────────────────────────────────────────────
import os, math, time, json, hashlib
from pathlib import Path
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm.auto import tqdm
import psutil

# ── 2. Config — edit these if needed ─────────────────────────────────────────
GDRIVE_ROOT      = "/content/drive/MyDrive/LLM_350M"   # your GDrive path
TARGET_TOKENS    = 7_000_000_000                        # 7B tokens
WORDS_PER_TOKEN  = 0.74                                 # GPT-2: ~1 token per 0.74 words
SHARD_SIZE_ROWS  = 50_000                               # rows per Parquet shard
MIN_TEXT_LENGTH  = 200                                  # skip very short docs (chars)
MAX_TEXT_LENGTH  = 100_000                              # skip extremely long docs (chars)
DATASET_NAME     = "HuggingFaceFW/fineweb-edu"
DATASET_SPLIT    = "sample-10BT"

# ── 3. Mount Google Drive ────────────────────────────────────────────────────
try:
    from google.colab import drive
    drive.mount("/content/drive")
    print("✓ Google Drive mounted")
except ImportError:
    print("⚠  Not in Colab — writing to local path instead")
    GDRIVE_ROOT = "./LLM_350M_local"

RAW_DIR      = Path(GDRIVE_ROOT) / "raw_shards"
META_DIR     = Path(GDRIVE_ROOT) / "metadata"
RAW_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

# ── 4. Pre-flight: inspect the dataset card BEFORE downloading ───────────────
print("\n" + "="*60)
print("PRE-FLIGHT INSPECTION")
print("="*60)

from huggingface_hub import dataset_info, DatasetCard
info = dataset_info(DATASET_NAME)
print(f"Dataset      : {DATASET_NAME}")
print(f"Split        : {DATASET_SPLIT}")
print(f"License      : {info.cardData.get('license', 'see card')}")
print(f"Last updated : {info.lastModified}")

# Estimate what fraction of the 10BT sample we need
# FineWeb-Edu 10BT ≈ 10B tokens → we need 70% of it
FRACTION_NEEDED = TARGET_TOKENS / 10_000_000_000
print(f"\nTarget tokens      : {TARGET_TOKENS:,}  ({TARGET_TOKENS/1e9:.1f}B)")
print(f"10BT sample size   : ~10,000,000,000 tokens")
print(f"Fraction to DL     : {FRACTION_NEEDED*100:.0f}%  (stop early, save bandwidth)")
print(f"Est. compressed DL : ~{FRACTION_NEEDED * 28:.0f} GB")
print(f"Est. raw text size : ~{FRACTION_NEEDED * 75:.0f} GB (uncompressed)")
print(f"Shard size         : {SHARD_SIZE_ROWS:,} rows → ~500 MB each")
print()

# Check disk space on GDrive
stat = psutil.disk_usage(GDRIVE_ROOT)
print(f"GDrive free space  : {stat.free / 1e9:.1f} GB")
needed_gb = FRACTION_NEEDED * 28 * 1.15   # 15% buffer
if stat.free / 1e9 < needed_gb:
    print(f"⚠  WARNING: May need {needed_gb:.0f} GB, only {stat.free/1e9:.1f} GB free!")
else:
    print(f"✓ Enough space ({stat.free/1e9:.0f} GB free, need ~{needed_gb:.0f} GB)")

# ── 5. Peek at a few rows before committing ──────────────────────────────────
print("\n" + "="*60)
print("SAMPLE ROWS (first 3 documents)")
print("="*60)

peek = load_dataset(
    DATASET_NAME,
    name=DATASET_SPLIT,
    split="train",
    streaming=True,
    trust_remote_code=True,
)

schema_printed = False
sample_tokens  = 0
for i, row in enumerate(peek):
    if i >= 3:
        break
    if not schema_printed:
        print(f"\nColumns: {list(row.keys())}")
        schema_printed = True
    words  = len(row["text"].split())
    tokens = words / WORDS_PER_TOKEN
    sample_tokens += tokens
    print(f"\n── Doc {i+1} ──────────────────────────────")
    print(f"  url        : {row.get('url', 'n/a')}")
    print(f"  text len   : {len(row['text']):,} chars")
    print(f"  word count : {words:,}")
    print(f"  est tokens : {tokens:,.0f}")
    print(f"  edu score  : {row.get('score', 'n/a')}")
    print(f"  text[0:200]: {row['text'][:200].strip()!r}")

avg_tokens_per_doc = sample_tokens / 3
est_total_docs_needed = int(TARGET_TOKENS / avg_tokens_per_doc)
est_shards_needed = math.ceil(est_total_docs_needed / SHARD_SIZE_ROWS)
print(f"\nEst. docs needed   : ~{est_total_docs_needed:,}")
print(f"Est. shards needed : ~{est_shards_needed}")
print()

# ── 6. Ask user to confirm before downloading ─────────────────────────────────
confirm = input("▶ Looks good? Type 'yes' to start downloading: ").strip().lower()
if confirm != "yes":
    print("Aborted. Re-run when ready.")
    sys.exit(0)

# ── 7. Stream & shard download ───────────────────────────────────────────────
print("\n" + "="*60)
print("DOWNLOADING & SHARDING")
print("="*60)

dataset = load_dataset(
    DATASET_NAME,
    name=DATASET_SPLIT,
    split="train",
    streaming=True,
    trust_remote_code=True,
)

shard_idx        = 0
total_tokens_est = 0
total_docs       = 0
total_bytes      = 0
skipped_short    = 0
skipped_long     = 0
shard_rows       = []
manifest         = []   # list of shard metadata for downstream scripts

# Resume support: find highest existing shard
existing = sorted(RAW_DIR.glob("shard_*.parquet"))
if existing:
    last = int(existing[-1].stem.split("_")[1])
    shard_idx = last + 1
    print(f"Resuming from shard {shard_idx} ({len(existing)} shards already saved)")
    # Load existing token count from manifest
    manifest_path = META_DIR / "download_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        total_tokens_est = sum(s["est_tokens"] for s in manifest)
        total_docs       = sum(s["num_rows"]  for s in manifest)
        total_bytes      = sum(s["bytes"]     for s in manifest)
        print(f"  Already have ~{total_tokens_est/1e9:.2f}B tokens, {total_docs:,} docs")

start_time = time.time()
pbar = tqdm(desc="Streaming docs", unit=" docs", dynamic_ncols=True)

def flush_shard(rows, idx):
    """Write a list of dicts to a compressed Parquet file."""
    path = RAW_DIR / f"shard_{idx:04d}.parquet"
    table = pa.Table.from_pydict({
        "text":  [r["text"]  for r in rows],
        "url":   [r.get("url",   "") for r in rows],
        "score": [r.get("score", 0.0) for r in rows],
    })
    pq.write_table(
        table, path,
        compression="zstd",
        compression_level=3,     # fast compress, still good ratio
        row_group_size=10_000,
    )
    size_bytes = path.stat().st_size
    # Compute a quick checksum for integrity verification later
    sha = hashlib.md5(path.read_bytes()).hexdigest()
    return path, size_bytes, sha

try:
    for row in dataset:
        text = row.get("text", "")
        tlen = len(text)

        # Quality filters
        if tlen < MIN_TEXT_LENGTH:
            skipped_short += 1
            continue
        if tlen > MAX_TEXT_LENGTH:
            skipped_long += 1
            continue

        shard_rows.append(row)
        word_count       = len(text.split())
        est_tokens       = word_count / WORDS_PER_TOKEN
        total_tokens_est += est_tokens
        total_bytes      += tlen
        total_docs       += 1
        pbar.update(1)
        pbar.set_postfix({
            "tokens": f"{total_tokens_est/1e9:.3f}B",
            "shard" : shard_idx,
        })

        # Flush shard when full
        if len(shard_rows) >= SHARD_SIZE_ROWS:
            path, size, sha = flush_shard(shard_rows, shard_idx)
            manifest.append({
                "shard"      : shard_idx,
                "path"       : str(path),
                "num_rows"   : len(shard_rows),
                "est_tokens" : int(sum(len(r["text"].split()) / WORDS_PER_TOKEN for r in shard_rows)),
                "bytes"      : size,
                "md5"        : sha,
                "timestamp"  : datetime.utcnow().isoformat(),
            })
            # Save manifest after every shard (crash-safe)
            (META_DIR / "download_manifest.json").write_text(
                json.dumps(manifest, indent=2)
            )
            shard_idx += 1
            shard_rows = []
            elapsed = time.time() - start_time
            speed   = total_docs / elapsed
            eta_s   = (est_total_docs_needed - total_docs) / max(speed, 1)
            tqdm.write(
                f"  Shard {shard_idx-1:04d} saved | "
                f"{total_tokens_est/1e9:.3f}B tokens | "
                f"ETA {eta_s/60:.0f} min"
            )

        # Stop once we have enough tokens
        if total_tokens_est >= TARGET_TOKENS:
            break

except KeyboardInterrupt:
    print("\nInterrupted — flushing partial shard…")

# Flush any remaining rows
if shard_rows:
    path, size, sha = flush_shard(shard_rows, shard_idx)
    manifest.append({
        "shard"      : shard_idx,
        "path"       : str(path),
        "num_rows"   : len(shard_rows),
        "est_tokens" : int(sum(len(r["text"].split()) / WORDS_PER_TOKEN for r in shard_rows)),
        "bytes"      : size,
        "md5"        : sha,
        "timestamp"  : datetime.utcnow().isoformat(),
    })

pbar.close()

# ── 8. Write final stats ─────────────────────────────────────────────────────
elapsed  = time.time() - start_time
manifest_path = META_DIR / "download_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2))

stats = {
    "completed_at"      : datetime.utcnow().isoformat(),
    "total_shards"      : len(manifest),
    "total_docs"        : total_docs,
    "total_tokens_est"  : int(total_tokens_est),
    "total_bytes_text"  : total_bytes,
    "skipped_short"     : skipped_short,
    "skipped_long"      : skipped_long,
    "elapsed_seconds"   : int(elapsed),
    "dataset"           : DATASET_NAME,
    "split"             : DATASET_SPLIT,
    "target_tokens"     : TARGET_TOKENS,
}
(META_DIR / "download_stats.json").write_text(json.dumps(stats, indent=2))

print("\n" + "="*60)
print("DOWNLOAD COMPLETE")
print("="*60)
print(f"  Shards saved   : {len(manifest)}")
print(f"  Total docs     : {total_docs:,}")
print(f"  Est. tokens    : {total_tokens_est/1e9:.2f}B")
print(f"  Raw text size  : {total_bytes/1e9:.2f} GB")
print(f"  Skipped short  : {skipped_short:,}")
print(f"  Skipped long   : {skipped_long:,}")
print(f"  Time elapsed   : {elapsed/60:.1f} min")
print(f"\n  Output dir     : {RAW_DIR}")
print(f"  Manifest       : {manifest_path}")
print(f"\n  Next step → run 02_tokenize_and_validate.py")