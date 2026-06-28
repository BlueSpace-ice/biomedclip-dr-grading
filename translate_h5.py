#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Translate the Chinese clinical text in the DR H5 files to English and write the
results back as NEW fields, matching the manuscript's preprocessing (GPT-style
translation that preserves pathological terms).

Provider : SiliconFlow (OpenAI-compatible)   base_url = https://api.siliconflow.cn/v1
Model    : configurable (default below) — VERIFY the exact string in the
           SiliconFlow model catalog before running.

Adds to each H5 file, per image key <k>:
    /diagnosis_en/<k>   English translation of /diagnosis/<k>
    /findings_en/<k>    English translation of /findings/<k>

Key properties
--------------
* Dedup: all frames of one exam share the same report text, so each unique
  Chinese string is translated ONCE (~2500 calls instead of 7500).
* Cache: translations are cached to JSON keyed by text hash, shared across the
  4 files -> safe resume, no double-billing on re-runs.
* Concurrency with retry/backoff for the network-bound API calls; the H5
  write-back is single-threaded (h5py is not thread-safe for writing).
* enable_thinking=False (Qwen3) so we get a clean translation, no reasoning.

SECURITY: the API key is read from the SILICONFLOW_API_KEY environment variable.
Never hard-code it. If you pasted a key anywhere public, ROTATE it now.

Usage
-----
  setx SILICONFLOW_API_KEY "sk-..."     (Windows, then reopen the shell)
  # or:  $env:SILICONFLOW_API_KEY="sk-..."   (PowerShell, current session)

  python translate_h5.py --model "Qwen/Qwen3-32B"            # translate all 4 files
  python translate_h5.py --sample 5                          # preview 5, no write
  python translate_h5.py --files dr_val.h5                   # one file only
"""

import os
import sys
import json
import time
import hashlib
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py

try:
    from openai import OpenAI
except ImportError:
    print("Please install the OpenAI SDK:  pip install openai", file=sys.stderr)
    raise

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    class tqdm:
        def __init__(self, total=None, initial=0, desc="", unit="", **_):
            self.n, self.total, self.desc = initial, total, desc
        def update(self, k=1):
            self.n += k
            if self.total:
                print(f"\r  {self.desc}: {self.n}/{self.total}", end="", flush=True)
        def close(self):
            print(flush=True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = r"F:\FFA_h5_dataset"
BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "Qwen/Qwen3-32B"   # VERIFY in the SiliconFlow catalog before use
CACHE_PATH = os.path.join(OUTPUT_DIR, "translation_cache.json")
H5_FILES = ["dr_train.h5", "dr_val.h5", "dr_internal_test.h5", "dr_external_test.h5"]

MAX_WORKERS = 4        # concurrent API calls; raise cautiously to respect rate limits
MAX_RETRIES = 5
TIMEOUT = 60

# Ophthalmology / FFA glossary embedded in the system prompt to lock terminology.
GLOSSARY = """微动脉瘤=microaneurysm; 新生血管=neovascularization; 毛细血管无灌注=capillary non-perfusion; 无灌注区=non-perfusion area; 荧光渗漏=fluorescein leakage; 渗漏=leakage; 出血=hemorrhage; 点状出血=dot hemorrhage; 硬性渗出=hard exudate; 棉绒斑=cotton-wool spot; 黄斑水肿=macular edema; 黄斑=macula; 视网膜=retina; 视盘=optic disc; 静脉串珠=venous beading; 视网膜内微血管异常=intraretinal microvascular abnormality (IRMA); 缺血=ischemia; 玻璃体积血=vitreous hemorrhage; 激光光凝斑=laser photocoagulation scar; 着染=staining; 充盈=filling; 臂视网膜循环时间=arm-retina circulation time; 视网膜动静脉循环时间=retinal arteriovenous circulation time; 渗漏增强=increased leakage; 微血管瘤=microaneurysm"""

SYSTEM_PROMPT = (
    "You are a professional medical translator specializing in ophthalmology and "
    "fundus fluorescein angiography (FFA). Translate the following Chinese clinical "
    "report text into accurate, fluent English. Preserve ALL pathological terms and "
    "their clinical meaning. Remove administrative metadata. Output ONLY the English "
    "translation as plain text: no preamble, no notes, no quotation marks, no Chinese. "
    "If the input is empty or contains no clinical content, output an empty string.\n"
    "Use this terminology consistently:\n" + GLOSSARY
)


# ---------------------------------------------------------------------------
# Cache (thread-safe)
# ---------------------------------------------------------------------------
_cache_lock = threading.Lock()


def load_cache():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            print("  [WARN] cache unreadable, starting fresh", flush=True)
    return {}


def save_cache(cache):
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=0)
    os.replace(tmp, CACHE_PATH)


def text_key(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------
def translate_one(client, model, text):
    """Translate a single string with retry/backoff. Empty in -> empty out."""
    if not text or not text.strip():
        return ""
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                max_tokens=1024,
                timeout=TIMEOUT,
                extra_body={"enable_thinking": False},  # Qwen3: no chain-of-thought
            )
            out = (resp.choices[0].message.content or "").strip()
            # strip accidental wrapping quotes
            if len(out) >= 2 and out[0] in "\"'" and out[-1] == out[0]:
                out = out[1:-1].strip()
            return out
        except Exception as e:
            wait = 2 ** attempt
            if attempt == MAX_RETRIES - 1:
                print(f"\n  [ERROR] giving up on a segment: {e}", flush=True)
                return None  # signal failure; caller leaves it untranslated
            time.sleep(wait)
    return None


def collect_unique_texts(h5_paths):
    """Gather all unique non-empty source strings across the given H5 files."""
    uniq = {}  # hash -> text
    for path in h5_paths:
        if not os.path.exists(path):
            continue
        with h5py.File(path, "r") as f:
            for grp in ("diagnosis", "findings"):
                if grp not in f:
                    continue
                g = f[grp]
                for k in g.keys():
                    s = bytes(g[k][()]).decode("utf-8", errors="replace")
                    if s.strip():
                        uniq.setdefault(text_key(s), s)
    return uniq


def translate_all_unique(client, model, uniq, cache):
    """Translate every unique string not already cached, with concurrency."""
    todo = [(h, t) for h, t in uniq.items() if h not in cache]
    print(f"  unique strings: {len(uniq)}, already cached: {len(uniq) - len(todo)}, "
          f"to translate: {len(todo)}", flush=True)
    if not todo:
        return

    pbar = tqdm(total=len(todo), desc="translating", unit="seg")
    done_since_save = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(translate_one, client, model, t): (h, t) for h, t in todo}
        for fut in as_completed(futs):
            h, t = futs[fut]
            res = fut.result()
            if res is not None:
                with _cache_lock:
                    cache[h] = res
                    done_since_save += 1
                    if done_since_save >= 25:
                        save_cache(cache)
                        done_since_save = 0
            pbar.update(1)
    save_cache(cache)
    pbar.close()


# ---------------------------------------------------------------------------
# Write-back into H5
# ---------------------------------------------------------------------------
def writeback(h5_path, cache):
    """Add /diagnosis_en and /findings_en, keyed identically to the source."""
    with h5py.File(h5_path, "a") as f:
        for src, dst in (("diagnosis", "diagnosis_en"), ("findings", "findings_en")):
            if src not in f:
                continue
            if dst not in f:
                f.create_group(dst)
            sg, dg = f[src], f[dst]
            done = set(dg.keys())
            keys = [k for k in sg.keys() if k not in done]
            if not keys:
                continue
            pbar = tqdm(total=len(keys), desc=f"{os.path.basename(h5_path)}:{dst}", unit="k")
            for k in keys:
                s = bytes(sg[k][()]).decode("utf-8", errors="replace")
                if not s.strip():
                    en = ""
                else:
                    en = cache.get(text_key(s))
                    if en is None:
                        # not translated (API failed earlier) -> skip, retry next run
                        pbar.update(1)
                        continue
                dg.create_dataset(k, data=en.encode("utf-8"))
                pbar.update(1)
            pbar.close()
        f.attrs["translation_model"] = "see translate_h5.py"
        f.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Translate H5 clinical text zh->en (SiliconFlow).")
    ap.add_argument("--out", default=OUTPUT_DIR)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--files", nargs="*", default=None,
                    help="subset of H5 filenames (default: all four)")
    ap.add_argument("--sample", type=int, default=0,
                    help="translate N example strings, print them, and exit (no write)")
    args = ap.parse_args()

    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        print("ERROR: set SILICONFLOW_API_KEY in your environment first.", file=sys.stderr)
        print('  PowerShell:  $env:SILICONFLOW_API_KEY="sk-..."', file=sys.stderr)
        return 2

    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    files = args.files or H5_FILES
    h5_paths = [os.path.join(args.out, fn) for fn in files]

    print("=" * 68)
    print(f"Translation via SiliconFlow  model={args.model}")
    print(f"  files: {', '.join(files)}")
    print("=" * 68)

    cache = load_cache()

    # ---- sample mode: eyeball quality before spending on the full run ----
    if args.sample > 0:
        uniq = collect_unique_texts(h5_paths)
        picks = list(uniq.values())[:args.sample]
        print(f"\nPreviewing {len(picks)} translations (not written):\n")
        for i, t in enumerate(picks, 1):
            en = translate_one(client, args.model, t)
            print(f"[{i}] 中文: {t[:120]}")
            print(f"    EN : {en}\n")
        return 0

    # ---- 1) gather unique strings, 2) translate, 3) write back ----
    print("\n[1/3] Collecting unique source strings...")
    uniq = collect_unique_texts(h5_paths)

    print("\n[2/3] Translating (cached + concurrent)...")
    translate_all_unique(client, args.model, uniq, cache)

    print("\n[3/3] Writing English fields back into H5...")
    for path in h5_paths:
        if os.path.exists(path):
            writeback(path, cache)

    # ---- report coverage ----
    missing = sum(1 for h in uniq if h not in cache)
    print("\n" + "=" * 68)
    print(f"Done. unique strings={len(uniq)}, translated={len(uniq) - missing}, "
          f"failed={missing}")
    if missing:
        print("Some segments failed (network/API). Re-run to retry only those.")
    print(f"Cache: {CACHE_PATH}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
