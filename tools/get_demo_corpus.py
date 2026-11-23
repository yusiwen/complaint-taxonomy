#!/usr/bin/env python3
"""Fetch the demo corpus, so nothing generated has to live in git.

Every corpus under `data/public/` is a download plus a conversion, which means it can be
rebuilt and should not be committed: a committed artefact drifts from the script that
makes it, and a 500-dialogue JSONL in the history is dead weight forever. This is the one
command that puts the demo corpus back.

    # English only: no model, no GPU, ~40 MB download, seconds
    uv run python tools/get_demo_corpus.py

    # add the machine-translated Chinese half (needs the local LLM endpoint)
    uv run python tools/get_demo_corpus.py --translate \
        --endpoint http://127.0.0.1:8000/v1/chat/completions

Outputs (all git-ignored)
    data/public/abcd_500_en.jsonl + abcd_500_labels.jsonl
    data/public/abcd_500_zh.jsonl                       (--translate)
    data/public/abcd_500_parallel*.jsonl                (--translate)

The download is cached under ~/.cache/complaint-taxonomy/ so repeated runs are offline.
ABCD is MIT licensed; the Chinese half is machine translation produced locally.
"""
import argparse
import os
import subprocess
import sys
import urllib.request

ABCD_URL = ("https://raw.githubusercontent.com/asappresearch/abcd/master/"
            "data/abcd_v1.1.json.gz")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.expanduser("~/.cache/complaint-taxonomy/abcd_v1.1.json.gz")


def ensure_download(path=None):
    """Return a local path to the ABCD archive, downloading it once if needed."""
    path = path or CACHE
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        print(f"[data] using cached archive {path} "
              f"({os.path.getsize(path) / 1e6:.0f} MB)")
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[data] downloading {ABCD_URL} -> {path} (about 37 MB)")
    tmp = path + ".part"
    with urllib.request.urlopen(ABCD_URL, timeout=300) as r, open(tmp, "wb") as f:
        f.write(r.read())
    os.replace(tmp, path)
    print(f"[data] {os.path.getsize(path) / 1e6:.0f} MB cached")
    return path


def run(cmd):
    print("[data] " + " ".join(os.path.basename(c) if i < 2 else c
                               for i, c in enumerate(cmd)))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("[fatal] command failed:\n" + (r.stderr or r.stdout)[-1500:])
    if r.stderr.strip():
        print(r.stderr.strip()[-400:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="dialogues to keep")
    ap.add_argument("--split", default="test",
                    help="ABCD split to sample from (test keeps it out of the fit sets)")
    ap.add_argument("--out", default="data/public/abcd_500")
    ap.add_argument("--src", default="", help="use an existing abcd_v1.1.json.gz")
    ap.add_argument("--translate", action="store_true",
                    help="also produce the Chinese half (needs --endpoint)")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--model", default="",
                    help="model name your endpoint expects; empty is fine")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    src = ensure_download(args.src)
    py = sys.executable
    base = [py, os.path.join(ROOT, "tools/translate_abcd.py"),
            "--src", src, "--split", args.split, "--n", str(args.n),
            "--out", args.out]
    if args.translate:
        run(base + ["--endpoint", args.endpoint, "--model", args.model,
                    "--workers", str(args.workers)])
        run([py, os.path.join(ROOT, "tools/make_parallel_benchmark.py"),
             "--en", f"{args.out}_en.jsonl", "--zh", f"{args.out}_zh.jsonl",
             "--labels", f"{args.out}_labels.jsonl",
             "--out", f"{args.out}_parallel"])
    else:
        run(base + ["--en-only"])

    print(f"\n[data] ready. Try:\n"
          f"  uv run python cluster_complaints.py --input {args.out}_en.jsonl \\\n"
          f"      --embed-backend fastembed --cue-mode off --k 10 "
          f"--kmeans-algo bisect --out out/demo\n"
          f"  uv run python tools/eval_against_labels.py "
          f"--labels {args.out}_labels.jsonl --conv-labels out/demo/conv_labels.jsonl")


if __name__ == "__main__":
    main()
