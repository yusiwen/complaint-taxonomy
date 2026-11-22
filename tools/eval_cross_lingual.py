#!/usr/bin/env python3
"""
Cross-lingual agreement check on a parallel corpus.

MASSIVE is parallel across languages: utterance id N is the *same* sentence in
every locale. That turns "does this embedding model actually align Chinese and
English" into a countable number instead of a vibe:

    pair agreement = share of parallel pairs whose two versions land in the
                     same cluster

Cluster-language mixing alone is a weaker signal (a cluster can be mixed while
the pairs inside it are shuffled), so report both.

Usage -- any corpus where one document has a parallel version in the other language,
and the labels file carries a shared `src_id` per pair:

    uv run python tools/translate_abcd.py --n 500 --out data/public/abcd_500
    uv run python tools/make_parallel_benchmark.py \
        --en data/public/abcd_500_en.jsonl --zh data/public/abcd_500_zh.jsonl \
        --labels data/public/abcd_500_labels.jsonl --out data/public/abcd_500_parallel
    uv run python cluster_complaints.py --input data/public/abcd_500_parallel.jsonl \
        --cue-mode off --embed-backend fastembed --k 10 --out out/parallel
    uv run python tools/eval_cross_lingual.py \
        --labels data/public/abcd_500_parallel_labels.jsonl \
        --conv-labels out/parallel/conv_labels.jsonl
"""
import argparse
import json
import re
from collections import Counter, defaultdict

CJK = re.compile(r"[\u4e00-\u9fa5]")


def read(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                o = json.loads(line)
                out[o["id"]] = o
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="jsonl with src_id + language")
    ap.add_argument("--conv-labels", required=True)
    ap.add_argument("--texts", default=None,
                    help="optional input jsonl, used to classify a cluster's language "
                         "from sample text instead of the id prefix")
    args = ap.parse_args()

    truth = read(args.labels)
    pred = read(args.conv_labels)

    def cluster_of(cid):
        p = pred.get(cid, {})
        if p.get("primary") is not None:
            return p["primary"]
        labels = p.get("labels") or []
        return max(labels, key=lambda d: (d["n"], -d["cluster"]))["cluster"] if labels \
            else None

    by_src = defaultdict(dict)                 # src_id -> {lang: cluster}
    cluster_langs = defaultdict(Counter)
    for cid, o in truth.items():
        c = cluster_of(cid)
        if c is None:
            continue
        lang = cid.split("_")[0]
        by_src[o.get("src_id", cid)][lang] = c
        cluster_langs[c][lang] += 1

    pairs = [(k, v) for k, v in by_src.items() if len(v) >= 2]
    agree = sum(1 for _, v in pairs if len(set(v.values())) == 1)
    print(f"parallel pairs found      {len(pairs)}")
    if pairs:
        print(f"pair agreement            {agree / len(pairs):.3f}  "
              f"({agree}/{len(pairs)} pairs share a cluster)")
    mixed = sum(1 for c in cluster_langs.values()
                if len([k for k, v in c.items() if v]) > 1)
    print(f"clusters with both langs  {mixed}/{len(cluster_langs)}")
    print()
    print(f"{'cluster':>8} {'zh':>5} {'en':>5}  size")
    for c, cnt in sorted(cluster_langs.items(), key=lambda kv: -sum(kv[1].values()))[:12]:
        print(f"{c:>8} {cnt.get('zh', 0):>5} {cnt.get('en', 0):>5}  "
              f"{sum(cnt.values()):>5}")


if __name__ == "__main__":
    main()
