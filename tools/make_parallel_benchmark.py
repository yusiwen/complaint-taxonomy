#!/usr/bin/env python3
"""Merge a parallel en/zh dialogue pair into one corpus for cross-lingual scoring.

Cross-lingual agreement only means something when both languages are clustered
**by the same centroids** — clustering the Chinese file and the English file
separately gives two unrelated cluster-numberings, and "cluster 3" in one run has
nothing to do with "cluster 3" in the other. So merge first, cluster once, then
ask whether the two versions of a dialogue share a cluster.

    uv run python tools/make_parallel_benchmark.py \
        --en data/public/abcd_500_en.jsonl \
        --zh data/public/abcd_500_zh.jsonl \
        --labels data/public/abcd_500_labels.jsonl \
        --out data/public/abcd_500_parallel

Outputs
    <out>.jsonl          both languages, ids prefixed en_/zh_
    <out>_labels.jsonl   per-language rows carrying the shared src_id
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--en", required=True)
    ap.add_argument("--zh", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="data/public/abcd_500_parallel")
    args = ap.parse_args()

    meta = {}
    with open(args.labels, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            meta[o["id"]] = o

    written = 0
    with open(f"{args.out}.jsonl", "w", encoding="utf-8") as fo, \
            open(f"{args.out}_labels.jsonl", "w", encoding="utf-8") as fl:
        for prefix, path in (("en_", args.en), ("zh_", args.zh)):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    o = json.loads(line)
                    m = meta.get(o["id"])
                    if m is None:
                        continue
                    cid = prefix + o["id"]
                    fo.write(json.dumps({"id": cid, "messages": o["messages"]},
                                        ensure_ascii=False) + "\n")
                    fl.write(json.dumps({"id": cid,
                                         "label": m.get("label"),
                                         "subflow": m.get("subflow"),
                                         "src_id": m.get("src_id") or o["id"]},
                                        ensure_ascii=False) + "\n")
                    written += 1

    print(f"wrote {written} documents ({written // 2} pairs) -> {args.out}.jsonl")


if __name__ == "__main__":
    main()
