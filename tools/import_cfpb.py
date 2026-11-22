#!/usr/bin/env python3
"""
Import the official CFPB Consumer Complaint Database and take a labelled sample.

Where the narratives actually live (verified 2026-11):
  * the public bulk export https://files.consumerfinance.gov/ccdb/complaints.csv.zip
    is ~350 MB zipped / 5.8 GB unzipped but ships **15 columns and no narrative
    column** -- "Consumer complaint narrative" is simply not in it;
  * the search API (data-research/consumer-complaints/search/api/v1/) ignores
    `has_narrative=true`, never returns `complaint_what_happened`, and rejects it
    as a `field` choice;
  * so the practical route is a mirror of the same database that keeps the
    narrative column, e.g.
    https://huggingface.co/datasets/davidheineman/consumer-finance-complaints-large
    (apache-2.0, 11 parquet shards, CFPB column names verbatim, 652k rows and
    ~197k narratives per shard, dates 2011-2025).

Usage
    # one shard is enough for a stratified 5k sample
    curl -sL -o /tmp/ds/cfpb0.parquet \
      https://huggingface.co/datasets/davidheineman/consumer-finance-complaints-large/resolve/main/data/train-00000-of-00011.parquet
    uv run --with pandas --with pyarrow python tools/import_cfpb.py \
      --source parquet --path /tmp/ds/cfpb0.parquet --n 5000 --out data/public/cfpb_issue

Sampling is stratified: keep the `--top-issues` most frequent Issue classes and
take at most `--per-issue` narratives from each, so every class is represented and
purity/NMI stay interpretable. This is class-balanced by construction; a
production corpus is far more long-tailed.

Outputs
    <out>.jsonl          {"id", "messages": [{"role": "user", "content": narrative}]}
    <out>_labels.jsonl   {"id", "label": <Issue>, "product", "sub_issue", "date"}
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict


def read_parquet(path):
    import pandas as pd
    return pd.read_parquet(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["parquet", "csv"], default="parquet")
    ap.add_argument("--path", default="-", help="input file, or - for stdin (csv only)")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--top-issues", type=int, default=25)
    ap.add_argument("--per-issue", type=int, default=200)
    ap.add_argument("--min-chars", type=int, default=100)
    ap.add_argument("--out", default="data/public/cfpb_issue_5k")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    if args.source == "parquet":
        df = read_parquet(args.path)
        df = df[["complaint_what_happened", "issue", "sub_issue", "product",
                 "date_received"]].copy()
        df["complaint_what_happened"] = df["complaint_what_happened"].fillna("").astype(str)
        df = df[df["complaint_what_happened"].str.len() >= args.min_chars]
        df = df[df["issue"].notna()]
        print(f"rows with a usable narrative: {len(df):,}", file=sys.stderr)
        counts = Counter(df["issue"])
        keep = [i for i, _ in counts.most_common(args.top_issues)]
        by_issue = defaultdict(list)
        for rec in df.itertuples(index=False):
            by_issue[rec.issue].append(rec)
        picked = []
        for issue in keep:
            rows = by_issue[issue]
            rng.shuffle(rows)
            picked.extend(rows[: args.per_issue])
        rng.shuffle(picked)
        picked = picked[: args.n]
        rows = [{"id": f"cfpb{i:05d}",
                 "narrative": r.complaint_what_happened,
                 "label": r.issue,
                 "sub_issue": r.sub_issue if isinstance(r.sub_issue, str) else "",
                 "product": r.product,
                 "date": str(r.date_received)}
                for i, r in enumerate(picked)]
    else:
        fh = sys.stdin if args.path == "-" else open(args.path, newline="", encoding="utf-8")
        reader = csv.DictReader(fh)
        rows = []
        for r in reader:
            n = (r.get("Consumer complaint narrative") or "").strip()
            if len(n) < args.min_chars:
                continue
            rows.append({"id": f"cfpb{len(rows):05d}", "narrative": n,
                         "label": (r.get("Issue") or "UNKNOWN").strip(),
                         "sub_issue": (r.get("Sub-issue") or "").strip(),
                         "product": (r.get("Product") or "").strip(),
                         "date": r.get("Date received", "")})
            if len(rows) >= args.n:
                break
        if not rows:
            raise SystemExit("no narratives in this source -- see the module docstring: "
                             "the official bulk CSV no longer includes the column")

    with open(f"{args.out}.jsonl", "w", encoding="utf-8") as f, \
            open(f"{args.out}_labels.jsonl", "w", encoding="utf-8") as g:
        for r in rows:
            f.write(json.dumps({"id": r["id"],
                                "messages": [{"role": "user",
                                              "content": r["narrative"][:4000]}]},
                               ensure_ascii=False) + "\n")
            g.write(json.dumps({"id": r["id"], "label": r["label"],
                                "product": r["product"],
                                "sub_issue": r["sub_issue"],
                                "date": r["date"]}, ensure_ascii=False) + "\n")

    dist = Counter(r["label"] for r in rows)
    print(f"wrote {len(rows)} docs over {len(dist)} issue classes -> {args.out}.jsonl",
          file=sys.stderr)
    for label, c in dist.most_common(args.top_issues):
        print(f"   {c:5d}  {label}", file=sys.stderr)


if __name__ == "__main__":
    main()
