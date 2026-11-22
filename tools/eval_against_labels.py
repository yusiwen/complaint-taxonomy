#!/usr/bin/env python3
"""
Score a clustering run against ground-truth labels.

Useful whenever a slice of the corpus has known categories (a public dataset, an
existing ticket taxonomy, or a hand-labelled sample): it answers "did the
clusters recover the real categories?" without needing any training.

Inputs
  --labels        jsonl with {"id": ..., "label": "..."} per document
  --conv-labels   the pipeline's out/<run>/conv_labels.jsonl
                  ({"id": ..., "labels": [{"cluster": c, "n": k}, ...]})

Metrics
  purity          per cluster take the most frequent true label; sum those counts.
                  Easy to inflate with many clusters, so always read it next to NMI.
  NMI             normalised mutual information, comparable across different k.
  ARI             adjusted Rand index: chance-corrected agreement between the
                  partition and the truth. 0 = chance level, 1 = identical, and
                  unlike purity it *penalises* splitting one class across many
                  clusters -- which is exactly what raising k does.
  coverage        share of documents the pipeline actually labelled.
  contingency     the largest clusters against their dominant true label.

Multi-label note: a document may carry several clusters. The dominant cluster
(highest n, ties broken by lowest id) is used for the metrics; the mean number of
labels per document is reported so the multi-label behaviour stays visible.
"""
import argparse
import json
import math
from collections import Counter, defaultdict


def read_jsonl(path, id_key="id"):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            out[o[id_key]] = o
    return out


def nmi(pairs):
    """Normalised mutual information between cluster ids and true labels."""
    n = len(pairs)
    if n == 0:
        return 0.0
    joint = Counter(pairs)
    cl = Counter(c for c, _ in pairs)
    lb = Counter(t for _, t in pairs)

    def mi():
        total = 0.0
        for (c, t), nij in joint.items():
            total += (nij / n) * math.log((nij * n) / (cl[c] * lb[t]))
        return total

    def h(counts):
        return -sum((v / n) * math.log(v / n) for v in counts.values())

    hc, ht = h(cl), h(lb)
    denom = math.sqrt(hc * ht) if hc > 0 and ht > 0 else 0.0
    return mi() / denom if denom else 0.0


def _comb2(v):
    return v * (v - 1) / 2.0


def ari(pairs):
    """Adjusted Rand index on the same (cluster, true label) pairs as nmi()."""
    n = len(pairs)
    if n < 2:
        return 0.0
    joint = Counter(pairs)
    cl = Counter(c for c, _ in pairs)
    lb = Counter(t for _, t in pairs)

    sum_ij = sum(_comb2(v) for v in joint.values())
    a = sum(_comb2(v) for v in cl.values())
    b = sum(_comb2(v) for v in lb.values())
    expected = a * b / _comb2(n)
    max_index = 0.5 * (a + b)
    denom = max_index - expected
    return (sum_ij - expected) / denom if denom else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--conv-labels", required=True)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    truth = {k: v["label"] for k, v in read_jsonl(args.labels).items() if "label" in v}
    pred = read_jsonl(args.conv_labels)

    pairs, unlabelled, n_labels = [], 0, []
    for cid, lab in truth.items():
        p = pred.get(cid, {})
        clusters = p.get("labels") or []
        n_labels.append(len(clusters))
        dom = p.get("primary")
        if dom is None:
            if not clusters:
                unlabelled += 1
                continue
            dom = max(clusters, key=lambda d: (d["n"], -d["cluster"]))["cluster"]
        pairs.append((dom, lab))

    if not pairs:
        raise SystemExit("no overlap between the two files")

    per_cluster = defaultdict(Counter)
    for c, t in pairs:
        per_cluster[c][t] += 1
    correct = sum(max(c.values()) for c in per_cluster.values())
    purity = correct / len(pairs)
    score = nmi(pairs)
    adj = ari(pairs)

    print(f"documents with truth      {len(truth)}")
    print(f"documents scored          {len(pairs)}  "
          f"(unlabelled by pipeline: {unlabelled}, "
          f"{(len(pairs) / max(len(truth), 1)) * 100:.1f}% coverage)")
    print(f"mean clusters per doc     {sum(n_labels) / max(len(n_labels), 1):.2f}")
    print(f"purity                    {purity:.3f}   "
          f"(baseline guessing the majority class: "
          f"{max(Counter(t for _, t in pairs).values()) / len(pairs):.3f})")
    print(f"NMI                       {score:.3f}")
    print(f"ARI                       {adj:.3f}   (0 = chance level, 1 = identical)")
    print()
    print(f"{'cluster':>8} {'size':>6}  {'dominant label':<32} {'share':>6}")
    for c, cnt in sorted(per_cluster.items(), key=lambda kv: -sum(kv[1].values()))[: args.top]:
        top_label, top_n = cnt.most_common(1)[0]
        size = sum(cnt.values())
        print(f"{c:>8} {size:>6}  {top_label[:32]:<32} {top_n / size * 100:5.1f}%")


if __name__ == "__main__":
    main()
