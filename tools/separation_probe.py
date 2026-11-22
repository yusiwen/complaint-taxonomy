#!/usr/bin/env python3
"""Is a label set learnable from the text at all?

Diagnostic to run BEFORE blaming the clustering pipeline. It answers three
different questions on the same embeddings:

  majority baseline    what you get by always predicting the most common class
  unsupervised purity  what k-means recovers with no labels (what the pipeline does)
  supervised ceiling   what a nearest-centroid classifier reaches under 5-fold CV,
                       i.e. how much of the label is actually in the text

If the supervised number is barely above the baseline, the labels are not a
function of the text and no amount of clustering - or training - will recover
them. That is a taxonomy problem, not a model problem.

Usage
    uv run python tools/separation_probe.py \
        --input data/public/cfpb_issue.jsonl \
        --labels data/public/cfpb_issue_labels.jsonl \
        --label-field label --embed-backend fastembed
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cluster_complaints import _sqdist, build_embedder, kmeans, l2norm  # noqa: E402


def load(input_path, labels_path, field):
    texts, labels = [], []
    truth = {}
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            truth[o["id"]] = o[field]
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if o["id"] in truth:
                texts.append(" ".join(m["content"] for m in o["messages"]))
                labels.append(truth[o["id"]])
    return texts, labels


def purity(pred, truth):
    from collections import Counter, defaultdict
    per = defaultdict(Counter)
    for c, t in zip(pred, truth):
        per[c][t] += 1
    return sum(max(v.values()) for v in per.values()) / len(pred)


def centroid_cv(X, y, folds=5, seed=0):
    """Supervised nearest-centroid accuracy with a deterministic fold split."""
    classes = sorted(set(y))
    idx = {c: i for i, c in enumerate(classes)}
    yi = np.array([idx[v] for v in y])
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    fold_of = np.empty(len(X), dtype=int)
    fold_of[order] = np.arange(len(X)) % folds

    acc = []
    for f in range(folds):
        tr, te = fold_of != f, fold_of == f
        if te.sum() == 0:
            continue
        C = []
        for c in range(len(classes)):
            m = tr & (yi == c)
            if not m.any():
                C.append(None)
                continue
            C.append(l2norm(X[m].mean(0, keepdims=True))[0])
        present = [i for i, c in enumerate(C) if c is not None]
        Cm = np.vstack([C[i] for i in present])
        D = _sqdist(X[te], Cm).argmin(1)
        pred = np.array([present[d] for d in D])
        acc.append(float((pred == yi[te]).mean()))
    return float(np.mean(acc)), float(np.std(acc)), len(classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--label-field", default="label")
    ap.add_argument("--embed-backend", default="fastembed")
    ap.add_argument("--embed-model", default="")
    ap.add_argument("--embed-prefix", default=None)
    ap.add_argument("--k", type=int, default=0, help="clusters; default = #classes")
    args = ap.parse_args()

    texts, y = load(args.input, args.labels, args.label_field)
    print(f"documents {len(texts)}  classes {len(set(y))}")

    emb = build_embedder(args, quiet=True)
    X = l2norm(np.asarray(emb(texts), dtype=np.float32))
    X = X - X.mean(0, keepdims=True)
    X = l2norm(X)

    from collections import Counter
    maj = Counter(y).most_common(1)[0][1] / len(y)

    k = args.k or len(set(y))
    lab, _, _ = kmeans(X, k, n_init=3)
    unsup = purity(lab, y)

    acc, sd, ncls = centroid_cv(X, y)

    print(f"majority baseline        {maj:.3f}")
    print(f"unsupervised (k={k:3d})     {unsup:.3f}   purity, no labels used")
    print(f"supervised ceiling       {acc:.3f} +/- {sd:.3f}   "
          f"nearest-centroid, 5-fold, {ncls} classes")
    print()
    if acc < maj * 1.3:
        print("VERDICT: the labels are barely a function of the text. Clustering "
              "cannot recover them,\n         and training a classifier would not "
              "either. Fix the taxonomy, not the model.")
    elif ac := (acc - maj) / max(1 - maj, 1e-9):
        print(f"VERDICT: labels are learnable (supervised reaches {acc:.2f} vs "
              f"{maj:.2f} baseline,\n         closing {ac * 100:.0f}% of the "
              f"gap). Unsupervised recovers {unsup:.2f}, so the gap to "
              f"{acc:.2f}\nis where a supervised classifier earns its keep.")


if __name__ == "__main__":
    main()
