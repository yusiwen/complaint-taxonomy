#!/usr/bin/env python3
"""How many labelled items do we actually need?

Answers "is 5,000 enough?" with a learning curve instead of an opinion: a fixed
held-out test set, a real trained classifier (ridge regression on one-hot targets
over frozen embeddings, closed form, numpy only), and accuracy at increasing
training sizes. It also reports the label-free baseline so two questions stay
separate:

    how well does *training* scale with labels   -> the linear-model curve
    how well does *clustering* do with no labels -> unsupervised purity

Read the curve, not the last row: if 2,000 and 5,000 land within noise of each
other, more labels of the same kind will not help -- the label definition is the
constraint, and that is what `separation_probe.py` measures.

Usage
    uv run python tools/learning_curve.py \
        --input data/public/cfpb_issue.jsonl \
        --labels data/public/cfpb_issue_labels.jsonl \
        --label-field label --sizes 250,500,1000,2000,3000,4000
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cluster_complaints import _sqdist, build_embedder, kmeans, l2norm  # noqa: E402


def load(input_path, labels_path, field):
    truth = {}
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            truth[o["id"]] = o[field]
    texts, labels = [], []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if o["id"] in truth:
                texts.append(" ".join(m["content"] for m in o["messages"]))
                labels.append(truth[o["id"]])
    return texts, labels


def embed_cached(texts, emb, cache):
    if cache and os.path.exists(cache):
        X = np.load(cache)
        if X.shape[0] == len(texts):
            print(f"[cache] {cache} {X.shape}", file=sys.stderr)
            return X
    X = l2norm(np.asarray(emb(texts), dtype=np.float32))
    X = l2norm(X - X.mean(0, keepdims=True))
    if cache:
        np.save(cache, X)
    return X


def fit_linear(X, y, classes, lam=1e-2):
    """Ridge regression on one-hot targets -- a linear classifier, closed form.

    Deliberately not gradient descent: an iterative solver needs a step size and
    an iteration count that both interact with `n`, and in an earlier version of
    this script a larger training set *lowered* accuracy purely because the
    optimiser stopped converging. The closed-form solution has no such knobs.
    """
    C = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((len(X), C), dtype=np.float32)
    Y[np.arange(len(X)), [idx[v] for v in y]] = 1.0
    A = X.T @ X + lam * np.eye(X.shape[1], dtype=np.float32)
    return np.linalg.solve(A, X.T @ Y)


def predict(X, W, classes):
    return [classes[i] for i in np.argmax(X @ W, axis=1)]


def accuracy(pred, truth):
    return float(np.mean([p == t for p, t in zip(pred, truth)]))


def pcf1(pred, truth, classes):
    """Macro F1 across classes, so a dominant class cannot hide a bad model."""
    f1s = []
    for c in classes:
        tp = sum(1 for p, t in zip(pred, truth) if p == c and t == c)
        fp = sum(1 for p, t in zip(pred, truth) if p == c and t != c)
        fn = sum(1 for p, t in zip(pred, truth) if p != c and t == c)
        if tp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn)
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


def centroid_pred(Xtr, ytr, Xte, classes):
    C = []
    for c in classes:
        m = np.array([yy == c for yy in ytr])
        C.append(l2norm(Xtr[m].mean(0, keepdims=True))[0] if m.any() else None)
    keep = [i for i, c in enumerate(C) if c is not None]
    Cm = np.vstack([C[i] for i in keep])
    return [classes[keep[d]] for d in _sqdist(Xte, Cm).argmin(1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--label-field", default="label")
    ap.add_argument("--sizes", default="250,500,1000,2000,3000,4000")
    ap.add_argument("--test-size", type=int, default=1000)
    ap.add_argument("--embed-backend", default="fastembed")
    ap.add_argument("--embed-model", default="")
    ap.add_argument("--cache", default="/tmp/lc_embeddings.npy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    texts, y = load(args.input, args.labels, args.label_field)
    classes = sorted(set(y))
    print(f"corpus {len(texts)} documents, {len(classes)} classes", file=sys.stderr)

    emb = build_embedder(args, quiet=True)
    X = embed_cached(texts, emb, args.cache)

    # stratified split: test set fixed, training pool the rest
    rng = np.random.default_rng(args.seed)
    by_class = {c: rng.permutation([i for i, v in enumerate(y) if v == c])
                for c in classes}
    per_test = max(1, args.test_size // len(classes))
    test_idx = np.concatenate([v[:per_test] for v in by_class.values()])
    pool_by_class = {c: v[per_test:] for c, v in by_class.items()}
    Xte, yte = X[test_idx], [y[i] for i in test_idx]
    print(f"fixed held-out test set: {len(Xte)} documents "
          f"({len(Xte) // len(classes)} per class)", file=sys.stderr)

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    print(f"{'train n':>8} {'LR acc':>8} {'LR macro-F1':>12} "
          f"{'centroid acc':>13} {'unsup purity':>13}")
    for n in sizes:
        per = max(1, n // len(classes))
        take = np.concatenate([v[:per] for v in pool_by_class.values()])
        if len(take) == 0:
            continue
        Xtr, ytr = X[take], [y[i] for i in take]
        W = fit_linear(Xtr, ytr, classes)
        lin_pred = predict(Xte, W, classes)
        cen_pred = centroid_pred(Xtr, ytr, Xte, classes)
        lab, _, _ = kmeans(Xtr, len(classes), n_init=3)
        per_cluster = {}
        for c, t in zip(lab, ytr):
            per_cluster.setdefault(int(c), Counter())[t] += 1
        pure = sum(max(v.values()) for v in per_cluster.values()) / len(ytr)
        print(f"{len(Xtr):>8} {accuracy(lin_pred, yte):>8.3f} "
              f"{pcf1(lin_pred, yte, classes):>12.3f} {accuracy(cen_pred, yte):>13.3f} "
              f"{pure:>13.3f}")

    maj = max(Counter(yte).values()) / len(yte)
    print(f"\nmajority-class baseline on the test set: {maj:.3f}")
    print("Read the shape: if the last two rows are within ~0.02, more labels of "
          "this kind will not help — worth comparing against separation_probe.py "
          "before buying annotation.")


if __name__ == "__main__":
    main()
