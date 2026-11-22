#!/usr/bin/env python3
"""Audit a label set before buying more annotation.

"Is my label definition good enough?" is measurable, and the answer is usually
visible in three places:

  1. **Cohesion** -- do a class's own documents sit closer to its centroid than to
     any other class's centroid? If not, the class is not a region of the text.
  2. **Confusion** -- which class pairs does a trained model swap? A pair with
     high centroid cosine *and* high mutual confusion is one class wearing two
     names; merging or re-drawing the boundary beats adding more labels.
  3. **Recall** -- classes a model never predicts are either too rare to learn or
     defined by something absent from the text (a form field, a customer segment).

Output is a work list for rewriting definitions, not a score.

Usage
    uv run python tools/label_audit.py \
        --input data/public/cfpb_issue.jsonl \
        --labels data/public/cfpb_issue_labels.jsonl \
        --label-field label
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cluster_complaints import build_embedder, l2norm  # noqa: E402


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


def fit_linear(X, y, classes, lam=1e-2):
    idx = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((len(X), len(classes)), dtype=np.float32)
    Y[np.arange(len(X)), [idx[v] for v in y]] = 1.0
    A = X.T @ X + lam * np.eye(X.shape[1], dtype=np.float32)
    return np.linalg.solve(A, X.T @ Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--label-field", default="label")
    ap.add_argument("--cache", default="/tmp/lc_embeddings.npy")
    ap.add_argument("--embed-backend", default="fastembed")
    ap.add_argument("--embed-model", default="")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    texts, y = load(args.input, args.labels, args.label_field)
    classes = sorted(set(y))
    counts = Counter(y)
    print(f"{len(texts)} documents, {len(classes)} classes "
          f"(sizes {min(counts.values())}-{max(counts.values())})")

    if args.cache and os.path.exists(args.cache):
        X = np.load(args.cache)
        if X.shape[0] != len(texts):
            X = None
        else:
            print(f"[cache] {args.cache} {X.shape}", file=sys.stderr)
    if not (args.cache and os.path.exists(args.cache) and
            X is not None and X.shape[0] == len(texts)):
        emb = build_embedder(args, quiet=False)
        X = np.asarray(emb(texts), dtype=np.float32)
        X = l2norm(l2norm(X) - l2norm(X).mean(0, keepdims=True))
        if args.cache:
            np.save(args.cache, X)

    # ---- 1. cohesion: own centroid vs nearest foreign centroid
    cent, keep, own = [], [], []
    for c in classes:
        m = np.array([v == c for v in y])
        if m.sum() >= 2:
            cv = l2norm(X[m].mean(0, keepdims=True))[0]
            cent.append(cv)
            own.append(float((X[m] @ cv).mean()))   # member-to-own-centroid cosine
            keep.append(c)
    C = np.vstack(cent)
    S = C @ C.T                                  # class-centroid-to-centroid cosine
    own = np.array(own)
    off = S.copy()
    np.fill_diagonal(off, -1.0)

    print(f"\n{'class':52s} {'n':>5} {'own':>6} {'nearest other':>28} {'cos':>6}")
    rows = []
    for i, _c in enumerate(keep):
        j = int(off[i].argmax())
        rows.append((keep[i], counts[keep[i]], own[i], keep[j], off[i, j]))
    for r in sorted(rows, key=lambda r: r[2])[: args.top]:
        print(f"{r[0][:52]:52s} {r[1]:>5} {r[2]:>6.3f} {r[3][:28]:>28} {r[4]:>6.3f}")
    print("  (own = cosine to own centroid; low own or own < cos means the class "
          "is not a region of the text)")

    # ---- 2. most similar class pairs
    pairs = []
    for i in range(len(keep)):
        for j in range(i + 1, len(keep)):
            pairs.append((S[i, j], keep[i], keep[j]))
    pairs.sort(reverse=True)
    print("\nmost similar class pairs (merge candidates):")
    for s, a, b in pairs[: args.top]:
        print(f"  {s:.3f}  {a[:46]:46s} | {b[:46]}")

    # ---- 3. confusion of a trained model on a held-out split
    rng = np.random.default_rng(0)
    by_class = defaultdict(list)
    for i, v in enumerate(y):
        by_class[v].append(i)
    tr, te = [], []
    for _c, idxs in by_class.items():
        idxs = rng.permutation(idxs)
        n_te = max(1, int(len(idxs) * args.test_frac))
        te.extend(idxs[:n_te].tolist())
        tr.extend(idxs[n_te:].tolist())
    W = fit_linear(X[tr], [y[i] for i in tr], classes)
    pred = [classes[i] for i in np.argmax(X[te] @ W, axis=1)]
    gold = [y[i] for i in te]
    acc = float(np.mean([p == g for p, g in zip(pred, gold)]))
    print(f"\nheld-out accuracy {acc:.3f} on {len(te)} documents "
          f"(majority baseline {max(Counter(gold).values()) / len(gold):.3f})")

    recall = {}
    for c in classes:
        idx = [i for i, g in enumerate(gold) if g == c]
        if idx:
            recall[c] = float(np.mean([pred[i] == c for i in idx]))
    print(f"\n{'class':52s} {'n':>4} {'recall':>7}")
    for c, r in sorted(recall.items(), key=lambda kv: kv[1])[: args.top]:
        n = sum(1 for g in gold if g == c)
        print(f"{c[:52]:52s} {n:>4} {r:>7.2f}")
    print("  (recall ≈ 0 means the class is never predicted: too rare, or defined "
          "by information that is not in the text)")

    conf = Counter((g, p) for g, p in zip(gold, pred) if g != p)
    print("\ntop confusions (gold -> predicted):")
    for (g, p), n in conf.most_common(args.top):
        print(f"  {n:>3}x  {g[:44]:44s} -> {p[:44]}")

    # ---- verdict
    merge = [(a, b, s) for s, a, b in pairs if s > 0.90][:6]
    dead = [c for c, r in recall.items() if r == 0.0]
    print("\nverdict")
    if merge:
        print(f"  {len(merge)} class pair(s) with centroid cosine > 0.90 -- merge or "
              f"write an explicit boundary rule:")
        for a, b, s in merge:
            print(f"    {s:.3f}  {a}  <->  {b}")
    if dead:
        print(f"  {len(dead)} class(es) never predicted -- check whether they are "
              f"defined by text at all:")
        for c in dead[:8]:
            print(f"    {c}")
    if not merge and not dead:
        print("  no obvious structural defect; the ceiling is probably label noise "
              "rather than class design")
    print("\nNext step: rewrite the definitions for the pairs/classes above, "
          "re-annotate 100 items twice, and check kappa before scaling up.")


if __name__ == "__main__":
    main()
