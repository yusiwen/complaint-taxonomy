#!/usr/bin/env python3
"""R5: train a classifier on the labelled data, and report what it is worth.

Clustering finds structure; it does not produce a decision rule you can deploy. Once
`make_labels.py` has labelled a few thousand conversations against the codebook, a
supervised model is both cheaper to run and far more accurate than assigning new
traffic to the nearest centroid. This trains a softmax regression on the document
embeddings -- deliberately the simplest thing that can work, because if a linear model
on these features cannot beat the majority class, no amount of classifier architecture
is the answer.

    uv run python tools/train_classifier.py \
        --input data/public/abcd_500_zh.jsonl --labels out/labels_zh/labels.jsonl \
        --out out/clf_zh

Reports stratified k-fold out-of-fold predictions, so every number comes from items the
model did not train on, plus the three baselines that decide whether the model is worth
deploying: the majority class, a nearest-centroid rule on the same embeddings, and the
agreement rate you would get by trusting the cluster assignment.

Saved: `model.npz` (weights + class names + embedding model) and `report.json`.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cluster_complaints import FastEmbedEmbedder, l2norm  # noqa: E402


def read_jsonl(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                o = json.loads(line)
                out[o["id"]] = o
    return out


def user_text(o):
    return " ".join(m["content"] for m in o.get("messages", [])
                    if m.get("role") == "user").strip()


def softmax_fit(X, y, n_classes, epochs=400, lr=0.5, l2=1e-4, seed=0):
    """Multinomial logistic regression by full-batch Adam. Deterministic."""
    n, d = X.shape
    W = np.zeros((d, n_classes), dtype=np.float64)
    b = np.zeros(n_classes, dtype=np.float64)
    Y = np.zeros((n, n_classes))
    Y[np.arange(n), y] = 1.0
    mW = np.zeros_like(W)
    vW = np.zeros_like(W)
    mb = np.zeros_like(b)
    vb = np.zeros_like(b)
    for t in range(1, epochs + 1):
        Z = X @ W + b
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        G = (P - Y) / n
        gW = X.T @ G + l2 * W
        gb = G.sum(axis=0)
        mW = 0.9 * mW + 0.1 * gW
        vW = 0.999 * vW + 0.001 * gW ** 2
        mb = 0.9 * mb + 0.1 * gb
        vb = 0.999 * vb + 0.001 * gb ** 2
        W -= lr * (mW / (1 - 0.9 ** t)) / (np.sqrt(vW / (1 - 0.999 ** t)) + 1e-8)
        b -= lr * (mb / (1 - 0.9 ** t)) / (np.sqrt(vb / (1 - 0.999 ** t)) + 1e-8)
    return W, b


def predict(X, W, b):
    return (X @ W + b).argmax(axis=1)


def predict_proba(X, W, b):
    """Softmax probabilities. The router thresholds on the max of these."""
    Z = X @ W + b
    Z -= Z.max(axis=1, keepdims=True)
    P = np.exp(Z)
    return P / P.sum(axis=1, keepdims=True)


def strat_folds(y, k, seed=0):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[j % k].append(i)
    return [np.array(sorted(f)) for f in folds]


def classification_report(y_true, y_pred, classes):
    n = len(y_true)
    acc = float((y_true == y_pred).mean()) if n else 0.0
    per_class = {}
    for ci, name in enumerate(classes):
        tp = int(((y_pred == ci) & (y_true == ci)).sum())
        fp = int(((y_pred == ci) & (y_true != ci)).sum())
        fn = int(((y_pred != ci) & (y_true == ci)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per_class[name] = {"precision": round(p, 3), "recall": round(r, 3),
                           "f1": round(f1, 3), "support": int((y_true == ci).sum())}
    macro_f1 = (sum(v["f1"] for v in per_class.values()) / len(classes)) if classes else 0.0
    macro_f1_sup = ([v["f1"] for v in per_class.values() if v["support"]] or [0.0])
    macro_f1_sup = sum(macro_f1_sup) / len(macro_f1_sup)
    conf = {}
    for t, p in zip(y_true, y_pred):
        conf.setdefault(classes[t], {})
        conf[classes[t]][classes[p]] = conf[classes[t]].get(classes[p], 0) + 1
    return {"accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
            "macro_f1_supported_classes": round(macro_f1_sup, 4),
            "per_class": per_class, "confusion": conf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--labels", required=True,
                    help="jsonl with id + primary (from make_labels.py) or + label")
    ap.add_argument("--label-field", default="primary")
    ap.add_argument("--drop-field", default="", help="e.g. valid: skip falsy rows")
    ap.add_argument("--embed-model",
                    default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/clf")
    args = ap.parse_args()

    convs = read_jsonl(args.input)
    raw = read_jsonl(args.labels)
    items = []
    for cid, o in raw.items():
        lab = o.get(args.label_field)
        if lab is None or lab == "" or lab == "null":
            continue
        if args.drop_field and not o.get(args.drop_field):
            continue
        if cid not in convs:
            continue
        text = user_text(convs[cid])
        if not text:
            continue
        items.append((cid, text, str(lab)))
    if len(items) < 40:
        sys.exit(f"[fatal] only {len(items)} usable labelled items")

    classes = sorted({lab for _, _, lab in items})
    cidx = {c: i for i, c in enumerate(classes)}
    y = np.array([cidx[lab] for _, _, lab in items])
    print(f"[R5] {len(items)} items, {len(classes)} classes, "
          f"{args.folds}-fold CV")

    emb = FastEmbedEmbedder(args.embed_model, cache_dir="~/.cache/fastembed")
    X = l2norm(np.asarray(emb([t for _, t, _ in items]), dtype=np.float32))
    X = X.astype(np.float64)

    oof = np.full(len(y), -1, dtype=int)
    oof_proba = np.zeros((len(y), len(classes)))
    for f, test_idx in enumerate(strat_folds(y, args.folds, args.seed)):
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
        W, b = softmax_fit(X[train_idx], y[train_idx], len(classes), seed=args.seed)
        oof_proba[test_idx] = predict_proba(X[test_idx], W, b)
        oof[test_idx] = oof_proba[test_idx].argmax(axis=1)
        print(f"  fold {f + 1}: train {len(train_idx)} / test {len(test_idx)}")

    rep = classification_report(y, oof, classes)
    maj = classes[int(np.bincount(y).argmax())]
    maj_acc = float((y == cidx[maj]).mean())

    # nearest centroid on the same features: what you get without a trained model
    nc_pred = np.full(len(y), -1, dtype=int)
    for test_idx in strat_folds(y, args.folds, args.seed):
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
        C = np.zeros((len(classes), X.shape[1]))
        keep = []
        for ci in range(len(classes)):
            m = train_idx[y[train_idx] == ci]
            if len(m):
                C[ci] = l2norm(X[m].mean(0, keepdims=True))[0]
                keep.append(ci)
        sim = X[test_idx] @ C.T
        nc_pred[test_idx] = sim.argmax(1)
    nc_acc = float((nc_pred == y).mean())

    print(f"\n[R5] majority class ({maj}): {maj_acc:.3f}")
    print(f"[R5] nearest centroid:        {nc_acc:.3f}")
    print(f"[R5] softmax regression:      {rep['accuracy']:.3f} "
          f"(macro-F1 {rep['macro_f1_supported_classes']:.3f})")
    gain = rep["accuracy"] - nc_acc
    print(f"[R5] gain over nearest centroid: {gain:+.3f}")

    print(f"\n{'class':<30} {'P':>6} {'R':>6} {'F1':>6} {'n':>5}")
    for name, m in sorted(rep["per_class"].items(), key=lambda kv: kv[1]["f1"]):
        print(f"{name[:30]:<30} {m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{m['f1']:>6.3f} {m['support']:>5}")

    os.makedirs(args.out, exist_ok=True)
    W, b = softmax_fit(X, y, len(classes), seed=args.seed)
    # Class centroids belong in the model file: the router thresholds on distance to
    # the nearest centre, and recomputing them from the weights is not possible.
    centroids = np.zeros((len(classes), X.shape[1]), dtype=np.float32)
    for ci in range(len(classes)):
        m = y == ci
        if m.any():
            centroids[ci] = l2norm(X[m].mean(0, keepdims=True))[0]
    np.savez_compressed(os.path.join(args.out, "model.npz"),
                        W=W.astype(np.float32), b=b.astype(np.float32),
                        centroids=centroids, classes=np.array(classes),
                        embed_model=np.array(args.embed_model))
    with open(os.path.join(args.out, "oof_items.jsonl"), "w", encoding="utf-8") as f:
        for i, (cid, _, lab) in enumerate(items):
            row = oof_proba[i]
            order = np.argsort(-row)
            f.write(json.dumps({
                "id": cid, "true": lab, "pred": classes[int(oof[i])],
                "prob": round(float(row[order[0]]), 6),
                "margin": round(float(row[order[0]] - row[order[1]])
                                if len(order) > 1 else 0.0, 6),
                "proba": {classes[j]: round(float(row[j]), 6)
                          for j in range(len(classes))},
            }, ensure_ascii=False) + "\n")

    report = {"n_items": len(items), "n_classes": len(classes),
              "classes": classes, "folds": args.folds,
              "majority_accuracy": round(maj_acc, 4),
              "nearest_centroid_accuracy": round(nc_acc, 4),
              "softmax": rep, "embed_model": args.embed_model,
              "labels": os.path.abspath(args.labels)}
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[R5] wrote {args.out}/model.npz and report.json")


if __name__ == "__main__":
    main()
