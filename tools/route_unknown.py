#!/usr/bin/env python3
"""R6: decide what the classifier is allowed to answer, and route the rest to humans.

A classifier that must always answer will force a brand-new complaint type into whichever
old class looks nearest, and that is how a taxonomy quietly rots: the novel class shows up
as "noise" spread over three existing categories, nobody sees it, and the metrics stay
flat because there is no ground truth for something nobody has defined yet.

So the classifier gets a confidence threshold and an explicit `UNKNOWN` answer. This tool
has three jobs:

  1. **Pick the threshold** from out-of-fold probabilities (`--oof`), reporting the trade
     the threshold makes: how much traffic is answered automatically, how accurate those
     answers are, and how big the human queue becomes.
  2. **Prove the routing works** (`--simulate-unknown <class>`): hold one class out of
     training entirely, then check that its items mostly fall below the threshold instead
     of being absorbed. That is the acceptance criterion in the design doc, and without
     this test "unknown handling" is an untested claim.
  3. **Apply it** (`--input` + `--model`): emit `label` or `UNKNOWN` per item for
     production, with the probability attached.

    # 1. threshold curve from out-of-fold probabilities
    uv run python tools/route_unknown.py --oof out/clf_zh/oof_items.jsonl

    # 2. does a class the model never saw get routed to humans?
    uv run python tools/route_unknown.py --input data/public/abcd_500_zh.jsonl \
        --labels out/labels_zh/labels.jsonl --simulate-unknown 账户基础信息查询

    # 3. apply
    uv run python tools/route_unknown.py --input new.jsonl --model out/clf_zh/model.npz \
        --threshold 0.55 --out out/routed.jsonl
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))


def read_jsonl(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                o = json.loads(line)
                out[o["id"]] = o
    return out


def curve(rows, thresholds):
    """Coverage against accuracy, per threshold, on out-of-fold predictions."""
    out = []
    for t in thresholds:
        kept = [r for r in rows if r["prob"] >= t]
        if not kept:
            out.append({"threshold": t, "coverage": 0.0, "n": 0,
                        "accepted_accuracy": None, "human_queue": 1.0,
                        "wrong_auto": 0})
            continue
        right = sum(1 for r in kept if r["pred"] == r["true"])
        out.append({"threshold": t,
                    "coverage": len(kept) / len(rows),
                    "n": len(kept),
                    "accepted_accuracy": right / len(kept),
                    "human_queue": 1 - len(kept) / len(rows),
                    "wrong_auto": len(kept) - right})
    return out


def print_curve(items, thresholds):
    """Print the trade-off table for one out-of-fold prediction set."""
    base = sum(1 for r in items if r["pred"] == r["true"]) / len(items)
    print(f"out-of-fold items: {len(items)}, accuracy if everything is answered: "
          f"{base:.3f}")
    print(f"{'thresh':>7} {'coverage':>9} {'answered acc':>13} {'human queue':>12} "
          f"{'wrong auto':>11}")
    for r in curve(items, thresholds):
        acc = "-" if r["accepted_accuracy"] is None else f"{r['accepted_accuracy']:.3f}"
        print(f"{r['threshold']:>7.2f} {r['coverage'] * 100:>8.0f}% {acc:>13} "
              f"{r['human_queue'] * 100:>11.0f}% {r['wrong_auto']:>11}")


def simulate_unknown(args):
    """Train without one class, then see whether that class is routed away.

    This is the honest test of unknown-handling: the held-out class is a real complaint
    type the model has never been taught, which is what a new taxonomy class looks like
    in production.
    """
    from train_classifier import predict_proba, softmax_fit, strat_folds

    from cluster_complaints import FastEmbedEmbedder, l2norm

    convs = read_jsonl(args.input)
    labels = read_jsonl(args.labels)
    items = []
    for cid, o in labels.items():
        lab = o.get("primary")
        if not lab or cid not in convs:
            continue
        text = " ".join(m["content"] for m in convs[cid].get("messages", [])
                        if m.get("role") == "user").strip()
        if text:
            items.append((cid, text, str(lab)))
    held = [it for it in items if it[2] == args.simulate_unknown]
    train = [it for it in items if it[2] != args.simulate_unknown]
    if len(held) < 3:
        sys.exit(f"[fatal] only {len(held)} items of class {args.simulate_unknown!r}")

    classes = sorted({it[2] for it in train})
    cidx = {c: i for i, c in enumerate(classes)}
    emb = FastEmbedEmbedder(args.embed_model, cache_dir="~/.cache/fastembed")
    Xtr = l2norm(np.asarray(emb([t for _, t, _ in train]), dtype=np.float32))
    Xho = l2norm(np.asarray(emb([t for _, t, _ in held]), dtype=np.float32))
    ytr = np.array([cidx[lab] for _, _, lab in train])

    # out-of-fold probabilities for the *kept* classes, to mirror deployment conditions
    oof = np.zeros((len(train), len(classes)))
    for test_idx in strat_folds(ytr, args.folds, args.seed):
        fit_idx = np.setdiff1d(np.arange(len(train)), test_idx)
        W, b = softmax_fit(Xtr[fit_idx].astype(np.float64), ytr[fit_idx],
                           len(classes), seed=args.seed)
        oof[test_idx] = predict_proba(Xtr[test_idx].astype(np.float64), W, b)

    W, b = softmax_fit(Xtr.astype(np.float64), ytr, len(classes), seed=args.seed)
    p_ho = predict_proba(Xho.astype(np.float64), W, b)
    max_ho = p_ho.max(axis=1)

    # A second, independent novelty score: cosine distance to the nearest class
    # centroid in embedding space. Softmax confidence can be high for a novel item that
    # happens to sit near an existing class; distance to the nearest centre is a
    # blunter but differently-shaped signal, so it is worth measuring before concluding
    # that "confidence does not detect novelty".
    C = np.zeros((len(classes), Xtr.shape[1]), dtype=np.float32)
    for ci in range(len(classes)):
        m = ytr == ci
        if m.any():
            C[ci] = l2norm(Xtr[m].mean(0, keepdims=True))[0]
    dist_ho = 1.0 - (Xho @ C.T).max(axis=1)
    dist_tr = 1.0 - (Xtr @ C.T).max(axis=1)

    print(f"[R6] simulated unseen class: {args.simulate_unknown!r} "
          f"({len(held)} items), trained on {len(classes)} classes / {len(train)} items")
    print(f"{'thresh':>7} {'routed away':>12} {'kept-classes answered':>22} "
          f"{'kept-classes acc':>17}")
    rows = []
    for t in args.thresholds:
        routed = float((max_ho < t).mean())
        kept_rows = [{"prob": float(oof[i].max()),
                      "pred": classes[int(oof[i].argmax())],
                      "true": classes[int(ytr[i])]} for i in range(len(train))]
        c = [r for r in kept_rows if r["prob"] >= t]
        cov = len(c) / len(kept_rows)
        acc = (sum(1 for r in c if r["pred"] == r["true"]) / len(c)) if c else None
        rows.append({"threshold": t, "unseen_routed": routed,
                     "kept_coverage": cov, "kept_accuracy": acc})
        print(f"{t:>7.2f} {routed * 100:>11.0f}% {cov * 100:>21.0f}% "
              f"{('-' if acc is None else f'{acc:.3f}'):>17}")
    print("\nA usable threshold sends most of the unseen class to humans while still "
          "answering the classes it was trained on. If both columns move together, the "
          "confidence signal is not separating novelty from uncertainty.")

    print(f"\n{'dist':>7} {'routed away':>12} {'kept-classes answered':>22} "
          f"{'kept-classes acc':>17}   (cosine distance to nearest class centre)")
    for t in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        routed = float((dist_ho >= t).mean())
        keep_rows = [{"d": float(dist_tr[i]),
                      "pred": classes[int(oof[i].argmax())],
                      "true": classes[int(ytr[i])]} for i in range(len(train))]
        c = [r for r in keep_rows if r["d"] < t]
        cov = len(c) / len(keep_rows)
        acc = (sum(1 for r in c if r["pred"] == r["true"]) / len(c)) if c else None
        print(f"{t:>7.2f} {routed * 100:>11.0f}% {cov * 100:>21.0f}% "
              f"{('-' if acc is None else f'{acc:.3f}'):>17}")
    return rows


def apply_router(args):
    from train_classifier import predict_proba

    from cluster_complaints import FastEmbedEmbedder, l2norm

    z = np.load(args.model, allow_pickle=True)
    W, b = z["W"].astype(np.float64), z["b"].astype(np.float64)
    classes = [str(c) for c in z["classes"]]
    model = str(z["embed_model"])
    centroids = (z["centroids"].astype(np.float32)
                 if "centroids" in z.files else None)
    if args.score == "distance" and centroids is None:
        sys.exit("[fatal] this model file carries no centroids; retrain with the "
                 "current train_classifier.py or use --score prob")

    convs = read_jsonl(args.input)
    ids = list(convs)
    texts = [" ".join(m["content"] for m in convs[c].get("messages", [])
                      if m.get("role") == "user").strip() for c in ids]
    keep = [i for i, t in enumerate(texts) if t]
    emb = FastEmbedEmbedder(model, cache_dir="~/.cache/fastembed")
    X = l2norm(np.asarray(emb([texts[i] for i in keep]), dtype=np.float32))
    P = predict_proba(X.astype(np.float64), W, b)

    novelty = None
    if args.score == "distance":
        novelty = 1.0 - (X @ centroids.T).max(axis=1)

    out_rows = []
    unknown = 0
    for row_i, item_i in enumerate(keep):
        p = P[row_i]
        j = int(p.argmax())
        conf = float(p[j])
        if args.score == "distance":
            # distance semantics are inverted on purpose: high novelty -> UNKNOWN
            route_away = float(novelty[row_i]) >= args.threshold
            conf = 1.0 - float(novelty[row_i])
        else:
            route_away = conf < args.threshold
        label = "UNKNOWN" if route_away else classes[j]
        if label == "UNKNOWN":
            unknown += 1
        out_rows.append({"id": ids[item_i], "label": label,
                         "confidence": round(conf, 4),
                         "best_guess": classes[j]})
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[R6] routed {len(out_rows)} items at threshold {args.threshold}: "
          f"{len(out_rows) - unknown} labelled, {unknown} UNKNOWN "
          f"({unknown / max(len(out_rows), 1):.0%} to humans)")
    print(f"[R6] wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="", help="out/clf/oof_items.jsonl")
    ap.add_argument("--input", default="")
    ap.add_argument("--labels", default="", help="needed for --simulate-unknown")
    ap.add_argument("--model", default="", help="clf model.npz for --input routing")
    ap.add_argument("--simulate-unknown", default="",
                    help="class to hold out entirely, to test routing of novelty")
    ap.add_argument("--score", choices=["prob", "distance"], default="prob",
                    help="prob: UNKNOWN when softmax confidence < threshold; "
                         "distance: UNKNOWN when cosine distance to the nearest class "
                         "centre >= threshold (measured about twice as good at routing "
                         "novelty away)")
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embed-model",
                    default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--out", default="out/routed.jsonl")
    args = ap.parse_args()

    if args.oof:
        items = list(read_jsonl(args.oof).values())
        print_curve(items, args.thresholds)
        return
    if args.simulate_unknown:
        if not (args.input and args.labels):
            sys.exit("[fatal] --simulate-unknown needs --input and --labels")
        simulate_unknown(args)
        return
    if args.input and args.model:
        apply_router(args)
        return
    sys.exit("[fatal] choose one: --oof, --input with --simulate-unknown, "
             "or --input with --model")


if __name__ == "__main__":
    main()
