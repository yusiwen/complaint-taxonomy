#!/usr/bin/env python3
"""Sweep k, report purity/NMI/ARI for each, and pick k by ARI rather than purity.

Why this exists: purity rises with k by construction (at k = n it is 1.0), so
choosing k by purity always walks towards over-segmentation. ARI is chance-
corrected and penalises splitting one class across clusters, so it is the honest
column to pick from. The pipeline is invoked as a subprocess per k, so the numbers
describe exactly what the pipeline produces (sentence-level clustering, orphan
rescue, document-level `primary`), not a simplified re-implementation.

    uv run python tools/k_sweep.py \
        --input data/public/abcd_500_zh.jsonl \
        --labels data/public/abcd_500_labels.jsonl \
        --k 4 6 8 10 12 16 20 30 40

Outputs (under --out, default out/ksweep)
    sweep.csv   one row per k
    sweep.json  same, plus the chosen k

Two different numbers both called "ARI" -- do not mix them up:
  * accuracy ARI  (column `ari`)     agreement with the true labels; the thing you
                                     want high, bounded by the supervised ceiling
  * stability ARI (column `stab`)    agreement between two runs at the SAME k with
                                     different k-means seeds; above ~0.85 the
                                     partition is reproducible, below ~0.7 the
                                     cluster boundaries are arbitrary
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_against_labels import ari, nmi  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_jsonl(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                o = json.loads(line)
                out[o["id"]] = o
    return out


def load_truth(path, label_map, field):
    truth = {}
    for k, v in read_jsonl(path).items():
        lab = v.get(field)
        if lab is None:
            continue
        truth[k] = label_map.get(lab, "_other") if label_map else lab
    return truth


def primary_of(conv_labels):
    pred = {}
    for cid, o in conv_labels.items():
        dom = o.get("primary")
        if dom is None:
            clusters = o.get("labels") or []
            if not clusters:
                continue
            dom = max(clusters, key=lambda d: (d["n"], -d["cluster"]))["cluster"]
        pred[cid] = dom
    return pred


def run_pipeline(args, k, outdir, seed):
    cmd = [sys.executable, os.path.join(ROOT, "cluster_complaints.py"),
           "--input", args.input, "--k", str(k), "--out", outdir,
           "--embed-backend", args.embed_backend,
           "--kmeans-seed", str(seed),
           "--cue-mode", args.cue_mode]
    if args.passthrough:
        cmd += args.passthrough
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = "\n".join((r.stdout + r.stderr).splitlines()[-6:])
        raise SystemExit(f"pipeline failed at k={k} seed={seed}:\n{tail}")
    return read_jsonl(os.path.join(outdir, "conv_labels.jsonl")), time.time() - t0


def ascii_curve(rows, width=40):
    """One marker per k on a shared axis -- the shape is the point, not the art."""
    vals = [r["ari"] for r in rows]
    lo, hi = min(vals), max(vals)
    span = hi - lo if hi > lo else 1.0
    out = []
    for r in rows:
        pos = int(round((r["ari"] - lo) / span * (width - 1)))
        line = [" "] * width
        line[pos] = "o"
        out.append(f"  k={r['k']:<3d} |{''.join(line)}| {r['ari']:.3f}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--labels", default="", help="ground truth to score against")
    ap.add_argument("--label-field", default="label")
    ap.add_argument("--label-map", default="",
                    help="json {fine_label: coarse_label}; scores a coarser axis "
                         "without re-clustering (unmapped labels -> '_other')")
    ap.add_argument("--k", nargs="+", type=int, default=[4, 6, 8, 10, 12, 16, 20, 30, 40])
    ap.add_argument("--embed-backend", default="fastembed")
    ap.add_argument("--cue-mode", default="off")
    ap.add_argument("--stability-runs", type=int, default=1,
                    help="runs per k with different k-means seeds (>=2 measures "
                         "stability ARI)")
    ap.add_argument("--stability-seed-step", type=int, default=100,
                    help="gap between the seed pools of consecutive runs; must be "
                         ">= n_init (5), otherwise the pools overlap and every run "
                         "reports a fake stability of 1.0")
    ap.add_argument("--out", default="out/ksweep")
    ap.add_argument("--passthrough", nargs=argparse.REMAINDER, default=[],
                    help="extra flags forwarded to cluster_complaints.py")
    args = ap.parse_args()

    label_map = {}
    if args.label_map:
        with open(args.label_map, encoding="utf-8") as f:
            label_map = json.load(f)
    truth = load_truth(args.labels, label_map, args.label_field) if args.labels else {}

    os.makedirs(args.out, exist_ok=True)
    rows = []
    print(f"{'k':>4} {'purity':>7} {'NMI':>6} {'ARI':>6} {'stab':>6} "
          f"{'cov':>5} {'labels/doc':>11} {'min':>4} {'empty':>6} {'sec':>5}")
    for k in args.k:
        subdir = os.path.join(args.out, f"k{k}")
        runs = []
        elapsed = 0.0
        for s in range(max(1, args.stability_runs)):
            seed = s * args.stability_seed_step
            cl, dt = run_pipeline(args, k,
                                  subdir if s == 0 else f"{subdir}_s{s}", seed)
            runs.append(primary_of(cl))
            elapsed += dt
        pred = runs[0]
        sizes = {}
        for c in pred.values():
            sizes[c] = sizes.get(c, 0) + 1
        # stability: how well a second seed reproduces the same partition
        stab = None
        if len(runs) > 1:
            # ARI is already symmetric in its two arguments, so compare each
            # extra run against the first one directly. (Mirroring the pairs
            # would build a different contingency table and understate ARI.)
            vals = []
            for j in range(1, len(runs)):
                pairs = [(pred[i], runs[j][i]) for i in pred if i in runs[j]]
                if pairs:
                    vals.append(ari(pairs))
            stab = sum(vals) / len(vals) if vals else None
        row = {"k": k, "stability": stab, "elapsed_s": round(elapsed, 1),
               "min_cluster": min(sizes.values()) if sizes else 0,
               "empty_clusters": k - len(sizes),
               "mean_labels": None, "purity": None, "nmi": None, "ari": None,
               "coverage": None}
        if truth:
            scored = [(pred[c], truth[c]) for c in truth if c in pred]
            per_cluster = {}
            for c, t in scored:
                per_cluster.setdefault(c, {}).setdefault(t, 0)
                per_cluster[c][t] += 1
            row["purity"] = sum(max(v.values()) for v in per_cluster.values()) / len(scored)
            row["nmi"] = nmi(scored)
            row["ari"] = ari(scored)
            row["coverage"] = len(scored) / max(len(truth), 1)
            nlab = [len(read_jsonl(os.path.join(subdir, "conv_labels.jsonl"))
                        .get(c, {}).get("labels") or []) for c in truth]
            row["mean_labels"] = sum(nlab) / max(len(nlab), 1)
        rows.append(row)
        print(f"{k:>4} {fmt(row['purity']):>7} {fmt(row['nmi']):>6} "
              f"{fmt(row['ari']):>6} {fmt(stab):>6} {fmt(row['coverage']):>5} "
              f"{fmt(row['mean_labels']):>11} {row['min_cluster']:>4} "
              f"{row['empty_clusters']:>6} {row['elapsed_s']:>5.1f}")

    with open(os.path.join(args.out, "sweep.csv"), "w", encoding="utf-8") as f:
        cols = ["k", "purity", "nmi", "ari", "stability", "coverage",
                "mean_labels", "min_cluster", "empty_clusters", "elapsed_s"]
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("" if r[c] is None else str(r[c]) for c in cols) + "\n")

    print()
    if not truth:
        print("no --labels given: sweep reports cluster geometry only, no ARI")
        return
    best = max(rows, key=lambda r: (r["ari"] if r["ari"] is not None else -1))
    print("ARI across k (pick the top of this curve, not the purity column):")
    print(ascii_curve(rows))
    print()
    print(f"best k by ARI: {best['k']}  (ARI {best['ari']:.3f}, purity "
          f"{best['purity']:.3f}, NMI {best['nmi']:.3f})")
    over = [r for r in rows if r["k"] > best["k"]]
    if over:
        gain = over[-1]["purity"] - best["purity"]
        print(f"purity keeps climbing past it: k={over[-1]['k']} shows purity "
              f"{over[-1]['purity']:.3f} (+{gain:.3f}) while ARI is "
              f"{over[-1]['ari']:.3f} -- that extra purity is over-segmentation, "
              f"not signal")
    with open(os.path.join(args.out, "sweep.json"), "w", encoding="utf-8") as f:
        json.dump({"best_k_by_ari": best["k"], "rows": rows}, f, indent=2)


def fmt(v):
    if v is None:
        return "-"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


if __name__ == "__main__":
    main()
