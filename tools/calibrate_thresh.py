#!/usr/bin/env python3
"""Sweep --knn-thresh and report quality against its real cost.

0.4 was tuned on one Chinese corpus. That number is not portable by assumption: the
threshold trades complaint recall against retained sentences, and where the optimum
sits depends on how much of *that* corpus is procedure, on the seed set, and on the
embedding model. This tool answers "does the threshold transfer?" per corpus instead
of asserting it.

Two things are reported together on purpose. Accuracy rises as more is filtered out,
right up to the point where real complaints start going with it; the cost column is
the share of conversations that end up with no sentence-level labels at all, which
silently pushes them onto the document-vector fallback.

    uv run python tools/calibrate_thresh.py \
        --input data/public/abcd.jsonl --labels data/public/abcd_labels.jsonl \
        --seeds data/seeds_proc_en.jsonl --thresh 0.3 0.4 0.5 --out out/calib_en

The recommendation is the best ARI among thresholds whose orphan share stays under
`--max-orphan` (default 15%), so a high-accuracy setting that guts the multi-label
output does not win by default.
"""
import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from eval_against_labels import ari, nmi  # noqa: E402

KEEP_RE = re.compile(r"keep (\d+)/(\d+) representatives")


def read_jsonl(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                o = json.loads(line)
                out[o["id"]] = o
    return out


def run(args, thresh, outdir):
    """Run the pipeline once. thresh=None means no seed filter at all."""
    cmd = [sys.executable, os.path.join(ROOT, "cluster_complaints.py"),
           "--input", args.input, "--out", outdir,
           "--k", str(args.k), "--kmeans-algo", args.kmeans_algo,
           "--embed-backend", args.embed_backend, "--cue-mode", "off",
           "--primary-mode", args.primary_mode]
    if thresh is not None:
        cmd += ["--seeds", args.seeds, "--knn-thresh", str(thresh)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"thresh={thresh} failed:\n"
                         + "\n".join(r.stderr.splitlines()[-6:]))
    kept = total = None
    for line in r.stdout.splitlines():
        m = KEEP_RE.search(line)
        if m:
            kept, total = int(m.group(1)), int(m.group(2))
    return os.path.join(outdir, "conv_labels.jsonl"), kept, total


def evaluate(conv_path, truth, field, label_map):
    pred, orphan, n_labels = {}, 0, []
    for cid, o in read_jsonl(conv_path).items():
        dom = o.get("primary")
        cl = o.get("labels") or []
        n_labels.append(len(cl))
        if not cl:
            orphan += 1
        if dom is None:
            if not cl:
                continue
            dom = max(cl, key=lambda d: (d["n"], -d["cluster"]))["cluster"]
        pred[cid] = dom
    pairs = []
    for cid, t in truth.items():
        if cid in pred:
            pairs.append((pred[cid], label_map.get(t, t) if label_map else t))
    if not pairs:
        return None
    per = {}
    for c, t in pairs:
        per.setdefault(c, {})
        per[c][t] = per[c].get(t, 0) + 1
    purity = sum(max(v.values()) for v in per.values()) / len(pairs)
    return {"n": len(pairs), "purity": purity, "nmi": nmi(pairs), "ari": ari(pairs),
            "orphan_share": orphan / max(len(n_labels), 1),
            "mean_labels": sum(n_labels) / max(len(n_labels), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--labels", default="")
    ap.add_argument("--label-field", default="label")
    ap.add_argument("--label-map", default="")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--thresh", nargs="+", type=float, default=[0.2, 0.3, 0.4, 0.5, 0.6])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--kmeans-algo", default="bisect")
    ap.add_argument("--embed-backend", default="fastembed")
    ap.add_argument("--primary-mode", default="doc")
    ap.add_argument("--max-orphan", type=float, default=0.15)
    ap.add_argument("--baseline", action="store_true",
                    help="also run once with no filter at all, for reference")
    ap.add_argument("--out", default="out/calib")
    args = ap.parse_args()

    truth, label_map = {}, {}
    if args.labels:
        truth = {k: v[args.label_field] for k, v in read_jsonl(args.labels).items()
                 if args.label_field in v}
    if args.label_map:
        with open(args.label_map, encoding="utf-8") as f:
            label_map = json.load(f)

    os.makedirs(args.out, exist_ok=True)
    rows = []
    if args.baseline:
        conv, kept, total = run(args, None, os.path.join(args.out, "nofilter"))
        r = evaluate(conv, truth, args.label_field, label_map) if truth else None
        rows.append({"thresh": None, "kept": kept, "total": total, **(r or {})})

    for th in args.thresh:
        conv, kept, total = run(args, th, os.path.join(args.out, f"t{th}"))
        r = evaluate(conv, truth, args.label_field, label_map) if truth else None
        rows.append({"thresh": th, "kept": kept, "total": total, **(r or {})})

    hdr = f"{'thresh':>7} {'kept':>13} {'purity':>7} {'NMI':>6} {'ARI':>6} " \
          f"{'labels/doc':>10} {'orphans':>8}"
    print(hdr)
    for r in rows:
        t = "none" if r["thresh"] is None else f"{r['thresh']:.2f}"
        kept = f"{r['kept']}/{r['total']}" if r["kept"] is not None else "-"
        print(f"{t:>7} {kept:>13} {r.get('purity', 0):>7.3f} "
              f"{r.get('nmi', 0):>6.3f} {r.get('ari', 0):>6.3f} "
              f"{r.get('mean_labels', 0):>10.2f} "
              f"{r.get('orphan_share', 0) * 100:>7.1f}%")

    scored = [r for r in rows if r["thresh"] is not None and "ari" in r]
    if scored:
        ok = [r for r in scored if r["orphan_share"] <= args.max_orphan] or scored
        best = max(ok, key=lambda r: r["ari"])
        print(f"\nbest ARI within an orphan budget of {args.max_orphan:.0%}: "
              f"thresh={best['thresh']:.2f} (ARI {best['ari']:.3f}, "
              f"NMI {best['nmi']:.3f}, purity {best['purity']:.3f}, "
              f"orphans {best['orphan_share']:.0%})")
        print("Compare that number with the same sweep on another corpus before "
              "treating the threshold as portable.")

    with open(os.path.join(args.out, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump({"input": os.path.abspath(args.input), "seeds": args.seeds,
                   "k": args.k, "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
