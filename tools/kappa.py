#!/usr/bin/env python3
"""R4a: Cohen's kappa between two annotation files, overall and per class.

The codebook only counts as agreed when two annotators, working from it independently,
reach kappa >= 0.7 on 100-200 shared items. Below 0.6 the definitions are not doing
their job and annotating more items just creates a bigger noise pile -- label nothing
until this passes.

    uv run python tools/kappa.py --a annotator1.jsonl --b annotator2.jsonl

Both files are jsonl with `{"id", "primary"}` (the per-item single label) and optionally
`{"labels": [...]}` for the per-class binary kappa. Ids are matched; only shared ids
count, and that number is printed because a kappa on 20 items is not evidence.

Two things this reports that a single kappa number hides:
  * per-class kappa -- one vague class can drag the average down while every other
    class is fine, and the fix is to rewrite that one definition, not the codebook;
  * the confusion cells behind each disagreement, so the arguments are concrete.
"""
import argparse
import json
import sys
from collections import Counter


def read(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                o = json.loads(line)
                out[o["id"]] = o
    return out


def cohen(pairs):
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    if pe >= 1:
        return 1.0
    return (po - pe) / (1 - pe)


def binary_kappa(items, label):
    pairs = [((label in i[0]), (label in i[1])) for i in items]
    return cohen(pairs)


def verdict(k):
    if k is None:
        return "no data"
    if k >= 0.8:
        return "good"
    if k >= 0.7:
        return "acceptable (the bar)"
    if k >= 0.6:
        return "moderate -- rewrite the weakest definitions before annotating more"
    return "insufficient -- do not label at scale"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="annotator A, jsonl with id/primary")
    ap.add_argument("--b", required=True, help="annotator B")
    ap.add_argument("--field", default="primary")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args()

    a, b = read(args.a), read(args.b)
    shared = [i for i in a if i in b]
    if not shared:
        sys.exit("[fatal] no shared ids between the two files")

    # A row either annotator left blank must not be compared at all: two blanks would
    # count as agreement and quietly inflate kappa. Dropped rows are reported, because
    # "we agree 100%" computed on the six items both people filled in is not a result.
    skipped, answered = [], []
    for cid in shared:
        va, vb = a[cid].get(args.field), b[cid].get(args.field)
        blank = (None, "", "null", "None")
        if va in blank or vb in blank:
            skipped.append(cid)
        else:
            answered.append((cid, va, vb))

    pairs = [(va, vb) for _, va, vb in answered]
    if not pairs:
        sys.exit("[fatal] every shared row is blank on at least one side")
    k = cohen(pairs)
    n_agree = sum(1 for x, y in pairs if x == y)
    print(f"shared items             {len(shared)}")
    if skipped:
        print(f"  skipped (blank on one side) {len(skipped)}  "
              f"-> compared {len(pairs)}")
        print("  [warn] blanks are excluded rather than counted as agreement; "
              "an annotator who skipped half the sheet cannot inflate kappa")
    print(f"raw agreement            {n_agree / len(pairs):.3f} ({n_agree}/{len(pairs)})")
    print(f"Cohen's kappa            {k:.3f}   [{verdict(k)}]")

    if len(shared) < 100:
        print(f"[warn] {len(shared)} items is below the 100-200 the design asks for; "
              f"treat this as a smoke test")

    # per-class binary kappa over the label sets, when both sides carry them
    if answered and all("labels" in a[i] and "labels" in b[i] for i, _, _ in answered):
        items = [(set(a[i]["labels"]), set(b[i]["labels"])) for i, _, _ in answered]
        classes = sorted({c for x, _ in items for c in x} | {c for _, y in items for c in y})
        print(f"\n{'class':<28} {'pos A':>6} {'pos B':>6} {'kappa':>7}")
        rows = []
        for c in classes:
            kc = binary_kappa(items, c)
            rows.append((kc, c, sum(1 for x, _ in items if c in x),
                         sum(1 for _, y in items if c in y)))
        for kc, c, ca, cb in sorted(rows, key=lambda r: r[0] if r[0] is not None else 1):
            print(f"{c[:28]:<28} {ca:>6} {cb:>6} "
                  f"{kc:>7.3f}" if kc is not None else f"{c[:28]:<28} {ca:>6} {cb:>6}    n/a")
        vals = [r[0] for r in rows if r[0] is not None]
        if vals:
            print(f"{'macro over classes':<28} {'':>6} {'':>6} {sum(vals) / len(vals):>7.3f}")

    # where they disagree
    conf = Counter((x, y) for x, y in pairs if x != y)
    if conf:
        print("\nlargest disagreements (A -> B):")
        for (x, y), n in conf.most_common(6):
            print(f"  {x!s:<28} -> {y!s:<28} {n}")


if __name__ == "__main__":
    main()
