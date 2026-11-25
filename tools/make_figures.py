#!/usr/bin/env python3
"""Draw the measured results of one run on one page, as a hand-written SVG.

Every number on the page is read from a file under `out/`:

    out/<run>/clusters.json          sentence counts, k, algorithm
    out/<run>/codebook.json          cluster id -> class label, coverage
    out/<labels>/summary.json        annotation: n, abstain, agreement, ARI
    out/<merged>/summary.json        the same for the merged codebook
    out/<clf>/report.json            accuracy, baselines, per-class P/R/F1, confusion

Nothing is copied out of README.md, so the figure cannot drift away from the
artefacts: a stale page is impossible, only a stale run is.

    uv run python tools/make_figures.py                  # docs/assets/results.svg
    uv run python tools/make_figures.py --html           # + a page that embeds it
    uv run python tools/make_figures.py --clf out/other/report.json --out /tmp/f.svg

No matplotlib and no plotting library: the SVG is written by hand so this stays
inside the repo's "numpy + stdlib only" rule, and the output is a single file
that renders in a browser, in a README, and in a diff. Three panels are always
drawn (cluster sizes, annotation outcome, classifier against its baselines);
the per-class and confusion panels are dropped with a printed note if the
classifier report for the run has not been produced yet.
"""
from __future__ import annotations

import argparse
import json
import sys
from html import escape
from pathlib import Path

W = 1000
PAD = 24
GRID = 30  # confusion-matrix cell size

STYLE = """
  text   { font-family: ui-sans-serif, -apple-system, "Segoe UI", "Noto Sans CJK SC",
                      "Source Han Sans SC", "Microsoft YaHei", Helvetica, Arial, sans-serif;
           font-size: 12.5px; }
  .mono  { font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace; }
  .bg    { fill: #ffffff; }
  .card  { fill: #ffffff; stroke: #d0d7de; }
  .hair  { stroke: #d8dee4; }
  .h1    { fill: #1f2328; font-size: 19px; font-weight: 650; }
  .h2    { fill: #1f2328; font-size: 13px; font-weight: 600; }
  .note  { fill: #57606a; font-size: 11px; }
  .lbl   { fill: #1f2328; font-size: 11.5px; }
  .val   { fill: #1f2328; font-size: 11px; }
  .big   { fill: #1f2328; font-size: 23px; font-weight: 650; }
  .track { fill: #eaeef2; }
  .b1    { fill: #0969da; }
  .b2    { fill: #8250df; }
  .b3    { fill: #1a7f37; }
  .warn  { fill: #bc4c00; }
  .grid  { stroke: #e6ebf1; }
  .hm    { fill: #0969da; }
  .cmin  { fill: #1f2328; font-size: 10.5px; }
  .cmout { fill: #ffffff; font-size: 10.5px; font-weight: 600; }
  @media (prefers-color-scheme: dark) {
    .bg    { fill: #0d1117; }
    .card  { fill: #161b22; stroke: #30363d; }
    .hair  { stroke: #21262d; }
    .h1    { fill: #e6edf3; }
    .h2    { fill: #e6edf3; }
    .note  { fill: #9198a1; }
    .lbl   { fill: #e6edf3; }
    .val   { fill: #9198a1; }
    .big   { fill: #e6edf3; }
    .track { fill: #21262d; }
    .b1    { fill: #4493f8; }
    .b2    { fill: #ab7df8; }
    .b3    { fill: #3fb950; }
    .warn  { fill: #db6d28; }
    .grid  { stroke: #21262d; }
    .hm    { fill: #4493f8; }
    .cmin  { fill: #e6edf3; }
  }
"""


# ---------------------------------------------------------------- input


def load_json(path: Path, what: str):
    if not path.is_file():
        sys.exit(f"[fatal] missing {what}: {path}\n        run the stage that writes it first")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path, what: str) -> list[dict]:
    if not path.is_file():
        sys.exit(f"[fatal] missing {what}: {path}")
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------- helpers


def bar(x, y, w, h, cls, radius=None) -> str:
    """A horizontal bar, zero-width safe."""
    if w <= 0:
        return ""
    rx = f' rx="{radius}"' if radius else ""
    return f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h}"{rx}/>'


def text(x, y, s, cls="lbl", anchor=None, size=None, opacity=None) -> str:
    attrs = [f'class="{cls}"', f'x="{x:.1f}"', f'y="{y:.1f}"']
    if anchor:
        attrs.append(f'text-anchor="{anchor}"')
    if size:
        attrs.append(f'font-size="{size}"')
    if opacity is not None:
        attrs.append(f'fill-opacity="{opacity}"')
    return f"<text {' '.join(attrs)}>{escape(str(s))}</text>"


def mono(x, y, s, cls="val", anchor=None, size=None) -> str:
    return text(x, y, s, "mono " + cls if cls else "mono", anchor=anchor, size=size)


def hline(x1, x2, y) -> str:
    return f'<line class="hair" x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke-width="1"/>'


# ---------------------------------------------------------------- panels


def panel_header(x, y, title, note, note_x_right=None) -> str:
    out = text(x, y, title, "h2")
    if note_x_right is None:
        out += text(x, y + 17, note, "note")
    else:
        out += text(note_x_right, y, note, "note", anchor="end")
    return out


def kpi_cards(run, ann, clf) -> str:
    cover = run["codebook"]["covered"]
    n_unique = run["clusters"]["n_unique_sentences"]
    cov_pct = 100.0 * cover / n_unique if n_unique else 0.0
    n_ann = ann["n"]
    abstain_pct = 100.0 * ann["abstain"] / n_ann if n_ann else 0.0

    cards = [
        (
            "S1 · cluster",
            f"k={run['clusters']['k']}",
            [
                f"{run['clusters']['n_sentences']:,} sentence labels · "
                f"{n_unique:,} unique sentences",
                f"{run['clusters']['kmeans_algo']} k-means · covered {cover:,} "
                f"({cov_pct:.0f}%)",
            ],
        ),
        (
            "S5 · annotate",
            f"{abstain_pct:.0f}% abstain",
            [
                f"{n_ann} items labelled · {ann['abstain']} abstain · "
                f"{n_ann - ann['abstain']} answered",
                f"agreement with cluster {ann['agree_with_cluster']:.2f} · "
                f"ARI vs truth {fmt_or_dash(ann['ari_vs_truth'])}",
            ],
        ),
        (
            "S6 · classify",
            f"{clf['softmax']['accuracy']:.4f}",
            [
                f"out-of-fold accuracy · macro-F1 {clf['softmax']['macro_f1']:.4f}",
                f"{clf['n_items']} items · {clf['folds']} folds · "
                f"nearest centroid {clf['nearest_centroid_accuracy']:.4f}",
            ],
        ),
    ]
    w = (W - 2 * PAD - 2 * 16) // 3
    out = []
    for i, (head, big, lines) in enumerate(cards):
        x = PAD + i * (w + 16)
        out.append(f'<rect class="card" x="{x}" y="84" width="{w}" height="106" rx="7"/>')
        out.append(text(x + 14, 108, head, "mono note"))
        out.append(text(x + 14, 140, big, "big"))
        for j, line in enumerate(lines):
            out.append(text(x + 14, 160 + j * 16, line, "note"))
    return "".join(out)


def panel_clusters(x, y, w, run) -> tuple[str, int]:
    classes = run["codebook"]["classes"]
    rows = sorted(classes, key=lambda c: c["cluster"])
    n_unique = run["clusters"]["n_unique_sentences"]
    max_size = max(c["size"] for c in rows)
    top = y + 66
    step = 30
    bar_x = x + 170
    bar_max = w - 170 - 78
    out = [text(x, y + 24, "S1 · cluster sizes", "h2"),
           text(x, y + 41, f"label from taxonomy.json · share of {n_unique:,} sentences", "note")]
    for i, c in enumerate(rows):
        yy = top + i * step
        blen = bar_max * c["size"] / max_size
        out.append(text(x + 12, yy + 8, c["label"], "lbl"))
        out.append(bar(bar_x, yy - 6, bar_max, 12, "track", 3))
        out.append(bar(bar_x, yy - 6, blen, 12, "b1", 3))
        out.append(mono(bar_x + blen + 6, yy + 8, str(c["size"]), "val"))
        out.append(mono(x + w - 12, yy + 8, f"{100.0 * c['size'] / n_unique:.1f}%",
                        "mono val", anchor="end"))
    height = top - y + (len(rows) - 1) * step + 18
    return "".join(out), height


def panel_baselines(x, y, w, clf) -> tuple[str, int]:
    rows = [
        ("majority class", clf["majority_accuracy"], "track"),
        ("nearest centroid", clf["nearest_centroid_accuracy"], "b2"),
        ("softmax, 5-fold OOF", clf["softmax"]["accuracy"], "b1"),
    ]
    bar_x = x + 150
    bar_max = w - 150 - 62
    out = [text(x, y + 24, "S6 · against its baselines", "h2")]
    for i, (name, val, cls) in enumerate(rows):
        yy = y + 66 + i * 32
        out.append(text(x + 12, yy + 8, name, "lbl"))
        out.append(bar(bar_x, yy - 6, bar_max, 12, "track", 3))
        out.append(bar(bar_x, yy - 6, bar_max * val, 12, cls, 3))
        out.append(mono(x + w - 12, yy + 8, f"{val:.4f}", "mono val", anchor="end"))
    gain = clf["softmax"]["accuracy"] - clf["nearest_centroid_accuracy"]
    over_majority = 100 * (clf["softmax"]["accuracy"] - clf["majority_accuracy"])
    out.append(text(x + 12, y + 164,
                    f"softmax adds {100 * gain:+.1f} points over the centroid, "
                    f"{over_majority:+.1f} over majority", "note"))
    return "".join(out), 176


def panel_annotation(x, y, w, ann, merged) -> tuple[str, int]:
    bars = [("zh_final codebook", ann), ("merged codebook", merged)]
    out = [text(x, y + 24, "S5 · who answered, per codebook", "h2"),
           text(x, y + 41, "the abstain bar is the honest part: a class the definitions do "
                           "not cover gets no answer", "note")]
    for i, (name, s) in enumerate(bars):
        n = s["n"]
        answered = n - s["abstain"]
        yy = y + 66 + i * 52
        out.append(text(x + 12, yy, f"{name} · {len(s['primary_distribution'])} classes · "
                                    f"{n} items", "lbl"))
        scale = (w - 24) / n if n else 0.0
        wa, wb = answered * scale, s["abstain"] * scale
        out.append(bar(x + 12, yy + 8, wa, 20, "b1", 3))
        out.append(bar(x + 12 + wa, yy + 8, wb, 20, "warn", 3))
        if wa > 60:
            out.append(text(x + 12 + wa / 2, yy + 22, f"{answered} answered", "cmout",
                            anchor="middle"))
        if wb > 60:
            out.append(text(x + 12 + wa + wb / 2, yy + 22, f"{s['abstain']} abstain", "cmout",
                            anchor="middle"))
    return "".join(out), 176


def panel_per_class(x, y, w, clf) -> tuple[str, int]:
    per = clf["softmax"]["per_class"]
    rows = sorted(per.items(), key=lambda kv: -kv[1]["support"])
    bar_x = x + 160
    bar_max = w - 160 - 108
    step, pitch, bh = 40, 12, 8
    out = [text(x, y + 24, "S6 · precision / recall / F1 per class", "h2")]
    swatches = [("precision", "b1"), ("recall", "b2"), ("F1", "b3")]
    for i, (name, cls) in enumerate(swatches):
        sx = x + w - 64 - (2 - i) * 78
        out.append(f'<rect class="{cls}" x="{sx}" y="{y + 14}" width="9" height="9" rx="2"/>')
        out.append(text(sx + 13, y + 23, name, "note"))
    for i, (name, m) in enumerate(rows):
        yy = y + 62 + i * step
        out.append(text(x + 12, yy + 10, name, "lbl"))
        for j, (key, cls) in enumerate([("precision", "b1"), ("recall", "b2"), ("f1", "b3")]):
            by = yy + j * pitch
            out.append(bar(bar_x, by, bar_max, bh, "track", 2))
            out.append(bar(bar_x, by, bar_max * m[key], bh, cls, 2))
            out.append(mono(bar_x + bar_max + 8, by + bh, f"{m[key]:.2f}", "mono val"))
        out.append(mono(x + w - 12, yy + 10, f"n={m['support']}", "mono val", anchor="end"))
    small = [n for n, m in rows if m["support"] < 5]
    note = (f"{len(small)} classes have fewer than 5 held-out items; a P/R/F1 of 0.00 there is "
            "one item, not a category that failed") if small else ""
    if note:
        out.append(text(x, y + 62 + len(rows) * step + 2, note, "note"))
    height = 62 + len(rows) * step + (24 if note else 0)
    return "".join(out), height


def panel_confusion(x, y, w, clf, label_of) -> tuple[str, int]:
    classes = clf["classes"]
    per = clf["softmax"]["per_class"]
    conf = clf["softmax"]["confusion"]
    ids = {name: f"C{label_of[name]}" for name in classes}
    order = sorted(classes, key=lambda c: label_of[c])
    gx = x + 74
    gy = y + 62
    n = len(order)
    out = [text(x, y + 24, "S6 · confusion matrix, out-of-fold", "h2"),
           text(x + w - 12, y + 24, "cell = item count · row-normalised shade · "
                                    "column = predicted", "note", anchor="end")]
    for j, name in enumerate(order):
        out.append(mono(gx + j * GRID + GRID / 2, gy - 7, ids[name], "mono note",
                        anchor="middle", size=10.5))
    for i, true_name in enumerate(order):
        cy = gy + i * GRID
        out.append(mono(gx - 8, cy + GRID / 2 + 4, ids[true_name], "mono note", anchor="end",
                        size=10.5))
        support = per[true_name]["support"]
        for j, pred_name in enumerate(order):
            cx = gx + j * GRID
            count = conf.get(true_name, {}).get(pred_name, 0)
            out.append(f'<rect class="grid" x="{cx}" y="{cy}" width="{GRID}" height="{GRID}" '
                       f'fill="none"/>')
            if not count:
                continue
            share = count / support if support else 0.0
            out.append(f'<rect class="hm" x="{cx + 1}" y="{cy + 1}" width="{GRID - 2}" '
                       f'height="{GRID - 2}" fill-opacity="{0.12 + 0.88 * share:.2f}"/>')
            cls = "cmout" if share > 0.45 else "cmin"
            out.append(text(cx + GRID / 2, cy + GRID / 2 + 4, str(count), cls, anchor="middle"))
    legend_x = x + 74 + n * GRID + 46
    out.append(text(legend_x, gy - 7, "cluster id → class", "note"))
    for i, name in enumerate(order):
        ly = gy + 14 + i * 24
        out.append(mono(legend_x, ly, ids[name], "mono note", size=10.5))
        out.append(text(legend_x + 28, ly, name, "lbl"))
        out.append(mono(legend_x + 176, ly, f"n={per[name]['support']}", "mono val"))
    return "".join(out), 62 + n * GRID + 26


# ---------------------------------------------------------------- assembly


def fmt_or_dash(v) -> str:
    return "--" if v is None else f"{v:.3f}"


def wrap_tokens(tokens, max_chars=100) -> list[str]:
    """Pack ' · '-joined tokens into lines of at most max_chars characters."""
    lines, cur = [], ""
    for token in tokens:
        piece = token if not cur else f" · {token}"
        if cur and len(cur) + len(piece) > max_chars:
            lines.append(cur)
            cur = token
        else:
            cur += piece
    if cur:
        lines.append(cur)
    return lines


def build(args) -> str:
    base = Path(args.out)
    run_dir = Path(args.run)
    clusters = load_json(run_dir / "clusters.json", "clusters.json")
    codebook = load_json(run_dir / "codebook.json", "codebook.json")
    ann = load_json(Path(args.labels), "annotation summary")
    merged = load_json(Path(args.merged), "merged annotation summary")
    clf_p = Path(args.clf)
    clf = load_json(clf_p, "classifier report") if clf_p.is_file() else None
    if clf is None:
        print(f"[warn] no classifier report at {clf_p}: per-class and confusion panels skipped")
    label_of = {c["label"]: c["cluster"] for c in codebook["classes"]}
    unknown = sorted(set(clf["classes"]) - set(label_of)) if clf else []
    if unknown:
        sys.exit("[fatal] the classifier report was trained on a different codebook: "
                 f"{unknown} are not in {run_dir}/codebook.json")

    run = {"clusters": clusters, "codebook": codebook}
    body = [kpi_cards(run, ann, clf) if clf else kpi_cards_no_clf(run, ann)]

    left_w, right_w = 472, W - PAD - 16 - (PAD + 472)
    left_x, right_x = PAD, PAD + 472 + 16
    y = 216
    sizes, h1 = panel_clusters(left_x, y, left_w, run)
    baselines, h2 = panel_baselines(right_x, y, right_w, clf) if clf else ("", 0)
    annot, h3 = panel_annotation(right_x, y + h2 + 18, right_w, ann, merged)
    body.append(f'<rect class="card" x="{left_x}" y="{y - 8}" width="{left_w}" '
                f'height="{h1 + 16}" rx="7"/>')
    body.append(f'<rect class="card" x="{right_x}" y="{y - 8}" width="{right_w}" '
                f'height="{h2 + h3 + 34 if clf else h3 + 16}" rx="7"/>')
    body += [sizes, baselines, annot]

    y = y + max(h1 + 16, h2 + h3 + 34) + 24
    if clf:
        pc, h4 = panel_per_class(PAD, y, W - 2 * PAD, clf)
        body.append(f'<rect class="card" x="{PAD}" y="{y - 8}" width="{W - 2 * PAD}" '
                    f'height="{h4 + 16}" rx="7"/>')
        body.append(pc)
        y += h4 + 16 + 24
        cm, h5 = panel_confusion(PAD, y, W - 2 * PAD, clf, label_of)
        body.append(f'<rect class="card" x="{PAD}" y="{y - 8}" width="{W - 2 * PAD}" '
                    f'height="{h5 + 16}" rx="7"/>')
        body.append(cm)
        y += h5 + 16

    footer_top = y + 40
    head = [
        text(PAD, 38, "complaint-taxonomy · measured results", "h1"),
        text(PAD, 58, f"{run_dir.name} · clustering (S1) · annotation (S5) · classifier (S6) · "
                      f"generated {clusters.get('created_utc', 'unknown')} by "
                      "tools/make_figures.py", "note"),
        hline(PAD, W - PAD, 74),
    ]
    sources = wrap_tokens([
        f"{run_dir}/clusters.json",
        f"{run_dir}/codebook.json",
        args.labels,
        args.merged,
        args.clf,
    ])
    height = footer_top + 30 + 14 * len(sources) + 10
    foot = [
        hline(PAD, W - PAD, footer_top),
        text(PAD, footer_top + 16,
             "every number is read from out/ · regenerating after a run is one command, so this "
             "page cannot disagree with the artefacts", "note"),
        text(PAD, footer_top + 32, "sources: " + sources[0], "mono note", size=9.5),
    ]
    for i, line in enumerate(sources[1:], start=1):
        foot.append(text(PAD, footer_top + 32 + i * 14, "         " + line, "mono note", size=9.5))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height:.0f}" '
           f'viewBox="0 0 {W} {height:.0f}" role="img" aria-labelledby="t d">\n'
           '<title id="t">Measured results of one complaint-taxonomy run</title>\n'
           '<desc id="d">Cluster sizes, annotation abstention, the classifier against its '
           'baselines, per-class precision/recall/F1 and the out-of-fold confusion matrix.</desc>\n'
           f"<style>{STYLE}</style>\n"
           f'<rect class="bg" width="{W}" height="{height:.0f}"/>\n'
           + "".join(head) + "\n" + "".join(body) + "\n" + "".join(foot) + "\n</svg>\n")
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_text(svg, encoding="utf-8")
    print(f"wrote {base}  ({W}x{height:.0f}, {len(svg) / 1024:.0f} KB)")
    if args.html:
        page = base.with_suffix(".html")
        page.write_text(HTML_PAGE.format(title="complaint-taxonomy · measured results", svg=svg),
                        encoding="utf-8")
        print(f"wrote {page}")
    return svg


def kpi_cards_no_clf(run, ann) -> str:
    """Two cards when the classifier has not been trained yet."""
    n_unique = run["clusters"]["n_unique_sentences"]
    cover = run["codebook"]["covered"]
    cards = [
        ("S1 · cluster", f"k={run['clusters']['k']}",
         [f"{run['clusters']['n_sentences']:,} sentence labels · {n_unique:,} unique",
          f"covered {cover:,} ({100.0 * cover / n_unique:.0f}%) by the codebook"]),
        ("S5 · annotate", f"{100.0 * ann['abstain'] / ann['n']:.0f}% abstain",
         [f"{ann['n']} items · {ann['abstain']} abstain",
          f"agreement with cluster {ann['agree_with_cluster']:.2f}"]),
    ]
    w = (W - 2 * PAD - 16) // 2
    out = []
    for i, (head, big, lines) in enumerate(cards):
        x = PAD + i * (w + 16)
        out.append(f'<rect class="card" x="{x}" y="84" width="{w}" height="106" rx="7"/>')
        out.append(text(x + 14, 108, head, "mono note"))
        out.append(text(x + 14, 140, big, "big"))
        for j, line in enumerate(lines):
            out.append(text(x + 14, 160 + j * 16, line, "note"))
    return "".join(out)


HTML_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  html, body {{ margin: 0; padding: 0; background: #fff; }}
  svg {{ display: block; width: 100%; height: auto; }}
  @media (prefers-color-scheme: dark) {{ html, body {{ background: #0d1117; }} }}
</style></head>
<body>
{svg}
</body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="out/zh_final", help="run directory with clusters/codebook")
    ap.add_argument("--labels", default="out/labels_zh/summary.json",
                    help="annotation summary for the run's own codebook")
    ap.add_argument("--merged", default="out/labels_merged/summary.json",
                    help="annotation summary for the merged codebook, for comparison")
    ap.add_argument("--clf", default="out/clf_zh/report.json", help="classifier report.json")
    ap.add_argument("--out", default="docs/assets/results.svg", help="SVG to write")
    ap.add_argument("--html", action="store_true", help="also write a page that embeds the SVG")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
