#!/usr/bin/env python3
"""R3: name every cluster and emit a taxonomy draft with its evidence.

The output is a codebook draft, not a verdict. Each entry carries the five fields the
design doc requires -- label, definition, includes, boundary, counterexample -- plus
the samples the model was shown and, when ground truth is supplied, the cluster's
dominant true label and its purity. That way a reviewer can check the claim against
the evidence instead of trusting a name.

    uv run python tools/name_clusters.py --run out/zh_k10 \
        --llm-endpoint http://127.0.0.1:8000/v1/chat/completions \
        --labels data/public/abcd_500_labels.jsonl \
        --label-map data/abcd_flow_coarse.json

Naming is slow (one call per cluster) and deterministic-ish, so `--resume` keeps the
entries already named in an existing taxonomy.json and only fills the gaps.

    # validate an existing taxonomy without calling the model
    uv run python tools/name_clusters.py --run out/zh_k10 --check

Exit code is 0 when every cluster has a label plus a boundary and counterexample,
1 when some entry still needs review -- so it can gate a release.
"""
import argparse
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import llm_naming  # noqa: E402


def read_jsonl(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                o = json.loads(line)
                out[o["id"]] = o
    return out


def primary_of(conv_labels):
    out = {}
    for cid, o in read_jsonl(conv_labels).items():
        dom = o.get("primary")
        if dom is None:
            cl = o.get("labels") or []
            if not cl:
                continue
            dom = max(cl, key=lambda d: (d["n"], -d["cluster"]))["cluster"]
        out[cid] = int(dom)
    return out


def truth_context(run, labels_path, label_field, label_map):
    """Per-cluster dominant true label, used as evidence and as prompt context."""
    cl_path = os.path.join(run, "conv_labels.jsonl")
    if not labels_path or not os.path.exists(cl_path):
        return {}, {}
    truth = {}
    for cid, o in read_jsonl(labels_path).items():
        lab = o.get(label_field)
        if lab is not None:
            truth[cid] = lab
    pred = primary_of(cl_path)
    per = {}
    for cid, c in pred.items():
        if cid in truth:
            per.setdefault(c, Counter())[truth[cid]] += 1
    ev, ctx = {}, {}
    for c, cnt in per.items():
        total = sum(cnt.values())
        top, n = cnt.most_common(1)[0]
        coarse = label_map.get(top, top) if label_map else top
        ev[c] = {"dominant_label": top, "label_purity": round(n / total, 3),
                 "coarse_label": coarse, "n_with_truth": total}
        ctx[c] = f"这一簇的对话里，真实类别最多的是 {coarse}（占 {n / total:.0%}）"
    return ev, ctx


def check(taxonomy_path):
    with open(taxonomy_path, encoding="utf-8") as f:
        tax = json.load(f)
    entries = tax.get("clusters") or []
    bad = []
    for e in entries:
        problems = llm_naming.validate(e)
        if not e.get("_ok", True):
            problems.append(f"model call failed: {e.get('_error', '?')}")
        if problems:
            bad.append((e.get("cluster"), e.get("label") or "(unnamed)", problems))
    print(f"{taxonomy_path}: {len(entries)} entries, {len(entries) - len(bad)} pass")
    for cl, label, problems in bad:
        print(f"  #{cl!s:<4} {label:<20} {'; '.join(problems)}")
    return 0 if entries and not bad else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="", help="directory holding clusters.json")
    ap.add_argument("--clusters", default="", help="clusters.json path (overrides --run)")
    ap.add_argument("--llm-endpoint", default="",
                    help="e.g. http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--llm-model", default="qwen38-27b-sglang")
    ap.add_argument("--labels", default="", help="ground truth: adds evidence + context")
    ap.add_argument("--label-field", default="label")
    ap.add_argument("--label-map", default="", help="json {fine: coarse}")
    ap.add_argument("--out", default="", help="default <run>/taxonomy.json")
    ap.add_argument("--context-from-labels", action="store_true",
                    help="put the cluster's dominant true label into the prompt; off "
                         "by default because it anchors the name to the answer key")
    ap.add_argument("--resume", action="store_true",
                    help="keep entries already named in an existing taxonomy.json")
    ap.add_argument("--check", action="store_true",
                    help="validate an existing taxonomy.json and exit")
    ap.add_argument("--no-update-clusters", action="store_true",
                    help="do not add the taxonomy_file pointer to clusters.json")
    args = ap.parse_args()

    clusters_path = args.clusters or os.path.join(args.run, "clusters.json")
    if not clusters_path or not os.path.exists(clusters_path):
        sys.exit(f"[fatal] clusters.json not found: {clusters_path!r}")
    run = args.run or os.path.dirname(clusters_path)
    out_path = args.out or os.path.join(run, "taxonomy.json")

    if args.check:
        sys.exit(check(out_path))

    if not args.llm_endpoint:
        sys.exit("[fatal] --llm-endpoint is required unless --check")

    with open(clusters_path, encoding="utf-8") as f:
        payload = json.load(f)
    clusters = payload.get("clusters") or []
    if not clusters:
        sys.exit("[fatal] no clusters in clusters.json")

    label_map = {}
    if args.label_map:
        with open(args.label_map, encoding="utf-8") as f:
            label_map = json.load(f)
    evidence, context = truth_context(run, args.labels, args.label_field, label_map)
    if context and not args.context_from_labels:
        # Labels are evidence for the reviewer, not input to the namer. Naming with
        # the answer key in the prompt produces a codebook that agrees with the
        # given taxonomy by construction, which then cannot be used to check what
        # the clustering found.
        context = {}

    done = {}
    if args.resume and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for e in json.load(f).get("clusters") or []:
                if e.get("_ok") and e.get("label"):
                    done[int(e["cluster"])] = e
        print(f"[R3] resuming: {len(done)} clusters already named")

    corpus_lang = llm_naming.sniff_lang(
        [t for c in clusters for t in (c.get("samples") or [])])
    print(f"[R3] corpus language: {corpus_lang} (one decision for the whole run, "
          f"not per cluster)")
    print(f"[R3] naming {len(clusters)} clusters from {clusters_path} "
          f"({payload.get('kmeans_algo', '?')} k={payload.get('k')}, "
          f"model={payload.get('embed_model', '?')})")
    entries = []
    for c in clusters:
        cl = int(c["cluster"])
        if cl in done:
            entries.append(done[cl])
            continue
        entry = llm_naming.name_cluster(args.llm_endpoint, args.llm_model,
                                        c.get("samples") or [],
                                        c.get("top_terms") or [],
                                        lang=corpus_lang,
                                        context=context.get(cl))
        entry["cluster"] = cl
        entry["size"] = int(c.get("size") or 0)
        entry["top_terms"] = list(c.get("top_terms") or [])[:8]
        entry["evidence"] = {
            "n_samples": len(c.get("samples") or []),
            "samples": (c.get("samples") or [])[:5],
            **evidence.get(cl, {}),
        }
        problems = llm_naming.validate(entry)
        # A Chinese corpus whose codebook entry comes back in English is a real
        # defect in the artefact even though the shape is fine -- annotators read
        # this, and mixed-language categories leak into the training labels.
        label = entry.get("label") or ""
        if corpus_lang == "zh" and label and not llm_naming._CJK.search(label):
            entry["_warn_language"] = True
            problems.append("label is not in the corpus language (zh)")
        flag = "ok" if not problems and entry.get("_ok") else "REVIEW"
        ev = entry.get("evidence") or {}
        dom = ev.get("dominant_label") or "-"
        pur = f"{ev['label_purity']:.0%}" if ev.get("label_purity") else "-"
        print(f"  #{cl:<3d} {entry.get('label') or '(failed)':<20} "
              f"size={entry['size']:<5} dom={dom:<18} pur={pur:<5} {flag}")
        entries.append(entry)

    taxonomy = {
        "k": payload.get("k"),
        "created_utc": payload.get("created_utc"),
        "llm_model": args.llm_model,
        "source_clusters": os.path.abspath(clusters_path),
        "kmeans_algo": payload.get("kmeans_algo"),
        "embed_model": payload.get("embed_model"),
        "clusters": entries,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, ensure_ascii=False, indent=2)

    if not args.no_update_clusters:
        payload["taxonomy_file"] = os.path.basename(out_path)
        with open(clusters_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    named = sum(1 for e in entries if e.get("label"))
    need = sum(1 for e in entries if llm_naming.validate(e) or not e.get("_ok", True))
    print(f"\n[R3] wrote {out_path}: {named}/{len(entries)} named, "
          f"{need} need review")
    if evidence:
        print(f"[R3] {len(evidence)}/{len(entries)} clusters carry a dominant true "
              f"label (shown as dom/pur above). A low purity means the cluster mixes "
              f"categories -- that is a finding about the clustering, not about the "
              f"name, and it belongs in the review rather than in the label.")
    sys.exit(1 if need else 0)


if __name__ == "__main__":
    main()
