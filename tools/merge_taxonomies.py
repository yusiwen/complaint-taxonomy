#!/usr/bin/env python3
"""Combine several taxonomies into one codebook.

A codebook grows: one clustering run finds the complaint topics, another run on a
differently filtered corpus finds the service requests that the filter removed, and the
deliverable is the union of both, minus the classes nobody wants to see in a taxonomy
("user gives their name", "conversation ends").

Measured reason this exists: filtering utterances down to complaints, then naming the
clusters, produced a codebook that refused to label 46% of the corpus — and the
abstentions were concentrated in shipping (68%) and order questions (66%) because those
utterances are *requests*, not complaints, so the filter had deleted them before any
cluster could represent them. The fix was not a better filter; it was a codebook that
covers requests too. That means merging the two runs' taxonomies.

    uv run python tools/merge_taxonomies.py \
        --taxonomy out/zh_final/taxonomy.json \
        --taxonomy out/cov_k20_raw/taxonomy.json \
        --drop-labels "对话结束确认,用户姓名或ID,服务结束致谢" \
        --sim-threshold 0.85 \
        --out out/codebook_merged.json

Rules, in order:
  * `--drop-labels` removes exact labels (CSV or repeated). Curation stays explicit —
    a regex over Chinese category names silently eats the classes you meant to keep
    (the cue/credential rules already taught that lesson).
  * near-duplicates are merged by embedding similarity of label+definition, keeping the
    first taxonomy that supplied them, so `--taxonomy yours --taxonomy theirs` decides
    precedence.
  * everything else is kept, with its origin recorded so a reviewer can trace a class
    back to the run and cluster it came from.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cluster_complaints import FastEmbedEmbedder, l2norm  # noqa: E402


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", action="append", required=True,
                    help="repeatable; earlier files win on duplicate/near-duplicate")
    ap.add_argument("--drop-labels", default="",
                    help="comma-separated exact labels to exclude")
    ap.add_argument("--sim-threshold", type=float, default=0.85,
                    help="merge two classes above this cosine similarity")
    ap.add_argument("--out", required=True)
    ap.add_argument("--embed-model",
                    default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    args = ap.parse_args()

    drop = {s.strip() for s in args.drop_labels.split(",") if s.strip()}
    entries, dropped, merged_similar = [], [], []

    for path in args.taxonomy:
        tax = load(path)
        for e in tax.get("clusters") or []:
            label = (e.get("label") or "").strip()
            if not label:
                continue
            if label in drop:
                dropped.append((label, "in --drop-labels"))
                continue
            entries.append({**e, "label": label,
                            "origin": {"taxonomy": os.path.abspath(path),
                                       "cluster": e.get("cluster")}})

    # exact duplicates first
    seen = {}
    unique = []
    for e in entries:
        if e["label"] in seen:
            merged_similar.append((e["label"], seen[e["label"]], "same label"))
            continue
        seen[e["label"]] = e["label"]
        unique.append(e)

    # then near-duplicates by definition embedding
    if len(unique) > 1 and args.sim_threshold < 1.0:
        emb = FastEmbedEmbedder(args.embed_model, cache_dir="~/.cache/fastembed")
        texts = [f"{e['label']}。{e.get('definition') or ''}" for e in unique]
        V = l2norm(np.asarray(emb(texts), dtype=np.float32))
        S = V @ V.T
        keep = []
        for i, e in enumerate(unique):
            clash = None
            for j in keep:
                if S[i, j] >= args.sim_threshold:
                    clash = j
                    break
            if clash is None:
                keep.append(i)
            else:
                merged_similar.append((e["label"], unique[clash]["label"],
                                       f"cos {S[i, clash]:.3f}"))
        unique = [unique[i] for i in keep]

    for idx, e in enumerate(unique):
        e["cluster"] = idx
        e["source_cluster"] = e.pop("origin", {}).get("cluster")
    models, algos = [], []
    for path in args.taxonomy:
        tax = load(path)
        if tax.get("llm_model"):
            models.append(tax["llm_model"])
        if tax.get("kmeans_algo"):
            algos.append(str(tax["kmeans_algo"]))

    out = {"k": len(unique),
           "llm_model": models[0] if len(set(models)) == 1 else ("mixed" if models else "n/a"),
           "kmeans_algo": algos[0] if len(set(algos)) == 1 else ("mixed" if algos else "n/a"),
           "n_sources": len(args.taxonomy),
           "sources": [os.path.abspath(p) for p in args.taxonomy],
           "merged_classes": len(unique),
           "dropped": [{"label": lab, "why": w} for lab, w in dropped],
           "near_duplicates_merged": [{"label": lab, "into": t, "why": w}
                                      for lab, t, w in merged_similar],
           "clusters": unique}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"merged {len(args.taxonomy)} taxonomies -> {len(unique)} classes "
          f"({len(out['dropped'])} dropped, {len(out['near_duplicates_merged'])} "
          f"duplicates) -> {args.out}")
    for d in out["dropped"]:
        print(f"  dropped  {d['label']} ({d['why']})")
    for d in out["near_duplicates_merged"]:
        print(f"  merged   {d['label']} -> {d['into']} ({d['why']})")
    print("\nclasses:")
    for e in unique:
        print(f"  {e['cluster']:>2}  {e['label']}")


if __name__ == "__main__":
    main()
