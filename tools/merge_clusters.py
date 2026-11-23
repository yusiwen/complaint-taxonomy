#!/usr/bin/env python3
"""Merge a fine clustering into a coarser codebook, bottom-up.

Why bottom-up: measured on this pipeline, clustering *coarsely* is worse than
clustering finely and merging. On the 500-dialogue benchmark a top-down root at
k=4 reached ARI 0.136 against the 4-group truth, while a k=30 partition merged
down to 4 groups reached 0.364. Splitting top-down inherits every mistake the root
makes and starves small branches; merging upward keeps the fine structure and only
has to decide which neighbours belong together.

The merge itself uses average linkage over cluster centroids (document-level
embeddings of each conversation), so it needs no labels. Optional `--label-*`
flags score the result, and `--llm-endpoint` drafts a codebook entry per group.

    uv run python tools/merge_clusters.py \
        --input data/public/abcd_500_zh.jsonl \
        --conv-labels out/ksweep_zh/k20/conv_labels.jsonl \
        --into 4 --out out/merge_zh_k20 \
        --labels data/public/abcd_500_labels.jsonl \
        --label-map data/abcd_flow_coarse.json

Outputs
    merge_map.json   {"fine_cluster": "group_id", ...}
    groups.json      per group: size, members, top terms, samples, dominant truth
    groups.md        readable codebook draft
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm_naming  # noqa: E402
from cluster_complaints import (  # noqa: E402
    FastEmbedEmbedder,
    bisect_kmeans,
    l2norm,
)

STOP = {
    "the", "and", "you", "your", "for", "that", "have", "with", "this", "but",
    "can", "will", "was", "are", "not", "from", "they", "our", "would", "could",
    "how", "what", "when", "get", "got", "just", "want", "need", "like", "help",
    "thanks", "thank", "know", "also", "been", "very", "about", "there",
    "的", "了", "是", "在", "我", "你", "他", "她", "它", "们", "这", "那",
    "有", "个", "吗", "呢", "吧", "啊", "就", "和", "要", "会", "不", "能",
}


def read_jsonl(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            o = json.loads(line)
            out[o["id"]] = o
    return out


def user_text(o):
    return " ".join(m["content"] for m in o["messages"] if m["role"] == "user")


def tokens(text):
    out = []
    for w in re.findall(r"[a-zA-Z][a-zA-Z']+", text.lower()):
        if len(w) >= 3 and w not in STOP:
            out.append(w)
    zh = re.findall(r"[\u4e00-\u9fff]+", text)
    for run in zh:
        for ch in run:
            if ch not in STOP:
                out.append(ch)
        for i in range(len(run) - 1):
            bigram = run[i:i + 2]
            if not any(c in STOP for c in bigram):
                out.append(bigram)
    return out


def top_terms(texts, n=8):
    c = Counter()
    for t in texts:
        c.update(set(tokens(t)))
    return [w for w, _ in c.most_common(n)]


def primary_of(conv_labels):
    out = {}
    for cid, o in read_jsonl(conv_labels).items():
        dom = o.get("primary")
        if dom is None:
            cl = o.get("labels") or []
            if not cl:
                continue
            dom = max(cl, key=lambda d: (d["n"], -d["cluster"]))["cluster"]
        out[cid] = dom
    return out


def merge_by_leaders(weights, m):
    """Cluster the fine cluster centroids themselves into m groups ("leaders").

    Plain average linkage chains badly here: when every centroid is a
    document-mean vector of the same domain, all pairwise similarities sit in a
    narrow band, so the greedy max-similarity merge grows one blob and swallows
    almost everything (measured: 13 of 16 clusters merged into a single group,
    ARI 0.000). Clustering the centroids with bisecting k-means cannot chain --
    every centroid lands in exactly one of m groups by construction -- and it is
    deterministic, so the merged codebook is reproducible.
    """
    import numpy as np
    C = l2norm(np.asarray(weights, dtype=np.float32))
    labels, _, _ = bisect_kmeans(C, m)
    return {g: [i for i in range(len(labels)) if labels[i] == g] for g in set(labels)}


def agglomerate(centroids, m, cos):
    """Average-linkage merge down to m groups (kept for comparison)."""
    groups = {i: [i] for i in range(len(centroids))}

    def sim(a, b):
        vals = [cos(centroids[x], centroids[y]) for x in groups[a] for y in groups[b]]
        return sum(vals) / len(vals)

    while len(groups) > m:
        ids = sorted(groups)
        best = None
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                key = (sim(ids[i], ids[j]), -ids[i], -ids[j])
                if best is None or key > best[0]:
                    best = (key, ids[i], ids[j])
        _, a, b = best
        groups[a] = groups[a] + groups.pop(b)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="same corpus the run used")
    ap.add_argument("--conv-labels", required=True, help="out/<run>/conv_labels.jsonl")
    ap.add_argument("--into", type=int, required=True, help="target number of groups")
    ap.add_argument("--out", default="out/merge")
    ap.add_argument("--embed-model",
                    default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--labels", default="", help="ground truth, for scoring only")
    ap.add_argument("--label-field", default="label")
    ap.add_argument("--label-map", default="", help="fine -> coarse label map")
    ap.add_argument("--linkage", choices=["leaders", "avg"], default="leaders",
                    help="leaders: cluster the fine centroids (deterministic, cannot chain); "
                         "avg: average linkage (chains badly on this kind of data)")
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--llm-endpoint", default="",
                    help="e.g. http://127.0.0.1:8000/v1/chat/completions (optional)")
    ap.add_argument("--llm-model", default="",
                    help="model name your endpoint expects; "
                         "empty is fine for most servers")
    args = ap.parse_args()

    convs = read_jsonl(args.input)
    pred = primary_of(args.conv_labels)
    ids = [cid for cid in pred if cid in convs]
    if not ids:
        raise SystemExit("conv-labels and input do not overlap")

    emb = FastEmbedEmbedder(args.embed_model, cache_dir="~/.cache/fastembed")
    vecs = l2norm(emb([user_text(convs[cid]) for cid in ids]))
    cluster_of = {cid: pred[cid] for cid in ids}
    fine = sorted(set(cluster_of.values()))

    members = {c: [i for i, cid in enumerate(ids) if cluster_of[cid] == c] for c in fine}
    centroids = {c: l2norm(vecs[members[c]].mean(0, keepdims=True))[0] for c in fine}

    def cos(a, b):
        return float(a @ b)

    ordered = [c for c in fine if len(members[c])]
    idx_centroids = {i: centroids[c] for i, c in enumerate(ordered)}
    m = min(args.into, len(ordered))
    if args.linkage == "leaders":
        groups = merge_by_leaders([idx_centroids[i] for i in range(len(ordered))], m)
    else:
        groups = agglomerate(idx_centroids, m, cos)

    # stable group ids: by merged size, biggest first
    def group_docs(g):
        return [i for c in groups[g] for i in members[ordered[c]]]

    order = sorted(groups, key=lambda g: (-len(group_docs(g)), g))
    gid = {g: f"G{n}" for n, g in enumerate(order)}

    truth, coarse = {}, {}
    lmap = {}
    if args.label_map:
        lmap = json.load(open(args.label_map, encoding="utf-8"))
    if args.labels:
        for cid, o in read_jsonl(args.labels).items():
            lab = o.get(args.label_field)
            if lab is None:
                continue
            truth[cid] = lab
            coarse[cid] = lmap.get(lab, "_other") if lmap else lab

    os.makedirs(args.out, exist_ok=True)
    merge_map = {str(ordered[c]): gid[g] for g in groups for c in groups[g]}
    with open(os.path.join(args.out, "merge_map.json"), "w", encoding="utf-8") as f:
        json.dump(merge_map, f, ensure_ascii=False, indent=2)

    out_groups = []
    for g in order:
        docs = group_docs(g)
        cids = [ids[i] for i in docs]
        texts = [user_text(convs[c]) for c in cids]
        dom = Counter(truth[c] for c in cids if c in truth)
        domcoarse = Counter(coarse[c] for c in cids if c in coarse)
        rec = {
            "group": gid[g],
            "size": len(cids),
            "fine_clusters": [ordered[c] for c in groups[g]],
            "top_terms": top_terms(texts),
            "samples": texts[: args.samples],
            "dominant_fine_label": dom.most_common(1)[0][0] if dom else None,
            "fine_consistency": round(dom.most_common(1)[0][1] / len(cids), 3) if dom else None,
            "dominant_coarse_label": domcoarse.most_common(1)[0][0] if domcoarse else None,
            "coarse_consistency": (round(domcoarse.most_common(1)[0][1] / len(cids), 3)
                                   if domcoarse else None),
        }
        if args.llm_endpoint:
            try:
                rec["codebook"] = llm_naming.name_cluster(
                    args.llm_endpoint, args.llm_model, rec["samples"],
                    rec["top_terms"])
            except Exception as exc:                              # noqa: BLE001
                rec["codebook"] = {"_ok": False, "_error": str(exc)[:120]}
        out_groups.append(rec)

    with open(os.path.join(args.out, "groups.json"), "w", encoding="utf-8") as f:
        json.dump({"into": args.into, "n_fine_clusters": len(ordered),
                   "groups": out_groups}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.out, "groups.md"), "w", encoding="utf-8") as f:
        f.write(f"# Merged codebook draft ({len(ordered)} fine clusters -> "
                f"{len(out_groups)} groups)\n\n")
        for rec in out_groups:
            cb = rec.get("codebook") or {}
            f.write(f"## {rec['group']} — {cb.get('label') or '(unnamed)'}  "
                    f"({rec['size']} 段)\n\n")
            if cb.get("definition"):
                f.write(f"{cb['definition']}\n\n")
            f.write(f"- 成员细簇：{rec['fine_clusters']}\n")
            f.write(f"- 高频词：{', '.join(rec['top_terms'][:8])}\n")
            if rec["dominant_fine_label"]:
                f.write(f"- 真值最多：`{rec['dominant_fine_label']}` "
                        f"占 {rec['fine_consistency']:.0%}\n")
            f.write("\n样本：\n\n")
            for s in rec["samples"][:5]:
                f.write(f"> {s[:160]}\n\n")

    print(f"{len(ordered)} fine clusters -> {len(out_groups)} groups "
          f"({args.out}/groups.json)")

    if truth:
        from eval_against_labels import ari
        merged = {cid: merge_map[str(cluster_of[cid])] for cid in ids if cid in truth}
        pairs_f = [(cluster_of[c], truth[c]) for c in ids if c in truth]
        pairs_mf = [(merged[c], truth[c]) for c in merged]
        print(f"  fine partition   vs fine truth   ARI {ari(pairs_f):.3f}")
        print(f"  merged groups    vs fine truth   ARI {ari(pairs_mf):.3f}")
        if lmap:
            pairs_mc = [(merged[c], coarse[c]) for c in merged]
            pairs_c = [(cluster_of[c], coarse[c]) for c in merged]
            print(f"  fine partition   vs coarse truth ARI {ari(pairs_c):.3f}")
            print(f"  merged groups    vs coarse truth ARI {ari(pairs_mc):.3f}")


if __name__ == "__main__":
    main()
