#!/usr/bin/env python3
"""R4a: turn a taxonomy into a codebook a human annotator can actually work from.

The design asks for six fields per class: name, definition, inclusion, boundary,
precedence, examples. Five of them come from the LLM naming pass. Precedence does not,
and pretending it does would be the classic codebook failure -- two classes that both
"cover" a case, with nothing saying which wins, guarantee disagreement later. So this
tool emits a precedence *slot* for every class, and fills in the candidate pairs a human
must decide, ranked by how similar the classes look in embedding space.

    uv run python tools/make_codebook.py --taxonomy out/zh_final/taxonomy.json \
        --out out/zh_final/codebook.md

Outputs
    codebook.md     one page per class + a precedence worksheet
    codebook.json   the same content, machine-readable
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def confusable_pairs(entries, top=6):
    """Pairs whose *definitions* sit closest in embedding space.

    Lexical overlap was tried first and produced pairs at 0.04-0.07 that ranked
    arbitrarily: Chinese definitions rarely share multi-character tokens, so Jaccard
    on words is noise. Embedding the label+definition and taking cosine similarity
    ranks by meaning instead, which is the thing that actually drives annotator
    confusion.
    """
    import numpy as np

    sys.path.insert(0, ROOT)
    from cluster_complaints import FastEmbedEmbedder, l2norm

    texts = [f"{e.get('label') or ''}。{e.get('definition') or ''}" for e in entries]
    emb = FastEmbedEmbedder(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        cache_dir="~/.cache/fastembed")
    V = l2norm(np.asarray(emb(texts), dtype=np.float32))
    S = V @ V.T
    scored = []
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            scored.append((float(S[i, j]), entries[i], entries[j]))
    scored.sort(key=lambda t: -t[0])
    return scored[:top]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--out", default="", help="default <taxonomy dir>/codebook.md")
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()

    with open(args.taxonomy, encoding="utf-8") as f:
        tax = json.load(f)
    entries = [e for e in tax.get("clusters", []) if e.get("label")]
    if not entries:
        sys.exit("[fatal] taxonomy has no labelled entries")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.taxonomy)),
                                   "codebook.md")

    algo = tax.get("kmeans_algo")
    if not algo:
        sib = os.path.join(os.path.dirname(os.path.abspath(args.taxonomy)),
                           "clusters.json")
        if os.path.exists(sib):
            with open(sib, encoding="utf-8") as f:
                algo = json.load(f).get("kmeans_algo")

    total = sum(int(e.get("size") or 0) for e in entries) or 1
    lines = [
        "# 投诉类别 codebook",
        "",
        f"来源：`{os.path.basename(args.taxonomy)}`，{len(entries)} 个类别，"
        f"覆盖 {total} 段对话。",
        f"生成模型：{tax.get('llm_model', 'n/a')}；聚类算法：{algo or 'n/a'}。",
        "",
        "> 这是一份草稿。`定义 / 包含 / 边界 / 例子` 由模型起草，`优先级` 必须人工填写——",
        "> 两个类别都能套上同一段对话时，谁赢必须写死在 codebook 里，否则标注一定分歧。",
        "",
        "## 类别速查",
        "",
        "| # | 类别 | 规模 | 占比 |",
        "|---|---|---|---|",
    ]
    for e in sorted(entries, key=lambda x: -int(x.get("size") or 0)):
        share = int(e.get("size") or 0) / total
        lines.append(f"| {e['cluster']} | {e['label']} | {e.get('size', 0)} | {share:.1%} |")

    lines += ["", "---", ""]
    for e in sorted(entries, key=lambda x: int(x["cluster"])):
        ev = e.get("evidence") or {}
        lines.append(f"## {e['label']}")
        lines.append("")
        lines.append(f"- 簇：`{e['cluster']}`｜样本量：{e.get('size', 0)} 段"
                     + (f"｜主导真值：`{ev.get('dominant_label')}` "
                        f"({ev.get('label_purity', 0):.0%})"
                        if ev.get("dominant_label") else ""))
        lines.append(f"- **定义**：{e.get('definition') or '（缺）'}")
        inc = e.get("includes") or []
        lines.append("- **包含**：" + ("；".join(inc) if inc else "（缺）"))
        lines.append(f"- **边界**：{e.get('boundary') or '（缺）'}")
        lines.append(f"- **反例**：{e.get('counterexample') or '（缺）'}")
        lines.append("- **优先级**：_待人工填写——与相邻类别冲突时谁赢_")
        terms = e.get("top_terms") or []
        if terms:
            lines.append(f"- 高频词：{'、'.join(terms[:6])}")
        samples = (e.get("evidence") or {}).get("samples") or []
        if samples:
            lines.append("- 例子：")
            for s in samples[: args.examples]:
                lines.append(f"  - {s[:180]}")
        lines.append("")

    pairs = confusable_pairs(entries)
    if pairs:
        lines += ["---", "", "## 优先级工作表（需人工决定）", "",
                  "按定义向量的余弦相似度排序。每组都要在 codebook 里写一条规则，",
                  "说明同时满足时的判定顺序，否则标注阶段的分歧会集中在这些组上。", "",
                  "| 定义相似度 | 类别 A | 类别 B | 优先级决定 |",
                  "|---|---|---|---|"]
        for overlap, a, b in pairs:
            lines.append(f"| {overlap:.2f} | {a['label']} | {b['label']} | _待填_ |")
        lines.append("")

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    payload = {"source_taxonomy": os.path.abspath(args.taxonomy),
               "n_classes": len(entries), "covered": total,
               "classes": [{k: e.get(k) for k in
                            ("cluster", "label", "size", "definition", "includes",
                             "boundary", "counterexample", "top_terms")} |
                           {"precedence": None} for e in entries],
               "precedence_worksheet": [
                   {"similarity": round(o, 3), "a": a["label"], "b": b["label"],
                    "decision": None} for o, a, b in pairs]}
    # The sidecar must never overwrite the taxonomy it was built from. Calling
    # `--out out/codebook_merged.md` on `out/codebook_merged.json` would otherwise
    # replace the input with the codebook and the next tool would find no classes.
    sidecar = os.path.splitext(out)[0] + ".json"
    if os.path.abspath(sidecar) == os.path.abspath(args.taxonomy):
        sidecar = os.path.splitext(out)[0] + ".codebook.json"
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    missing = sum(1 for e in entries
                  if not (e.get("definition") and e.get("boundary")
                          and e.get("counterexample")))
    print(f"wrote {out}: {len(entries)} classes, {missing} incomplete, "
          f"{len(pairs)} precedence pairs to decide")
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
