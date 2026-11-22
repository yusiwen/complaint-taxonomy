#!/usr/bin/env python3
"""R4: label conversations by applying the taxonomy *definitions*, not cluster ids.

The difference matters. Labelling with cluster ids ties every label to one run of one
clustering: retrain, change k, and the ids mean something else. Labelling with the
definitions from `taxonomy.json` produces labels that survive a re-clustering, because
the model is asked "does this conversation match this definition", not "which cluster
would this have landed in". That is also what makes the output reviewable — a label can
be argued with, a cluster id cannot.

    uv run python tools/make_labels.py \
        --input data/public/abcd_500_zh.jsonl \
        --taxonomy out/zh_k10/taxonomy.json \
        --llm-endpoint http://127.0.0.1:8000/v1/chat/completions \
        --limit 500 --out out/labels_zh

Outputs (under --out)
    labels.jsonl   {"id", "labels": [...], "primary", "confidence", "valid"}
    summary.json   distribution, abstain/invalid rate, agreement with the clustering

Agreement is measured two ways and both are reported, because they answer different
questions:
  * against the cluster the conversation was in (`agree_with_cluster`) — a sanity check;
    large disagreement means either the definitions or the clustering is wrong, and the
    per-conversation rows say which;
  * against ground truth, mapped back through cluster -> dominant true label
    (`ari_vs_truth`) when `--truth-labels` is given. This is only as good as that
    mapping, which is why it is reported next to the raw row count.
"""
import argparse
import json
import os
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import llm_naming  # noqa: E402

_zh_rules = (
    "你是客服投诉标注员。下面是已经定义好的投诉类别，每一个都给了定义、包含情形和边界。\n"
    "请判断给定的这段客服对话属于哪些类别。\n"
    "规则：\n"
    "1. 只能使用下面列出的类别名，逐字照抄，不要自己造类别；\n"
    "2. 只根据用户说的话判断，客服的话只是上下文；\n"
    "3. 一段对话可以有多个投诉点，给 1-3 个类别，最多 3 个；\n"
    "4. primary 是最主要的那一个；\n"
    "5. 如果这段对话完全没有投诉内容（全是确认、致谢、给姓名/账号），"
    'primary 给 null，labels 给空数组；\n'
    "6. confidence 只能是 high / medium / low。\n"
    "严格输出 JSON，不要解释、不要加代码块："
    '{"labels": ["类别名"], "primary": "类别名或null", "confidence": "high"}'
)
_en_rules = (
    "You are a complaint annotation assistant. Below are defined complaint categories, "
    "each with a definition, inclusions and a boundary.\n"
    "Decide which categories this customer-service conversation belongs to.\n"
    "Rules:\n"
    "1. Use only the category names listed below, verbatim; invent nothing;\n"
    "2. judge from what the customer says; the agent turns are context only;\n"
    "3. a conversation may contain several complaints: give 1-3 labels, at most 3;\n"
    "4. primary is the main one;\n"
    "5. if there is no complaint content at all (only confirmations, thanks, "
    "name/account details), set primary to null and labels to [];\n"
    "6. confidence is high / medium / low.\n"
    "Output JSON only, no prose, no fences:"
    ' {"labels": ["Category name"], "primary": "Category name or null", '
    '"confidence": "high"}'
)

_lock = threading.Lock()
stats = Counter()


def read_jsonl(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                o = json.loads(line)
                out[o["id"]] = o
    return out


def category_block(taxonomy, lang):
    lines = []
    for e in taxonomy["clusters"]:
        if not e.get("label"):
            continue
        bits = [f"### {e['label']}"]
        if e.get("definition"):
            bits.append(f"定义：{e['definition']}" if lang == "zh"
                        else f"Definition: {e['definition']}")
        if e.get("includes"):
            joined = "；".join(e["includes"][:3]) if lang == "zh" else "; ".join(e["includes"][:3])
            bits.append(("包含：" if lang == "zh" else "Includes: ") + joined)
        if e.get("boundary"):
            bits.append(("边界：" if lang == "zh" else "Boundary: ") + e["boundary"])
        lines.append("\n".join(bits))
    return "\n\n".join(lines)


def conversation_text(o, max_chars=900):
    turns = [m["content"] for m in o.get("messages", []) if m.get("role") == "user"]
    text = " / ".join(t.strip() for t in turns if t.strip())
    return text[:max_chars]


def label_one(endpoint, model, rules, cats, allowed, text, timeout=180,
              temperature=0.0):
    prompt = (f"{rules}\n\n{cats}\n\n—— 对话（用户说的话）——\n{text}\n\n"
              "只输出 JSON。")
    for attempt in range(2):
        try:
            raw = llm_naming.chat(endpoint, model,
                                  [{"role": "user", "content": prompt}],
                                  max_tokens=300, timeout=timeout,
                                  temperature=temperature)
        except Exception as exc:                                   # noqa: BLE001
            if attempt == 1:
                return {"labels": [], "primary": None, "confidence": "low",
                        "valid": False, "error": f"request failed: {exc}"[:120]}
            continue
        obj = None
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                obj = None
        if obj is not None:
            labels = obj.get("labels") or []
            if isinstance(labels, str):
                labels = [labels]
            labels = [str(x).strip() for x in labels if str(x).strip()]
            prim = obj.get("primary")
            prim = str(prim).strip() if prim not in (None, "null", "") else None
            ok = all(x in allowed for x in labels) and (prim is None or prim in allowed)
            if ok:
                return {"labels": labels[:3], "primary": prim,
                        "confidence": str(obj.get("confidence") or "").lower(),
                        "valid": True}
            # one retry with the offending names called out
            prompt += (f"\n\n注意：上次输出用了不存在的类别名 "
                       f"{[x for x in labels if x not in allowed]}。"
                       f"只能用这些名字：{sorted(allowed)}")
            continue
        prompt += "\n\n注意：上次输出不是合法 JSON，请只输出 JSON。"
    return {"labels": [], "primary": None, "confidence": "low", "valid": False,
            "error": "could not get a valid label set"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--llm-endpoint", required=True)
    ap.add_argument("--llm-model", default="qwen38-27b-sglang")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--truth-labels", default="",
                    help="ground truth jsonl, for the ari_vs_truth diagnostic")
    ap.add_argument("--truth-field", default="label")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="nonzero gives a second, independent rater for the kappa "
                         "self-consistency check")
    ap.add_argument("--out", default="out/labels")
    args = ap.parse_args()

    with open(args.taxonomy, encoding="utf-8") as f:
        taxonomy = json.load(f)
    entries = [e for e in taxonomy["clusters"] if e.get("label")]
    if not entries:
        sys.exit("[fatal] taxonomy has no labelled clusters")
    allowed = {e["label"] for e in entries}
    lang = llm_naming.sniff_lang([e.get("definition", "") + e.get("label", "")
                                 for e in entries])
    rules = _zh_rules if lang == "zh" else _en_rules
    cats = category_block(taxonomy, lang)

    convs = read_jsonl(args.input)
    ids = list(convs)
    if args.limit:
        ids = ids[: args.limit]
    print(f"[R4] {len(ids)} conversations x {len(allowed)} categories ({lang})")

    def work(cid):
        rec = label_one(args.llm_endpoint, args.llm_model, rules, cats, allowed,
                        conversation_text(convs[cid]),
                        temperature=args.temperature)
        with _lock:
            stats["valid" if rec["valid"] else "invalid"] += 1
            if rec["valid"] and not rec["labels"]:
                stats["abstain"] += 1
        return cid, rec

    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (cid, rec) in enumerate(pool.map(work, ids), 1):
            results[cid] = rec
            if i % 50 == 0:
                print(f"  {i}/{len(ids)} done, stats={dict(stats)}")

    # Cluster ids come from the run the taxonomy was built on. `name_clusters.py`
    # records that path; the pipeline's inline `--name-clusters` writes taxonomy.json
    # into the run itself and does not, so fall back to the taxonomy's own directory.
    run = os.path.dirname(taxonomy.get("source_clusters") or "") or \
        os.path.dirname(os.path.abspath(args.taxonomy))
    cl_path = os.path.join(run, "conv_labels.jsonl")
    cluster_primary = {}
    if not os.path.exists(cl_path):
        print(f"[warn] no conv_labels.jsonl in {run!r}: the cluster-agreement and "
              f"truth-mapped checks cannot run", file=sys.stderr)
    if os.path.exists(cl_path):
        for cid, o in read_jsonl(cl_path).items():
            cluster_primary[cid] = o.get("primary")
    # cluster -> label, for the agreement check
    cl_to_label = {int(e["cluster"]): e["label"] for e in entries}

    agree = tot = 0
    for cid, rec in results.items():
        cl = cluster_primary.get(cid)
        if cl is None or not rec["valid"]:
            continue
        tot += 1
        if cl_to_label.get(int(cl)) == rec["primary"]:
            agree += 1

    truth, ari_vs_truth = {}, None
    if args.truth_labels:
        truth = {k: v[args.truth_field] for k, v in read_jsonl(args.truth_labels).items()
                 if args.truth_field in v}
        # map each taxonomy label back to the true label its cluster is dominated by
        label_to_truth = {}
        if os.path.exists(cl_path) and truth:
            per = {}
            for cid, cl in cluster_primary.items():
                if cid in truth:
                    per.setdefault(int(cl), Counter())[truth[cid]] += 1
            for e in entries:
                cnt = per.get(int(e["cluster"]))
                if cnt:
                    label_to_truth[e["label"]] = cnt.most_common(1)[0][0]
        pairs = [(label_to_truth.get(rec["primary"]), truth[cid])
                 for cid, rec in results.items()
                 if rec["valid"] and rec["primary"] in label_to_truth and cid in truth]
        if pairs:
            from eval_against_labels import ari
            ari_vs_truth = ari(pairs)
            stats["ari_pairs"] = len(pairs)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "labels.jsonl"), "w", encoding="utf-8") as f:
        for cid in ids:
            rec = results[cid]
            f.write(json.dumps({"id": cid, **rec}, ensure_ascii=False) + "\n")

    dist = Counter(rec["primary"] for rec in results.values()
                   if rec["valid"] and rec["primary"])
    summary = {
        "n": len(ids),
        "valid": stats["valid"],
        "invalid": stats["invalid"],
        "abstain": stats["abstain"],
        "agree_with_cluster": round(agree / tot, 4) if tot else None,
        "n_compared_to_cluster": tot,
        "ari_vs_truth": None if ari_vs_truth is None else round(ari_vs_truth, 4),
        "n_ari_pairs": stats["ari_pairs"],
        "primary_distribution": dist.most_common(),
        "taxonomy": os.path.abspath(args.taxonomy),
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[R4] valid {stats['valid']}/{len(ids)}, invalid {stats['invalid']}, "
          f"abstained {stats['abstain']}")
    if summary["agree_with_cluster"] is not None:
        print(f"[R4] agreement with the cluster each item came from: "
              f"{summary['agree_with_cluster']:.3f} "
              f"({agree}/{tot})")
    if ari_vs_truth is not None:
        print(f"[R4] ARI vs ground truth (via cluster->label mapping): "
              f"{ari_vs_truth:.3f} on {stats['ari_pairs']} items")
    print(f"[R4] wrote {args.out}/labels.jsonl and summary.json")


if __name__ == "__main__":
    main()
