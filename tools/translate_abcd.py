#!/usr/bin/env python3
"""Build a Chinese/English parallel complaint-dialogue benchmark from ABCD.

No public corpus is bilingual *and* dialogue-shaped *and* complaint-labelled, so
this manufactures the missing half: take N ABCD dialogues, translate every turn
into Simplified Chinese with the local Qwen3.8-27B (SGLang, thinking disabled),
and keep the original English alongside. Same ids, same turn order, same `flow`
label — which makes cross-lingual agreement a countable number: does the Chinese
version of a dialogue land in the same cluster as its English source?

Translation is batched one dialogue per request (numbered lines in, numbered
lines out) because 500 requests beats 7,000. Every response is validated on line
count; a mismatch retries once, then falls back to per-turn translation.

Honesty note: this is machine translation, so the Chinese side carries MT noise.
Disagreements measure "embedding alignment under realistic paraphrase", not a
human-parallel corpus. Keep that in mind when reading the agreement rate.

Usage

    uv run python tools/translate_abcd.py --n 500 --split test \
        --endpoint http://127.0.0.1:8000/v1/chat/completions \
        --model <name-if-your-server-needs-one> \
        --out data/public/abcd_500

Outputs
    <out>_en.jsonl / <out>_zh.jsonl   {"id", "messages": [...]} (turn-aligned)
    <out>_labels.jsonl                {"id", "label": flow, "subflow", "src_id"}
"""
import argparse
import gzip
import json
import os
import re
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cluster_complaints import clean  # noqa: E402

SYSTEM = ("你是专业的英译中译者。把用户给的每一行英文翻译成简体中文，"
          "逐行输出，保留行首编号。不要合并或拆分行，不要输出解释或任何额外文字。")

_lock = threading.Lock()
stats = {"ok": 0, "retry": 0, "fallback": 0}


def call(endpoint, model, lines, timeout=180):
    body = json.dumps({
        "model": model,
        "temperature": 0,
        "max_tokens": 40 * len(lines) + 200,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n".join(f"{i + 1}. {t}" for i, t in enumerate(lines))},
        ],
    }).encode()
    req = urllib.request.Request(endpoint, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"].get("content") or ""


def parse(out, n):
    """Return exactly n translated lines, or None if the shape is wrong."""
    got = {}
    for line in out.splitlines():
        m = re.match(r"^\s*(\d+)[.、:)]\s*(.+?)\s*$", line)
        if m:
            got[int(m.group(1))] = m.group(2)
    if len(got) != n or sorted(got) != list(range(1, n + 1)):
        return None
    return [got[i] for i in range(1, n + 1)]


def translate_dialogue(endpoint, model, lines):
    for attempt in range(2):
        try:
            parsed = parse(call(endpoint, model, lines), len(lines))
        except Exception as exc:                                   # noqa: BLE001
            parsed = None
            if attempt == 1:
                print(f"  request failed: {exc}", file=sys.stderr)
        if parsed:
            with _lock:
                stats["ok" if attempt == 0 else "retry"] += 1
            return parsed
    # last resort: one request per line
    out = []
    for t in lines:
        try:
            got = call(endpoint, model, [t], timeout=90).strip()
            out.append(got.splitlines()[-1] if got else t)
        except Exception:                                          # noqa: BLE001
            out.append(t)
    with _lock:
        stats["fallback"] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/tmp/abcd/abcd_v1.1.json.gz")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--model", default="",
                    help="model name your endpoint expects; empty is fine")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="data/public/abcd_500")
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: only translate the first K dialogues")
    ap.add_argument("--en-only", action="store_true",
                    help="write only the English half and the labels; no model needed")
    ap.add_argument("--src-default", default="",
                    help="override the corpus path (used by get_demo_corpus.py)")
    args = ap.parse_args()

    opener = gzip.open if args.src.endswith(".gz") else open
    with opener(args.src, "rt", encoding="utf-8") as f:
        data = json.load(f)

    import random
    rng = random.Random(args.seed)
    convs = list(data[args.split])
    rng.shuffle(convs)
    convs = convs[: args.n or len(convs)]
    if args.limit:
        convs = convs[: args.limit]
    print(f"translating {len(convs)} dialogues from split={args.split}", file=sys.stderr)

    # one task per dialogue: mask PII first so ids/emails are not translated
    tasks = []
    for c in convs:
        pairs = [(("user" if who == "customer" else "agent"), clean(text))
                 for who, text in c["original"]]
        tasks.append((c, pairs))

    def work(item):
        c, pairs = item
        if args.en_only:
            return c, pairs, [t for _, t in pairs]
        lines = [t for _, t in pairs]
        zh = translate_dialogue(args.endpoint, args.model, lines)
        return c, pairs, zh

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, res in enumerate(pool.map(work, tasks), 1):
            results.append(res)
            if i % 50 == 0:
                print(f"  {i}/{len(tasks)} dialogues, stats={stats}", file=sys.stderr)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    zh_handle = None if args.en_only else open(f"{args.out}_zh.jsonl", "w",
                                               encoding="utf-8")
    with open(f"{args.out}_en.jsonl", "w", encoding="utf-8") as fe, \
            open(f"{args.out}_labels.jsonl", "w", encoding="utf-8") as fl:
        kept = 0
        for c, pairs, zh in results:
            if len(zh) != len(pairs):
                continue
            cid = f"abcd_{args.split}_{c['convo_id']}"
            en_msgs = [{"role": r, "content": t} for r, t in pairs]
            zh_msgs = [{"role": r, "content": t} for (r, _), t in zip(pairs, zh)]
            fe.write(json.dumps({"id": cid, "messages": en_msgs}, ensure_ascii=False) + "\n")
            if zh_handle is not None:
                zh_handle.write(json.dumps({"id": cid, "messages": zh_msgs},
                                           ensure_ascii=False) + "\n")
            fl.write(json.dumps({"id": cid,
                                 "src_id": c["convo_id"],
                                 "label": c["scenario"].get("flow"),
                                 "subflow": c["scenario"].get("subflow")},
                                ensure_ascii=False) + "\n")
            kept += 1
    if zh_handle is not None:
        zh_handle.close()

    shape = "English-only" if args.en_only else "parallel en/zh"
    print(f"wrote {kept} {shape} dialogues -> {args.out}_en"
          f"{'' if args.en_only else '|_zh'}.jsonl | stats={stats}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
