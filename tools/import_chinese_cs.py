#!/usr/bin/env python3
"""Import a Chinese customer-service dialogue corpus (Chinese-Ambiguous-Reference).

Why this one: it is the only openly licensed (MIT) corpus of *real* Chinese
customer-service dialogues that is directly downloadable. It is small (1,104
conversations) and the domain is food/shopping rather than complaints, so treat it
as a **smoke test for the Chinese code path** (speaker roles, sentence splitting,
PII masking, clustering on zh) and not as a labelled complaint benchmark.

Source: https://github.com/ygan/Chinese-Ambiguous-Reference
        All Conversation.json (2.1 MB, MIT)

Each record has `total_split` — a list of turns rendered as `说话人：“内容”`, with a
few integer placeholders that mark something internal to the original study.

Usage

    curl -sLO "https://raw.githubusercontent.com/ygan/Chinese-Ambiguous-Reference/main/All%20Conversation.json"
    uv run python tools/import_chinese_cs.py --src "All Conversation.json" \
        --out data/public/zh_cs

Outputs
    <out>.jsonl          {"id", "messages": [{"role": "user"|"agent", "content"}]}
    <out>_meta.jsonl     {"id", "domain", "shop"}
"""
import argparse
import json
import re
import sys

# Anything that is clearly the customer side. Everything else (server, agent,
# shop assistant, cashier) is the service side.
CUSTOMER = re.compile(r"顾客|客户|买家|消费者|客人")
SPEAKER = re.compile(r"^\s*([^：:\"]{1,12})[：:]\s*(.*)$")


def split_turns(turns):
    out = []
    for t in turns:
        if not isinstance(t, str):
            continue
        m = SPEAKER.match(t)
        if not m:
            if out:                       # continuation of the previous turn
                out[-1][1] += t.strip()
            continue
        who, text = m.group(1), m.group(2)
        text = text.strip().strip("“”\"「」")
        if not text:
            continue
        role = "user" if CUSTOMER.search(who) else "agent"
        if out and out[-1][0] == role:
            out[-1][1] += text
        else:
            out.append([role, text])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="All Conversation.json")
    ap.add_argument("--out", default="data/public/zh_cs")
    args = ap.parse_args()

    with open(args.src, encoding="utf-8") as f:
        data = json.load(f)

    n = 0
    with open(f"{args.out}.jsonl", "w", encoding="utf-8") as f, \
            open(f"{args.out}_meta.jsonl", "w", encoding="utf-8") as m:
        for i, rec in enumerate(data):
            turns = split_turns(rec.get("total_split") or [])
            users = [t for t in turns if t[0] == "user"]
            if len(turns) < 2 or not users:
                continue
            cid = f"zhcs{i:05d}"
            f.write(json.dumps({"id": cid,
                                "messages": [{"role": r, "content": txt}
                                             for r, txt in turns]},
                               ensure_ascii=False) + "\n")
            m.write(json.dumps({"id": cid,
                                "domain": rec.get("shop") or "",
                                "merchandise": rec.get("merchandise") or "",
                                "location": rec.get("location") or ""},
                               ensure_ascii=False) + "\n")
            n += 1

    print(f"wrote {n} dialogues -> {args.out}.jsonl", file=sys.stderr)


if __name__ == "__main__":
    main()
