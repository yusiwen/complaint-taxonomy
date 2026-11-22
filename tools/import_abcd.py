#!/usr/bin/env python3
"""Import ABCD (Action-Based Conversations Dataset) as bilingual-ready dialogue jsonl.

Why this dataset: it is the closest public match to "user and agent complaint
conversations" —
  * real human-human customer-service dialogues, 10,042 (8,034 train / 1,004 dev /
    1,004 test), MIT licensed;
  * explicit `agent` / `customer` speaker labels, so the pipeline's "customer turns
    only" stage works unchanged;
  * a conversation-level `flow` (10 classes) that *is* a complaint taxonomy:
    product_defect, shipping_issue, purchase_dispute, order_issue, account_access,
    manage_account, troubleshoot_site, subscription_inquiry, storewide_query,
    single_item_query — plus ~55 finer `subflow` labels;
  * an agent `guidelines.json` + `ontology.json`, i.e. the procedure book, which is
    a useful second axis for clustering.

It is English only. For Chinese see `import_chinese_cs.py` (Chinese-Ambiguous-
Reference, MIT, ~1k dialogues).

Download (37 MB, one file, no auth):

    mkdir -p /tmp/abcd && cd /tmp/abcd
    curl -sLO https://raw.githubusercontent.com/asappresearch/abcd/master/data/abcd_v1.1.json.gz

Usage

    uv run python tools/import_abcd.py --splits train dev \
        --out data/public/abcd --limit 0

Outputs
    <out>.jsonl          {"id", "messages": [{"role": "user"|"agent", "content"}]}
    <out>_labels.jsonl   {"id", "label": <flow>, "subflow": <subflow>, "split"}
"""
import argparse
import gzip
import json
import os


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/tmp/abcd/abcd_v1.1.json.gz")
    ap.add_argument("--splits", nargs="*", default=["train", "dev"])
    ap.add_argument("--limit", type=int, default=0,
                    help="max conversations per split, 0 = all")
    ap.add_argument("--out", default="data/public/abcd")
    args = ap.parse_args()

    data = load(args.src)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n = 0
    with open(f"{args.out}.jsonl", "w", encoding="utf-8") as f, \
            open(f"{args.out}_labels.jsonl", "w", encoding="utf-8") as g:
        for split in args.splits:
            convs = data.get(split, [])
            if args.limit:
                convs = convs[: args.limit]
            for c in convs:
                cid = f"{split}_{c['convo_id']}"
                # normalise ABCD speakers onto the pipeline's role names
                msgs = [{"role": "user" if who == "customer" else "agent",
                         "content": text}
                        for who, text in c["original"]]
                f.write(json.dumps({"id": cid, "messages": msgs},
                                   ensure_ascii=False) + "\n")
                g.write(json.dumps({"id": cid,
                                    "label": c["scenario"].get("flow"),
                                    "subflow": c["scenario"].get("subflow"),
                                    "split": split}, ensure_ascii=False) + "\n")
                n += 1

    print(f"wrote {n} conversations over {len(args.splits)} split(s) -> {args.out}.jsonl")


if __name__ == "__main__":
    main()
