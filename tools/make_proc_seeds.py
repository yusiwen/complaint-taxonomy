#!/usr/bin/env python3
"""Build a complaint-vs-procedure seed set for the embedding kNN discriminator.

The rule filter in `procedural.py` removes the clear-cut procedure (bare names,
credential lines, acknowledgements) but leaves what is wrapped in enough real words
to look like a sentence -- about 28% of the corpus on the 500-dialogue run. Rules do
not generalise to that. A kNN over labelled examples does: `SeedFilter` already
exists in the pipeline and measures F1 0.81 where the cue regex scores 0.11.

Seeds come from two places:
  * hand-written coverage of the three procedure families plus chit-chat, so the
    set does not depend on any one corpus;
  * the samples of the clusters that naming identified as procedure, so the set
    contains the *residual* long forms that the rules miss. Those samples are the
    whole point -- they are what a regex cannot see.

    # after a run whose clusters were named
    uv run python tools/make_proc_seeds.py --run out/zh_k10 \
        --procedure-clusters 0,2,9 --out data/seeds_proc_zh.jsonl

    # then filter with it
    cluster_complaints.py --input corpus.jsonl --k 10 \
        --seeds data/seeds_proc_zh.jsonl --knn-thresh 0.4 --out out/filtered

The `label` field is what `SeedFilter` reads: only "complaint" counts as a positive
vote, so every procedure and chit-chat example carries another label.
"""
import argparse
import json
import os

ZH_HAND = [
    # acknowledgements and confirmations
    ("好的，谢谢", "procedure"),
    ("可以，没问题", "procedure"),
    ("嗯嗯，明白了", "procedure"),
    ("知道了，就这样吧", "procedure"),
    ("不用了，谢谢", "procedure"),
    ("没有了，就这些", "procedure"),
    ("是的，没错", "procedure"),
    # thanks and closings
    ("非常感谢你的帮助", "procedure"),
    ("谢谢，祝您愉快", "procedure"),
    ("好的，那就不麻烦你了", "procedure"),
    # names and identity
    ("我的名字是阿尔伯特·桑德斯", "procedure"),
    ("我叫乔伊斯·吴", "procedure"),
    ("用户名是 ap259678", "procedure"),
    ("我的邮箱是 <EMAIL>", "procedure"),
    ("订单号是 <N>", "procedure"),
    # procedure wrapped in real words (the residuals rules miss)
    ("请核实一下我的账户信息，我的名字是诺曼·布沙尔，地址是 <N> 号", "procedure"),
    ("我这边可以提供订单号 <N> 和邮箱 <EMAIL>，你查一下", "procedure"),
    ("好的，我确认一下，我的会员是黄金会员", "procedure"),
    ("我先登录看看，稍等一下", "procedure"),
    # complaint, worded closely to the procedure examples above (precision guards)
    ("我登录不上我的账户，密码一直报错", "complaint"),
    ("我的账户被锁了，进不去", "complaint"),
    ("会员费扣了两次，要求退款", "complaint"),
    ("地址填对了但一直显示配送失败", "complaint"),
]

EN_HAND = [
    ("ok, thanks", "procedure"),
    ("yes, that's all", "procedure"),
    ("no, nothing else", "procedure"),
    ("sounds good, bye", "procedure"),
    ("thanks a lot for your help", "procedure"),
    ("my name is Albert Sanders", "procedure"),
    ("my account id is <N>", "procedure"),
    ("i can give you the order id <N> and my email <EMAIL>", "procedure"),
    ("sure, i will check my account and get back to you", "procedure"),
    ("i cannot log into my account, the password keeps failing", "complaint"),
    ("my account is locked and i cannot get in", "complaint"),
    ("i was charged the subscription fee twice, i want a refund", "complaint"),
    ("the address is correct but delivery keeps failing", "complaint"),
]


def read_seeds(path):
    out = []
    if not path or not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def mine(run, wanted):
    """Pull the samples of the named procedural clusters out of a run."""
    path = os.path.join(run, "clusters.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    out = []
    for c in payload.get("clusters") or []:
        if int(c.get("cluster", -1)) in wanted:
            for s in (c.get("samples") or [])[:8]:
                out.append({"text": s, "label": "procedure"})
    return out


def dedupe(rows):
    seen, out = set(), []
    for r in rows:
        key = (r["text"].strip(), r["label"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="", help="run whose clusters were named")
    ap.add_argument("--procedure-clusters", default="",
                    help="comma-separated cluster ids that naming found procedural")
    ap.add_argument("--base", default="", help="existing seed file to extend")
    ap.add_argument("--lang", choices=["zh", "en"], default="zh")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = read_seeds(args.base)
    if args.run and args.procedure_clusters:
        wanted = {int(x) for x in args.procedure_clusters.split(",") if x.strip()}
        rows += mine(args.run, wanted)
    hand = ZH_HAND if args.lang == "zh" else EN_HAND
    rows += [{"text": t, "label": lab} for t, lab in hand]
    rows = dedupe(rows)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_c = sum(1 for r in rows if r["label"] == "complaint")
    print(f"{args.out}: {len(rows)} seeds ({n_c} complaint, "
          f"{len(rows) - n_c} procedure/other)")


if __name__ == "__main__":
    main()
