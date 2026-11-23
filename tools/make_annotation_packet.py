#!/usr/bin/env python3
"""Build the double-annotation packet: blank sheets two humans can label independently.

The codebook is only agreed once two annotators, working alone from the same definitions,
reach Cohen's kappa >= 0.7 on 100-200 shared items. Getting there needs a packet that
cannot leak the answer, so this tool writes only what an annotator is allowed to see:

  * the user-side turns of each conversation (the agent's turns are context, and the
    pipeline judges from the customer's words -- showing the agent's script invites
    copying the procedure book instead of applying the definitions);
  * **no** ground-truth flow, **no** LLM label, **no** cluster id, and no `primary`
    already filled in;
  * a shuffled order, seeded, so the two annotators cannot compare positions.

It also writes the codebook next to the sheets, because annotating from memory of the
codebook is how kappa gets inflated: both people must be reading the same text.

    uv run python tools/make_annotation_packet.py \
        --input data/public/abcd_500_zh.jsonl \
        --codebook out/codebook_merged.md \
        --labels data/public/abcd_500_labels.jsonl \
        --out annotation --n 150 --pilot 30

Outputs (under --out)
    README.md            protocol, time budget, how it is scored
    codebook.md          copy of the definitions being tested
    pilot_items.jsonl   30 items  -> annotate first, then fix the codebook
    pilot_A.jsonl / pilot_B.jsonl
    items.jsonl         150 items -> the kappa that counts
    ann_A.jsonl / ann_B.jsonl
    support.json        class distribution + which classes have too few items to judge
"""
import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter


def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def user_text(o, max_chars=1200):
    turns = [m["content"].strip() for m in o.get("messages", [])
             if m.get("role") == "user" and m.get("content")]
    return " / ".join(turns)[:max_chars]


def blank(ids, path):
    with open(path, "w", encoding="utf-8") as f:
        for cid in ids:
            # labels left empty on purpose: filling them in from the truth is the whole
            # failure mode this packet exists to prevent
            f.write(json.dumps({"id": cid, "primary": "", "labels": [],
                                "note": ""}, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--codebook", required=True, help="codebook.md the annotators read")
    ap.add_argument("--labels", default="", help="ground truth, used only for support stats")
    ap.add_argument("--out", default="annotation")
    ap.add_argument("--n", type=int, default=150, help="items in the scored set")
    ap.add_argument("--pilot", type=int, default=30, help="items for the calibration pass")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label-field", default="label")
    args = ap.parse_args()

    convs = [c for c in read_jsonl(args.input) if user_text(c)]
    if len(convs) < args.n + args.pilot:
        sys.exit(f"[fatal] only {len(convs)} usable conversations")

    rng = random.Random(args.seed)
    order = list(convs)
    rng.shuffle(order)
    pilot = order[: args.pilot]
    main_items = order[args.pilot: args.pilot + args.n]

    os.makedirs(args.out, exist_ok=True)
    shutil.copyfile(args.codebook, os.path.join(args.out, "codebook.md"))

    for name, items in (("pilot", pilot), ("", main_items)):
        items_path = os.path.join(args.out, f"{name + '_' if name else ''}items.jsonl")
        with open(items_path, "w", encoding="utf-8") as f:
            for o in items:
                f.write(json.dumps({"id": o["id"], "text": user_text(o)},
                                   ensure_ascii=False) + "\n")
        stem = f"{name}_" if name else "ann_"
        blank([o["id"] for o in items], os.path.join(args.out, f"{stem}A.jsonl"))
        blank([o["id"] for o in items], os.path.join(args.out, f"{stem}B.jsonl"))

    support = {}
    if args.labels:
        truth = {o["id"]: o.get(args.label_field) for o in read_jsonl(args.labels)}
        dist = Counter(truth.get(o["id"]) for o in main_items)
        thin = [str(k) for k, v in dist.items() if k is not None and v < 5]
        support = {"distribution": {str(k): v for k, v in dist.items()},
                   "too_few_for_per_class_kappa": sorted(thin)}
        with open(os.path.join(args.out, "support.json"), "w", encoding="utf-8") as f:
            json.dump(support, f, ensure_ascii=False, indent=2)

    n_classes = 0
    with open(args.codebook, encoding="utf-8") as f:
        for line in f:
            if line.startswith("## ") and not line.startswith("## 类别速查") \
                    and "优先级" not in line:
                n_classes += 1

    with open(os.path.join(args.out, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"""# 双标标注包

目标：两个标注员**各自独立**按 `codebook.md` 标注同一批对话，算出 Cohen's kappa，
判定 codebook 是否可用（门槛 kappa ≥ 0.7）。

## 材料

| 文件 | 说明 |
|---|---|
| `codebook.md` | 要检验的类别定义，**只按这个标**，不要凭印象 |
| `pilot_items.jsonl` | 30 条，校准用 |
| `pilot_A.jsonl` / `pilot_B.jsonl` | 上面 30 条的空答卷 |
| `items.jsonl` | {len(main_items)} 条，**决定 kappa 的那一批** |
| `ann_A.jsonl` / `ann_B.jsonl` | 上面 {len(main_items)} 条的空答卷 |

`items.jsonl` 里只有用户说的话，没有真值、没有模型标注、没有聚类编号——这是故意的。
客服的话也没给：定义是按"用户抱怨/要求什么"写的，看到客服话术会让人去抄流程而不是套定义。

## 流程

**第一步：校准（30 条，约 40 分钟，两人各自做，不要讨论）**

1. 读一遍 `codebook.md`（约 15 分钟）。
2. 标 `pilot_items.jsonl` 的 30 条，写进各自的 `pilot_X.jsonl`：`primary` 填**一个**
   类别名（必须与 codebook 里的名字**逐字一致**），可选 `labels` 填 1–3 个。
3. 都没有把握就填 `"UNKNOWN"`，并接受它会被算作分歧——弃权是合法答案，但不要滥用。
4. 两人交回后算一次 kappa，把分歧集中在哪些类别对上找出来。

**第二步：补 codebook（我来做，30 分钟）**

根据第一步的分歧，把 `codebook.md` 里 6 组"优先级待填"的类别对写死（例如
`物流时效与进度查询` vs `订单状态查询` 谁优先），并修正被误解的定义。**校准集的数据作废**，
下一批用全新的 {len(main_items)} 条。

**第三步：正式（{len(main_items)} 条，每人约 2–3 小时）**

1. 用补完的 codebook 标 `items.jsonl`，写进 `ann_X.jsonl`。
2. 两个人之间**不允许交流**，不允许看对方的答卷。
3. 单条 40–60 秒；不要连续标超过 45 分钟，疲劳会直接压低 kappa。

## 判定标准

| 指标 | 门槛 | 含义 |
|---|---|---|
| 总体 Cohen's kappa | **≥ 0.70** | codebook 可用，可以开始放量标注 |
| 0.60–0.70 | 修定义 | 找出 per-class kappa 最低的类别重写，不必重标全部 |
| < 0.60 | 停 | codebook 还不能用，标得越多噪声越大 |

每类少于 5 条的，per-class kappa 不可信，只看总体。

## 计分

```bash
uv run python tools/kappa.py --a annotation/ann_A.jsonl --b annotation/ann_B.jsonl
```

输出：总体 kappa、原始一致率、每类 kappa、以及最大的分歧格（A 判什么 → B 判什么），
分歧格子就是下一次改 codebook 的清单。

## 两条纪律

- **不要看真值。** 语料有真实类别字段，但标的时候只能看 codebook——用真值会得到一个漂亮
  却没意义的 kappa，因为它测的是"两人能不能猜到原数据集的口径"，不是 codebook 是否可读。
- **不要互相商量。** 商量出来的 kappa 不是信度，是共识。
""" .replace("{len(main_items)}", str(len(main_items))))

    print(f"wrote {args.out}/: pilot {len(pilot)} items, scored set {len(main_items)} items")
    print(f"  codebook classes: ~{n_classes}")
    if support:
        dist = support["distribution"]
        print(f"  scored-set class distribution: {dist}")
        if support["too_few_for_per_class_kappa"]:
            print(f"  classes with <5 items (per-class kappa unreliable): "
                  f"{support['too_few_for_per_class_kappa']}")
    print(f"\nnext: two annotators fill {args.out}/pilot_A.jsonl and pilot_B.jsonl, "
          f"then\n  uv run python tools/kappa.py --a {args.out}/pilot_A.jsonl "
          f"--b {args.out}/pilot_B.jsonl")


if __name__ == "__main__":
    main()
