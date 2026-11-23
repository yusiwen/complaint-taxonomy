#!/usr/bin/env python3
"""Add English fields to a taxonomy so the annotation page can be read in English.

The corpus and the class names are Chinese; the definitions were written in Chinese. An
annotator who reads English more comfortably cannot work from them, and inventing a
second codebook by hand risks the two annotators reading different rules — which is the
one thing that must not differ. So the translation is mechanical, stored *next to* the
Chinese in the same file, and both annotators load the same copy.

The canonical label stays the Chinese string in every artefact: `label_en` is a display
alias only. `make_labels.py`, `kappa.py` and the exported sheets compare `label`, so a
translation can never split a class in two.

    uv run python tools/translate_codebook.py \
        --taxonomy out/codebook_merged.json \
        --endpoint http://127.0.0.1:8000/v1/chat/completions

Writes the same file back, adding label_en / definition_en / includes_en / boundary_en /
counterexample_en. Idempotent: entries that already have `label_en` are skipped unless
`--force` is given.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import llm_naming  # noqa: E402

FIELDS = ("label", "definition", "includes", "boundary", "counterexample")

PROMPT = """把下面这个中文投诉类别翻译成英文，供标注员阅读。

要求：
1. `label` 翻成业务能直接用的英文类别名，不超过 6 个词；
2. `definition` / `boundary` / `counterexample` 直译，保持判断依据不变，不要加入原文没有的规则；
3. `includes` 是数组，逐条翻译，条目数不变；
4. 保持中文原文的严格程度——原文说"必须"就不要译成"通常"。

输入：
{payload}

严格输出同样结构的 JSON，不要解释、不要加代码块。"""


def translate(endpoint, model, entry, timeout=180):
    payload = {f: entry.get(f) for f in FIELDS}
    prompt = PROMPT.replace("{payload}", json.dumps(payload, ensure_ascii=False, indent=2))
    for attempt in range(2):
        try:
            raw = llm_naming.chat(endpoint, model,
                                  [{"role": "user", "content": prompt}],
                                  max_tokens=800, timeout=timeout)
        except Exception as exc:                                   # noqa: BLE001
            if attempt == 1:
                return None, f"request failed: {exc}"[:120]
            continue
        parsed = llm_naming.parse_fields(raw)
        if parsed:
            return parsed, None
    return None, "could not parse JSON from the model answer"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--model", default="",
                    help="model name your endpoint expects; empty is fine for most")
    ap.add_argument("--force", action="store_true",
                    help="retranslate entries that already carry label_en")
    ap.add_argument("--out", default="", help="default: rewrite --taxonomy in place")
    args = ap.parse_args()

    with open(args.taxonomy, encoding="utf-8") as f:
        tax = json.load(f)

    done = skipped = failed = 0
    for e in tax.get("clusters") or []:
        if not e.get("label"):
            continue
        if e.get("label_en") and not args.force:
            skipped += 1
            continue
        fields, err = translate(args.endpoint, args.model, e)
        if err:
            failed += 1
            print(f"  #{e.get('cluster')} {e['label']} -> FAILED: {err}", file=sys.stderr)
            continue
        for f in FIELDS:
            val = fields.get(f)
            if f == "includes":
                val = val if isinstance(val, list) else ([val] if val else [])
            e[f + "_en"] = val
        done += 1
        print(f"  {e['label']}  ->  {e.get('label_en')}")

    out = args.out or args.taxonomy
    with open(out, "w", encoding="utf-8") as f:
        json.dump(tax, f, ensure_ascii=False, indent=2)
    print(f"\ntranslated {done}, skipped {skipped} (already had label_en), "
          f"failed {failed} -> {out}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
