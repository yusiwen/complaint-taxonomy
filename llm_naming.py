#!/usr/bin/env python3
"""Ask a local or hosted LLM to name a cluster, in one place.

Shared by `cluster_complaints.py` (optional inline naming), `tools/merge_clusters.py`
(merged-group codebook entries) and `tools/name_clusters.py` (R3 taxonomy). It was
duplicated in the merge tool first; keeping one copy means the prompt, the JSON
contract and the parse fallbacks cannot drift apart between artefacts that are
supposed to end up in the same codebook.

The prompt asks for the five fields the design doc requires of a taxonomy entry:
label, definition, what it includes, where its boundary is, and a counterexample.
A cluster name alone is not usable for annotation -- the annotator needs to know
what falls *outside* the class, otherwise the next 2,000 labels are guesses.

Nothing here decides whether the names are any good. Every entry the model returns
is a draft to be edited, and `name_clusters.py` keeps the samples it was shown
next to the answer so a reviewer can check the claim.
"""
import json
import re
import urllib.request

FIELDS = ("label", "definition", "includes", "boundary", "counterexample")

_CJK = re.compile(r"[\u4e00-\u9fff]")
_ASCII = re.compile(r"[A-Za-z]")

_RULES = {
    "zh": (
        "你是客服投诉分类体系的设计者。下面是一个聚类得到的簇的代表样本，"
        "它们被 embedding 聚在一起，但还没有名字。请判断这些样本共同在讲什么，"
        "写出这个类别的定义。\n"
        "要求：\n"
        "1. 只用简体中文；\n"
        "2. label 不超过 12 个字，是业务能直接用的类别名，不要写\"类A\"\"簇1\"这种占位；\n"
        "3. definition 说明什么算这一类，用一句话讲清判断依据；\n"
        "4. includes 给 2-3 个典型情形；\n"
        "5. boundary 说明它跟相邻类别怎么区分；\n"
        "6. counterexample 给一个看起来像但不属于这一类的例子；\n"
        "7. 如果样本本身就不一致（混了多个主题），在 definition 里直接说明，"
        "不要硬凑成一个类别。\n"
        "严格输出 JSON，不要输出解释、不要加 markdown 代码块。"
    ),
    "en": (
        "You design customer-complaint taxonomies. Below are the representative "
        "samples of one cluster produced by embedding-based clustering; it has no "
        "name yet. Decide what these samples have in common and define the class.\n"
        "Rules:\n"
        "1. English only;\n"
        "2. label: at most 6 words, a business-usable category name -- never a "
        "placeholder like 'Cluster 1';\n"
        "3. definition: one sentence that says what makes something belong here;\n"
        "4. includes: 2-3 typical cases;\n"
        "5. boundary: how it differs from the neighbouring category;\n"
        "6. counterexample: something that looks similar but is not in this class;\n"
        "7. if the samples are not actually one topic, say so in definition instead "
        "of forcing a category.\n"
        "Output JSON only, no prose, no markdown fences."
    ),
}

SCHEMA = ('{"label": "...", "definition": "...", "includes": ["...", "..."], '
          '"boundary": "...", "counterexample": "..."}')


def sniff_lang(texts):
    """Which prompt to use: the corpus decides, not the operator.

    Call this on the whole corpus, not per cluster. A Chinese cluster full of user
    names, emails and brand words can out-count its own CJK characters, and a
    per-cluster vote then flips that one entry into English -- which is exactly the
    defect that showed up the first time this ran (cluster 3 of a Chinese corpus
    came back as "Customer Identity Verification Details").

    The threshold is deliberately generous towards CJK for the same reason: a
    Chinese corpus with heavy Latin proper nouns should still be treated as Chinese.
    """
    cjk = sum(len(_CJK.findall(t)) for t in texts)
    ascii_ = sum(len(_ASCII.findall(t)) for t in texts)
    return "zh" if cjk and cjk * 3 >= ascii_ else "en"


def build_prompt(samples, terms, lang="zh", context=None):
    """Assemble the user message: rules, style, high-frequency terms, samples."""
    lines = [_RULES[lang], "", f"输出格式（严格）：{SCHEMA}", ""]
    if context:
        lines.append(f"上下文：{context}")
    if terms:
        lines.append("高频词：" + "、".join(terms[:10]))
    lines.append("样本：")
    for i, s in enumerate(samples, 1):
        text = " ".join(str(s).split())
        lines.append(f"{i}. {text[:300]}")
    return "\n".join(lines)


def chat(endpoint, model, messages, max_tokens=600, timeout=240, temperature=0.0):
    """One non-streaming chat call. Thinking is disabled: we want the JSON only."""
    body = json.dumps({
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": messages,
    }).encode()
    req = urllib.request.Request(endpoint, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    msg = data["choices"][0]["message"]
    return msg.get("content") or ""


def parse_fields(text):
    """Pull the five fields out of a model answer, or return None."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        # models like to emit trailing commas and unescaped newlines
        cleaned = re.sub(r",\s*([}\]])", r"\1", m.group(0))
        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    out = {}
    for field in FIELDS:
        val = obj.get(field)
        if val is None:
            out[field] = [] if field == "includes" else ""
        elif field == "includes":
            out[field] = [str(v) for v in val] if isinstance(val, list) else [str(val)]
        else:
            out[field] = str(val)
    return out


def name_cluster(endpoint, model, samples, terms=(), lang=None, context=None,
                 timeout=240):
    """Name one cluster. Returns the five fields plus `_ok` / `_error`.

    A failed parse is reported, not papered over: an empty label that looks like a
    field is worse than an explicit error, because the codebook then carries a hole
    nobody notices.
    """
    lang = lang or sniff_lang(list(samples) or ["zh"])
    prompt = build_prompt(list(samples), list(terms), lang, context)
    for attempt in range(2):
        try:
            text = chat(endpoint, model, [{"role": "user", "content": prompt}],
                        timeout=timeout)
        except Exception as exc:                                   # noqa: BLE001
            if attempt == 1:
                return {"_ok": False, "_error": f"request failed: {exc}"[:200],
                        **{f: ([] if f == "includes" else "") for f in FIELDS}}
            continue
        fields = parse_fields(text)
        if fields:
            if fields["includes"] and not isinstance(fields["includes"], list):
                fields["includes"] = [fields["includes"]]
            return {"_ok": True, "_raw_chars": len(text), **fields}
    return {"_ok": False, "_error": "could not parse JSON from the model answer",
            **{f: ([] if f == "includes" else "") for f in FIELDS}}


def validate(entry):
    """What a reviewer should be able to demand of one taxonomy entry."""
    problems = []
    if not entry.get("label"):
        problems.append("missing label")
    if len(entry.get("definition") or "") < 8:
        problems.append("definition too short to be usable")
    if not entry.get("boundary"):
        problems.append("missing boundary")
    if not entry.get("counterexample"):
        problems.append("missing counterexample")
    if not entry.get("includes"):
        problems.append("missing includes")
    return problems


def name_all(clusters, endpoint, model, context=None, timeout=240, log=None):
    """Name every cluster in a `clusters.json` payload, one call each.

    `clusters` is the list the pipeline writes: each item carries `cluster`,
    `size`, `top_terms` and `samples`. The samples are passed through untouched so
    the answer can always be audited against the evidence it was derived from.
    """
    entries = []
    corpus_lang = sniff_lang([t for c in clusters for t in (c.get("samples") or [])])
    for c in clusters:
        samples = c.get("samples") or []
        entry = name_cluster(endpoint, model, samples, c.get("top_terms") or [],
                             lang=corpus_lang, context=context, timeout=timeout)
        entry["cluster"] = int(c.get("cluster"))
        entry["size"] = int(c.get("size") or 0)
        entry["top_terms"] = list(c.get("top_terms") or [])[:8]
        entry["evidence"] = {
            "n_samples": len(samples),
            "samples": samples[:5],
        }
        if log:
            log(f"  #{entry['cluster']:<3d} {entry.get('label') or '(failed)':<18}"
                f" {len(entry['evidence']['samples'])} samples")
        entries.append(entry)
    return entries
