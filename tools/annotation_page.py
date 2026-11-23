#!/usr/bin/env python3
"""Render a self-contained annotation page: dropdowns, autosave, JSONL export.

Typing category names by hand is a transcription task, and transcription errors count as
disagreement in the kappa — the number then measures typing, not definitions. So the
labels become a dropdown (plus UNKNOWN), and the page writes exactly the JSONL that
`tools/kappa.py` reads.

Deliberately a single offline HTML file with no server and no network:

  * two annotators must not see each other's work, and a shared server would show both;
  * it works from a file:// path, on any machine, with no install;
  * answers live in localStorage keyed by annotator name, so a reload or a crash during a
    2-3 hour session loses nothing, and two people on one machine do not collide.

The page shows the codebook beside the item, because the whole point is to apply the
definitions — and it never shows ground truth, an LLM label or a cluster id.

    uv run python tools/make_annotation_packet.py ... --html

Writes `annotate_pilot.html` and `annotate_main.html` next to the sheets.
"""
import json
import os

TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>投诉标注 — {set_name}</title>
<style>
  :root {{
    --bg: #0f1115; --card: #171a21; --line: #2a2f3a; --fg: #e6e8ee;
    --muted: #9aa3b2; --accent: #4c8dff; --ok: #3fb950; --warn: #d29922;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--bg); color: var(--fg);
         font: 15px/1.6 -apple-system, "Segoe UI", "Noto Sans SC", sans-serif; }}
  header {{ position: sticky; top: 0; z-index: 5; background: #12151b;
            border-bottom: 1px solid var(--line); padding: 10px 16px;
            display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }}
  header h1 {{ font-size: 15px; margin: 0; font-weight: 600; }}
  .grow {{ flex: 1; }}
  .bar {{ height: 6px; background: var(--line); border-radius: 3px; width: 180px;
          overflow: hidden; }}
  .bar > i {{ display: block; height: 100%; background: var(--ok); width: 0%; }}
  main {{ display: grid; grid-template-columns: 1fr 380px; gap: 16px; padding: 16px;
          max-width: 1400px; margin: 0 auto; }}
  @media (max-width: 1000px) {{ main {{ grid-template-columns: 1fr; }} }}
  .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px;
           padding: 16px; }}
  .meta {{ color: var(--muted); font-size: 12px; display: flex; gap: 12px; }}
  .text {{ white-space: pre-wrap; word-break: break-word; margin: 12px 0 16px;
           font-size: 15px; line-height: 1.75; max-height: 46vh; overflow: auto; }}
  label.f {{ display: block; color: var(--muted); font-size: 12px; margin: 12px 0 4px; }}
  select, input, textarea, button {{ font: inherit; color: var(--fg);
    background: #10131a; border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; }}
  select {{ width: 100%; }}
  select.set {{ border-color: var(--accent); }}
  button {{ cursor: pointer; }}
  button.primary {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
  button:disabled {{ opacity: .45; cursor: default; }}
  .row {{ display: flex; gap: 8px; align-items: center; }}
  .chip {{ font-size: 12px; color: var(--muted); border: 1px solid var(--line);
           border-radius: 999px; padding: 2px 8px; }}
  .chip.ok {{ color: var(--ok); border-color: var(--ok); }}
  .chip.warn {{ color: var(--warn); border-color: var(--warn); }}
  aside {{ position: sticky; top: 64px; align-self: start; max-height: calc(100vh - 84px);
           overflow: auto; }}
  aside h2 {{ font-size: 13px; margin: 0 0 8px; color: var(--muted); font-weight: 600; }}
  .cb-item h3 {{ font-size: 13px; margin: 12px 0 4px; }}
  .cb-item p {{ margin: 2px 0; font-size: 12.5px; color: #c3c9d5; }}
  .cb-item .k {{ color: var(--muted); }}
  code {{ background: #10131a; padding: 1px 5px; border-radius: 4px; font-size: 12px; }}
  .blind {{ font-size: 12px; color: var(--muted); }}
  dialog {{ background: var(--card); color: var(--fg); border: 1px solid var(--line);
            border-radius: 10px; max-width: 720px; }}
  .sum {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;
          margin: 6px 0 2px; }}
  .sum div {{ background: #10131a; border: 1px solid var(--line); border-radius: 8px;
              padding: 10px; text-align: center; }}
  .sum b {{ display: block; font-size: 20px; }}
  .sum span {{ font-size: 12px; color: var(--muted); }}
  .warnbox p {{ margin: 8px 0; font-size: 13px; padding: 8px 10px; border-radius: 8px; }}
  .warnbox .bad {{ background: rgba(255,80,80,.12); border: 1px solid #a33; color: #ffb3b3; }}
  .warnbox .soft {{ background: rgba(210,153,34,.12); border: 1px solid var(--warn);
                    color: #f0d090; }}
  .warnbox .ok {{ background: rgba(63,185,80,.12); border: 1px solid var(--ok); color: #9be3a5; }}
  textarea {{ width: 100%; height: 220px; font-family: ui-monospace, monospace;
              font-size: 12px; }}
</style>
</head>
<body>
<header>
  <h1>投诉标注 · {set_name}</h1>
  <span class="chip" id="pos"></span>
  <div class="bar"><i id="bar"></i></div>
  <span class="chip" id="count"></span>
  <span class="grow"></span>
  <input id="who" placeholder="标注员（A 或 B）" style="width:130px" value="">
  <button class="primary" id="exp">导出 JSONL</button>
</header>

<main>
  <section>
    <div class="card">
      <div class="meta">
        <span id="cid"></span><span id="chars"></span>
        <span class="chip" id="state"></span>
      </div>
      <div class="text" id="text"></div>

      <label class="f" for="sel">类别（下拉选择，选不准就选 UNKNOWN）</label>
      <select id="sel"></select>

      <label class="f" for="note">备注（可选，写你不确定的地方）</label>
      <input id="note" placeholder="例如：像'订单状态查询'也像'物流时效'">

      <div class="row" style="margin-top:14px">
        <button id="prev">← 上一条</button>
        <button id="next">下一条 →</button>
        <span class="grow"></span>
        <button id="firstun">跳到下一条未标</button>
      </div>
      <p class="blind">只用这份 codebook 判断，参考真值或互相商量都会让 kappa 失去意义。
        进度自动保存在本机浏览器。快捷键：↑↓ 切换条目，Alt+←/→ 同，Enter 下一条。</p>
    </div>
  </section>

  <aside class="card">
    <h2>CODEFOOK（{n_classes} 类）</h2>
    <input id="q" placeholder="过滤类别…" style="width:100%;margin-bottom:8px">
    <div id="cb"></div>
  </aside>
</main>

<dialog id="dlg">
  <h2 style="margin-top:0">导出检查</h2>
  <div id="sum" class="sum"></div>
  <div id="warn" class="warnbox"></div>
  <div class="row" style="margin-top:12px;flex-wrap:wrap">
    <button class="primary" id="dl">确认下载</button>
    <button id="fillun" style="display:none"></button>
    <button id="jump" style="display:none">跳到第一条未标</button>
    <button id="close">关闭</button>
  </div>
  <details style="margin-top:12px">
    <summary class="blind">JSONL 预览（文件名 <code id="fname"></code>）</summary>
    <textarea id="out" readonly style="height:160px;margin-top:8px"></textarea>
  </details>
</dialog>

<script>
const ITEMS = {items_json};
const LABELS = {labels_json};
const CODEBOOK = {codebook_json};
const SET = "{set_name}";
const PREFIX = "{prefix}";
const UNKNOWN = "UNKNOWN";

const $ = (id) => document.getElementById(id);
let i = 0;
const state = {{}};   // id -> {{primary, note}}

function key() {{
  return "complaint-taxonomy:" + SET + ":" + (($("who").value || "anon").trim() || "anon");
}}

function load() {{
  try {{
    const raw = localStorage.getItem(key());
    if (raw) {{ Object.assign(state, JSON.parse(raw)); }}
  }} catch (e) {{ console.warn("state load failed", e); }}
}}
function save() {{
  localStorage.setItem(key(), JSON.stringify(state));
}}

function draw() {{
  const it = ITEMS[i];
  $("pos").textContent = (i + 1) + " / " + ITEMS.length;
  $("cid").textContent = it.id;
  $("chars").textContent = it.text.length + " 字";
  $("text").textContent = it.text;
  const cur = state[it.id] || {{ primary: "", note: "" }};
  $("sel").value = cur.primary || "";
  $("note").value = cur.note || "";
  $("sel").classList.toggle("set", !!cur.primary);
  $("state").textContent = cur.primary ? "已标" : "未标";
  $("state").className = "chip " + (cur.primary ? "ok" : "warn");
  const done = ITEMS.filter((x) => (state[x.id] || {{}}).primary).length;
  $("count").textContent = "已标 " + done + " / " + ITEMS.length;
  $("bar").style.width = (100 * done / ITEMS.length) + "%";
  $("prev").disabled = i === 0;
  $("next").disabled = i === ITEMS.length - 1;
}}

function setPrimary(v) {{
  const id = ITEMS[i].id;
  state[id] = Object.assign({{}}, state[id] || {{}}, {{ primary: v }});
  save(); draw();
}}

function initSelect() {{
  const sel = $("sel");
  sel.innerHTML = "";
  const o0 = document.createElement("option");
  o0.value = ""; o0.textContent = "— 请选择 —";
  sel.appendChild(o0);
  for (const l of LABELS) {{
    const o = document.createElement("option");
    o.value = l; o.textContent = l;
    sel.appendChild(o);
  }}
  const ou = document.createElement("option");
  ou.value = UNKNOWN; ou.textContent = "UNKNOWN（没有类别适用）";
  sel.appendChild(ou);
}}

function initCodebook() {{
  const box = $("cb");
  box.innerHTML = "";
  for (const c of CODEBOOK) {{
    const d = document.createElement("div");
    d.className = "cb-item";
    d.dataset.q = (c.label + " " + (c.definition || "")).toLowerCase();
    const parts = [];
    parts.push("<h3>" + esc(c.label) + "</h3>");
    if (c.definition) parts.push("<p>" + esc(c.definition) + "</p>");
    if (c.includes && c.includes.length)
      parts.push('<p><span class="k">包含：</span>' + esc(c.includes.join("；")) + "</p>");
    if (c.boundary)
      parts.push('<p><span class="k">边界：</span>' + esc(c.boundary) + "</p>");
    if (c.counterexample)
      parts.push('<p><span class="k">反例：</span>' + esc(c.counterexample) + "</p>");
    d.innerHTML = parts.join("");
    box.appendChild(d);
  }}
}}

function esc(s) {{
  const map = {{ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }};
  return String(s).replace(/[&<>"]/g, (m) => map[m]);
}}

function toJSONL() {{
  const lines = [];
  for (const it of ITEMS) {{
    const s = state[it.id] || {{}};
    const p = s.primary || "";
    const labels = (p && p !== UNKNOWN) ? [p] : [];
    lines.push(JSON.stringify({{ id: it.id, primary: p || null, labels: labels,
                                note: s.note || "" }}));
  }}
  return lines.join("\\n") + "\\n";
}}

function precheck() {{
  const total = ITEMS.length;
  const labelled = ITEMS.filter((x) => (state[x.id] || {{}}).primary).length;
  const unknown = ITEMS.filter((x) => (state[x.id] || {{}}).primary === UNKNOWN).length;
  const notes = ITEMS.filter((x) => ((state[x.id] || {{}}).note || "").trim()).length;
  return {{ total, labelled, unknown, notes,
           unlabelled: total - labelled,
           ratio: labelled ? unknown / labelled : 0 }};
}}

let armed = false;   // second click required when UNKNOWN is suspiciously high

function done() {{
  const c = precheck();
  armed = false;
  $("out").value = toJSONL();
  $("fname").textContent = PREFIX + "_" + (($("who").value || "X").trim() || "X") + ".jsonl";

  $("sum").innerHTML =
    "<div><b>" + c.total + "</b><span>总条数</span></div>" +
    "<div><b>" + c.labelled + "</b><span>已标</span></div>" +
    "<div><b>" + c.unlabelled + "</b><span>未标</span></div>" +
    "<div><b>" + (c.ratio * 100).toFixed(0) + "%</b><span>UNKNOWN 占比</span></div>";

  const w = [];
  if (c.unlabelled > 0) {{
    w.push('<p class="bad">还有 <b>' + c.unlabelled + '</b> 条未标。未标的条目两边都会写 null，'
         + 'kappa 会把它当成"一致"从而虚高——所以先标完，或者显式把它们标成 UNKNOWN。</p>');
  }}
  if (c.ratio > 0.3) {{
    w.push('<p class="soft">UNKNOWN 占已标的 <b>' + (c.ratio * 100).toFixed(0) + '%</b>。'
         + '弃权是合法答案，但两人都大量弃权同样会把 kappa 推高——确认这是你想表达的意思。</p>');
  }}
  if (!w.length) {{
    w.push('<p class="ok">全部已标，UNKNOWN 占比 '
         + (c.ratio * 100).toFixed(0) + '%，可以导出。</p>');
  }}
  $("warn").innerHTML = w.join("");

  $("dl").disabled = c.unlabelled > 0;
  $("dl").textContent = c.unlabelled > 0 ? "先标完再导出"
    : (c.ratio > 0.3 ? "确认下载（UNKNOWN 偏高）" : "确认下载");
  $("fillun").style.display = c.unlabelled > 0 ? "" : "none";
  $("fillun").textContent = "把剩下 " + c.unlabelled + " 条标为 UNKNOWN";
  $("jump").style.display = c.unlabelled > 0 ? "" : "none";

  $("dlg").showModal();
}}

function fillUnknown() {{
  for (const it of ITEMS) {{
    if (!(state[it.id] || {{}}).primary) {{
      state[it.id] = Object.assign({{}}, state[it.id] || {{}}, {{ primary: UNKNOWN }});
    }}
  }}
  save(); draw(); done();
}}

function download() {{
  const c = precheck();
  if (c.unlabelled > 0) return;
  if (c.ratio > 0.3 && !armed) {{
    armed = true;
    $("dl").textContent = "再点一次确认（UNKNOWN " + (c.ratio * 100).toFixed(0) + "%）";
    return;
  }}
  const blob = new Blob([toJSONL()], {{ type: "application/x-ndjson" }});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = PREFIX + "_" + (($("who").value || "X").trim() || "X") + ".jsonl";
  a.click();
  URL.revokeObjectURL(a.href);
}}

// --- wiring
initSelect(); initCodebook(); load(); draw();

$("sel").addEventListener("change", (e) => {{ setPrimary(e.target.value); }});
$("note").addEventListener("change", (e) => {{
  const id = ITEMS[i].id;
  state[id] = Object.assign({{}}, state[id] || {{}}, {{ note: e.target.value }});
  save();
}});
$("who").addEventListener("change", () => {{ load(); draw(); }});
$("next").onclick = () => {{ if (i < ITEMS.length - 1) {{ i++; draw(); $("sel").focus(); }} }};
$("prev").onclick = () => {{ if (i > 0) {{ i--; draw(); $("sel").focus(); }} }};
$("firstun").onclick = () => {{
  const n = ITEMS.findIndex((x) => !(state[x.id] || {{}}).primary);
  i = n < 0 ? ITEMS.length - 1 : n; draw(); $("sel").focus();
}};
$("exp").onclick = done;
$("dl").onclick = download;
$("fillun").onclick = fillUnknown;
$("jump").onclick = () => {{ $("firstun").click(); $("dlg").close(); }};
$("close").onclick = () => $("dlg").close();
$("q").addEventListener("input", (e) => {{
  const q = e.target.value.trim().toLowerCase();
  for (const d of $("cb").children)
    d.style.display = (!q || d.dataset.q.includes(q)) ? "" : "none";
}});

document.addEventListener("keydown", (e) => {{
  const tag = (e.target.tagName || "").toLowerCase();
  if (tag === "textarea" || tag === "input") return;
  if (e.key === "Enter") {{ $("next").click(); e.preventDefault(); }}
  else if (e.key === "ArrowDown" && !e.altKey) {{ $("next").click(); e.preventDefault(); }}
  else if (e.key === "ArrowUp" && !e.altKey) {{ $("prev").click(); e.preventDefault(); }}
  else if (e.key === "ArrowRight" && e.altKey) {{ $("next").click(); e.preventDefault(); }}
  else if (e.key === "ArrowLeft" && e.altKey) {{ $("prev").click(); e.preventDefault(); }}
}});
</script>
</body>
</html>
"""


def read_items(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def render(set_name, items, labels, codebook, out_path, prefix="ann"):
    html = TEMPLATE.format(
        set_name=set_name,
        prefix=prefix,
        items_json=json.dumps(items, ensure_ascii=False),
        labels_json=json.dumps(labels, ensure_ascii=False),
        codebook_json=json.dumps(codebook, ensure_ascii=False),
        n_classes=len(labels),
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return os.path.getsize(out_path)


def codebook_cards(taxonomy):
    """The five fields that matter to an annotator, in the order they read them."""
    cards = []
    for e in taxonomy.get("clusters") or []:
        if e.get("label"):
            cards.append({"label": e["label"],
                          "definition": e.get("definition") or "",
                          "includes": e.get("includes") or [],
                          "boundary": e.get("boundary") or "",
                          "counterexample": e.get("counterexample") or ""})
    return cards
