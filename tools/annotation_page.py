#!/usr/bin/env python3
"""Render a self-contained annotation page: dropdowns, autosave, JSONL export, zh/en UI.

Typing category names by hand is a transcription task, and transcription errors count as
disagreement in the kappa — the number then measures typing, not definitions. So the
labels become a dropdown (plus UNKNOWN), and the page writes exactly the JSONL that
`tools/kappa.py` reads.

Deliberately a single offline HTML file with no server and no network:

  * two annotators must not see each other's work, and a shared server would show both;
  * it works from a file:// path, on any machine, with no install;
  * answers live in localStorage keyed by annotator name, so a reload or a crash during a
    2-3 hour session loses nothing, and two people on one machine do not collide.

Language: the interface switches between Chinese and English, and the codebook shows the
English translation when `tools/translate_codebook.py` has added one. **The exported file
always carries the canonical Chinese label**, because that is what the codebook, the
labelling tool and the kappa compare on — English is a display alias only.

The template uses @@PLACEHOLDER@@ rather than str.format: an HTML file full of CSS and JS
braces needs every brace doubled under format(), which is exactly how template bugs get
introduced.
"""
import json
import os

TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>@@TITLE@@</title>
<style>
  :root {
    --bg: #0f1115; --card: #171a21; --line: #2a2f3a; --fg: #e6e8ee;
    --muted: #9aa3b2; --accent: #4c8dff; --ok: #3fb950; --warn: #d29922;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--fg);
         font: 15px/1.6 -apple-system, "Segoe UI", "Noto Sans SC", sans-serif; }
  header { position: sticky; top: 0; z-index: 5; background: #12151b;
           border-bottom: 1px solid var(--line); padding: 10px 16px;
           display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  header h1 { font-size: 15px; margin: 0; font-weight: 600; }
  .grow { flex: 1; }
  .bar { height: 6px; background: var(--line); border-radius: 3px; width: 150px;
         overflow: hidden; }
  .bar > i { display: block; height: 100%; background: var(--ok); width: 0%; }
  main { display: grid; grid-template-columns: 1fr 400px; gap: 16px; padding: 16px;
         max-width: 1440px; margin: 0 auto; }
  @media (max-width: 1000px) { main { grid-template-columns: 1fr; } }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
          padding: 16px; }
  .meta { color: var(--muted); font-size: 12px; display: flex; gap: 12px;
          align-items: center; }
  .text { white-space: pre-wrap; word-break: break-word; margin: 12px 0 16px;
          font-size: 15px; line-height: 1.75; max-height: 46vh; overflow: auto; }
  label.f { display: block; color: var(--muted); font-size: 12px; margin: 12px 0 4px; }
  select, input, textarea, button { font: inherit; color: var(--fg);
    background: #10131a; border: 1px solid var(--line); border-radius: 8px;
    padding: 8px 10px; }
  select { width: 100%; }
  select.set { border-color: var(--accent); }
  button { cursor: pointer; }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  button:disabled { opacity: .45; cursor: default; }
  .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .chip { font-size: 12px; color: var(--muted); border: 1px solid var(--line);
          border-radius: 999px; padding: 2px 8px; }
  .chip.ok { color: var(--ok); border-color: var(--ok); }
  .chip.warn { color: var(--warn); border-color: var(--warn); }
  .lang { display: flex; border: 1px solid var(--line); border-radius: 8px;
          overflow: hidden; }
  .lang button { border: 0; border-radius: 0; padding: 6px 10px; font-size: 12px;
                 background: transparent; }
  .lang button.on { background: var(--accent); color: #fff; }
  aside { position: sticky; top: 64px; align-self: start;
          max-height: calc(100vh - 84px); overflow: auto; }
  aside h2 { font-size: 13px; margin: 0 0 8px; color: var(--muted); font-weight: 600; }
  .cb-item h3 { font-size: 13px; margin: 12px 0 2px; }
  .cb-item h3 small { display: block; color: var(--muted); font-weight: 400;
                      font-size: 11.5px; margin-top: 1px; }
  .cb-item p { margin: 2px 0; font-size: 12.5px; color: #c3c9d5; }
  .cb-item .k { color: var(--muted); }
  code { background: #10131a; padding: 1px 5px; border-radius: 4px; font-size: 12px; }
  .blind { font-size: 12px; color: var(--muted); }
  dialog { background: var(--card); color: var(--fg); border: 1px solid var(--line);
           border-radius: 10px; max-width: 720px; }
  .sum { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;
         margin: 6px 0 2px; }
  .sum div { background: #10131a; border: 1px solid var(--line); border-radius: 8px;
             padding: 10px; text-align: center; }
  .sum b { display: block; font-size: 20px; }
  .sum span { font-size: 12px; color: var(--muted); }
  .warnbox p { margin: 8px 0; font-size: 13px; padding: 8px 10px; border-radius: 8px; }
  .warnbox .bad { background: rgba(255,80,80,.12); border: 1px solid #a33; color: #ffb3b3; }
  .warnbox .soft { background: rgba(210,153,34,.12); border: 1px solid var(--warn);
                   color: #f0d090; }
  .warnbox .ok { background: rgba(63,185,80,.12); border: 1px solid var(--ok);
                 color: #9be3a5; }
  textarea { width: 100%; font-family: ui-monospace, monospace; font-size: 12px; }
</style>
</head>
<body>
<header>
  <h1 id="h1"></h1>
  <span class="chip" id="pos"></span>
  <div class="bar"><i id="bar"></i></div>
  <span class="chip" id="count"></span>
  <span class="grow"></span>
  <input id="who" style="width:150px">
  <div class="lang">
    <button id="lang-zh">中文</button><button id="lang-en">EN</button>
  </div>
  <button class="primary" id="exp"></button>
</header>

<main>
  <section>
    <div class="card">
      <div class="meta">
        <span id="cid"></span><span id="chars"></span>
        <span class="chip" id="state"></span>
      </div>
      <div class="text" id="text"></div>

      <label class="f" id="lbl-cat" for="sel"></label>
      <select id="sel"></select>

      <label class="f" id="lbl-note" for="note"></label>
      <input id="note">

      <div class="row" style="margin-top:14px">
        <button id="prev"></button>
        <button id="next"></button>
        <span class="grow"></span>
        <button id="firstun"></button>
      </div>
      <p class="blind" id="blind"></p>
    </div>
  </section>

  <aside class="card">
    <h2 id="cb-title"></h2>
    <input id="q" style="width:100%;margin-bottom:8px">
    <div id="cb"></div>
  </aside>
</main>

<dialog id="dlg">
  <h2 id="dlg-title" style="margin-top:0"></h2>
  <div id="sum" class="sum"></div>
  <div id="warn" class="warnbox"></div>
  <div class="row" style="margin-top:12px">
    <button class="primary" id="dl"></button>
    <button id="fillun" style="display:none"></button>
    <button id="jump" style="display:none"></button>
    <button id="close"></button>
  </div>
  <details style="margin-top:12px">
    <summary class="blind"><span id="pv-label"></span><code id="fname"></code></summary>
    <textarea id="out" readonly style="height:160px;margin-top:8px"></textarea>
  </details>
</dialog>

<script>
const ITEMS = @@ITEMS@@;
const CODEBOOK = @@CODEBOOK@@;
const SET_ZH = @@SET_ZH@@;
const SET_EN = @@SET_EN@@;
const PREFIX = "@@PREFIX@@";
const HAS_EN = CODEBOOK.some((c) => !!c.label_en);
const UNKNOWN = "UNKNOWN";

const I18N = {
  zh: {
    title: "投诉标注 · " + SET_ZH,
    h1: "投诉标注 · " + SET_ZH,
    who: "标注员（A 或 B）",
    exp: "导出 JSONL",
    chars: "字",
    labelled: "已标",
    unlabelled: "未标",
    lblCat: "类别（下拉选择，选不准就选 UNKNOWN）",
    lblNote: "备注（可选，写你不确定的地方）",
    noteHint: "例如：像“订单状态查询”也像“物流时效”",
    prev: "← 上一条",
    next: "下一条 →",
    firstun: "跳到下一条未标",
    blind: "只用这份 codebook 判断，参考真值或互相商量都会让 kappa 失去意义。"
         + "进度自动保存在本机浏览器。快捷键：↑↓ 切换条目，Alt+←/→ 同，Enter 下一条。",
    cbTitle: "CODEBOOK（" + CODEBOOK.length + " 类）",
    filter: "过滤类别…",
    pick: "— 请选择 —",
    unknown: "UNKNOWN（没有类别适用）",
    dlgTitle: "导出检查",
    total: "总条数",
    unknownRatio: "UNKNOWN 占比",
    dlOk: "确认下载",
    dlBlocked: "先标完再导出",
    dlHigh: "确认下载（UNKNOWN 偏高）",
    dlArmed: "再点一次确认（UNKNOWN ",
    fillun: "把剩下 N 条标为 UNKNOWN",
    jump: "跳到第一条未标",
    close: "关闭",
    pv: "JSONL 预览（文件名 ",
    canon: "导出用的是中文类别名，与 codebook 一致。",
    warnUnlabelled: "还有 N 条未标。未标的条目两边都会写 null，kappa 会把它当成“一致”"
                  + "从而虚高——先标完，或者显式把它们标成 UNKNOWN。",
    warnUnknown: "UNKNOWN 占已标的 X%。弃权是合法答案，但两人都大量弃权同样会把 kappa "
               + "推高——确认这是你想表达的意思。",
    allDone: "全部已标，UNKNOWN 占比 X%，可以导出。",
    includes: "包含",
    boundary: "边界",
    counterexample: "反例",
  },
  en: {
    title: "Complaint annotation · " + SET_EN,
    h1: "Complaint annotation · " + SET_EN,
    who: "Annotator (A or B)",
    exp: "Export JSONL",
    chars: "chars",
    labelled: "labelled",
    unlabelled: "unlabelled",
    lblCat: "Category (pick from the list; choose UNKNOWN if nothing fits)",
    lblNote: "Note (optional — what made it hard)",
    noteHint: "e.g. reads like both 'Order status' and 'Delivery timing'",
    prev: "← Previous",
    next: "Next →",
    firstun: "Jump to next unlabelled",
    blind: "Judge from this codebook only. Checking the ground truth, or discussing with "
         + "the other annotator, makes the kappa meaningless. Progress is saved in this "
         + "browser. Keys: Up/Down or Alt+Left/Right to move, Enter for the next item.",
    cbTitle: "CODEBOOK (" + CODEBOOK.length + " classes)",
    filter: "Filter classes…",
    pick: "— choose —",
    unknown: "UNKNOWN (no category fits)",
    dlgTitle: "Export check",
    total: "total",
    unknownRatio: "UNKNOWN",
    dlOk: "Confirm download",
    dlBlocked: "Label everything first",
    dlHigh: "Confirm download (UNKNOWN high)",
    dlArmed: "Click again to confirm (UNKNOWN ",
    fillun: "Mark the remaining N as UNKNOWN",
    jump: "Jump to first unlabelled",
    close: "Close",
    pv: "JSONL preview (file ",
    canon: "The export carries the canonical Chinese labels.",
    warnUnlabelled: "N items are still unlabelled. A blank is written as null on both "
                  + "sides and the kappa would count it as agreement, inflating the number"
                  + " — finish them, or mark them UNKNOWN explicitly.",
    warnUnknown: "UNKNOWN is X% of what you labelled. Abstaining is legitimate, but two "
               + "annotators abstaining a lot also pushes kappa up — confirm this is what "
               + "you mean.",
    allDone: "Everything is labelled, UNKNOWN at X%. Ready to export.",
    includes: "Includes",
    boundary: "Boundary",
    counterexample: "Not this",
  },
};

const $ = (id) => document.getElementById(id);
let i = 0;
let lang = localStorage.getItem("ct-annot-lang") || "zh";
const state = {};
const t = (k) => (I18N[lang][k] !== undefined ? I18N[lang][k] : k);
const withNum = (k, n) => t(k).replace("N", n);
const withPct = (k, p) => t(k).replace("X", p);

function key() {
  return "complaint-taxonomy:" + SET_ZH + ":" + (($("who").value || "anon").trim() || "anon");
}
function load() {
  try {
    const raw = localStorage.getItem(key());
    if (raw) Object.assign(state, JSON.parse(raw));
  } catch (e) { console.warn("state load failed", e); }
}
function save() { localStorage.setItem(key(), JSON.stringify(state)); }

function esc(s) {
  const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" };
  return String(s === null || s === undefined ? "" : s).replace(/[&<>"]/g, (m) => map[m]);
}

function drawSelect() {
  const sel = $("sel");
  const cur = (state[ITEMS[i].id] || {}).primary || "";
  sel.innerHTML = "";
  const o0 = document.createElement("option");
  o0.value = ""; o0.textContent = t("pick");
  sel.appendChild(o0);
  for (const c of CODEBOOK) {
    const o = document.createElement("option");
    o.value = c.label;                                  // canonical, always Chinese
    o.textContent = (lang === "en" && c.label_en)
      ? c.label_en + " — " + c.label : c.label;
    sel.appendChild(o);
  }
  const ou = document.createElement("option");
  ou.value = UNKNOWN; ou.textContent = t("unknown");
  sel.appendChild(ou);
  sel.value = cur;
}

function drawCodebook() {
  const box = $("cb");
  box.innerHTML = "";
  for (const c of CODEBOOK) {
    const en = lang === "en" && !!c.label_en;
    const d = document.createElement("div");
    d.className = "cb-item";
    d.dataset.q = ((c.label || "") + " " + (c.definition || "") + " "
                   + (c.label_en || "") + " " + (c.definition_en || "")).toLowerCase();
    const def = en ? c.definition_en : c.definition;
    const inc = en ? c.includes_en : c.includes;
    const bd = en ? c.boundary_en : c.boundary;
    const ce = en ? c.counterexample_en : c.counterexample;
    const parts = [];
    const colon = lang === "en" ? ": " : "：";
    parts.push("<h3>" + esc(en ? c.label_en : c.label)
               + (en ? "<small>" + esc(c.label) + "</small>" : "") + "</h3>");
    if (def) parts.push("<p>" + esc(def) + "</p>");
    if (inc && inc.length)
      parts.push('<p><span class="k">' + t("includes") + colon + "</span>"
                 + esc(inc.join(lang === "en" ? "; " : "；")) + "</p>");
    if (bd) parts.push('<p><span class="k">' + t("boundary") + colon + "</span>"
                       + esc(bd) + "</p>");
    if (ce) parts.push('<p><span class="k">' + t("counterexample") + colon + "</span>"
                       + esc(ce) + "</p>");
    d.innerHTML = parts.join("");
    box.appendChild(d);
  }
  const q = $("q").value.trim().toLowerCase();
  for (const d of box.children)
    d.style.display = (!q || d.dataset.q.includes(q)) ? "" : "none";
}

function applyLang() {
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  document.title = t("title");
  $("h1").textContent = t("h1");
  $("who").placeholder = t("who");
  $("exp").textContent = t("exp");
  $("lbl-cat").textContent = t("lblCat");
  $("lbl-note").textContent = t("lblNote");
  $("note").placeholder = t("noteHint");
  $("prev").textContent = t("prev");
  $("next").textContent = t("next");
  $("firstun").textContent = t("firstun");
  $("blind").textContent = t("blind");
  $("cb-title").textContent = t("cbTitle");
  $("q").placeholder = t("filter");
  $("dlg-title").textContent = t("dlgTitle");
  $("close").textContent = t("close");
  $("jump").textContent = t("jump");
  $("pv-label").textContent = t("pv");
  $("lang-zh").classList.toggle("on", lang === "zh");
  $("lang-en").classList.toggle("on", lang === "en");
  $("lang-en").title = HAS_EN ? "" : "this packet has no English codebook";
  drawSelect(); drawCodebook(); draw();
  if ($("dlg").open) done();
}

function draw() {
  const it = ITEMS[i];
  $("pos").textContent = (i + 1) + " / " + ITEMS.length;
  $("cid").textContent = it.id;
  $("chars").textContent = it.text.length + " " + t("chars");
  $("text").textContent = it.text;
  const cur = state[it.id] || { primary: "", note: "" };
  $("sel").value = cur.primary || "";
  $("note").value = cur.note || "";
  $("sel").classList.toggle("set", !!cur.primary);
  $("state").textContent = cur.primary ? t("labelled") : t("unlabelled");
  $("state").className = "chip " + (cur.primary ? "ok" : "warn");
  const doneN = ITEMS.filter((x) => (state[x.id] || {}).primary).length;
  $("count").textContent = t("labelled") + " " + doneN + " / " + ITEMS.length;
  $("bar").style.width = (100 * doneN / ITEMS.length) + "%";
  $("prev").disabled = i === 0;
  $("next").disabled = i === ITEMS.length - 1;
}

function setPrimary(v) {
  const id = ITEMS[i].id;
  state[id] = Object.assign({}, state[id] || {}, { primary: v });
  save(); draw();
}

function toJSONL() {
  const lines = [];
  for (const it of ITEMS) {
    const s = state[it.id] || {};
    const p = s.primary || "";
    const labels = (p && p !== UNKNOWN) ? [p] : [];
    lines.push(JSON.stringify({ id: it.id, primary: p || null, labels: labels,
                               note: s.note || "" }));
  }
  return lines.join("\\n") + "\\n";
}

function precheck() {
  const total = ITEMS.length;
  const labelled = ITEMS.filter((x) => (state[x.id] || {}).primary).length;
  const unknown = ITEMS.filter((x) => (state[x.id] || {}).primary === UNKNOWN).length;
  const notes = ITEMS.filter((x) => ((state[x.id] || {}).note || "").trim()).length;
  return { total, labelled, unknown, notes, unlabelled: total - labelled,
           ratio: labelled ? unknown / labelled : 0 };
}

let armed = false;

function done() {
  const c = precheck();
  armed = false;
  $("out").value = toJSONL();
  $("fname").textContent = PREFIX + "_" + (($("who").value || "X").trim() || "X") + ".jsonl";
  const pct = (c.ratio * 100).toFixed(0);

  $("sum").innerHTML =
    "<div><b>" + c.total + "</b><span>" + t("total") + "</span></div>" +
    "<div><b>" + c.labelled + "</b><span>" + t("labelled") + "</span></div>" +
    "<div><b>" + c.unlabelled + "</b><span>" + t("unlabelled") + "</span></div>" +
    "<div><b>" + pct + "%</b><span>" + t("unknownRatio") + "</span></div>";

  const w = [];
  if (c.unlabelled > 0)
    w.push('<p class="bad">' + withNum("warnUnlabelled", c.unlabelled) + "</p>");
  if (c.ratio > 0.3)
    w.push('<p class="soft">' + withPct("warnUnknown", pct) + "</p>");
  if (!w.length)
    w.push('<p class="ok">' + withPct("allDone", pct) + "</p>");
  if (lang === "en") w.push('<p class="soft">' + t("canon") + "</p>");
  $("warn").innerHTML = w.join("");

  $("dl").disabled = c.unlabelled > 0;
  $("dl").textContent = c.unlabelled > 0 ? t("dlBlocked")
    : (c.ratio > 0.3 ? t("dlHigh") : t("dlOk"));
  $("fillun").style.display = c.unlabelled > 0 ? "" : "none";
  $("fillun").textContent = withNum("fillun", c.unlabelled);
  $("jump").style.display = c.unlabelled > 0 ? "" : "none";
  if (!$("dlg").open) $("dlg").showModal();
}

function fillUnknown() {
  for (const it of ITEMS) {
    if (!(state[it.id] || {}).primary)
      state[it.id] = Object.assign({}, state[it.id] || {}, { primary: UNKNOWN });
  }
  save(); draw(); done();
}

function download() {
  const c = precheck();
  if (c.unlabelled > 0) return;
  if (c.ratio > 0.3 && !armed) {
    armed = true;
    $("dl").textContent = t("dlArmed") + (c.ratio * 100).toFixed(0) + "%)";
    return;
  }
  const blob = new Blob([toJSONL()], { type: "application/x-ndjson" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = PREFIX + "_" + (($("who").value || "X").trim() || "X") + ".jsonl";
  a.click();
  URL.revokeObjectURL(a.href);
}

$("sel").addEventListener("change", (e) => setPrimary(e.target.value));
$("note").addEventListener("change", (e) => {
  const id = ITEMS[i].id;
  state[id] = Object.assign({}, state[id] || {}, { note: e.target.value });
  save();
});
$("who").addEventListener("change", () => { load(); draw(); });
$("next").onclick = () => { if (i < ITEMS.length - 1) { i++; draw(); $("sel").focus(); } };
$("prev").onclick = () => { if (i > 0) { i--; draw(); $("sel").focus(); } };
$("firstun").onclick = () => {
  const n = ITEMS.findIndex((x) => !(state[x.id] || {}).primary);
  i = n < 0 ? ITEMS.length - 1 : n; draw(); $("sel").focus();
};
$("exp").onclick = done;
$("dl").onclick = download;
$("fillun").onclick = fillUnknown;
$("jump").onclick = () => { $("firstun").click(); $("dlg").close(); };
$("close").onclick = () => $("dlg").close();
function setLang(next) {
  lang = next;
  localStorage.setItem("ct-annot-lang", next);
  applyLang();
}
$("lang-zh").onclick = () => setLang("zh");
$("lang-en").onclick = () => setLang("en");
$("q").addEventListener("input", () => drawCodebook());

document.addEventListener("keydown", (e) => {
  const tag = (e.target.tagName || "").toLowerCase();
  if (tag === "textarea" || tag === "input") return;
  if (e.key === "Enter") { $("next").click(); e.preventDefault(); }
  else if (e.key === "ArrowDown" && !e.altKey) { $("next").click(); e.preventDefault(); }
  else if (e.key === "ArrowUp" && !e.altKey) { $("prev").click(); e.preventDefault(); }
  else if (e.key === "ArrowRight" && e.altKey) { $("next").click(); e.preventDefault(); }
  else if (e.key === "ArrowLeft" && e.altKey) { $("prev").click(); e.preventDefault(); }
});

load(); applyLang();
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


def render(set_name_zh, items, codebook, out_path, prefix="ann", set_name_en=""):
    """Write one page.

    `items` are {"id", "text"}; `codebook` cards carry the Chinese fields and, when
    `tools/translate_codebook.py` has run, the `*_en` twins.
    """
    html = TEMPLATE
    subs = {
        "@@TITLE@@": "投诉标注 · " + set_name_zh,
        "@@ITEMS@@": json.dumps(items, ensure_ascii=False),
        "@@CODEBOOK@@": json.dumps(codebook, ensure_ascii=False),
        "@@SET_ZH@@": json.dumps(set_name_zh, ensure_ascii=False),
        "@@SET_EN@@": json.dumps(set_name_en or set_name_zh, ensure_ascii=False),
        "@@PREFIX@@": prefix,
    }
    for k, v in subs.items():
        html = html.replace(k, v)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return os.path.getsize(out_path)


def codebook_cards(taxonomy):
    """The five fields an annotator reads, in both languages when available."""
    cards = []
    # a committed codebook stores its entries under `classes`; a run taxonomy under
    # `clusters` — both are the same thing to the page
    for e in (taxonomy.get("clusters") or taxonomy.get("classes") or []):
        if not e.get("label"):
            continue
        card = {"label": e["label"],
                "definition": e.get("definition") or "",
                "includes": e.get("includes") or [],
                "boundary": e.get("boundary") or "",
                "counterexample": e.get("counterexample") or ""}
        for f in ("label", "definition", "boundary", "counterexample"):
            if e.get(f + "_en"):
                card[f + "_en"] = e[f + "_en"]
        if e.get("includes_en"):
            card["includes_en"] = e["includes_en"]
        cards.append(card)
    return cards
