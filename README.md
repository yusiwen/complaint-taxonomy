# complaint-taxonomy

Cluster complaint conversation logs (customer support chats) into complaint
categories, in **Chinese and English**.

No GPU is required for the clustering itself: k-means assignment on 1M x
1024-dim vectors runs at roughly 1.3M documents/second on a single CPU core.
Embeddings can come from a local ONNX model on CPU, from an HTTP endpoint, or
from a GPU service -- the pipeline does not care.

The project is managed by [uv](https://docs.astral.sh/uv/): dependencies live in
`pyproject.toml`, the resolved set in `uv.lock`, and the interpreter in
`.python-version`. Nothing is installed into the host Python.

This README covers *what runs and what it measured*. The technical route, the
reasoning behind each design choice (with the alternatives that were rejected),
the module map and the roadmap to a production classifier live in
[`docs/DESIGN.md`](docs/DESIGN.md).

- [What it does](#what-it-does)
- [Quick start](#quick-start)
- [Environment](#environment)
- [Embedding backends](#embedding-backends)
- [Bilingual support](#bilingual-support)
- [How it works](#how-it-works)
- [Parameters](#parameters)
- [Measured results](#measured-results)
- [Evaluating your own corpus](#evaluating-your-own-corpus)
- [Output files](#output-files)
- [Limitations](#limitations)
- [Repository layout](#repository-layout)
- [Design and roadmap](docs/DESIGN.md) — route, decisions, component map, phases

## What it does

Support chat logs break naive clustering in four specific ways. This pipeline
addresses each one explicitly.

| Problem | Symptom | Handling |
|---|---|---|
| **P1 Boilerplate** | "Thanks for contacting support" / "please handle this or I will complain" swallow an entire cluster (48.4% of sentences in a measured ablation) | customer turns only, document-frequency filter, MinHash near-duplicate dedupe; the dropped template lines are exported as `appeal_phrases.json` because they carry the *appeal* signal |
| **P2 Multi-label** | one conversation raises several complaints; a hard partition splits it arbitrarily | cluster **sentences**, split on conjunctions, aggregate to conversation level as multi-label with counts |
| **P3 Non-complaint traffic** | enquiries and chit-chat get clustered as complaints | embedding kNN over hand-labelled seed sentences; the keyword regex is only a fallback |
| **P4 Orphaned conversations** | every sentence of a conversation was filtered away, so it loses its label | rescue by full-text nearest centroid, vetoable by the same kNN |

Output: clusters with representative sentences and top terms -- designed to be
handed to an LLM (or a human) for naming as
`{label, definition, boundary condition, counterexample}`, plus a per-conversation
multi-label file.

## Quick start

```bash
uv sync --extra embed          # creates .venv from pyproject.toml + uv.lock
```

**On real dialogue data.** No corpus is committed — every one of them is a download plus
a conversion. One command fetches the demo set (40 MB, cached under
`~/.cache/complaint-taxonomy/`, no model needed):

```bash
uv run python tools/get_demo_corpus.py        # 500 ABCD dialogues -> data/public/
uv run python cluster_complaints.py \
    --input data/public/abcd_500_en.jsonl --out out/demo \
    --embed-backend fastembed --cue-mode off --k 10 --kmeans-algo bisect
uv run python tools/eval_against_labels.py \
    --labels data/public/abcd_500_labels.jsonl \
    --conv-labels out/demo/conv_labels.jsonl | head -6
```

That measured purity **0.444**, NMI **0.423**, ARI **0.286** against a majority baseline
of 0.126. Add `--translate --endpoint ...` to the fetch command for the Chinese half; that
is the corpus the cross-lingual and filtering numbers below were measured on.

**Offline smoke test, no model and no network** (`--stub` swaps in a lexical
placeholder embedder; it exercises segmentation, filtering, dedupe and multi-label
aggregation, and says nothing about semantics):

```bash
uv run python cluster_complaints.py --input data/public/abcd_500_zh.jsonl \
    --stub --k auto --out out/smoke
```

**Your own logs**, against an OpenAI-compatible embedding endpoint:

```bash
uv run python cluster_complaints.py \
    --input /path/to/logs.jsonl --out out/run1 \
    --embed-url http://127.0.0.1:8000/v1/embeddings \
    --embed-model BAAI/bge-m3 \
    --seeds data/seeds_zh.jsonl --k auto
```

Input format -- one JSON object per line, one conversation each:

```json
{"id": "c1", "messages": [{"role": "user", "content": "..."},
                          {"role": "agent", "content": "..."}]}
{"id": "c2", "user_text": "..."}
```

Both shapes are accepted; the loader also understands `conversation`, `turns`,
`from`/`speaker`, and `customer_text`.

## Environment

| Item | Value |
|---|---|
| manager | `uv` (lock file committed, interpreter pinned by `.python-version`) |
| Python | 3.12 (resolved from the system interpreter, no host install) |
| base deps | `numpy` only -- k-means, silhouette, MinHash, kNN are hand-written |
| `--extra embed` | `fastembed` (ONNX + tokenizers) for local CPU embeddings |
| `--extra torch` | `torch==2.12.1` (+cu130, ~4.4 GB of wheels) |

```bash
uv sync                               # pipeline only (numpy)
uv sync --extra embed                 # + fastembed, the normal setup
uv sync --extra embed --extra torch   # + CUDA build, verified on aarch64 + CUDA 13
uv run python tools/gpu_check.py      # proves the GPU actually computes
uv run ruff check .                   # lint (ruff is in the dev group)
```

Notes:

- Dependencies never touch the host Python: everything lands in `./.venv`
  (202 MB without torch, 4.6 GB with it).
- `torch` is pinned to **2.12.1** deliberately: that is the build verified on
  this box. It comes from plain PyPI -- `manylinux_2_28_aarch64` wheels that pull
  `cuda-toolkit`, `nvidia-cublas`, `nvidia-cudnn-cu13`, `nvidia-cusparselt-cu13`
  etc. as separate packages. No PyTorch index URL or `--extra-index-url` needed.
- The pipeline itself does not need torch; the extra exists for GPU work and for
  swapping the ONNX embedder for sentence-transformers. numpy, torch and
  fastembed were verified to coexist in one process.

Verified with `tools/gpu_check.py`:

```
torch   2.12.1+cu130   cuda build 13.0   cudnn 92000
device  <gpu name>  cc=12.1  mem=130.6 GB  sm=48
anchor  ones@ones = 4096 (exact)
matmul  4096x4096 bf16   1.5 ms/iter   89 TFLOP/s
anchor  bf16 gemm vs fp32 gemm, same inputs: rel err 2.1e-03
```

One trap worth knowing when writing your own GPU smoke tests: a random
bf16 dot product of length 4096 legitimately differs from an fp64 reference by
~10% *relative* error (|sum| ~ sqrt(n) while sum|terms| ~ n), so that comparison
looks like a broken kernel when nothing is wrong. Compare bf16 against fp32 on
the *same* inputs instead, and add an exact anchor such as `ones @ ones`.

## Embedding backends

| Backend | Flag | Where it runs | Notes |
|---|---|---|---|
| fastembed | `--embed-backend fastembed` | CPU, in-process | ONNX-quantised sentence-transformers models, no torch |
| api | `--embed-backend api` | GPU server / cloud | `bge-m3`, `Qwen3-Embedding`, anything OpenAI-compatible |
| stub | `--stub` | nowhere | character n-gram hashing; verifies plumbing only |

Measured on this machine (Cortex-X925 aarch64, CPU only, 20 cores). F1 is the
seed-kNN complaint filter on 40 hand-written hard cases, best threshold per model:

| Model | dim | Size | Throughput | zh F1 | en F1 |
|---|---|---|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 0.22 GB | **177 sent/s** | 0.88 | 0.86 |
| `paraphrase-multilingual-mpnet-base-v2` | 768 | 1.0 GB | 70 sent/s | 0.88 | 0.85 |
| `intfloat/multilingual-e5-large` | 1024 | 2.24 GB | 22 sent/s | 0.85 | **0.93** |
| stub (lexical placeholder) | -- | -- | 60k sent/s | 0.84 | 0.80 |

Cross-lingual behaviour is measured on the ABCD parallel benchmark (500 dialogues
translated to Chinese, see [Measured results](#measured-results)): MiniLM keeps
translation-equivalent Chinese and English together (pair agreement **0.776**), while
e5-large splits a mixed corpus by language (2/8 clusters contained both). The earlier
MASSIVE single-sentence check measured 0.795 — see [`docs/ARCHIVE.md`](docs/ARCHIVE.md).

**Recommendation**

- Mixed Chinese + English corpus: **`paraphrase-multilingual-MiniLM-L12-v2`**.
  It is 8x faster than e5-large and it is the only family tested here that keeps
  zh/en in the same clusters.
- English-dominant corpus, quality over speed: `multilingual-e5-large`, which
  scores highest on English (0.93) but at 22 sentences/s -- 300k sentences take
  about 4 hours instead of 30 minutes.
- `mpnet-base-v2` buys nothing over MiniLM at 2.7x the throughput cost.

**Why e5-large splits by language.** Its vectors are strongly anisotropic:
measured cross-lingual cosine between a Chinese complaint and its English
counterpart is 0.847, against 0.830 for two *unrelated* complaints -- only a
0.017 gap. Within-language surface similarity is larger than the cross-lingual
semantic gap, so k-means splits by language. Mean-centring the vectors does not
fix it. The `paraphrase-multilingual` models show a 0.061 gap on the same probe
(0.192 vs 0.131), which is why they stay aligned.

E5 models are also trained with asymmetric prefixes. The pipeline applies
`passage: ` automatically for `intfloat/*e5*` models (`--embed-prefix` overrides).

## Bilingual support

Language handling is per sentence, not per file: the CJK ratio of each sentence
decides whether the Chinese or English rules are applied. A mixed-language
corpus therefore needs no flag. Three language-specific pieces exist:

| Component | Chinese | English |
|---|---|---|
| Sentence boundaries | `。！？；` + newline | `.!?;` + whitespace |
| Conjunction splitting | 而且 并且 同时 另外 还有 顺便 以及 再加上 | and also, plus, additionally, furthermore, on top of that, as well as, besides that |
| Cue keywords (P3 fallback) | 投诉 退款 破损 … | complain refund broken late missing overcharged … |

The minimum sentence length is per language as well (Chinese 3 chars, English 6):
six *characters* is a fragment in English but a complete command in Chinese
("叫我起床"). A single threshold silently dropped 135 of 800 Chinese utterances
in the MASSIVE test.

**The embedding model must be multilingual**, and it must actually align the two
languages -- see the table above. Do not use an English-only model
(`all-MiniLM`, `bge-small-en`): it clusters English well and Chinese randomly.

Shipped data:

```
data/seeds_zh.jsonl  data/seeds_en.jsonl          complaint vs non-complaint seeds
data/seeds_proc_zh.jsonl  data/seeds_proc_en.jsonl  procedure seeds (the better filter)
data/abcd_flow_coarse.json                        10 flows -> 4 coarse groups
data/public/                                      generated corpora (git-ignored)
```

## How it works

```
conversations
   │  customer-side turns only, PII masked to placeholders
   ▼
sentences  ── cue/keyword prefilter (only when no --seeds)
   │
   ├─ document-frequency filter (drop template lines, export them)
   ├─ MinHash near-duplicate dedupe → representatives
   ▼
embeddings (representatives only)  ── seed kNN veto (P3)
   ▼
mean-centre → L2 normalise → k-means++ (k auto by silhouette, or fixed)
   │
   ├─ expand labels back to every original sentence
   ├─ aggregate to conversation multi-label (with counts)
   ├─ document-level `primary` label (whole conversation, nearest centroid)
   └─ orphan rescue for conversations left without any label
```

Four details that are easy to get wrong and are handled here:

- **L2 normalisation before k-means.** k-means minimises Euclidean distance, so
  without normalisation cluster membership follows vector norm -- long documents
  and boilerplate clump together regardless of meaning.
- **Labels are expanded back after dedupe.** Collapsing 50 occurrences of a
  sentence to one representative is what makes it cheap, but without the reverse
  mapping 49 conversations silently lose their label.
- **Newlines are sentence boundaries.** Whitespace normalisation must not
  collapse them, or a closing line glues onto the preceding complaint and forms
  an almost-unique hybrid that survives every template filter.
- **Punctuation-less text needs a hard wrap.** CFPB narratives measure 301 chars
  with zero periods and zero commas; without a word-boundary fallback a whole
  document becomes one vector and its topics are averaged away (measured: 2.4
  chunks/doc before, 6.5 after).

## Parameters

| Flag | Default | Meaning |
|---|---|---|
| `--k` | `auto` | cluster count, or let silhouette choose |
| `--k-max` | 60 | upper bound when `--k auto` |
| `--embed-backend` | `api` | `api` \| `fastembed` \| `stub` |
| `--embed-model` | backend-specific | `bge-m3` (api) / `paraphrase-multilingual-MiniLM-L12-v2` (fastembed) |
| `--embed-url` | env `EMBED_URL` | endpoint for the `api` backend |
| `--embed-prefix` | auto | prefix per text; `passage: ` for e5-family, none otherwise |
| `--embed-cache` | `~/.cache/fastembed` | where the fastembed backend keeps downloaded ONNX weights. **Must be persistent** — fastembed's own default is `<TMPDIR>/fastembed_cache`, which is a scratch directory that gets pruned, so the weights would be silently re-downloaded (or an offline run broken) |
| `--seeds` | – | labelled sentences `{text,label}`; enables the kNN stage |
| `--knn-k` | 7 | neighbours in the vote |
| `--knn-thresh` | 0.4 | complaint-weight share required. **Stub = 0.40, real models = 0.45–0.60**; sweep per corpus |
| `--cue-mode` | `auto` | `auto`: cue off when `--seeds` is given (measured: cue + kNN < kNN alone) |
| `--primary-mode` | `doc` | `doc`: embed the whole conversation for the single document label (0.29 vs 0.19 purity); `mean`: average the sentence vectors, cheaper |
| `--df-max` | 0.05 | drop sentences appearing in more than this share of conversations; 0 disables |
| `--df-min-support` | 20 | absolute floor for `--df-max`, protects small corpora |
| `--no-center` | off | disable mean centring before normalisation |
| `--no-dedupe` | off | disable MinHash dedupe |
| `--no-conj-split` | off | disable conjunction splitting (double complaints lose a label) |
| `--no-rescue` | off | disable orphan rescue |
| `--kmeans-cap` | 40000 | max sentences used to fit k-means (assignment still uses all) |
| `--samples-per-cluster` | 12 | representatives written per cluster |
| `--stub` | off | alias for `--embed-backend stub` |

## Measured results

All numbers below were produced on this machine with `--embed-backend fastembed`
(`paraphrase-multilingual-MiniLM-L12-v2`, 384d).

### Datasets, and what each one is good for

No public corpus is simultaneously bilingual, dialogue-shaped and complaint-
labelled. The workable set:

| Dataset | Language | Shape | Labels | Licence | Use it for |
|---|---|---|---|---|---|
| [**ABCD**](https://github.com/asappresearch/abcd) | en | real agent↔customer dialogues, 10,042 (8,034 train) | 10 `flow` + 96 `subflow` — a genuine complaint taxonomy | MIT, 37 MB direct download | **the main benchmark**: dialogue shape *and* learnable labels |
| CFPB complaints | en | single-turn narratives, 5,000 sampled | 25 `Issue` (intake-form bookkeeping) | public | the counter-example: what an unlearnable label set looks like |
| MASSIVE | 51 incl. zh | single-utterance commands | 60 intents | CC-BY-4.0 | historical: cross-lingual alignment (see `docs/ARCHIVE.md`) |
| [Chinese-Ambiguous-Reference](https://github.com/ygan/Chinese-Ambiguous-Reference) | zh | real service dialogues, 1,103 | none | MIT, 2.1 MB | smoke-testing the Chinese code path |
| JDDC 2.1 | zh | 246k real JD support sessions | none | application only | the best Chinese corpus, if procurement is possible |

Importers: `tools/import_abcd.py`, `tools/import_cfpb.py`,
`tools/import_chinese_cs.py`; the parallel benchmark is built by
`tools/translate_abcd.py` + `tools/make_parallel_benchmark.py`. MASSIVE is no longer
imported — see [`docs/ARCHIVE.md`](docs/ARCHIVE.md).

### ABCD, 8,034 real customer-service dialogues

Pipeline at `k=10`, `--cue-mode off`: purity **0.481** (majority baseline 0.109),
NMI **0.409**, coverage 100%.

Learnability probe on the same embeddings — this is the number that matters:

| label set | classes | baseline | unsupervised | supervised ceiling |
|---|---|---|---|---|
| ABCD `flow` | 10 | 0.109 | 0.388 | **0.800** |
| ABCD `subflow` | 96 | 0.037 | 0.319 | **0.735** |
| CFPB `Issue` | 25 | 0.040 | 0.288 | 0.449 |

ABCD's `flow` closes 78% of the gap over the majority baseline; CFPB's `Issue`
closes 43%. Same embedder, same probe, same pipeline — the difference is entirely
the label definition. `flow`/`subflow` are what the customer actually ran into
(return, defect, shipping, dispute); `Issue` is what an intake form recorded.

The Chinese path was verified end to end on 1,103 real zh dialogues (4,853 user
sentences, language detection zh=4853 / en=14, clusters readable); that corpus has
no category labels, so it qualifies the code path, not the taxonomy.

### Bilingual parallel benchmark, built from ABCD (500 dialogues × 2 languages)

`tools/translate_abcd.py` translates the customer and agent turns of 500 held-out
ABCD dialogues into Simplified Chinese with the local Qwen3.8-27B (thinking
disabled, PII masked before translation, one request per dialogue with numbered
lines, line count validated, per-turn fallback). Result: **500/500 pairs aligned,
0 turn-count mismatches**. Machine translation, so the Chinese side carries MT
noise — read the agreement rate as "alignment under realistic paraphrase".

Both languages are clustered **together** (one set of centroids — separate runs
would have unrelated cluster numbering):

| corpus | purity | NMI |
|---|---|---|
| en only (500) | 0.386 | 0.378 |
| zh only (500) | 0.414 | 0.373 |
| **merged en+zh (1,000)** | **0.413** | **0.365** |

Majority baseline 0.126. Chinese scores *at least* as well as English, so the
translation did not degrade the clusters; merging the two languages costs almost
nothing (0.413 vs 0.386/0.414).

**Cross-lingual agreement: 0.776** (388 of 500 pairs land in the same cluster),
7 of 8 non-empty clusters contain both languages.

For reference, single-utterance MASSIVE measures 0.795 on the same embedder. A
10–15-turn dialogue that has additionally been machine-translated loses about two
points — dialogue form and MT noise are not the problem they are often assumed to
be.

Learnability of the Chinese side alone: baseline 0.13, unsupervised 0.53,
supervised ceiling **0.732** (±0.044 over 5 folds) on 500 documents, against 0.800
for the full 8,034-document English set — a sample-size effect, not a language
effect (see the learning curve above).

### Choosing k: sweep ARI, never purity

`tools/k_sweep.py` runs the real pipeline once per `k` and reports purity, NMI, ARI
and seed stability side by side. On the 500-dialogue Chinese set (`k` 4→40, two
runs per `k` with disjoint k-means seed pools):

| k | purity | NMI | ARI | stability |
|---|---|---|---|---|
| 4 | 0.284 | 0.306 | 0.142 | **1.000** |
| 6 | 0.344 | 0.317 | 0.161 | **0.978** |
| 8 | 0.352 | 0.320 | 0.166 | 0.819 |
| 10 | 0.414 | 0.373 | 0.225 | 0.863 |
| 12 | 0.392 | 0.336 | 0.194 | 0.553 |
| 16 | 0.450 | 0.378 | 0.223 | 0.835 |
| 20 | 0.472 | 0.413 | **0.256** | 0.656 |
| 30 | 0.502 | 0.418 | 0.240 | 0.576 |
| 40 | **0.550** | 0.436 | 0.245 | 0.561 |

Purity keeps climbing to k=40 while ARI peaks at k=20 and then flattens: everything
past k=20 is over-segmentation. Choosing `k` by purity always walks that way.

The two columns called ARI mean different things and both matter: **accuracy ARI**
against the true labels peaks at k=20 (0.256), while **stability ARI** — how well a
second k-means seed (with a disjoint init pool) reproduces the partition — collapses
exactly where accuracy is highest, from ~0.86 at k=10 to ~0.66 at k=20 and ~0.56 at
k=40. The usable window is k=10: ARI 0.225 at stability 0.863. Above that you are
buying a few hundredths of ARI with partitions that no longer reproduce.

That trade-off is the honest answer to "is the clustering stable": at small `k` it
is essentially reproducible (k=4 and k=6 score 1.000 and 0.978), and it degrades as
the boundaries get finer, which is what weak separability looks like from the
inside. Same-seed reruns are bit-identical (ARI 1.000), so this is genuine seed
sensitivity, not a bug.

### Do the ARI levers combine? (measured, not assumed)

Three levers are usually proposed: more data, a coarser label axis, hierarchical
clustering. Measured on this pipeline (`data/abcd_flow_coarse.json` maps the 10
flows onto 4 groups):

| lever | change | accuracy ARI | verdict |
|---|---|---|---|
| L1 more data | 500 → 8,034 dialogues | 0.225 → **0.301** | works (corpus differs: zh-MT vs en, so partly confounded) |
| L2 coarser axis | same partition, 10 → 4 truth classes | 0.225 → 0.241 | **almost nothing**, while purity jumps 0.414 → 0.626 and NMI *drops* 0.373 → 0.288 |
| L3 top-down hierarchy | root k=4, then re-cluster each super-cluster | 0.136 vs the 4-group truth | **worse than flat**, and one super-cluster left 6 usable sentences, tripping the pipeline's guard |
| L3 bottom-up merge, free | k=20 partition, merge centroids into 4 groups | 0.191 vs the 4-group truth | better than top-down, **worse than not merging at all** (0.221) |
| L3 bottom-up merge, oracle | merge decisions taken from the true labels | **0.364** vs the 4-group truth | the ceiling merging could reach if the merge decisions were informed |

L2 is not a quality lever at all — it changes the evaluation axis, not the
partition. Coarsening inflates purity hardest (+51%) and moves ARI by 7%, which
makes purity on a coarse axis the most misleading number in the set. Do not coarsen
the label set to make a number look better; coarsen it because the business needs
coarse categories.

L3 is only worth doing bottom-up, and even then the free version does not pay: a
k=20 partition scores 0.221 against the coarse truth on its own, and mechanically
merging it into 4 groups drops that to 0.191. What the oracle row says is that the
*loss is in the merge decisions*, not in the granularity — 0.364 is waiting there
for whoever makes better decisions. That is a job for naming and human review, not
for linkage arithmetic, which is why `tools/merge_clusters.py` spends its effort on
a codebook draft (per-group name, definition, what is included and excluded) rather
than on pretending the geometry can decide the taxonomy.

None of the three fixes the real constraint: the partition is only seed-stable at
0.55–0.86 once k is fine enough to be useful, and the supervised ceiling on the
same text is ARI 0.519. The levers rearrange a search whose representation is the
limit.

### Reproducibility: same data, same k, same clusters

Two separate problems hid behind "the cluster ids are not reproducible", and they
need different fixes.

**Ids followed the init order.** k-means labels clusters by the order it happened to
seed them, so two runs that agree perfectly on the partition still disagree on which
cluster is "cluster 3" — anything keyed by cluster id (a saved model, a codebook, a
dashboard) silently breaks. Fixed in the pipeline: after the fit, clusters are
renumbered by size, with ties broken on the rounded centroid, so "cluster 0" is
always the biggest cluster and the numbering is content-determined. It prints
`[3.1] renumbered N clusters by size` when it does anything.

**The partition itself moved with the seed.** `--kmeans-algo bisect` is a divisive
(bisecting) k-means that never draws a random number: it splits the cluster with the
largest SSE along that block's top principal axis, and refines with Lloyd iterations.
Measured on the same corpus, k=10:

| config | stability across seeds | purity | NMI | ARI |
|---|---|---|---|---|
| `lloyd`, `--kmeans-inits 5` | 0.863 | 0.414 | 0.373 | 0.225 |
| `lloyd`, `--kmeans-inits 20` | 0.738 | 0.412 | 0.363 | 0.227 |
| `lloyd`, `--kmeans-inits 50` | 0.928 | 0.394 | 0.350 | 0.212 |
| `bisect` | **1.000** | 0.410 | 0.361 | 0.207 |

`bisect` is bit-identical across seeds (500/500 conversations get the same `primary`
id) for 0.018 of ARI, and more restarts help `lloyd` (0.863 -> 0.928 at 50) without
matching it. Use `bisect` when results must be reproducible or when a saved codebook
has to keep pointing at the same clusters; keep `lloyd` for exploratory runs.

A note on the stability column everywhere in this README: it is ARI between two runs
at the same `k` with disjoint init pools (`--stability-seed-step`, default 100, must
be at least `n_init` — overlapping pools make every run look perfectly stable). An
earlier version of `k_sweep.py` mirrored the pairs before computing ARI, which builds
a different contingency table and *understates* stability; the numbers here are the
corrected ones.

### Merging fine clusters into a coarse codebook

    uv run python tools/merge_clusters.py \
        --input data/public/abcd_500_zh.jsonl \
        --conv-labels out/ksweep_zh/k20/conv_labels.jsonl \
        --into 4 --out out/merge_zh \
        --labels data/public/abcd_500_labels.jsonl \
        --label-map data/abcd_flow_coarse.json \
        --llm-endpoint http://127.0.0.1:8000/v1/chat/completions

The default `--linkage leaders` clusters the fine centroids themselves with
bisecting k-means, which cannot chain and is deterministic. `--linkage avg` is kept
for comparison and is a documented trap: on document-mean vectors every pairwise
similarity sits in a narrow band, so average linkage grew one blob that swallowed 13
of 16 clusters and scored ARI 0.000 against the truth. Clustering leaders is
immune to that by construction.

Output is a draft, not a verdict: `groups.json` (sizes, member clusters, top terms,
dominant truth label, codebook fields) and `groups.md` for reading. With
`--llm-endpoint` the local 27B writes each group's entry, e.g.

    ## G0 — 订阅账户状态与费用查询  (204 段)
    用户咨询订阅服务的当前状态、费用明细、扣款记录或账户权益，并请求核实或处理相关账务问题。

The groups it proposes are sensible prose but scored 0.191 against the truth, so
treat the merge as a starting point and edit it. The measured ceiling for informed
merges (0.364) is the reason this tool exists; the gap between 0.191 and 0.364 is the
work a human or an LLM naming pass has to do.

### Incremental use: save the centroids, assign new data

`--save-model` writes a 19 KB npz (k=10, d=384) holding the centroids, the centre
vector, and the metadata: embedding model, its prefix, the resolved cue mode, the
`primary` mode, k, and the per-cluster summaries. `--apply-model` then labels new
conversations against it, skipping every corpus-relative stage (boilerplate DF
filter, near-duplicate dedupe, k-means) and re-running only the per-sentence path.

The centre vector is part of the model, not an implementation detail. Every vector
is mean-centred before normalisation at fit time; an apply run that skipped
centring, or centred on its own batch mean, would land in a different space while
all the shapes still matched — the assignments would be wrong with no error to
show for it. Same reason the cue mode is stored: it decides which sentences exist.

    # fit once
    cluster_complaints.py --input data/public/abcd_500_zh.jsonl --k 10 \
        --kmeans-algo bisect --cue-mode off --embed-backend fastembed \
        --out out/zh_k10 --save-model out/zh_k10.npz

    # label tomorrow's conversations with it
    cluster_complaints.py --input new_batch.jsonl --apply-model out/zh_k10.npz \
        --embed-backend fastembed --out out/zh_k10_new

Two measurements, both on the 500-dialogue corpus:

| check | result |
|---|---|
| round trip: fit 500 -> save -> apply the same 500 | **500/500 identical labels, ARI 1.000** |
| transfer: fit on 403, apply to 97 unseen, compare with a full 500 refit | **ARI 0.874** |

The round trip is exact because `bisect` is deterministic and the apply path
replicates the document-level assignment. The transfer number is not 1.0 and should
not be: the 403-dialogue model has different centroids from a 500-dialogue one. For
scale, on those same 97 held-out dialogues the three routes scored ARI 0.178
(apply), 0.179 (clustered alone) and 0.182 (full refit, rows restricted) against the
true flows — the transfer loses almost nothing in accuracy, which is what makes
incremental maintenance viable.

One trap worth recording: the first version of `--apply-model` defaulted the cue
filter on when the model file did not say otherwise. The cue regex keeps only ~2%
of Chinese sentences, so the batch produced 75 sentences for 97 conversations and
still reported success — every label came from the document-level fallback. Apply
now inherits the cue mode from the model file and prints which one it used.

### Naming the clusters (R3): taxonomy draft with its evidence

    # inline, right after clustering
    cluster_complaints.py --input corpus.jsonl --k 10 --kmeans-algo bisect \
        --out out/zh_k10 --save-model out/zh_k10.npz \
        --name-clusters --llm-endpoint http://127.0.0.1:8000/v1/chat/completions

    # or afterwards, on an existing run, with ground truth as evidence
    tools/name_clusters.py --run out/zh_k10 --resume \
        --llm-endpoint http://127.0.0.1:8000/v1/chat/completions \
        --labels data/public/abcd_500_labels.jsonl \
        --label-map data/abcd_flow_coarse.json

    # gate: exit 0 only if every entry has a label, a boundary and a counterexample
    tools/name_clusters.py --run out/zh_k10 --check

`llm_naming.py` at the repo root holds the shared prompt, the JSON contract and the
parse fallbacks; the pipeline, `name_clusters.py` and `merge_clusters.py` all use it
so the codebook pieces cannot drift apart. Each entry in `taxonomy.json` carries the
five fields the design requires — `label`, `definition`, `includes`, `boundary`,
`counterexample` — plus the samples the model was shown and, when `--labels` is
given, the cluster's dominant true label and its purity. A reviewer checks the claim
against the evidence instead of trusting a name.

Three decisions worth knowing about:

* **The language is decided once per run, not per cluster.** The first version
  sniffed each cluster separately, and a Chinese cluster full of customer names and
  brand words came back in English ("Customer Identity Verification Details").
  `sniff_lang` now runs over the whole corpus and is generous towards CJK, and a
  label that still comes back in the wrong script is flagged for review.
* **Ground truth never enters the prompt by default.** `--context-from-labels`
  exists, but off is the default: a codebook named with the answer key in the prompt
  agrees with the given taxonomy by construction and can no longer be used to check
  what the clustering actually found. Labels are for the reviewer.
* **A failed parse is an error, not an empty field.** `validate()` demands a label,
  a usable definition, a boundary, a counterexample and at least one inclusion; a
  hole nobody notices is worse than a visible failure.

What naming exposes is more useful than the names. On the 500-dialogue run, 4 of the
10 clusters came out as conversational procedure rather than complaint content:

| cluster | label | size | dominant truth (purity) |
|---|---|---|---|
| 0 | 订单支付与状态查询 | 456 | order_issue (22%) |
| 1 | 自助排查与等待反馈 | 369 | troubleshoot_site (77%) |
| 2 | 购物流程受阻 | 350 | purchase_dispute (29%) |
| 3 | 用户身份凭证信息 | 338 | shipping_issue (49%) |
| 4 | 客户姓名信息 | 325 | storewide_query (83%) |
| 5 | 用户确认与结束 | 307 | — |
| 6 | 服装商品属性咨询 | 263 | single_item_query (61%) |
| 7 | 否定反馈与无效确认 | 256 | purchase_dispute (100%) |
| 8 | 账户凭证遗忘或丢失 | 253 | account_access (48%) |
| 9 | 用户致谢与结束语 | 231 | single_item_query (100%) |

Clusters 3, 4, 5 and 9 — identity/credential exchange, names, confirmations, thanks
— hold **1,201 of 3,148 sentences (38%)**. The boilerplate DF filter cannot catch
them: every customer's name and account id is unique, so they never repeat often
enough to look templated. This is the scale of the problem the ARI numbers were
pointing at, and it is the next thing to fix: a procedural-utterance filter (PII
shapes, short confirmations, thanks) before clustering, or a sentence-level
classifier for "is this a complaint at all". Naming the clusters is what made the
size of that bucket visible.

### Procedural utterances: the 38% that were not complaints

Naming the clusters showed four of ten were procedure, not complaint — credential
exchange, bare names, acknowledgements, thanks. `procedural.py` drops them before
clustering, with `--drop-procedural` (off by default so earlier numbers stay
reproducible). Three rule families, applied after PII masking so the placeholders are
visible:

| reason | what it catches | example |
|---|---|---|
| `credential` | field labels plus identifying values: two or more of them anywhere in the utterance, or one with almost nothing else left after removal | `我的名字是 阿尔伯特·桑德斯，我的账户ID是 <N>` |
| `name` | a bare person name: two to four CJK characters with no function word, or a title-cased Latin name | `诺曼·布沙尔`, `Crystal Minh` |
| `ack` | acknowledgement, thanks, closing: nothing survives removing a closed vocabulary of ack words, thanks, particles and well-wishes | `好的，那就这样吧，谢谢你` |

Two decisions that came out of getting this wrong first:

* **A complaint cue protects the utterance.** `我无法访问我的账户` contains 账户, and the
  first version dropped it as credentials. Any utterance matching the complaint cue
  list skips the credential and name rules entirely.
* **Short forms are the easy half.** Dropping only the clear-cut short lines removed
  30% of sentences and left the cluster structure untouched — the procedural share
  stayed at 39% because the long instances (`我的名字是 X，我的邮箱是 Y，订单号 Z`) were
  still there. The density rule ("two or more identifying items in one utterance")
  is what moved the numbers.

Measured on the 500-dialogue set, `k=10`, `bisect`:

| config | sentences | purity | NMI | ARI | procedural clusters |
|---|---|---|---|---|---|
| `--primary-mode doc`, filter off | 4,627 | 0.410 | 0.361 | 0.207 | 38% |
| `--primary-mode doc`, filter on | 3,053 | **0.420** | **0.365** | **0.240** | ~28% |
| `--primary-mode mean`, filter off | 4,627 | 0.354 | 0.214 | 0.101 | — |
| `--primary-mode mean`, filter on | 3,053 | 0.362 | 0.242 | 0.141 | — |

Dropping 34% of sentences (1,574: 520 credential, 842 ack, 212 name) lifts ARI by 16%
in `doc` mode and by 40% in `mean` mode. The filter helps exactly where sentence
content drives the label; `doc` mode embeds the whole conversation, so its `primary`
barely notices, and the gain there comes from the centroids moving rather than from
the document vectors changing.

The remaining ~28% is the honest limit of rules: what survives is procedural content
wrapped in enough real words to look like a sentence. Closing that gap needs a
semantic discriminator, not more regex — the pipeline already has the mechanism
(`--seeds` with a complaint-vs-procedure seed set drives an embedding kNN, measured
F1 0.81 against 0.11 for the cue regex). Everything dropped is exported to
`procedural_dropped.json` with its reason, so the filter's decisions can be audited
instead of assumed.

### Semantic discriminator: seeds do what regex cannot

The rules leave about 28% procedure behind — content wrapped in enough real words to
look like a sentence. `SeedFilter` already existed in the pipeline (an embedding kNN
over labelled seed sentences); it needed a seed set aimed at procedure.

`tools/make_proc_seeds.py` builds one from two sources: hand-written coverage of the
three procedure families and of chit-chat, plus **the samples of the clusters naming
identified as procedural**. That second source is what matters — it puts the residual
long forms into the seed set, which is exactly what a keyword list cannot reach.
Result: `data/seeds_proc_zh.jsonl` (71 seeds: 16 complaint, 55 procedure/other) and
`data/seeds_proc_en.jsonl` (37: 16/21).

    tools/make_proc_seeds.py --run out/zh_k10 --procedure-clusters 0,2,9 \
        --base data/seeds_zh.jsonl --lang zh --out data/seeds_proc_zh.jsonl

    cluster_complaints.py --input corpus.jsonl --k 10 --kmeans-algo bisect \
        --seeds data/seeds_proc_zh.jsonl --knn-thresh 0.4 --out out/filtered

Same 500 dialogues, `k=10`, `bisect`, `--primary-mode doc`:

| filter | reps kept | purity | NMI | ARI | mean labels/doc | procedural clusters |
|---|---|---|---|---|---|---|
| none | 3,148 | 0.410 | 0.361 | 0.207 | 5.02 | 38% |
| rules (`--drop-procedural`) | 2,673 | 0.420 | 0.365 | 0.240 | 4.12 | ~28% |
| seeds, thresh 0.3 | 1,305 | 0.438 | 0.404 | 0.237 | — | — |
| **seeds, thresh 0.4** | 1,118 | **0.450** | **0.416** | **0.244** | **1.81** | **0%** |
| seeds, thresh 0.5 | 864 | 0.412 | 0.361 | 0.207 | — | — |
| rules + seeds 0.4 | 1,033 | 0.450 | 0.399 | 0.239 | — | — |

Naming the thresh-0.4 run is the qualitative proof — all ten clusters are complaint
topics, and the procedural ones are gone:

    支付与订阅异常 / 购物受阻与价格争议 / 订单商品误购或错发 / 用户负面情绪宣泄 /
    方案无效或问题未解决 / 网站访问速度过慢 / 寻求替代解决方案 /
    退款进度与规则咨询 / 账号凭证遗忘 / 账户基础信息查询

Two things to be honest about. **Aggressiveness:** thresh 0.4 keeps 35% of the
representatives, and 59 of 500 conversations (12%) lose every sentence — their
`primary` still comes from the document vector, but they contribute nothing to the
multi-label side. If per-complaint-point labels matter, that is a real cost, and
`mean clusters per doc` falling 5.02 -> 1.81 is the same fact from the other side.
**The filters do not compound:** rules plus seeds scores 0.239 against seeds alone at
0.244, because the rules have already taken the easy cases and the kNN then removes
some real complaints. Pick one. Seeds at 0.4 is the better single choice, and it is
also the one that leaves no procedural clusters standing.

### Is the threshold portable? No — calibrate it per corpus

`--knn-thresh 0.4` was tuned on the Chinese 500-dialogue set. Run the same sweep on the
English 8,034-dialogue corpus and the optimum moves:

    tools/calibrate_thresh.py --input data/public/abcd.jsonl \
        --labels data/public/abcd_labels.jsonl --seeds data/seeds_proc_en.jsonl \
        --thresh 0.2 0.3 0.4 0.5 0.6 0.7 --baseline --out out/calib_en

| corpus | thresh | kept | purity | NMI | ARI | orphans |
|---|---|---|---|---|---|---|
| zh 500 | none | 3,148 | 0.410 | 0.361 | 0.207 | 0% |
| zh 500 | 0.30 | 1,305 | 0.438 | 0.404 | 0.237 | 7.6% |
| zh 500 | **0.40** | 1,118 | **0.450** | **0.416** | **0.244** | 11.8% |
| zh 500 | 0.50 | 864 | 0.412 | 0.361 | 0.207 | 21.0% |
| en 8,034 | none | 45,219 | 0.411 | 0.377 | 0.272 | 0% |
| en 8,034 | 0.20 | 32,574 | 0.461 | 0.411 | 0.291 | 0.5% |
| en 8,034 | 0.30 | 28,084 | 0.459 | 0.405 | 0.284 | 2.2% |
| en 8,034 | 0.40 | 23,087 | 0.465 | 0.408 | 0.279 | 5.1% |
| en 8,034 | **0.50** | 16,773 | **0.523** | **0.449** | **0.337** | 12.6% |
| en 8,034 | 0.60 | 11,143 | 0.495 | 0.419 | 0.284 | 27.3% |
| en 8,034 | 0.70 | 6,926 | 0.426 | 0.349 | 0.235 | 44.9% |

The Chinese curve peaks at 0.40 and collapses by 0.50; the English curve keeps rising
through 0.50 and only collapses at 0.60. Using 0.40 on English leaves 21% relative ARI
on the table (0.279 against 0.337 at 0.50). **The threshold does not transfer, and
neither does the shape of the curve.**

What does transfer is the procedure: sweep the threshold on your own corpus and take
the best ARI *subject to an orphan budget*. On both corpora that lands on the last
point before the collapse, at an orphan share of about 12%, which is where accuracy
starts buying itself with conversations that no longer have any sentence-level label.
`tools/calibrate_thresh.py` reports both columns side by side for exactly that reason,
and prints the recommendation under a 15% default budget.

Two side findings from this sweep. The deterministic `bisect` algorithm costs more
accuracy on the larger English corpus (unfiltered ARI 0.272 against 0.301 for `lloyd`,
versus 0.207/0.225 on the Chinese one), so the reproducibility decision has a
corpus-dependent price. And the Chinese corpus rewards filtering less than English
(+18% ARI against +24%), which fits the fact that it is machine-translated: MT leaves
procedure looking more like complaint prose.

### Labelling with definitions (R4), the codebook (R4a), the classifier (R5)

**R4 — `tools/make_labels.py`.** Each conversation is labelled against the *definitions*
in `taxonomy.json`, never against cluster ids, so the labels stay meaningful after the
clustering changes. The prompt allows abstention, because a taxonomy built from
complaint clusters should refuse to label a non-complaint rather than force it into the
nearest category.

    tools/make_labels.py --input data/public/abcd_500_zh.jsonl \
        --taxonomy out/zh_final/taxonomy.json \
        --llm-endpoint http://127.0.0.1:8000/v1/chat/completions \
        --truth-labels data/public/abcd_500_labels.jsonl --out out/labels_zh

500 conversations: **500/500 valid, 0 invalid parses, 232 abstentions (46%)**, ARI
against ground truth 0.445 on the 266 labelled items — against 0.244 for the clustering
on the same corpus, which is the whole argument for the extra step.

The abstentions are the interesting part, and they break down by ground-truth flow:

| true flow | abstain rate | reading |
|---|---|---|
| single_item_query | 97% | a product question, not a complaint — correct refusal |
| storewide_query | 81% | same |
| shipping_issue | 68% | **a real complaint with no home in the codebook** |
| order_issue | 66% | **same** |
| subscription_inquiry | 44% | mixed |
| purchase_dispute | 15% | covered |
| account_access | 14% | covered |
| product_defect | 7% | covered |
| troubleshoot_site | 4% | covered |

So the abstain rate is two different signals at once. The top two rows are the taxonomy
doing its job. Rows three and four are a **coverage gap**: at k=10 the clustering never
produced distinct delivery and order-status categories, so the definitions have nothing
to offer and the model correctly declines. A 46% abstain rate is not a failure of R4; it
is R4 telling you the codebook is incomplete, which is exactly the information needed
before annotating at scale.

**R4a — `tools/make_codebook.py` and `tools/kappa.py`.** The codebook generator emits
the five fields the naming pass produced plus an explicit **precedence slot** per class,
and a worksheet of the 6 class pairs whose definitions sit closest in embedding space
(0.55–0.73 cosine), because precedence is the one field an LLM must not invent: two
classes that both "cover" a case, with nothing saying which wins, guarantee disagreement
later. Ranking those pairs by embedding similarity rather than word overlap was a fix —
lexical Jaccard on Chinese definitions produced pairs at 0.04–0.07 that ranked
arbitrarily, while the embedding ranking surfaces the ones an annotator will actually
confuse (方案无效或问题未解决 vs 寻求替代解决方案 at 0.73).

On inter-annotator agreement, measured two ways:

| comparison | raw agreement | Cohen's kappa |
|---|---|---|
| same model, independent runs (temperature 0 vs 0.5) | 0.936 | **0.913 [good]** |
| definition-driven labels vs the cluster each item came from | 0.340 | 0.280 |

The first is the useful proxy: the labelling *task* is performed reliably (macro over
classes 0.850, weakest class 0.508). The second is not an annotation-quality number at
all — its largest cells are `None -> <category>`, i.e. the 232 abstentions. The LLM and
the nearest-centroid assignment simply have different abstention policies, and the flow
breakdown above says the LLM is the one being right.

`kappa.py` is the real gate and it takes two human annotation files; **R4a's
acceptance criterion (kappa ≥ 0.7 on 100–200 double-annotated items) is unverified**, and
saying otherwise would be inventing a number. What is in place: the codebook, the
tool, and a runnable command for the two annotators who have to produce the data.

**R5 — `tools/train_classifier.py`.** Softmax regression on the document embeddings, 5-fold
stratified CV, out-of-fold predictions only, plus the two baselines that decide whether
it is worth deploying:

| method | accuracy | macro-F1 |
|---|---|---|
| majority class | 0.220 | — |
| nearest centroid (no training) | 0.754 | — |
| **softmax regression** | **0.810** | **0.522** |

Per class it is fine where there is data (账号凭证遗忘 0.96, 订单商品错购或瑕疵 0.87,
网站访问速度过慢 0.85) and zero where there are two or three examples — the macro-F1
is the honest headline, and the learning curve says ~2,000 labels is where this stops
being sample-starved.

One caveat that matters more than the accuracy: those labels were produced by a model
reading definitions that were themselves derived from the clustering's own samples, so
0.810 partly measures self-consistency. The mechanics are verified — train, hold out,
beat the baselines, save weights — but the deployable number has to come from labels
that were not derived from the clustering, meaning human annotation or a different
model family.

### R6 — routing what the model should not answer

A classifier that must always answer will fold a brand-new complaint type into whichever
old class looks nearest, and the taxonomy rots quietly: the novel class appears as noise
spread over three existing categories, the metrics stay flat because nobody defined
ground truth for it, and nobody looks. `tools/route_unknown.py` gives the classifier an
`UNKNOWN` answer and three jobs: pick the threshold, prove the routing works, and apply it.

    # threshold curve (out-of-fold), novelty test, production routing
    route_unknown.py --oof out/clf_zh/oof_items.jsonl
    route_unknown.py --input corpus.jsonl --labels labels.jsonl \
        --simulate-unknown 网站访问速度过慢
    route_unknown.py --input new.jsonl --model out/clf_zh/model.npz \
        --score distance --threshold 0.30 --out out/routed.jsonl

**The threshold trade, from out-of-fold probabilities** (268 labelled items):

| threshold | coverage | accuracy on answered | human queue | wrong auto-labels |
|---|---|---|---|---|
| 0.30 | 100% | 0.810 | 0% | 51 |
| 0.50 | 88% | 0.852 | 12% | 35 |
| **0.55** | **84%** | **0.871** | **16%** | 29 |
| 0.70 | 75% | 0.900 | 25% | 20 |
| 0.90 | 52% | 0.935 | 48% | 9 |

**Does an unseen class actually get routed?** Hold one class out of training entirely and
see where its items land — the acceptance criterion, tested instead of assumed:

| unseen class (items) | softmax 0.55: routed / kept coverage | distance 0.30: routed / kept coverage |
|---|---|---|
| 网站访问速度过慢 (29, distinctive) | 38% / 85% | **79% / 82%** |
| 账户基础信息查询 (11, adjacent to 账号凭证遗忘) | 18% / 88% | **36% / 82%** |

Two findings, one of which is a correction to the obvious plan:

**Softmax confidence is a familiarity score, not a novelty score.** At the operating point
that keeps 84% coverage it routes away only 18–38% of a class it has never seen, because
the softmax is confident whenever an item sits *near* any class — which is exactly what a
novel-but-related complaint does. **Cosine distance to the nearest class centroid is about
twice as good at the same coverage**, so that is the score to threshold on.

**Adjacent novelty still leaks.** At distance 0.30 the distinctive class routes 79% to
humans but the adjacent one only 36%: semantic neighbours of an existing class are, by
construction, close to it. No threshold fixes that, so the safeguard has to be the human
queue itself — review what lands in `UNKNOWN`, and re-cluster the absorbed cases each
quarter rather than trusting the router to notice. With 11 and 29 held-out items these
numbers are a direction, not a calibration.

### CFPB, 5,000 real complaints, 25 real `Issue` classes

Sample: 5,000 narratives from the CFPB consumer complaint database, 200 per Issue
class across the 25 most frequent classes (deliberately balanced). Coverage 100%,
97% of documents received 2+ sentence-level labels.

| method | purity | NMI |
|---|---|---|
| majority baseline (1/25) | 0.040 | – |
| **pipeline**, `--primary-mode mean` | 0.192 | 0.197 |
| **pipeline**, `--primary-mode doc` (default) | 0.194 | **0.250** |
| document-level clustering, no chunking (`tools/separation_probe.py`) | **0.288** | – |
| **supervised ceiling** (nearest-centroid, 5-fold, same embeddings) | **0.449** | – |

The decisive row is the last one: even a *supervised* classifier given these
labels only reaches 0.449. The `Issue` field is the intake-form bookkeeping
category, not a description of what happened -- two narratives with the same
Issue often read nothing alike, and two with different Issues often read
identically. Unsupervised recovers **64%** of what supervision achieves
(0.288/0.449), so the clustering is not the bottleneck; the label is.

Control on the same data with the 13 *product* classes present:

| method | purity |
|---|---|
| majority baseline | 0.200 |
| unsupervised (k=13) | **0.532** |
| supervised ceiling | 0.554 |

Unsupervised reaches 96% of the supervised ceiling here. Same pipeline, same
embedder -- the only change is that product type is genuinely expressed in the
text. **This is the effect to check before trusting any complaint taxonomy.**

Two operational consequences:

1. If you need **one category per conversation**, cluster documents directly
   (0.288) rather than sentences-then-aggregate (0.19–0.25). `--primary-mode doc`
   is the cheap approximation of that and is the default.
2. Chunking is for **multi-label issue extraction**, not for document labelling.
   Finer chunking improves topic discovery and degrades single-label assignment.

### Historical experiments

Three earlier experiment sets — the MASSIVE cross-lingual check, the cue-regex versus
seed-kNN filter study, and the synthetic structural ablation — are recorded in
[`docs/ARCHIVE.md`](docs/ARCHIVE.md), together with the list of tooling removed in the
cleanup and whether each result is still reproducible. Their conclusions are in the
design: the cue regex is not a filter, the DF filter breaks the boilerplate cluster,
conjunction splitting is what makes multi-label work, and the seed kNN is what decides
complaint versus non-complaint.

## Evaluating your own corpus

Before trusting any clustering, run the separation probe. It answers whether your
labels are a function of the text at all, using three numbers on the same
embeddings:

```bash
uv run python tools/separation_probe.py \
    --input my_logs.jsonl --labels my_labels.jsonl --label-field category
```

```
majority baseline        0.040          always predict the most common class
unsupervised (k= 25)     0.288          what clustering recovers, no labels used
supervised ceiling       0.449 +/- 0.007  nearest-centroid, 5-fold
```

- supervised ≈ baseline → the taxonomy is not in the text; clustering and training
  will both fail. Fix the label definition (CFPB's `Issue` is such a case).
- unsupervised ≈ supervised → clustering already extracts everything learnable;
  a trained classifier buys almost nothing (CFPB's `product` field: 0.532 vs 0.554).
- a wide gap → that gap is exactly what a supervised classifier is worth, and it
  is the case for labelling a few thousand items and training.

Other evaluation tools:

```bash
# purity / NMI / per-cluster dominant label against any labelled slice
uv run python tools/eval_against_labels.py --labels L.jsonl --conv-labels out/run/conv_labels.jsonl

# parallel-corpus cross-lingual agreement (needs a parallel corpus)
uv run python tools/eval_cross_lingual.py --labels L.jsonl --conv-labels out/run/conv_labels.jsonl
```

### How many labels do you need

`tools/learning_curve.py` answers that with a curve instead of an opinion: a fixed
held-out set, a real trained linear classifier (ridge on one-hot targets, closed
form, numpy only), and accuracy as training data grows. Both label sets below are
on the same 5,000 CFPB documents and the same frozen embeddings; only the label
definition changes.

CFPB `Issue`, 25 classes, held-out 1,000 (40/class), majority baseline 0.040:

| train n | linear acc | macro-F1 | centroid acc | unsupervised purity |
|---|---|---|---|---|
| 250 | 0.224 | 0.228 | **0.371** | 0.328 |
| 500 | 0.333 | 0.331 | **0.407** | 0.324 |
| 1,000 | 0.421 | 0.407 | **0.431** | 0.302 |
| 2,000 | **0.462** | 0.440 | 0.443 | 0.306 |
| 3,000 | **0.482** | 0.462 | 0.453 | 0.320 |
| 4,000 | **0.487** | 0.466 | 0.449 | 0.312 |

CFPB `product`, 13 classes, held-out 811 (62/class), baseline 0.094:

| train n | linear acc | macro-F1 | centroid acc | unsupervised purity |
|---|---|---|---|---|
| 190 | 0.351 | 0.278 | **0.499** | 0.463 |
| 732 | 0.518 | 0.405 | 0.538 | 0.469 |
| 1,318 | **0.562** | 0.431 | 0.538 | 0.523 |
| 2,133 | **0.575** | 0.436 | 0.541 | 0.541 |

Three readings:

1. **`Issue` saturates by ~2,000 labels.** 2,000 → 4,000 adds 0.025, 3,000 →
   4,000 adds 0.005. More labels of the same kind will not break 0.49; the label
   definition, not the volume, is the ceiling (cross-check with
   `separation_probe.py`).
2. **Below ~1,000 labels, a centroid classifier beats a trained linear model**
   (0.371 vs 0.224 at n=250). Training only pays once there are ~2,000 labels.
   Annotating a few hundred items and training on them is the wrong order.
3. **The better the label, the less training buys.** On `product` unsupervised
   purity is already 0.54 against a 0.58 trained ceiling; on `Issue` the gap is
   0.31 → 0.49, but the trained model still mislabels more than half the corpus.

### Auditing the label set itself

`tools/label_audit.py` answers "is this label set worth annotating more?" before
the annotation budget is spent. Three signals, one work list:

```bash
uv run python tools/label_audit.py \
    --input data/public/cfpb_issue.jsonl \
    --labels data/public/cfpb_issue_labels.jsonl --label-field label
```

On the CFPB `Issue` sample it reports that every class's members sit **closer to
another class's centroid than to their own** (own cohesion 0.21–0.31 versus
0.60–0.90 to the nearest neighbour), names the confusable pairs that should be
merged or given a boundary rule, and lists the classes whose held-out recall
collapses to ~0.2. That is what "the label definition is the bottleneck" looks
like as a measurement rather than an opinion.

A definition is only good enough when all six of these exist:

| Field | Why it is required |
|---|---|
| name | a term the business actually uses |
| definition | one sentence that must be true, decidable from the text alone |
| inclusion criteria | what has to appear in the text to qualify |
| exclusion / boundary | the neighbouring class it gets confused with, and how to tell them apart |
| precedence | which class wins when a case satisfies two definitions |
| 2–3 positive + 2–3 negative examples | what separates this class from its neighbour in practice |

Acceptance before scaling annotation: double-annotate 100–200 items and require
Cohen's kappa ≥ 0.7; then re-run `separation_probe.py` and require the supervised
ceiling to rise well above the gap measured before the rewrite, and
`learning_curve.py` to still saturate by ~2,000 labels. Only then spend the
annotation budget.

Rebuilding the CFPB sample (see `tools/import_cfpb.py` for why the official bulk
CSV cannot be used directly):
```bash
curl -sL -o /tmp/cfpb0.parquet \
  https://huggingface.co/datasets/davidheineman/consumer-finance-complaints-large/resolve/main/data/train-00000-of-00011.parquet
uv run --with pandas --with pyarrow python tools/import_cfpb.py \
  --source parquet --path /tmp/cfpb0.parquet --n 5000 --top-issues 25 --per-issue 200
```

## Output files

`out/<run>/clusters.json`

```json
{"k": 6,
 "n_unique_sentences": 28,
 "n_sentences": 130,
 "clusters": [{"cluster": 3, "size": 8, "share": 0.2857,
               "top_terms": ["refund", "still", "back"],
               "samples": ["..."]}]}
```

`out/<run>/conv_labels.jsonl` -- one line per conversation:

```json
{"id": "c0059", "labels": [{"cluster": 1, "n": 1}, {"cluster": 4, "n": 1}],
 "primary": 1, "rescued": false}
```

`n` is the number of that conversation's sentences assigned to the cluster, so
downstream code can require `n >= 2` for a confident label. `primary` is the
document-level label (see `--primary-mode`). `rescued` marks labels that came from
the orphan fallback.

`out/<run>/appeal_phrases.json` -- the template lines that were filtered out,
ordered by how many conversations contained them. These are the *appeal*
dimension ("I want a refund / I want compensation") and are worth a second,
orthogonal pass.

## Limitations

- **A label that is not expressed in the text cannot be recovered by any method.**
  Measured on CFPB: the `Issue` taxonomy tops out at 0.449 supervised accuracy on
  25 classes. Run `tools/separation_probe.py` on your own labels before blaming
  the pipeline.
- **The evaluation set is balanced by construction** (200 per class) while a real
  corpus is long-tailed. Purity on production data will look different; NMI is
  the more comparable number.
- **40 hand-written cases cannot rank models reliably.** All three real models
  land within 0.03 F1 of each other on that set, and the best threshold is
  selected on the same 40 cases it is scored on. Trust the cross-lingual metric
  and throughput instead.
- `--knn-thresh` and `--df-max` are corpus-dependent. Sweep them.
- A conversation is assumed to be about complaints. If your logs mix sales,
  logistics enquiries and complaints, the seed set must include the "other" class
  explicitly, or the kNN will force everything into a complaint cluster.
- `top_terms` is a crude lift x sqrt(count) measure on character n-grams
  (Chinese) and words (English). It is a hint for the naming step, not a label.
- The pipeline produces **clusters, not categories**. Naming them is a separate
  step: sample 20--50 representatives per cluster and ask a model for
  `{label, definition, boundary condition, counterexample}`.

## Repository layout

```
pyproject.toml                dependencies, extras (embed / torch), ruff config
uv.lock                       resolved dependency set, committed
.python-version               3.12
README.md                     what runs, how to run it, what it measured
docs/DESIGN.md                technical route, pipeline internals, decisions, roadmap
cluster_complaints.py         the pipeline (numpy + stdlib only)
data/
  seeds_zh.jsonl seeds_en.jsonl            complaint vs non-complaint seeds for --seeds
  seeds_proc_zh.jsonl seeds_proc_en.jsonl  procedure seeds (R3b, the better filter)
  abcd_flow_coarse.json       10 ABCD flows -> 4 coarse groups (lever experiments)
  public/                                  imported datasets (git-ignored)
tools/
  get_demo_corpus.py          fetch ABCD and build the demo corpus (nothing committed)
  import_cfpb.py              CFPB database -> labelled sample (see module docstring)
  import_abcd.py              ABCD customer-service dialogues -> en jsonl + flow labels
  import_chinese_cs.py        Chinese-Ambiguous-Reference -> zh jsonl (smoke test)
  translate_abcd.py           ABCD dialogues -> Simplified Chinese (local LLM)
  make_parallel_benchmark.py  merge en/zh pairs into one corpus for cross-lingual scoring
  k_sweep.py                  sweep k, report purity/NMI/ARI/stability, pick k by ARI
  merge_clusters.py           merge fine clusters into a coarse codebook (bottom-up)
  name_clusters.py            R3: name every cluster, write taxonomy.json + evidence
  make_proc_seeds.py          build a complaint-vs-procedure seed set for --seeds kNN
  calibrate_thresh.py         sweep --knn-thresh per corpus against its orphan cost
  make_labels.py              R4: label conversations against taxonomy definitions
  make_codebook.py            R4a: taxonomy.json -> codebook.md + precedence worksheet
  kappa.py                    R4a: Cohen's kappa, overall / per class / disagreements
  train_classifier.py         R5: softmax regression on embeddings + baselines + report
  route_unknown.py            R6: confidence/distance routing of novelty to humans
  eval_against_labels.py      purity / NMI / contingency against any labelled slice
  eval_cross_lingual.py       parallel-pair agreement + per-cluster language mix
  separation_probe.py         is the label learnable from the text at all?
  learning_curve.py           how many labels are needed before accuracy saturates
  label_audit.py              which classes are ambiguous, mergeable or non-textual
  gpu_check.py                torch/CUDA smoke test with real correctness anchors
out/                          run outputs (git-ignored)
.venv/                        uv-managed environment (git-ignored)
```
