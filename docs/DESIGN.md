# Design: technical route, pipeline internals, decisions, roadmap

This document owns the *reasoning*: what this project is, how the pipeline is
built, why each choice was made, and what is deliberately not built. Measured
numbers live in [`README.md`](../README.md#measured-results) and are not repeated
here — a second copy would drift on the first edit.

Language rule: English is the source of truth for every document and comment.
If a translation is ever added it is a `-zh_cn` twin of the same file with
identical structure and numbers, never a rewrite.

- [1. What this is](#1-what-this-is)
- [2. Technical route](#2-technical-route)
- [3. Pipeline stages](#3-pipeline-stages)
- [4. Component map](#4-component-map)
- [5. Decision log](#5-decision-log)
- [6. Evaluation methodology](#6-evaluation-methodology)
- [7. Roadmap](#7-roadmap)
- [8. Operating procedure](#8-operating-procedure)

## 1. What this is

A **corpus discovery tool** for complaint logs: it takes unlabelled support
conversations, finds the complaint categories that actually occur in them, and
emits a per-conversation multi-label file plus per-cluster evidence for naming.

It is explicitly **not** a trained classifier and **not** a production
categoriser. The reason is in [§2](#2-technical-route): you cannot train a
categoriser before you know what the categories are, and the categories in a
ticket system are usually wrong.

Position in the intended architecture:

```
 phase 0  DISCOVER   this project: clusters + evidence -> a taxonomy a human signs off
 phase 1  LABEL      an LLM applies that taxonomy to a few thousand conversations
 phase 2  CLASSIFY   a small supervised model serves production traffic
 phase 3  MAINTAIN   re-run phase 0 quarterly to find new categories
```

Only phase 0 exists. [§7](#7-roadmap) records what is planned and what was
rejected with evidence.

## 2. Technical route

**Unsupervised discovery on frozen embeddings, plus a tiny hand-labelled kNN for
the complaint/non-complaint decision. No gradient descent anywhere.**

Two facts force this shape:

1. **No labels exist at the start.** The output of this stage *is* the label set.
   Supervised training cannot bootstrap itself.
2. **Cluster IDs are unstable across runs** (measured earlier in the project:
   two runs on the same corpus with different seeds agreed at ARI 0.73–0.88 on
   synthetic data). A cluster id is a position, not a category — it may organise
   the corpus today and mean something else tomorrow.

Consequences that show up throughout the pipeline:

- The embedding model is **frozen**. Swapping it is a configuration change, not a
  training run, and it is chosen by measurement (throughput + cross-lingual
  agreement), not by leaderboard rank.
- The only object computed from the user's data is a `k x d` centroid matrix
  (70 KB at `k=46, d=384`). There is no checkpoint, no optimiser state, no epoch.
- Everything downstream of the clustering is about producing **evidence for a
  human** (representative samples, top terms, multi-label counts), because the
  output of phase 0 is a decision, not a prediction.

## 3. Pipeline stages

Ordering matters; several stages exist only because of a failure in the one
before them. Shapes assume `N` conversations, `S` sentences, `U` unique
sentences (`U <= S`), `d` embedding dim.

The runtime prints stage markers `[1] [1.5] [1.6] [2] [2.5] [3] [4.2] [4.5] [4]`;
they group onto the rows below rather than mapping one-to-one.

| # | Stage | In → Out | Mnemonic / knobs | Failure it prevents |
|---|---|---|---|---|
| 1 | Ingest user side | jsonl → `(cid, text)` | `iter_convs`, `user_turns` | agent turns are template text and dominate every centroid |
| 2 | PII mask | text → text with `<ID> <AMT> <DURATION> <N> <DATE>` | `clean`, 16 regexes | one order number per document makes the vector about that number |
| 3 | Sentence split | text → `S` sentences | `split_sentences`, `_hard_wrap`, per-language `min_len` | long documents average their topics away; punctuation-less text never splits |
| 4 | Conjunction split | sentence → 1–2 sentences | `CONJ_ZH`, `CONJ_EN`, `TRAIL_CONN` | a two-complaint sentence lands between two centroids |
| 5 | Cue prefilter *(optional)* | `S` → `S'` | `--cue-mode`, `looks_like_complaint` | non-complaint traffic eats clusters — **only used when no `--seeds`** |
| 6 | Boilerplate filter | `S'` → `S''`, plus `appeal_phrases.json` | `--df-max`, `--df-min-support` | one template line becomes a mega-cluster (measured 48.4% of sentences) |
| 7 | Near-dup dedupe | `S''` → `U` representatives + `rep_of` map | `minhash_sig`, `dedupe_near`, `--no-dedupe` | paraphrase variants of the same template survive stage 6 |
| 8 | Embed | `U` → `X` (`U x d`, float32) | `build_embedder`, `--embed-backend/--embed-model/--embed-prefix` | — (the only step that wants a GPU or an API) |
| 9 | Mean-centre + L2 | `X` → unit-norm `X` | `--no-center` | anisotropic spaces compress distances until surface form wins |
| 10 | Seed kNN veto | `X` + seeds → keep-mask | `--seeds`, `--knn-k`, `--knn-thresh` | enquiries and chit-chat labelled as complaints |
| 11 | k-means++ | `X[kept]` → `C` (`k x d`), labels | `kmeans`, `--k`, `--k-max`, `--kmeans-cap` | — |
| 12 | Expand labels | kept labels + `rep_of` → `labels_all` (`S`, `-1` = filtered) | — | collapses 50 duplicates to 1 representative then loses 49 conversations |
| 13 | Conversation multi-label | `labels_all` → `{cluster: n}` per conversation | — | one conversation can raise several complaints |
| 14 | Document label | whole conversation → nearest centroid | `--primary-mode`, `l2norm`, `_sqdist` | sentence-count argmax is a coin flip on long documents |
| 15 | Orphan rescue | conversations with no labels → nearest centroid, kNN veto | `--no-rescue` | filters may remove every sentence of a conversation |
| 16 | Write | → `clusters.json`, `conv_labels.jsonl`, `appeal_phrases.json` | `--samples-per-cluster` | — |

Two shape details worth knowing before touching the code:

- k-means is **fitted** on a subsample (`--kmeans-cap`, default 40k sentences) but
  **assignment** always runs over every kept sentence.
- `labels_all` has length `S` (the pre-filter sentence count) and stores `-1` for
  sentences removed by stages 5–10. Multi-label aggregation skips `-1`.

## 4. Component map

| File | Owns |
|---|---|
| `cluster_complaints.py` | the whole pipeline; numpy + stdlib only, one optional dep (`fastembed`) imported lazily |
| `tools/import_cfpb.py` | CFPB database → labelled sample (and the docstring recording *where narratives actually are*) |
| `tools/separation_probe.py` | "is this label learnable from the text at all" — run before blaming the pipeline |
| `tools/eval_against_labels.py` | purity / NMI / per-cluster dominant label against any labelled slice |
| `tools/eval_cross_lingual.py` | parallel-pair agreement, per-cluster language mix |
| `tools/gpu_check.py` | torch/CUDA smoke test with correctness anchors |

There is no package, no service, no daemon: `pyproject.toml` sets
`[tool.uv] package = false` and every entry point is a script.

## 5. Decision log

Each entry: decision → alternative rejected → the evidence that settled it.
Reversibility is noted because some of these would be expensive to undo.

**D1 — Discover with clustering, interpret with an LLM. Never let the LLM
partition.**
Rejected: pairwise LLM comparison, or scoring every document against k centroids.
Arithmetic settles it: 1M documents x 200 centroids is ~2x10^8 calls, three orders
of magnitude more expensive than the whole embedding pass, and the partition
changes with the prompt. *Reversible.*

**D2 — Cluster sentences, aggregate to conversations.**
Rejected: cluster whole conversations. A multi-turn log averages several topics
into one vector; measured on the synthetic fixture, double-issue conversations
could carry a second label in 0/18 cases without the split and 5–7/18 with it.
*Reversible but re-tuning `--k` becomes necessary.*

**D3 — k-means for a known category count, not HDBSCAN.**
Rejected: HDBSCAN as the default. HDBSCAN is the right tool when `k` is unknown
and densities vary, and it should be the choice for pure exploration; k-means was
kept because the operating procedure is "fit `k` to the taxonomy size we expect,
then name clusters", and because a fixed `k` makes run-to-run comparison possible.
*Reversible; `--k auto` already covers the "no idea" case crudely.*

**D4 — Seed kNN instead of a keyword regex for the complaint/non-complaint
decision.**
Rejected: grow the cue regex. Measured on 40 hard cases, the regex scores
F1 ≈ 0.11 in both languages while a 24-seed kNN scores 0.84 / 0.69 (stub) and
0.88 / 0.86 (MiniLM). *Reversible; the regex remains as the no-seeds fallback.*

**D5 — The two filters are mutually exclusive, not stacked.**
Rejected: cue ∪ kNN as a "high recall" ensemble. Measured: the union is worse than
kNN alone (F1 0.62 vs 0.81) because the regex's false positives survive it. Hence
`--cue-mode auto` disables the cue filter whenever `--seeds` is given.
*Reversible via `--cue-mode prefilter`.*

**D6 — Keep CPU-only embeddings as the default path.**
Rejected: require a GPU. k-means on 1M x 1024 vectors assigns at ~1.3M documents/s
on one core, so the clustering never needs a GPU; only embedding does, and a
0.22 GB ONNX model on CPU is fast enough for the corpora this is aimed at
(measured throughput in README). `--embed-backend api` covers GPU servers.
*Reversible.*

**D7 — Multilingual embeddings from the `paraphrase-multilingual` family, not
`e5-large`, for mixed corpora.**
Rejected: `intfloat/multilingual-e5-large` as the default. It scores best on
English, but on a parallel Chinese/English corpus it separates by language: the
measured cross-lingual gap between "same sentence, other language" and "unrelated
sentence" is 0.017 for e5-large versus 0.061 for MiniLM, and mean-centring does
not repair it. *Reversible: it is a flag.*

**D8 — Emit both a multi-label view and one document label.**
Rejected: pick one. They answer different questions and measurably disagree on
long text: finer chunking improves topic discovery and degrades single-label
assignment. `labels` (per sentence) and `primary` (whole conversation) coexist for
that reason. *Reversible.*

**D9 — Do not train anything in phase 0.**
Rejected: fine-tune an embedding model on cluster pseudo-labels before validating
the taxonomy. That freezes clustering errors into a model and, per the CFPB
experiment, would encode a label set that is only ~45% recoverable from text even
under supervision. *Reversible; phase 2 is where training belongs.*

## 6. Evaluation methodology

Three tools, three questions. Numbers live in README §Measured results.

| Question | Tool | What it must show |
|---|---|---|
| Did the clusters recover the real categories? | `eval_against_labels.py` | purity **and** NMI; purity alone inflates with `k` |
| Does the embedder actually align Chinese and English? | `eval_cross_lingual.py` | parallel-pair agreement, not just "clusters look mixed" |
| Is the label set a function of the text at all? | `separation_probe.py` | majority baseline vs unsupervised vs supervised ceiling |

Rules the project follows when reporting:

1. **Always report the majority-class baseline next to purity.** On a balanced
   25-class sample the baseline is 0.04; on a production corpus it is 0.4.
2. **Purity and NMI together, or neither.** Purity rises mechanically with `k`.
3. **State whether the sample is balanced.** The shipped CFPB sample is 200/class
   by construction; production corpora are long-tailed, so purity will differ.
4. **A supervised ceiling is the honest upper bound.** If a nearest-centroid
   classifier on the same embeddings barely beats the baseline, the label set —
   not the pipeline — is the problem, and that verdict belongs in the README.

## 7. Roadmap

Status is "delivered" only when the acceptance command passes, not when the code
is written. Nothing here is scheduled; this is the agreed order.

| # | Item | State | Acceptance |
|---|---|---|---|
| R1 | Centroid persistence: `--save-model model.npz` writing `C`, `mu`, `k`, embed model/prefix, cue mode, cluster summaries | **built** (19 KB for k=10/d=384) | `--apply-model` on the same corpus reproduces the fit run's labels: measured **500/500 identical, ARI 1.000** |
| R2 | Incremental assignment: `--apply-model` path that skips dedupe/k-means | **built** | held-out transfer: a 403-dialogue model labelling 97 unseen dialogues agrees with a full refit at **ARI 0.874** |
| R3 | Cluster naming: `tools/name_clusters.py` feeding `clusters.json` samples to an LLM, emitting `taxonomy.json` (`{id, label, definition, boundary, counterexample}`) | **built** (`llm_naming.py` shared with the pipeline's `--name-clusters` and the merge tool; per-entry evidence, `--resume`, `--check` gate) | every cluster gets a name and ≥1 boundary example; a human signs off — **measured 10/10 named, 10/10 pass validation on the 500-dialogue run** |
| R3a | Procedural-utterance filter: `procedural.py` + `--drop-procedural`, exporting every dropped line with its reason | **built** (rules for credentials/PII, bare names, acknowledgements) | removes the non-complaint clusters naming exposed — **measured 34% dropped, ARI 0.207 → 0.240, procedural share 38% → ~28%**; the residual needs a semantic discriminator, not more regex |
| R3b | Semantic discriminator: `tools/make_proc_seeds.py` builds a complaint-vs-procedure seed set (hand-written + the samples of clusters naming found procedural) driving `--seeds` kNN | **built** | removes the residual the rules miss — **measured ARI 0.244, NMI 0.416, purity 0.450, and 0 of 10 clusters procedural**; costs 12% of conversations their sentence-level labels, and does not compound with R3a |
| R3c | Threshold calibration: `tools/calibrate_thresh.py` sweeps `--knn-thresh` per corpus, reporting accuracy next to the orphan share it costs | **built** | proves the threshold is corpus-specific rather than assuming it — **zh optimum 0.40 (ARI 0.244), en optimum 0.50 (ARI 0.337)**; the procedure transfers even though the number does not |
| R4 | Taxonomy-based labelling: `tools/make_labels.py` producing `(conversation, labels)` for a few thousand items, with the LLM applying *definitions*, not cluster ids | **built** (abstention allowed, invalid-name retry, agreement + truth-mapped ARI reported) | ≥90% agreement with human spot-checks on 10% of items — **unverified** (no human); measured 500/500 valid parses, 46% abstention, ARI 0.445 vs truth on labelled items |
| R4a | Codebook: one page per class with name / definition / inclusion / boundary / precedence / examples, validated by double annotation | **partly built** (`tools/make_codebook.py` emits the five fields + a precedence worksheet; `tools/kappa.py` computes overall/per-class kappa and confusion cells) | Cohen's kappa ≥ 0.7 on 100–200 double-annotated items — **unverified, needs two humans**; model self-consistency measures 0.913 (macro 0.850), which is an upper bound, not a substitute |
| R5 | Classifier: `tools/train_classifier.py` (logistic regression on embeddings first, SetFit second) with per-class F1 and a confusion matrix | **built** (softmax regression, stratified k-fold out-of-fold, majority and nearest-centroid baselines, saved weights) | beats the phase-0 `primary` label on held-out data by a margin worth deploying — **measured accuracy 0.810 vs 0.754 nearest centroid and 0.220 majority, macro-F1 0.522**; labels are cluster-derived, so the deployable number needs independent annotation |
| R6 | Unknown-class handling: threshold on classifier confidence, route low-confidence to humans | **built** (`tools/route_unknown.py`: OOF threshold curve, `--simulate-unknown` holdout test with both softmax and centroid-distance scores, and a `--model` apply mode) | a new complaint type is routed to humans instead of being forced into an old class — **only partly met**: distance-to-centroid at 0.30 routes 79% of a distinctive unseen class and 36% of a semantically adjacent one at 82% coverage; softmax confidence alone reaches 18–38%, so novelty routing needs the distance score plus human review of `UNKNOWN` |

Explicitly **not planned**:

- **HDBSCAN as the default.** Useful for pure exploration and listed as an
  alternative in D3, but it does not fit the "fit the expected taxonomy size"
  procedure. Reach for it when `k` is genuinely unknown.
- **A torch/sentence-transformers embedding backend.** `fastembed` (ONNX) already
  covers local CPU inference and the `api` backend covers torch servers; adding a
  third path buys no capability. `--extra torch` stays installed for GPU work.
- **Training the embedder in this repo.** Belongs to phase 2 and needs phase 1
  labels first (D9).

## 8. Operating procedure

**First run on a new corpus**

1. `separation_probe.py` on any labelled slice you have (even 300 items). If the
   labels are not recoverable from text, stop: the taxonomy needs fixing first.
2. Run with `--cue-mode off` if the corpus is already known to be complaints;
   otherwise build 30–200 seeds and pass `--seeds`.
3. `--k` from the expected taxonomy size; cross-check the silhouette table printed
   by `--k auto`.
4. Read `clusters.json`: 20–50 samples per cluster, ask "is this one thing?".
   Merge or split, re-run. This loop, not the model, is where the taxonomy comes
   from.
5. Tune in this order: `--df-max` → `--knn-thresh` → `--k`. The first two change
   what enters clustering; changing `k` before them just reshuffles noise.

**Recurring runs**

- Full refit when the corpus shifts materially (new product line, new channel,
  new intake form).
- Until R1/R2 are built there is **no incremental path**: every run re-embeds and
  re-clusters from scratch. Budget accordingly.
- Keep `out/<run>/` — `clusters.json` is the evidence record for a delivered
  taxonomy and is what R3 would consume.

**What breaks silently**

| Symptom | Likely cause |
|---|---|
| one cluster holds ≫30% of sentences | boilerplate not filtered (`--df-max` too high, or templates are unique per conversation) |
| clusters split cleanly by language | embedding model is not cross-lingually aligned (see D7), or `--no-center` is set |
| every conversation gets 4+ labels | chunks are too short / `k` too large relative to the corpus |
| most conversations unlabelled | cue filter too strict; use `--seeds` or `--cue-mode off` |
| purity ≈ baseline on a labelled slice | the label set is not in the text — run `separation_probe.py` |
