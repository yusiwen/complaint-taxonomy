# Archive: experiments whose tooling has been removed

These results still explain why the pipeline looks the way it does, so they are kept.
What is gone is the code and data that produced them, because once a conclusion stopped
changing, keeping the machinery only made the repository harder to read.

Removed in this cleanup:

| removed | what it was | why | still reproducible? |
|---|---|---|---|
| `tools/make_demo.py` | generated a 130-conversation synthetic corpus for a structural self-test | real corpora (ABCD, CFPB) replaced it for anything that mattered | no — regenerate a fixture yourself if you need one |
| `tools/make_filter_cases.py` | built the 40 hand-written cue-filter hard cases | the cue regex lost to the seed kNN and stopped being a design decision | no |
| `tools/eval_filter.py` | scored a filter against those cases | same | no |
| `tools/compare_embedders.py` | one table per embedder: throughput plus filter P/R/F1 on those cases | its quality column measured the removed filter study, and it could not run once the fixtures were gone | no — but throughput is still visible in the embedder table in the README |
| `tools/import_massive.py` | imported the MASSIVE parallel corpus | cross-lingual alignment is now measured on the ABCD parallel benchmark, which is dialogue-shaped and in the same pipeline | yes — the dataset is still public (CC-BY-4.0), the importer is in git history |
| `data/public/cfpb_issue*.jsonl` | 5,000 sampled CFPB complaints | derived; `tools/import_cfpb.py` rebuilds it from the public database | yes |
| `data/public/zh_cs*.jsonl` | Chinese service-dialogue smoke-test corpus | served its purpose (proving the zh code path) | yes — `tools/import_chinese_cs.py` |
| `data/public/abcd_500*.jsonl` (all of them) | the 500-dialogue demo corpus, English and machine-translated Chinese, plus the merged parallel set | a download plus a conversion; committed artefacts drift from the script that makes them | yes — `tools/get_demo_corpus.py`, add `--translate` for the Chinese half |
| `data/filter_test_*.jsonl` | fixtures for the removed filter study | see `make_filter_cases.py` | no |

**Nothing under `data/public/` is committed.** Every corpus there is a download plus a
Python conversion, so it belongs in `.gitignore` like any other build product: a committed
copy drifts from the script that generates it, and it keeps weight in the history forever.
`tools/get_demo_corpus.py` rebuilds the demo set in one command (and caches the 37 MB ABCD
archive under `~/.cache/complaint-cluster/`).

What *is* committed: the hand-written seed sentences (`data/seeds*.jsonl`,
`data/seeds_proc*.jsonl`) and `data/abcd_flow_coarse.json`. Those are source data — they
are not generated from anything in this repository, and the pipeline reads them.

---

## MASSIVE zh-CN + en-US (cross-lingual, parallel corpus)

800 parallel pairs, 60 intents, k=20:

| metric | value |
|---|---|
| parallel-pair agreement (zh and en of one sentence in the same cluster) | **0.795** |
| clusters containing both languages | 19/20 |
| intent NMI | 0.694 |
| purity | 0.611 (majority baseline 0.146) |
| clean recoveries | `weather_query` 100%, `news_query` 98%, `play_music` 72%, `datetime_query` 76% |

0.795 was the direct, countable answer to "does the embedder actually align Chinese and
English". It was later superseded by the ABCD parallel benchmark (0.776 on 10–15-turn
dialogues that had also been machine-translated), which is the harder test and the one
the README now reports.

## Non-complaint filter, 40 hand-written hard cases

20 complaints written *without* any cue keyword, 20 non-complaints *containing* cue
keywords (policy questions, "has my refund arrived yet", ...).

| Embedder | Threshold | Chinese P / R / F1 | English P / R / F1 |
|---|---|---|---|
| cue regex alone | – | 0.12 / 0.10 / **0.11** | 0.15 / 0.10 / **0.12** |
| stub | 0.40 | 0.76 / 0.95 / 0.84 | 0.53 / 1.00 / 0.69 |
| MiniLM | 0.40 (default) | 0.68 / 0.95 / 0.79 | 0.61 / 1.00 / 0.75 |
| MiniLM | 0.50 zh / 0.60 en (best) | 0.83 / 0.95 / **0.88** | 0.79 / 0.95 / **0.86** |
| e5-large | 0.45 (best) | 0.74 / 1.00 / 0.85 | 0.87 / 1.00 / **0.93** |
| any | union (cue or kNN) | below kNN alone | below kNN alone |

Conclusions worth keeping:

- A keyword regex is useless on adversarial text in both languages (F1 ~0.11);
  the seed kNN is what makes P3 work.
- **Never stack the two stages.** The union is worse than kNN alone, because the regex's
  false positives survive it.
- The best threshold moves with the embedder: 0.40 for the lexical stub, 0.45–0.60 for
  real models. Sweep it on a holdout, not on the set you report.
- The response is a step, not a slope. e5-large measures 0.73 / 0.82 (zh / en) at 0.40,
  then 0.85 / 0.93 at 0.45, 0.50 and 0.55 alike. Do not interpolate.

## Structural ablation (synthetic sanity check)

130 synthetic conversations, 6 topics, `--stub` embedder. These validate **structure**,
not semantics.

| Configuration | max cluster share | conversations with 2+ labels | double-issue → 2 labels |
|---|---|---|---|
| no filtering, no dedupe, no conjunction split | 48.4% | 68% (inflated by the boilerplate cluster) | 0/18 |
| + document-frequency filter | 26.4% | 18% | 0/18 |
| + near-duplicate dedupe | 29.0% | 16% | 0/18 |
| + conjunction splitting (default) | 33.3% | 16% | 5/18 |
| + seed kNN | 25.9% | 24% | 7/18 |

The conclusions that survived into the design: the document-frequency filter is what
breaks the boilerplate mega-cluster (48% → 26%), conjunction splitting is what makes a
double complaint produce two labels (0/18 → 5/18), and the seed kNN then removes what the
DF filter cannot see.
