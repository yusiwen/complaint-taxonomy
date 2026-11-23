#!/usr/bin/env python3
"""
Complaint conversation log clustering -- bilingual (Chinese / English).

Pure numpy + stdlib. No GPU, no torch: k-means assignment on 1M x 1024 vectors
takes about a second on one CPU core, so only the embedding step needs a GPU or
an API.

Three structural problems in support chat logs, and how each is handled:

  P1  Boilerplate sentences form a mega-cluster.
      -> user-side turns only, document-frequency filter, MinHash near-duplicate
         dedupe. Dropped template sentences are not discarded: they are exported
         to appeal_phrases.json because they carry the *appeal* signal.

  P2  One conversation raises several complaints, so a hard partition is wrong.
      -> cluster sentences (not conversations), split on conjunctions as well as
         punctuation, aggregate to conversation level as multi-label with counts.

  P3  Non-complaint traffic gets clustered as if it were a complaint.
      -> two-stage: an embedding kNN over hand-labelled seed sentences decides.
         The cue regex is only a fallback when no seeds are supplied, because
         measured on hard cases a cue regex alone scores F1 0.11 while the seed
         kNN scores F1 0.81, and the union of the two is worse than kNN alone.

Bilingual support: cue keywords, conjunction splitting and sentence splitting are
per-language, and the language of every sentence is detected individually, so a
mixed-language corpus works without a flag. The embedding model must be
multilingual -- see README.md.

Usage
  # offline self-test, no model and no network required
  python cluster_complaints.py --input data/demo_zh.jsonl --stub --k auto --out out/zh

  # real run against an OpenAI-compatible embedding endpoint
  python cluster_complaints.py --input logs.jsonl --out out/run1 \
      --embed-url http://127.0.0.1:8000/v1/embeddings \
      --embed-model BAAI/bge-m3 --seeds data/seeds_zh.jsonl --k auto

Input: jsonl, one conversation per line. Supported shapes:
  {"id": "c1", "messages": [{"role": "user"|"agent", "content": "..."}]}
  {"id": "c1", "user_text": "..."}
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request
import zlib

import numpy as np

from procedural import is_procedural  # noqa: E402

# Roles treated as the customer side. Agent/assistant turns are discarded: they
# are template text, highly repetitive, and they pollute every centroid.
USER_ROLES = {"user", "customer", "client", "客户", "用户"}
PLACEHOLDER = re.compile(r"<\w+>")


# --------------------------------------------------------------- ingest

def iter_convs(path):
    """Yield (conversation_id, user_side_text) for every line of a jsonl file."""
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] line {lineno} is not json, skipped", file=sys.stderr)
                continue
            yield obj.get("id", f"L{lineno}"), user_turns(obj)


def user_turns(obj):
    """Concatenate every customer-side turn of one conversation."""
    msgs = obj.get("messages") or obj.get("conversation") or obj.get("turns")
    if isinstance(msgs, list) and msgs:
        out = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or m.get("from") or m.get("speaker") or "").lower()
            if role in USER_ROLES:
                out.append(str(m.get("content") or m.get("text") or ""))
        if out:
            return "\n".join(out)
    for key in ("user_text", "customer_text", "user", "query", "text"):
        if obj.get(key):
            return str(obj[key])
    return ""


# --------------------------------------------------------------- cleaning

# Unique identifiers must not reach the embedding model: with one order number
# per document every vector is dominated by that token instead of by meaning.
PII_PATTERNS = [
    (re.compile(r"https?://\S+|www\.\S+"), " <URL> "),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), " <EMAIL> "),
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), " <PHONE> "),
    (re.compile(r"(?<!\d)\d{12,}(?!\d)"), " <ID> "),
    (re.compile(r"(?<![\d.])\d{1,3}(?:[,.]\d{3})+(?:\.\d+)?(?![\d.])"), " <AMT> "),
    (re.compile(r"\d+(?:\.\d+)?\s*(?:元|块|万元|人民币)"), " <AMT> "),
    (re.compile(r"(?:USD|\$|EUR|€|£)\s?\d+(?:[.,]\d+)?", re.I), " <AMT> "),
    (re.compile(r"\d+(?:\.\d+)?\s?(?:dollars|euros|pounds|bucks|yuan|rmb)\b", re.I),
     " <AMT> "),
    (re.compile(r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?"), " <DATE> "),
    (re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"), " <DATE> "),
    (re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s"
                r"\d{1,2}(?:st|nd|rd|th)?\b", re.I), " <DATE> "),
    (re.compile(r"\d+\s*(?:天|小时|分钟|个工作日)"), " <DURATION> "),
    (re.compile(r"\b\d+\s?(?:days?|hours?|hrs?|minutes?|mins?|weeks?|business days?)\b",
                re.I), " <DURATION> "),
    (re.compile(r"[「『\"“][^」』\"”]{2,20}[」』\"”]"), " <QUOTE> "),
    (re.compile(r"\"[^\"]{2,30}\""), " <QUOTE> "),
    # Any leftover bare number. Without this, two otherwise identical sentences
    # that differ only by a count produce different MinHash signatures and both
    # escape near-duplicate dedupe.
    (re.compile(r"(?<![<\w])\d+(?![>\w])"), " <N> "),
]

# Sentence boundaries. Chinese and Latin punctuation are handled separately so
# that decimals such as 3.5 do not split a sentence.
SENT_SPLIT_ZH = re.compile(r"(?<=[。！？；\n])")
SENT_SPLIT_EN = re.compile(r"(?<=[.!?;])\s+")
CJK = re.compile(r"[\u4e00-\u9fa5]")

# A sentence carrying two complaints must be split, otherwise its vector sits
# halfway between two centroids and belongs to neither cluster.
CONJ_ZH = re.compile(r"(?:而且|并且|同时|另外|还有|顺便|以及|再加上|再说了)")
CONJ_EN = re.compile(r"\s(?:and also|plus|additionally|furthermore|on top of that|"
                     r"as well as|besides that)\s", re.I)
# Splitting on a connector leaves it dangling on the left half ("... late and").
TRAIL_CONN = re.compile(r"(?:\s+(?:and|but|so|because|also|plus)|\s*(?:和|以及|还|就))$",
                        re.I)


def detect_lang(text):
    """Per-sentence language detection: 'zh' when CJK characters dominate."""
    stripped = re.sub(r"\s+", "", text)
    if not stripped:
        return "en"
    return "zh" if len(CJK.findall(stripped)) / len(stripped) >= 0.15 else "en"


def clean(text):
    """Mask identifiers and normalise whitespace, preserving newlines.

    Newlines are sentence boundaries: collapsing them into spaces would glue a
    closing line onto the preceding complaint and create an unsplittable,
    almost-unique hybrid sentence that survives every template filter.
    """
    for pat, repl in PII_PATTERNS:
        text = pat.sub(repl, text)
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r" *\n[ \n]*", "\n", text)
    return text.strip()


def split_by_lang(text):
    """Split a cleaned text into sentence candidates of both languages."""
    out = []
    for chunk in SENT_SPLIT_ZH.split(text):
        if not chunk:
            continue
        out.extend(p for p in SENT_SPLIT_EN.split(chunk) if p)
    return out


def _hard_wrap(s, max_len):
    """Break a long run-on segment at word boundaries.

    Needed for text with no punctuation at all: CFPB complaint narratives are
    typically one redacted blob (measured: 301 chars, zero periods, zero commas).
    Without this the whole document becomes a single vector and its topics are
    averaged away.
    """
    out = []
    while len(s) > max_len:
        cut = s.rfind(" ", max_len // 2, max_len)
        if cut <= 0:
            cut = s.rfind("\u3000", 1, max_len)
        if cut <= 0:
            cut = max_len
        out.append(s[:cut].strip())
        s = s[cut:].strip()
    if s:
        out.append(s)
    return out


def split_sentences(text, max_len=200, conj=True):
    """Sentence segmentation with an extra split on complaint conjunctions.

    The minimum length is per language: six *characters* is a fragment in English
    but a complete command in Chinese ("叫我起床" is 4 chars). Using one threshold
    silently drops a large slice of the Chinese corpus -- measured on MASSIVE, a
    6-char floor discarded 135 of 800 Chinese utterances against 0 of 800 English.
    """
    out = []
    for raw in split_by_lang(text):
        s = raw.strip()
        if not s:
            continue
        min_len = 3 if detect_lang(s) == "zh" else 6
        pieces = [s]
        if conj:
            pat = CONJ_ZH if detect_lang(s) == "zh" else CONJ_EN
            parts = [TRAIL_CONN.sub("", p.strip(" ，,、;；")).strip()
                     for p in pat.split(s)]
            parts = [p for p in parts if p]
            if len(parts) > 1 and all(len(p) >= min_len for p in parts):
                pieces = parts
        for s2 in pieces:
            if len(s2) < min_len:
                continue
            if len(s2) <= max_len:
                out.append(s2)
                continue
            for chunk in re.split(r"[，,、]\s*", s2):
                chunk = chunk.strip()
                if len(chunk) < min_len:
                    continue
                if len(chunk) <= max_len:
                    out.append(chunk)
                else:
                    out.extend(p for p in _hard_wrap(chunk, max_len)
                               if len(p) >= min_len)
    return out


# Stage-1 fallback filter. High recall by design: it is only used when no seed
# sentences are supplied; with seeds the embedding kNN decides instead.
CUE_ZH = re.compile(
    r"投诉|退款|退货|换货|退钱|赔|补偿|差评|举报|不满意|太差|垃圾|骗|虚假|"
    r"没收到|未收到|没到|还没|超时|延迟|太慢|催|拒|擅自|私自|扣款|多收|乱收费|"
    r"态度|敷衍|推诿|不回|没人管|解决不了|为什么|怎么回事|要求|必须|立刻|马上|"
    r"质量|破损|坏了|不能用|无法|发错|少发|漏发|过期|变质|假货|正品|问题|"
    r"开胶|脱线|褪色|异味|缩水|起球|失灵|死机|闪退|登录不上|扣两次|差额|"
    r"收不到|点不了|打不开|充不进|不配送|不给|凭什么"
)
CUE_EN = re.compile(
    r"complain|complaint|refund|return|reimburse|compensat|chargeback|"
    r"late|delay|slow|never arrived|not received|still waiting|still not|"
    r"broken|damaged|defect|cracked|torn|leak|shrunk|faded|scratch|"
    r"missing|wrong item|wrong colou?r|wrong size|short shipped|"
    r"overcharg|double charg|charged twice|extra fee|unauthori[sz]ed|"
    r"rude|ignor|no response|unhelpful|unresponsive|passed around|"
    r"unacceptable|terrible|awful|worst|disgust|scam|fraud|fake|counterfeit|"
    r"expired|stale|not working|doesn't work|does not work|stopped working|"
    r"won't turn on|won't start|cannot log in|can't log in|crash|"
    r"escalate|manager|supervisor|dispute|unhappy|not satisfied|problem|issue",
    re.I,
)


def looks_like_complaint(sentence):
    """Stage-1 cue check, language aware. Deliberately over-inclusive."""
    pat = CUE_ZH if detect_lang(sentence) == "zh" else CUE_EN
    return bool(pat.search(sentence))


# --------------------------------------------------------------- embeddings

class ApiEmbedder:
    """OpenAI-compatible /v1/embeddings client with batching and backoff."""

    def __init__(self, url, model, api_key="", timeout=120, batch=32, retries=4):
        self.url, self.model, self.timeout = url, model, timeout
        self.batch, self.retries = batch, retries
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def _post(self, texts):
        body = json.dumps({"model": self.model, "input": texts}).encode()
        req = urllib.request.Request(self.url, data=body, headers=self.headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read())
        rows = sorted(data["data"], key=lambda d: d.get("index", 0))
        return np.asarray([d["embedding"] for d in rows], dtype=np.float32)

    def __call__(self, texts):
        out = []
        for i in range(0, len(texts), self.batch):
            chunk = texts[i:i + self.batch]
            for attempt in range(self.retries):
                try:
                    out.append(self._post(chunk))
                    break
                except Exception as exc:                       # noqa: BLE001
                    if attempt == self.retries - 1:
                        raise
                    print(f"[retry {attempt + 1}] {exc}", file=sys.stderr)
                    time.sleep(2 ** attempt)
        return np.vstack(out)


class StubEmbedder:
    """Deterministic hashing embedder, for offline plumbing tests only.

    It is lexical (character n-grams), not semantic. It verifies that the
    pipeline runs end to end; it says nothing about clustering quality.
    """

    def __init__(self, dim=1024):
        self.dim = dim

    def _one(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        t = re.sub(r"\s+", "", PLACEHOLDER.sub(" ", text))
        for n in (2, 3):
            for i in range(len(t) - n + 1):
                v[zlib.crc32(t[i:i + n].encode()) % self.dim] += 1.0
        nrm = np.linalg.norm(v)
        return v / nrm if nrm else v

    def __call__(self, texts):
        return np.vstack([self._one(t) for t in texts])


class FastEmbedEmbedder:
    """Real multilingual embeddings on CPU via ONNX (fastembed, no torch).

    fastembed ships ONNX-quantised builds of sentence-transformers models and
    runs them on onnxruntime, so the whole pipeline stays numpy + one small dep.

    E5-family models are trained with asymmetric prefixes: passages and queries
    live in different regions of the space. For clustering every sentence is a
    passage, so the prefix must be applied consistently -- mixing prefixed and
    unprefixed text measurably degrades the clusters.
    """

    # Models that expect a "passage: " / "query: " prefix.
    PREFIXED = ("intfloat/multilingual-e5", "intfloat/e5-")

    def __init__(self, model, prefix=None, threads=None, batch=32, cache_dir=None):
        from fastembed import TextEmbedding  # imported lazily: optional dep
        kwargs = {"model_name": model}
        if threads:
            kwargs["threads"] = threads
        # fastembed defaults to <TMPDIR>/fastembed_cache, which on this box is a
        # scratch directory that gets pruned -- a 220 MB model would be silently
        # re-downloaded (or an offline run broken) every time. Pin it to a stable
        # global cache instead.
        if cache_dir:
            kwargs["cache_dir"] = os.path.expanduser(cache_dir)
        self._m = TextEmbedding(**kwargs)
        self.model = model
        self.batch = batch
        if prefix is None:
            prefix = "passage: " if model.startswith(self.PREFIXED) else ""
        self.prefix = prefix

    def __call__(self, texts):
        payload = [self.prefix + t for t in texts] if self.prefix else list(texts)
        return np.asarray(list(self._m.embed(payload, batch_size=self.batch)),
                          dtype=np.float32)


def build_embedder(args, quiet=False):
    """Pick the embedding backend: real ONNX model, HTTP API, or offline stub."""
    backend = getattr(args, "embed_backend", None) or "api"
    if getattr(args, "stub", False):
        backend = "stub"
    if backend == "stub":
        if not quiet:
            print("[2] WARNING stub embedder: lexical character n-grams, "
                  "plumbing test only -- numbers say nothing about quality")
        return StubEmbedder()
    if backend == "fastembed":
        model = args.embed_model or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        if not quiet:
            print(f"[2] embedding backend: fastembed (ONNX, CPU) model={model}")
        return FastEmbedEmbedder(model,
                                 prefix=getattr(args, "embed_prefix", None),
                                 cache_dir=getattr(args, "embed_cache", None)
                                 or "~/.cache/fastembed")
    if not quiet:
        print(f"[2] embedding backend: HTTP {args.embed_url or 'http://127.0.0.1:8000/v1/embeddings'}")
    return ApiEmbedder(args.embed_url or "http://127.0.0.1:8000/v1/embeddings",
                       args.embed_model or "BAAI/bge-m3", args.api_key)


class SeedFilter:
    """Similarity-weighted kNN over hand-labelled seed sentences.

    A handful of labelled examples generalises far better than an ever-growing
    keyword regex, and unlike a regex its decision is a distance in embedding
    space, so it inherits the model's bilingual and paraphrase handling.
    """

    def __init__(self, seeds, embedder, k=7, center=None):
        texts = [s["text"] for s in seeds]
        self.labels = np.array([1 if s["label"] == "complaint" else 0
                                for s in seeds], dtype=np.int8)
        raw = np.asarray(embedder(texts), dtype=np.float32)
        if center is not None:                 # same centring as the corpus
            raw = raw - center
        self.X = l2norm(raw)
        self.k = max(1, min(k, len(texts)))

    def __call__(self, Xq, thresh=0.4):
        S = Xq @ self.X.T
        k = self.k
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        vote = np.zeros(len(Xq))
        wsum = np.zeros(len(Xq))
        for i in range(len(Xq)):
            w = np.maximum(S[i, idx[i]], 0.0)
            vote[i] = float((w * self.labels[idx[i]]).sum())
            wsum[i] = float(w.sum())
        return np.divide(vote, np.maximum(wsum, 1e-9)) >= thresh


# --------------------------------------------------------------- dedupe

def minhash_sig(text, n=3, perms=32):
    """MinHash signature over character n-grams: equal signatures ~ duplicates."""
    t = PLACEHOLDER.sub("", re.sub(r"\s+", "", text))
    sh = {zlib.crc32(t[i:i + n].encode()) for i in range(max(1, len(t) - n + 1))}
    if not sh:
        return tuple([0] * perms)
    h = np.fromiter(sh, dtype=np.uint64, count=len(sh))
    a = (np.arange(1, perms + 1, dtype=np.uint64) * 2654435761) | 1
    b = np.array([zlib.crc32(bytes([p])) for p in range(perms)], dtype=np.uint64)
    return tuple(int(x) for x in ((h[None, :] * a[:, None] + b[:, None]).min(axis=1)))


def dedupe_near(sents, max_n=200_000):
    """Collapse near-identical sentences to one representative each.

    Returns (reps, rep_of). reps are the indices that get embedded and clustered;
    rep_of maps every original sentence to its representative. The mapping
    matters: without it a sentence reused by 50 conversations keeps its label in
    only one of them.
    """
    if len(sents) > max_n:
        print(f"[warn] {len(sents)} sentences > {max_n}, near-dup dedupe skipped",
              file=sys.stderr)
        return list(range(len(sents))), list(range(len(sents)))
    seen, reps, rep_of = {}, [], []
    for i, s in enumerate(sents):
        key = minhash_sig(s)
        if key in seen:
            rep_of.append(seen[key])
            continue
        seen[key] = i
        reps.append(i)
        rep_of.append(i)
    return reps, rep_of


# --------------------------------------------------------------- clustering

def l2norm(X):
    """L2 normalise. Mandatory before k-means: otherwise cluster membership
    follows vector norm (boilerplate, long documents) instead of meaning."""
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def _sqdist(X, C):
    """Squared Euclidean distances without materialising a 3-D array."""
    return (X * X).sum(1)[:, None] + (C * C).sum(1)[None, :] - 2.0 * (X @ C.T)


def kmeans(X, k, n_init=5, iters=100, seed0=0):
    """k-means++ with restarts, keeping the lowest-inertia solution."""
    best = None
    for s in range(n_init):
        r = np.random.default_rng(seed0 + s)
        C = np.empty((k, X.shape[1]), dtype=np.float32)
        C[0] = X[r.integers(len(X))]
        d2 = np.full(len(X), np.inf, dtype=np.float32)
        for i in range(1, k):
            d2 = np.minimum(d2, ((X - C[i - 1]) ** 2).sum(1))
            tot = float(d2.sum())
            C[i] = X[r.choice(len(X), p=None if tot <= 0 else d2 / tot)]
        for _ in range(iters):
            lab = _sqdist(X, C).argmin(1)
            newC = C.copy()
            for j in range(k):
                m = lab == j
                if m.any():
                    newC[j] = X[m].mean(0)
            if np.allclose(newC, C, atol=1e-5):
                C = newC
                break
            C = newC
        D = _sqdist(X, C)
        inertia = float(D.min(1).sum())
        if best is None or inertia < best[0]:
            best = (inertia, D.argmin(1), C)
    return best[1], best[2], best[0]


def _split2(P):
    """Deterministic 2-way split of a point block via its top principal axis.

    Power iteration from the mean-shifted data: no random state, so the same
    block always splits the same way. The split is then refined with Lloyd
    iterations, which is what makes it a real k-means split rather than a
    projection heuristic.
    """
    mu = P.mean(0)
    Q = P - mu
    v = Q.sum(0)
    if not np.any(v):
        v = Q[0] if len(Q) else np.zeros(P.shape[1], dtype=np.float32)
    for _ in range(20):
        v = Q.T @ (Q @ v)
        nv = float(np.linalg.norm(v))
        if nv < 1e-12:
            break
        v = v / nv
    proj = Q @ v
    if float(proj.max() - proj.min()) < 1e-12:
        half = len(P) // 2                      # degenerate block: split by index
        return np.arange(len(P)) < half, np.arange(len(P)) >= half
    m = proj >= float(np.median(proj))
    for _ in range(30):
        if not m.any() or m.all():
            break
        ca, cb = P[m].mean(0), P[~m].mean(0)
        new = ((P - ca) ** 2).sum(1) <= ((P - cb) ** 2).sum(1)
        if np.array_equal(new, m):
            break
        m = new
    if not m.any() or m.all():
        half = len(P) // 2
        m = np.arange(len(P)) < half
    return m, ~m


def bisect_kmeans(X, k):
    """Divisive (bisecting) k-means: deterministic, and it builds a hierarchy.

    Lloyd's k-means renumbers clusters depending on where the random init landed,
    and on weakly separated text data different seeds settle into different local
    optima (measured stability ARI 0.4-0.8). Bisecting k-means removes both
    problems: the cluster with the largest SSE is split, the split direction comes
    from that block's top principal axis, and nothing is random. Same data, same k
    -> same partition and same numbering, which is what makes runs comparable.

    Steinbach et al. 2000, "A comparison of document clustering techniques".
    """
    n = len(X)
    if k <= 1 or n <= k:
        lab = np.arange(n, dtype=np.int64)
        C = X.mean(0, keepdims=True) if n else np.zeros((1, X.shape[1]), np.float32)
        return lab, C, 0.0

    blocks = [np.arange(n)]
    while len(blocks) < k:
        sse = []
        for idx in blocks:
            if len(idx) < 2:
                sse.append(-1.0)                # never split a singleton
                continue
            mu = X[idx].mean(0)
            sse.append(float(((X[idx] - mu) ** 2).sum()))
        j = int(np.argmax(sse))
        if sse[j] <= 0:
            break
        inner_a, inner_b = _split2(X[blocks[j]])
        idx = blocks[j]
        blocks[j] = idx[inner_a]
        blocks.append(idx[inner_b])

    labels = np.zeros(n, dtype=np.int64)
    for c, idx in enumerate(blocks):
        labels[idx] = c
    m = len(blocks)
    C = np.zeros((m, X.shape[1]), dtype=np.float32)
    for c, idx in enumerate(blocks):
        C[c] = X[idx].mean(0)
    inertia = float(((X - C[labels]) ** 2).sum())
    return labels, C, inertia


def fit_clusters(X, k, algo="lloyd", n_init=5, seed0=0):
    """One entry point for both clustering algorithms (same return shape)."""
    if algo == "bisect":
        return bisect_kmeans(X, k)
    return kmeans(X, k, n_init=n_init, seed0=seed0)


def canonical_order(sizes, C):
    """Renumber clusters so an id means the same thing across runs.

    k-means ids follow the init order, so two runs that agree on the partition
    still disagree on which cluster is "cluster 3". Sort by size (so "cluster 0"
    is always the biggest) and break ties on the rounded centroid, which is
    content-determined rather than init-determined.
    """
    keys = [(-int(sizes[j]), tuple(np.round(C[j], 3).tolist()))
            for j in range(len(sizes))]
    return sorted(range(len(sizes)), key=lambda j: keys[j])


def silhouette(X, labels, sample=800, seed=0):
    """Mean silhouette on a sample. Around 0 means there is no structure to find."""
    r = np.random.default_rng(seed)
    idx = r.choice(len(X), min(sample, len(X)), replace=False)
    Xs, ls = X[idx], labels[idx]
    D = np.sqrt(np.maximum(_sqdist(Xs, Xs), 0.0))
    scores = []
    for i in range(len(Xs)):
        same = ls == ls[i]
        same[i] = False
        if not same.any():
            continue
        a = D[i, same].mean()
        b = min(D[i, ls == c].mean() for c in np.unique(ls) if c != ls[i])
        scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def top_terms(texts, labels, k, topn=8):
    """Cluster vocabulary (lift x sqrt count) to feed the LLM naming step."""
    from collections import Counter

    glob = Counter()
    per = [Counter() for _ in range(k)]
    for t, lab in zip(texts, labels):
        t = PLACEHOLDER.sub(" ", t)
        toks = [w.lower() for w in re.findall(r"[\u4e00-\u9fa5]{2,4}|[a-zA-Z]{3,}", t)]
        glob.update(toks)
        per[lab].update(toks)
    N = sum(glob.values()) or 1
    out = []
    for j in range(k):
        size = sum(per[j].values()) or 1
        scored = []
        for w, c in per[j].items():
            if c < 2:
                continue
            lift = (c / size) / max(glob[w] / N, 1e-9)
            scored.append((lift * (c ** 0.5), w))
        scored.sort(reverse=True)
        out.append([w for _, w in scored[:topn]])
    return out


# --------------------------------------------------------------- main

MODEL_VERSION = 1


def save_model(path, C, mu, embedder, args, k, clusters, n_sent, n_conv,
               cue_mode):
    """Persist everything needed to assign new conversations to these centroids.

    The centre vector is part of the model, not an implementation detail: at fit
    time every vector is mean-centred before normalisation, so a later run that
    skips centring (or centres on its own corpus mean) silently lands in a
    different space and the nearest-centroid assignment is wrong even though the
    shapes match.
    """
    meta = {
        "version": MODEL_VERSION,
        "k": int(k),
        "dim": int(C.shape[1]),
        "embed_model": getattr(embedder, "model", "") or args.embed_model,
        "embed_prefix": getattr(embedder, "prefix", "") or "",
        "embed_backend": args.embed_backend,
        "centred": mu is not None,
        "primary_mode": args.primary_mode,
        "kmeans_algo": args.kmeans_algo,
        "cue_mode": cue_mode,
        "canonical_ids": True,
        "n_fit_sentences": int(n_sent),
        "n_fit_conversations": int(n_conv),
        "clusters": clusters,
    }
    payload = {"C": C.astype(np.float32),
               "meta": np.array(json.dumps(meta, ensure_ascii=False))}
    if mu is not None:
        payload["center"] = mu.astype(np.float32)
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **payload)
    print(f"[R1] saved model -> {path} ({os.path.getsize(path) / 1024:.0f} KB, "
          f"k={k}, centred={mu is not None}, "
          f"model={meta['embed_model'] or 'n/a'})")


def load_model(path):
    """Read back what save_model wrote."""
    with np.load(path, allow_pickle=False) as z:
        C = z["C"].astype(np.float32)
        mu = z["center"].astype(np.float32) if "center" in z.files else None
        meta = json.loads(str(z["meta"]))
    return C, mu, meta


def apply_model(args):
    """Assign conversations to a saved model. No fitting, no clustering.

    Deliberately skips the corpus-relative stages of the fit path (boilerplate DF
    filter, near-duplicate dedupe, k-means): they depend on the *set* of
    conversations, and a new batch is not that set. What must be identical is the
    per-sentence path -- cleaning, sentence splitting, the cue filter, the
    embedding model, its prefix, and the centre vector -- because those decide
    which point in space a sentence occupies.
    """
    C, mu, meta = load_model(args.apply_model)
    k, dim = C.shape[0], C.shape[1]
    fit_model = meta.get("embed_model") or ""
    print(f"[R2] model {args.apply_model}: k={k}, dim={dim}, "
          f"fit on {meta.get('n_fit_conversations')} conversations, "
          f"centred={meta.get('centred')}")

    if args.embed_model and fit_model and args.embed_model != fit_model:
        sys.exit(f"[fatal] --embed-model {args.embed_model!r} does not match the "
                 f"model file ({fit_model!r}); the vectors would not be comparable")
    if not args.embed_model:
        args.embed_model = fit_model
    if args.embed_prefix is None:
        args.embed_prefix = meta.get("embed_prefix") or ""
    if meta.get("centred") and mu is None:
        sys.exit("[fatal] model was fitted with mean centring but carries no centre")

    # The cue filter decides which sentences exist, so it must match the fit run.
    # Inherit it from the model file unless the caller asks for something else:
    # measuring on this corpus, the cue regex alone keeps only ~2% of Chinese
    # sentences, which would silently reduce an apply run to document-level
    # guesses while still looking successful.
    recorded_cue = meta.get("cue_mode")
    if args.cue_mode == "auto":
        cue_mode = recorded_cue or "prefilter"
        if recorded_cue:
            print(f"[R2] cue filter {cue_mode} (inherited from the model file)")
    else:
        cue_mode = args.cue_mode
        if recorded_cue and cue_mode != recorded_cue:
            print(f"[warn] --cue-mode {cue_mode} differs from the fit run "
                  f"({recorded_cue}); sentences may land differently",
                  file=sys.stderr)
    use_cue = (cue_mode != "off") and not args.no_cue_filter

    convs, sents, owner, conv_text = [], [], [], {}
    for cid, text in iter_convs(args.input):
        if not text.strip():
            continue
        convs.append(cid)
        conv_text[cid] = text
        for sent in split_sentences(clean(text), conj=not args.no_conj_split):
            if use_cue and not looks_like_complaint(sent):
                continue
            sents.append(sent)
            owner.append(cid)
    if not convs:
        sys.exit("[fatal] no conversations in --input")

    embedder = build_embedder(args)
    per_conv, rescued, X = {}, set(), None
    if sents:
        Xraw = np.asarray(embedder(sents), dtype=np.float32)
        X = l2norm(Xraw - mu if mu is not None else Xraw)
        for cid, j in zip(owner, _sqdist(X, C).argmin(1)):
            per_conv.setdefault(cid, {})
            per_conv[cid][int(j)] = per_conv[cid].get(int(j), 0) + 1

    primary = {}
    if meta.get("primary_mode", "doc") == "doc":
        Zraw = np.asarray(embedder([clean(conv_text[c]) for c in convs]),
                          dtype=np.float32)
        Z = l2norm(Zraw - mu if mu is not None else Zraw)
        for c, j in zip(convs, _sqdist(Z, C).argmin(1)):
            primary[c] = int(j)
    elif X is not None:
        acc = {}
        for cid, v in zip(owner, X):
            acc[cid] = acc[cid] + v if cid in acc else v.astype(np.float32).copy()
        if acc:
            cids = list(acc)
            M = l2norm(np.vstack([acc[c] for c in cids]))
            for c, j in zip(cids, _sqdist(M, C).argmin(1)):
                primary[c] = int(j)

    for cid in convs:
        if cid not in per_conv and cid in primary:
            per_conv[cid] = {primary[cid]: 1}
            rescued.add(cid)

    with open(os.path.join(args.out, "conv_labels.jsonl"), "w",
              encoding="utf-8") as f:
        for cid in convs:
            d = per_conv.get(cid, {})
            items = [{"cluster": c, "n": n}
                     for c, n in sorted(d.items(), key=lambda kv: -kv[1])]
            f.write(json.dumps({"id": cid, "labels": items,
                                "primary": primary.get(cid),
                                "rescued": cid in rescued},
                               ensure_ascii=False) + "\n")

    hit = {int(j) for j in primary.values()}
    print(f"[R2] assigned {len(per_conv)}/{len(convs)} conversations from "
          f"{len(sents)} sentences -> {args.out}/conv_labels.jsonl")
    print(f"[R2] clusters hit: {len(hit)}/{k}"
          + (f", never hit: {sorted(set(range(k)) - hit)}" if len(hit) < k else "")
          + (f"; {len(rescued)} conversations labelled from their full text only"
             if rescued else ""))
    if len(hit) < k:
        print("[note] unseen clusters are expected on a small batch; a cluster "
              "with no members is not a broken model", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(
        description="Cluster complaint conversation logs into complaint categories.")
    ap.add_argument("--input", required=True,
                    help="input jsonl, one conversation per line")
    ap.add_argument("--out", default="out", help="output directory")
    ap.add_argument("--k", default="auto", help="number of clusters, or 'auto'")
    ap.add_argument("--k-max", type=int, default=60, help="upper bound for auto k")
    ap.add_argument("--kmeans-algo", choices=["lloyd", "bisect"], default="lloyd",
                    help="lloyd: k-means++ with restarts (fast, seed-dependent); "
                         "bisect: deterministic divisive k-means -- same data and k "
                         "always give the same partition and numbering")
    ap.add_argument("--kmeans-inits", type=int, default=5,
                    help="restarts for the lloyd algorithm; more restarts make the "
                         "chosen optimum (and therefore the numbering) more stable")
    ap.add_argument("--kmeans-seed", type=int, default=0,
                    help="k-means++ random seed; vary it to measure partition "
                         "stability (same corpus, different seed)")
    ap.add_argument("--embed-url", default=os.environ.get("EMBED_URL", ""),
                    help="OpenAI-compatible /v1/embeddings endpoint")
    ap.add_argument("--embed-model", default=os.environ.get("EMBED_MODEL", ""),
                    help="multilingual embedding model, e.g. BAAI/bge-m3 "
                         "(api backend) or intfloat/multilingual-e5-large "
                         "(fastembed backend)")
    ap.add_argument("--embed-backend", default="api",
                    choices=["api", "fastembed", "stub"],
                    help="api: HTTP /v1/embeddings; fastembed: local ONNX on CPU; "
                         "stub: offline lexical placeholder")
    ap.add_argument("--embed-prefix", default=None,
                    help="prefix prepended to every text; default auto "
                         "('passage: ' for e5 models, none otherwise)")
    ap.add_argument("--embed-cache", default="~/.cache/fastembed",
                    help="where the fastembed backend keeps downloaded ONNX "
                         "weights; must be a persistent path, not a temp dir")
    ap.add_argument("--api-key", default=os.environ.get("EMBED_API_KEY", ""))
    ap.add_argument("--seeds", default="",
                    help="labelled seed sentences jsonl: {text, label}")
    ap.add_argument("--knn-k", type=int, default=7,
                    help="kNN neighbours used for voting")
    ap.add_argument("--knn-thresh", type=float, default=0.4,
                    help="complaint weight share needed to keep a sentence; "
                         "re-tune per corpus, sweep 0.3-0.6")
    ap.add_argument("--stub", action="store_true",
                    help="offline plumbing test with a fake lexical embedder")
    ap.add_argument("--cue-mode", default="auto",
                    choices=["auto", "prefilter", "off"],
                    help="auto: cue off when --seeds is given (measured: cue + kNN "
                         "is worse than kNN alone), cue on otherwise")
    ap.add_argument("--no-cue-filter", action="store_true")
    ap.add_argument("--no-conj-split", action="store_true",
                    help="disable conjunction splitting (double complaints lose "
                         "their second label)")
    ap.add_argument("--df-max", type=float, default=0.05,
                    help="drop sentences appearing in more than this share of "
                         "conversations; 0 disables")
    ap.add_argument("--df-min-support", type=int, default=20,
                    help="absolute floor for --df-max, protects small corpora")
    ap.add_argument("--no-dedupe", action="store_true",
                    help="disable near-duplicate dedupe")
    ap.add_argument("--no-center", action="store_true",
                    help="disable mean centring before L2 normalisation; keeps "
                         "E5-family models from splitting a mixed corpus by language")
    ap.add_argument("--no-rescue", action="store_true",
                    help="disable orphan rescue (full-text nearest-centroid fallback)")
    ap.add_argument("--primary-mode", default="doc", choices=["doc", "mean"],
                    help="how the single document-level label is computed: 'doc' "
                         "embeds the whole conversation and assigns it to the nearest "
                         "centroid (measured 0.29 vs 0.19 purity on CFPB, at the cost "
                         "of one embedding per conversation); 'mean' averages the "
                         "sentence vectors (cheaper)")
    ap.add_argument("--drop-procedural", action="store_true",
                    help="drop utterances that carry no complaint content: "
                         "credential/PII exchanges, bare person names, "
                         "acknowledgements and thanks (measured: 32%% of a "
                         "customer-service corpus); dropped lines are exported")
    ap.add_argument("--no-proc-names", dest="proc_names", action="store_false",
                    help="keep the person-name rule out of --drop-procedural")
    ap.add_argument("--name-clusters", action="store_true",
                    help="after clustering, ask an LLM to name every cluster and "
                         "write taxonomy.json (R3); needs --llm-endpoint")
    ap.add_argument("--llm-endpoint", default="",
                    help="OpenAI-compatible /v1/chat/completions for cluster naming")
    ap.add_argument("--llm-model", default="",
                    help="whatever your chat endpoint calls the model; empty "
                         "sends an empty name, which most OpenAI-compatible "
                         "servers accept")
    ap.add_argument("--save-model", default="",
                    help="write the fitted centroids + metadata (npz) so new "
                         "conversations can be assigned later with --apply-model")
    ap.add_argument("--apply-model", default="",
                    help="assign --input to a saved model instead of clustering; "
                         "skips every corpus-relative stage (DF filter, dedupe, "
                         "k-means) and only re-runs the per-sentence path")
    ap.add_argument("--samples-per-cluster", type=int, default=12)
    ap.add_argument("--kmeans-cap", type=int, default=40000,
                    help="max sentences used to fit k-means (assignment uses all)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.apply_model:
        apply_model(args)
        return

    # 1) ingest -> sentences. Agent turns are dropped here (P1).
    cue_mode = args.cue_mode
    if cue_mode == "auto":
        cue_mode = "off" if args.seeds else "prefilter"
    use_cue = (cue_mode != "off") and not args.no_cue_filter
    if cue_mode == "off" and args.seeds:
        print("[1] cue prefilter off: --seeds drives the decision "
              "(measured: cue + kNN < kNN alone)")

    convs, sents, owner = [], [], []
    conv_text = {}
    n_all = 0
    proc_dropped = []
    for cid, text in iter_convs(args.input):
        if not text.strip():
            continue
        convs.append(cid)
        conv_text[cid] = text
        for s in split_sentences(clean(text), conj=not args.no_conj_split):
            n_all += 1
            if use_cue and not looks_like_complaint(s):
                continue
            if args.drop_procedural:
                flag, why = is_procedural(s, drop_names=args.proc_names)
                if flag:
                    proc_dropped.append({"text": s, "reason": why})
                    continue
            sents.append(s)
            owner.append(cid)
    n_zh = sum(1 for s in sents if detect_lang(s) == "zh")
    print(f"[1] {len(convs)} conversations / {n_all} user sentences "
          f"-> {len(sents)} after filter (zh={n_zh}, en={len(sents) - n_zh})")

    if proc_dropped:
        reasons = {}
        for d in proc_dropped:
            reasons[d["reason"]] = reasons.get(d["reason"], 0) + 1
        print(f"[1.4] procedural filter: {len(proc_dropped)} sentences dropped "
              f"({len(proc_dropped) / max(n_all, 1):.0%} of {n_all}) {reasons}")
        with open(os.path.join(args.out, "procedural_dropped.json"), "w",
                  encoding="utf-8") as f:
            json.dump(sorted(proc_dropped, key=lambda d: d["reason"]), f,
                      ensure_ascii=False, indent=2)

    # 1.5) P1: drop template sentences, but export them. A line like "please
    # handle this or I will complain" is the appeal signal, not noise.
    appeal = {}
    if args.df_max > 0 and sents:
        def sig(s):
            return re.sub(r"<\w+>|\d+|\s+", "", s)

        conv_sigs = {}
        for cid, s in zip(owner, sents):
            conv_sigs.setdefault(cid, set()).add(sig(s))
        df = {}
        for ss in conv_sigs.values():
            for s in ss:
                df[s] = df.get(s, 0) + 1
        nconv = max(len(conv_sigs), 1)
        keep = []
        for i, s in enumerate(sents):
            d = df.get(sig(s), 0)
            if d >= args.df_min_support and d / nconv > args.df_max:
                appeal.setdefault(sig(s), [s, 0])[1] += 1
            else:
                keep.append(i)
        if appeal:
            dropped = len(sents) - len(keep)
            print(f"[1.5] boilerplate filter: {dropped} sentences dropped "
                  f"(df > {args.df_max:.0%} and >= {args.df_min_support} conversations)")
            if dropped > 0.6 * len(sents):
                print("[warn] over 60% of sentences dropped -- corpus may be "
                      "heavily templated; raise --df-max or pass --df-max 0",
                      file=sys.stderr)
            sents = [sents[i] for i in keep]
            owner = [owner[i] for i in keep]

    # 1.6) P1 continued: paraphrases of the same template.
    owner_all = owner          # original sentence -> conversation, before dedupe
    if not args.no_dedupe and len(sents) > 50:
        reps, rep_of = dedupe_near(sents)
    else:
        reps, rep_of = list(range(len(sents))), list(range(len(sents)))
    if len(reps) < len(sents):
        print(f"[1.6] near-dup dedupe: {len(sents)} sentences -> {len(reps)} "
              f"representatives (only representatives are embedded)")
    uni = [sents[i] for i in reps]

    if appeal:
        with open(os.path.join(args.out, "appeal_phrases.json"), "w",
                  encoding="utf-8") as f:
            json.dump([{"text": v[0], "conversations": v[1]} for v in
                       sorted(appeal.values(), key=lambda v: -v[1])[:200]],
                      f, ensure_ascii=False, indent=2)

    if len(sents) < 20:
        sys.exit(f"[fatal] only {len(sents)} usable sentences -- check the schema "
                 f"or the cue filter")

    # 2) embed representatives only
    embedder = build_embedder(args)
    t0 = time.time()
    Xraw = np.asarray(embedder(uni), dtype=np.float32)
    dt = time.time() - t0
    # Mean centring before normalisation. E5-family vectors are strongly
    # anisotropic (any two sentences sit at cosine ~0.85), which flattens
    # Euclidean distances until k-means splits on surface form -- in a mixed
    # corpus that means it splits by LANGUAGE. Subtracting the corpus mean
    # restores the spread. Cheap, and harmless for isotropic models.
    mu = Xraw.mean(0, keepdims=True) if not args.no_center else None
    X = l2norm(Xraw - mu if mu is not None else Xraw)
    print(f"[2] embedded {X.shape[0]} x {X.shape[1]} in {dt:.1f}s "
          f"({X.shape[0] / dt:.0f} sentences/s)"
          + ("" if mu is None else "  [mean-centred]"))

    # 2.5) P3 stage 2: the seed kNN decides what is actually a complaint
    kept = np.ones(len(uni), dtype=bool)
    sf = None
    if args.seeds:
        with open(args.seeds, encoding="utf-8") as f:
            seeds = [json.loads(line) for line in f if line.strip()]
        sf = SeedFilter(seeds, embedder, k=args.knn_k, center=mu)
        if not seeds:
            sys.exit(f"[fatal] seed file {args.seeds} is empty")
        m = sf(X, thresh=args.knn_thresh)
        print(f"[2.5] seed kNN: keep {int(m.sum())}/{len(m)} representatives")
        if m.sum() >= 20:
            kept = m
        else:
            print("[warn] kNN kept <20 sentences, falling back to the cue result",
                  file=sys.stderr)

    X = X[kept]
    uni = [s for s, k_ in zip(uni, kept) if k_]
    kept_reps = [r for r, k_ in zip(reps, kept) if k_]
    pos_of_rep = {r: p for p, r in enumerate(kept_reps)}

    # 3) fit k-means (subsample for very large corpora, assign everything)
    Xf = X
    if len(X) > args.kmeans_cap:
        Xf = X[np.random.default_rng(0).choice(len(X), args.kmeans_cap, replace=False)]

    if args.k == "auto":
        cands = [c for c in (6, 10, 15, 20, 30, 40, 50, 60, 80, 100)
                 if c <= min(args.k_max, max(6, len(Xf) // 20))]
        best = None
        for k_ in cands:
            lab, _, _ = fit_clusters(Xf, k_, args.kmeans_algo,
                                    max(2, args.kmeans_inits // 2),
                                    args.kmeans_seed)
            sil = silhouette(Xf, lab)
            print(f"      k={k_:3d}  silhouette={sil:+.3f}")
            if best is None or sil > best[0]:
                best = (sil, k_)
        k, sil = best[1], best[0]
        print(f"[3] chose k={k} (silhouette {sil:+.3f})")
    else:
        k = int(args.k)
        print(f"[3] k={k} (fixed)")

    _, C, _ = fit_clusters(Xf, k, args.kmeans_algo, args.kmeans_inits,
                           args.kmeans_seed)
    labels = _sqdist(X, C).argmin(1)
    sizes = np.bincount(labels, minlength=k)

    # 3.1) canonical ids: k-means numbering follows the init order, so the same
    # partition can come out as different ids on every run. Renumber by size.
    perm = np.array(canonical_order(sizes, C), dtype=np.int64)
    if not np.array_equal(perm, np.arange(k)):
        inv = np.empty(k, dtype=np.int64)
        inv[perm] = np.arange(k)
        C = C[perm]
        labels = inv[labels]
        sizes = np.bincount(labels, minlength=k)
        print(f"[3.1] renumbered {k} clusters by size (canonical ids)")
    terms = top_terms(uni, labels, k)

    # Expand representative labels back onto every original sentence (P2).
    labels_all = np.array([labels[pos_of_rep[r]] if r in pos_of_rep else -1
                           for r in rep_of], dtype=np.int64)

    order = {}
    for j in range(k):
        idx = np.where(labels == j)[0]
        if len(idx) == 0:
            continue
        d = ((X[idx] - C[j]) ** 2).sum(1)
        order[j] = idx[np.argsort(d)][: args.samples_per_cluster].tolist()

    clusters = [{
        "cluster": int(j),
        "size": int(sizes[j]),
        "share": round(float(sizes[j]) / max(len(labels), 1), 4),
        "top_terms": terms[j],
        "samples": [uni[i] for i in order[j]],
    } for j in sorted(order, key=lambda j: -sizes[j])]
    # Provenance, so a run can be reproduced and so the codebook can point back at
    # the exact algorithm, seed and model file that produced these clusters.
    provenance = {
        "kmeans_algo": args.kmeans_algo,
        "kmeans_seed": int(args.kmeans_seed),
        "kmeans_inits": int(args.kmeans_inits),
        "embed_model": getattr(embedder, "model", "") or args.embed_model,
        "embed_prefix": getattr(embedder, "prefix", "") or "",
        "embed_backend": args.embed_backend,
        "centred": mu is not None,
        "primary_mode": args.primary_mode,
        "cue_mode": cue_mode,
        "model_file": os.path.abspath(args.save_model) if args.save_model else None,
        "n_fit_conversations": len(convs),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    clusters_json = {"k": k,
                     "n_unique_sentences": int(len(uni)),
                     "n_sentences": int(len(labels_all)),
                     **provenance,
                     "clusters": clusters}
    with open(os.path.join(args.out, "clusters.json"), "w", encoding="utf-8") as f:
        json.dump(clusters_json, f, ensure_ascii=False, indent=2)

    if args.name_clusters:
        if not args.llm_endpoint:
            sys.exit("[fatal] --name-clusters needs --llm-endpoint")
        import llm_naming
        print(f"[3.2] naming {k} clusters with "
              f"{args.llm_model or '(server default)'}")
        entries = llm_naming.name_all(clusters, args.llm_endpoint, args.llm_model,
                                      log=print)
        taxonomy = {"k": k,
                    "created_utc": provenance["created_utc"],
                    "llm_model": args.llm_model,
                    "kmeans_algo": args.kmeans_algo,
                    "embed_model": provenance["embed_model"],
                    "source_clusters": os.path.abspath(
                        os.path.join(args.out, "clusters.json")),
                    "clusters": entries}
        with open(os.path.join(args.out, "taxonomy.json"), "w",
                  encoding="utf-8") as f:
            json.dump(taxonomy, f, ensure_ascii=False, indent=2)
        named = sum(1 for e in entries if e.get("_ok") and e.get("label"))
        print(f"[3.2] taxonomy.json: {named}/{k} clusters named; "
              f"{sum(1 for e in entries if llm_naming.validate(e))} entries need review")
        clusters_json["taxonomy_file"] = "taxonomy.json"
        with open(os.path.join(args.out, "clusters.json"), "w",
                  encoding="utf-8") as f:
            json.dump(clusters_json, f, ensure_ascii=False, indent=2)

    if args.save_model:
        save_model(args.save_model, C, mu, embedder, args, k, clusters,
                   len(uni), len(convs), cue_mode)

    # 4) conversation-level multi-label with counts (P2)
    per_conv = {}
    for cid, j in zip(owner_all, labels_all):
        if j < 0:
            continue
        per_conv.setdefault(cid, {})
        per_conv[cid][int(j)] = per_conv[cid].get(int(j), 0) + 1

    # 4.2) document-level primary label.
    # Two ways to get one label per document, and they are NOT equivalent on long
    # text: averaging the sentence vectors ("mean") is cheap but inherits the
    # chunking noise; embedding the whole conversation and assigning that ("doc")
    # measured 0.29 vs 0.19 purity on CFPB's 25 issue classes. The per-sentence
    # labels still carry the multi-label information either way.
    vec_of_rep = {r: X[p] for r, p in pos_of_rep.items()}
    acc = {}
    for cid, r in zip(owner_all, rep_of):
        v = vec_of_rep.get(r)
        if v is None:
            continue
        if cid in acc:
            acc[cid] += v
        else:
            acc[cid] = v.astype(np.float32).copy()

    primary = {}
    doc_ids, Zdoc = [], None
    if args.primary_mode == "doc":
        doc_ids = [c for c in convs if c in conv_text]
        if doc_ids:
            Zraw = np.asarray(embedder([clean(conv_text[c]) for c in doc_ids]),
                              dtype=np.float32)
            Zdoc = l2norm(Zraw - mu if mu is not None else Zraw)
            for c, j in zip(doc_ids, _sqdist(Zdoc, C).argmin(1)):
                primary[c] = int(j)
            print(f"[4.2] document labels: {len(primary)} conversations embedded whole "
                  f"(primary-mode=doc)")
    elif acc:
        cids = list(acc)
        M = l2norm(np.vstack([acc[c] for c in cids]))
        for c, j in zip(cids, _sqdist(M, C).argmin(1)):
            primary[c] = int(j)
    doc_pos = {c: i for i, c in enumerate(doc_ids)}

    # 4.5) Orphan rescue: the filters may have removed every sentence of a
    # conversation. Label it from its full text, but let the seed kNN veto,
    # otherwise every chit-chat conversation is forced into a complaint cluster.
    rescued_ids = set()
    if not args.no_rescue:
        orphans = [c for c in convs if c not in per_conv]
        if orphans:
            if Zdoc is not None:
                todo = [c for c in orphans if c in doc_pos]
                Zo = (np.vstack([Zdoc[doc_pos[c]] for c in todo]) if todo
                      else np.zeros((0, C.shape[1]), dtype=np.float32))
            else:
                todo = orphans
                Zraw = np.asarray(embedder([clean(conv_text[c]) for c in todo]),
                                  dtype=np.float32)
                Zo = l2norm(Zraw - mu if mu is not None else Zraw)
            keep_o = np.ones(len(todo), dtype=bool)
            if sf is not None and len(todo):
                keep_o = sf(Zo, thresh=args.knn_thresh)
            for cid, j, keep in zip(todo, (_sqdist(Zo, C).argmin(1) if len(todo) else []),
                                    keep_o):
                if not keep:
                    continue
                per_conv.setdefault(cid, {})
                per_conv[cid][int(j)] = 1
                primary.setdefault(cid, int(j))
                rescued_ids.add(cid)
            print(f"[4.5] orphan rescue: {len(orphans)} conversations lost all "
                  f"sentences, {len(rescued_ids)} labelled"
                  + (" (rest vetoed by the seed kNN)" if sf is not None else ""))

    with open(os.path.join(args.out, "conv_labels.jsonl"), "w", encoding="utf-8") as f:
        for cid in convs:
            d = per_conv.get(cid, {})
            items = [{"cluster": c, "n": n}
                     for c, n in sorted(d.items(), key=lambda kv: -kv[1])]
            f.write(json.dumps({"id": cid, "labels": items,
                                "primary": primary.get(cid),
                                "rescued": cid in rescued_ids},
                               ensure_ascii=False) + "\n")

    empty = int((sizes == 0).sum())
    multi = sum(1 for d in per_conv.values() if len(d) > 1)
    print(f"[4] wrote {args.out}/clusters.json and {args.out}/conv_labels.jsonl")
    if empty:
        print(f"[warn] {empty} empty clusters -- k is too large for this corpus")
    print(f"[stat] conversations with 2+ complaint points: {multi}/{len(per_conv)} "
          f"({multi / max(len(per_conv), 1) * 100:.0f}%)")
    print("\nTop clusters:")
    for c in clusters[:10]:
        print(f"  #{c['cluster']:<3d} {c['size']:6d} ({c['share'] * 100:5.1f}%)  "
              f"{'/'.join(c['top_terms'][:5])}")


if __name__ == "__main__":
    main()
