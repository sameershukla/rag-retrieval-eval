# RAG Retrieval Evaluation

A small, readable lab for the two decisions that make or break a RAG system
*before* the LLM is ever called:

1. **Retrieval** — did we find the relevant evidence? Measured with **Recall@K**.
2. **Chunking** — did we cut the documents so the evidence *can* be found?
   Measured with the same metric, so chunking strategies can be compared directly.

Everything runs on CPU in seconds. The datasets are deliberately tiny so every
number in this README can be verified by hand.

---

## Contents

- [Layout](#layout)
- [Quickstart](#quickstart)
- [Part 1 — Recall@K](#part-1--recallk)
- [Part 2 — Semantic chunking](#part-2--semantic-chunking)
- [Part 3 — Evaluating a chunking strategy](#part-3--evaluating-a-chunking-strategy)
- [What to take away](#what-to-take-away)

---

## Layout

```text
data/
  dataset.py     6 AWS Glue docs + 4 labelled queries   (Part 1: retriever eval)
  corpus.py      6 policy/prose docs + 10 labelled questions (Part 3: chunking eval)
retrievers/
  bm25_retriever.py    lexical retrieval (rank_bm25)
  dense_retriever.py   embedding retrieval (all-MiniLM-L6-v2)
chunking/
  semantic_chunking.py  embedding-based boundary detection, one function per step
evaluation/
  recall_at_k.py                    the metric, on its own
  run_evaluation.py                 BM25 vs dense            -> Part 1
  semantic_chunking_evaluation.py   fixed-size vs semantic   -> Part 3
```

Two datasets on purpose. `dataset.py` has one-sentence documents, so chunking is
irrelevant and the retriever is the only variable. `corpus.py` has multi-topic
documents, so chunking becomes the variable.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

All three entry points expect the repo root on the path:

```bash
# Part 1: BM25 vs dense retrieval, Recall@3
PYTHONPATH=. python evaluation/run_evaluation.py

# Part 2: watch semantic chunking run, step by step, on one document
PYTHONPATH=. python chunking/semantic_chunking.py

# Part 3: does semantic chunking beat fixed-size chunking?
PYTHONPATH=. python evaluation/semantic_chunking_evaluation.py
```

The first run downloads `all-MiniLM-L6-v2` (~90 MB) from the Hugging Face Hub.

---

# Part 1 — Recall@K

## What problem does Recall@K solve?

Before an LLM can answer a question, the retriever must find the relevant evidence.

Recall@K answers this question:

> Of all the documents known to be relevant, how many did the retriever find
> within its top K results?

For example, `Recall@3` checks whether the relevant documents appeared anywhere
within the first three retrieved results.

## Formula

```text
Recall@K = Relevant documents retrieved in top K
           -------------------------------------
           Total relevant documents
```

A score of:

- `1.00` means the retriever found 100% of the relevant documents.
- `0.50` means it found 50% of the relevant documents.
- `0.00` means it found none of them.

## Ground truth

Each evaluation query has a set of documents that humans have identified as
relevant. This is called the **ground truth**.

```python
{
    "query": "How can I reduce AWS Glue startup time?",
    "relevant_doc_ids": {"doc-2", "doc-5"},
}
```

Here, two documents are considered relevant. A retriever must find both to
achieve perfect recall for this query.

## Example 1: Recall@3 = 0.50

```text
Query: How can I reduce AWS Glue startup time?
Relevant:  ['doc-2', 'doc-5']
Retrieved: ['doc-2', 'doc-4', 'doc-1']
Recall@3: 0.50
```

There are two relevant documents:

| Document | Retrieved in top 3? |
| --- | --- |
| `doc-2` | Yes |
| `doc-5` | No |

The retriever found one of the two relevant documents:

```text
Recall@3 = 1 / 2 = 0.50
```

Although `doc-2` was ranked first, recall is only `0.50` because `doc-5` was missed.

## Example 2: Recall@3 = 1.00

```text
Query: Why is my Glue job waiting before it starts?
Relevant:  ['doc-4']
Retrieved: ['doc-4', 'doc-2', 'doc-3']
Recall@3: 1.00
```

Only one document is relevant, and it was retrieved:

```text
Recall@3 = 1 / 1 = 1.00
```

The result would still have `Recall@3 = 1.00` if `doc-4` appeared second or third:

```text
Retrieved: ['doc-2', 'doc-3', 'doc-4']
```

Recall@K considers the entire top-K result set. It does not measure the exact
position of a relevant document.

## Why 1.00 does not mean "ranked first"

The word **recall** refers to coverage, not ranking position.

Recall@3 asks:

> Did we retrieve all relevant documents somewhere in the top three?

It does not ask:

> Did we rank the best document first?

Use metrics such as **MRR** or **NDCG** when ranking position matters.

## Macro vs micro averaging

The sample execution produces these query-level scores:

```text
0.50, 1.00, 1.00, 1.00
```

**Macro average** (what `run_evaluation.py` prints) averages the per-query scores:

```text
(0.50 + 1.00 + 1.00 + 1.00) / 4 = 0.875  ->  0.88
```

Read it as: on a typical query, the retriever found 88% of that query's relevant
documents. Every query counts equally, no matter how many relevant documents it has.

**Micro average** pools all relevant documents first, then divides once:

```text
4 relevant documents found / 5 relevant documents total = 0.80
```

Read it as: across the whole label set, 80% of the relevant evidence was retrieved.
Queries with more relevant documents pull harder.

Neither is more correct — but a benchmark that does not say which one it used is
not reproducible. Macro answers *"how well do we serve the average question?"*;
micro answers *"how much of the evidence do we surface overall?"*

## Why BM25 and dense retrieval can receive the same score

Both retrievers score **Mean Recall@3 = 0.88** on this dataset, and both miss the
same thing — `doc-5` on query 1 — despite ranking the other results differently:

```text
Relevant: ['doc-2', 'doc-5']

BM25:  ['doc-2', 'doc-4', 'doc-3']
Dense: ['doc-2', 'doc-4', 'doc-1']
```

Both found only `doc-2`, so both receive `1 / 2 = 0.50`.

Recall@K ignores the non-relevant documents, except for the fact that they occupy
positions that could have held relevant ones. Two retrievers agree on the metric
whenever they find the *same number* of relevant documents — not when they behave
the same way.

That failure is also a labelling question worth noticing. `doc-5`
("Provisioned capacity provides dedicated resources for predictable workloads")
is labelled relevant to reducing startup time because provisioned capacity avoids
cold starts — a connection that requires domain knowledge and appears nowhere in
the wording. Neither lexical nor embedding similarity can bridge that gap; this is
what a reranker or query expansion is for.

## What Recall@K does not measure

Recall@K does not tell us:

- Whether the relevant document was ranked first or third.
- How many retrieved documents were irrelevant.
- Whether the LLM used the retrieved evidence correctly.
- Whether the generated answer was factually correct.

Other metrics answer different questions:

| Metric | Question answered |
| --- | --- |
| Recall@K | Did we retrieve the available relevant evidence? |
| Precision@K | How much of the retrieved result set was relevant? |
| MRR | How early did the first relevant result appear? |
| NDCG | Did the retriever place more relevant results near the top? |

---

# Part 2 — Semantic chunking

Retrieval can only return what chunking made findable. A chunk is the unit that
gets embedded, retrieved and pasted into the prompt, so a boundary in the wrong
place either splits one answer across two chunks or buries it among unrelated text.

**Fixed-size chunking** cuts every N characters. It knows nothing about the text
and will happily cut mid-sentence. **Semantic chunking** puts the boundaries where
the *meaning* changes.

## The algorithm

`chunking/semantic_chunking.py` is written as one function per step:

| Step | Function | What it does |
| --- | --- | --- |
| 1 | `split_into_sentences` | Split on whitespace after `.` `!` `?`. The sentence is the unit of meaning. |
| 2 | `embed_sentences` | One normalised vector per sentence (384-d, `all-MiniLM-L6-v2`). |
| 3 | `compare_neighbours` | Cosine similarity between each **adjacent** pair — `n - 1` numbers. |
| 4+5 | `group_sentences` | Walk in order; cut where similarity drops below the threshold. |
| — | `pick_threshold` | Decide what "drops too low" means for *this* document. |

Because only neighbours are compared, this is a **topic-change detector**, not
clustering. Sentence order is preserved and chunks are always contiguous.

Embeddings are normalised to length 1, so the cosine similarity is a plain dot
product:

```python
float(np.dot(embeddings[i], embeddings[i + 1]))
```

### The threshold is relative, not absolute

```python
def pick_threshold(similarities, cut_fraction=0.15):
    return float(np.percentile(similarities, cut_fraction * 100))
```

The weakest 15% of neighbour similarities become cuts. A hard-coded constant like
`0.55` looks tidy but silently breaks on a new encoder or a new corpus, because
similarity distributions are not comparable across models. A percentile adapts.

The trade-off: `cut_fraction` controls the *number* of cuts, not their quality. On
a document with no real topic shift it will still cut somewhere — at the least
similar pair, which may be an arbitrary seam. It also makes chunk count roughly
proportional to document length, which is predictable but not adaptive.

## Watch it run

```bash
PYTHONPATH=. python chunking/semantic_chunking.py
```

`show_steps` prints the similarity of each sentence to its predecessor and marks
the cuts, on a call-log document from `data/corpus.py`:

```text
support-call-2291

step 1  10 sentences
step 2  10 embeddings of 384 dimensions
step 3  9 neighbour similarities
step 4  keep together when similarity >= 0.13
step 5  cut when similarity <  0.13

               Call opened at 09:14.
    0.09  CUT  Customer says the unit stopped charging about a week after delivery ...
    0.24       I checked the order and delivery was confirmed on the 3rd ...
    0.18       Worth noting the cable itself is only covered for 6 months ...
    0.23       Customer asked whether they could just return it instead of claiming warranty ...
    0.12  CUT  They were not happy about that.
    0.14       I offered the warranty route and they agreed.
    0.34       Opened claim WC-4417 in the portal ...
    0.31       Customer then asked about the parcel they sent back last month ...
    0.40       Call closed 09:31, customer satisfied by the end.
```

Three things are worth reading off that output:

- **The cuts are defensible.** The second one separates diagnosis from resolution.
- **Neighbour similarities are low overall** (0.09–0.40). Adjacent sentences in a
  real call log are not near-duplicates, which is exactly why an absolute
  threshold would have been guesswork.
- **The first cut produces a useless chunk.** `"Call opened at 09:14."` is a
  21-character chunk of its own. It is *correct* — a timestamp really is unlike
  the sentence after it — and still bad for retrieval. Part 3 shows what it costs.

---

# Part 3 — Evaluating a chunking strategy

```bash
PYTHONPATH=. python evaluation/semantic_chunking_evaluation.py
```

Same corpus, same questions, same retriever, same K. The **only** variable is
where the boundaries fall, so any difference in Recall@K is the chunking's doing.

## Ground truth at the evidence level

Part 1 labelled relevance by document id. That cannot work here: chunk ids change
with every strategy, so they are not comparable across strategies. What does not
change is the *sentences that must reach the model*:

```python
{
    "id": "q5",
    "query": "How long before a missing parcel counts as lost?",
    "evidence": [
        "A parcel is treated as lost after 14 days with no carrier scan.",
        "We reship lost parcels at no cost once the carrier confirms the loss.",
    ],
}
```

Recall@K is then the fraction of a question's evidence sentences that appear
**verbatim** somewhere in the concatenated top-K chunks. Each question carries two
evidence sentences, so a per-question score is `0.00`, `0.50` or `1.00`.

The check is a normalised substring match (lowercase, punctuation stripped), which
makes it strict in a useful way: a chunk boundary that severs an evidence sentence
means that sentence is **not** found, even if both halves were retrieved. That is
the right call — half a sentence is often not usable evidence.

## The corpus is mixed on purpose

`data/corpus.py` holds six documents of two kinds:

| Kind | Documents | Why it is there |
| --- | --- | --- |
| `structured` | returns, shipping, warranty policies | Markdown headings mark exactly where topics change — semantic chunking should find them. |
| `prose` | two call logs, one incident note | No headings, topics drift mid-paragraph — the case headings cannot help with. |

A corpus of only one kind cannot distinguish the strategies.

## Results

```text
question       fixed 400      semantic
--------------------------------------
q1                  1.00          1.00
q2                  1.00          1.00
q3                  1.00          1.00
q4                  1.00          1.00
q5                  0.50          1.00
q6                  1.00          1.00
q7                  0.50          0.00
q8                  1.00          1.00
q9                  1.00          1.00
q10                 1.00          1.00

strategy              recall@3   chunks
---------------------------------------
fixed 400                 0.90       15
semantic                  0.90       13
```

**A tie — and the tie is the interesting result.** Semantic chunking is widely
sold as a straight upgrade. Here it wins one question, loses another, and lands
in exactly the same place. Reporting only the bottom line would hide everything
that actually happened.

### Where semantic chunking wins: q5

Fixed-size chunking cut at character 400, which landed in the middle of the
evidence sentence:

```text
retrieved [1]  " days with no carrier scan. We reship lost parcels at no cost once ..."
```

`"A parcel is treated as lost after 14"` is stranded at the end of the previous
chunk, which did not make the top 3. The sentence was retrieved in *halves* and
counted as missing — correctly, because neither half answers the question.

Semantic chunking respects sentence boundaries by construction, so it retrieved
the whole `## Lost parcels` section as one 149-character chunk and scored `1.00`.

### Where semantic chunking loses: q7

This is the orphan chunk from Part 2, and it is worth reading closely.

```text
q7: "What happened in the ingestion incident and what did it cost us?"

rank  score  size  chunk
   1  0.693   40   "Rough notes from the ingestion incident."
   2  0.260  331   "## What is excluded  Accidental damage, liquid damage ..."
   3  0.226  415   "Customer called about an international order placed on the 12th ..."
  ...
   6  0.132  818   "The nightly job reported success but the document count ... "   <- all the evidence
```

Two failure modes compound:

1. **A tiny chunk is a similarity magnet.** The 40-character opening sentence is
   almost pure topic words, so it scores `0.693` — nearly triple the runner-up —
   while containing no answer at all. Short chunks have nothing to dilute them,
   so they dominate cosine ranking. It consumed a top-3 slot and returned nothing.
2. **A large chunk is diluted.** The remaining 818 characters, which hold *both*
   evidence sentences, mix the parser bug, the guard that was added, the 4×
   reindex cost and the context-budget concern. Averaged into one vector, the
   match to any single question is weak — it ranks **6th at 0.132**, below three
   chunks about warranty exclusions and international shipping.

Fixed-size chunking scored `0.50` here not by being smarter but by being blunt:
its 400-character cut kept the opening sentence attached to the parser bug, so
retrieving that chunk delivered one evidence sentence for free.

### Chunk-size distribution

This is the underlying trade, and it is why the incident above happens:

| Strategy | Chunks | Min | Mean | Max |
| --- | --- | --- | --- | --- |
| fixed 400 | 15 | 47 | 293 | 400 |
| semantic | 13 | 21 | 337 | 818 |

Fixed-size chunking gives up meaning to buy a predictable context budget.
Semantic chunking buys meaningful boundaries and gives up that predictability —
a 39× spread between its smallest and largest chunk. Both ends of that spread
cost recall: the 21- and 40-character chunks rank on topic words alone, and the
818-character chunk is too mixed to rank at all.

### What would actually fix it

The failures above are not arguments against semantic chunking. They point at
the guardrails a production implementation needs, none of which are in this repo:

- **Merge undersized chunks** into a neighbour (a `min_chunk_chars` floor). This
  alone removes both similarity magnets.
- **Cap oversized chunks** by re-splitting at the weakest internal boundary, so
  the 818-character chunk becomes two rankable ones.
- **Overlap adjacent chunks** by a sentence or two, so evidence that straddles a
  boundary survives.
- **Split on headings first** for `structured` documents, then chunk semantically
  within each section. Headings are ground truth about topic structure; ignoring
  them and re-deriving boundaries from embeddings is strictly worse.
- **Retrieve small, expand to the parent** — rank on narrow chunks, then pass the
  surrounding section to the LLM. This decouples "what ranks well" from "what the
  model needs to read", which is the real tension in every result above.

## Reading these numbers honestly

Both datasets are tiny by design, so the calculations can be checked by hand.
Ten questions at three possible scores each means one question moving is worth
`0.10` of mean recall — comfortably larger than the gap between the strategies.
**This experiment cannot establish that either strategy is better.** What it can
do is show precisely *how* each one fails, which is the more transferable lesson.

A production evaluation would:

- Use a larger, representative corpus with realistic distractor documents.
- Use human-reviewed relevance labels over many representative queries.
- Sweep K (Recall@1, @3, @5, @10) rather than reporting a single K.
- Sweep the knobs too — `cut_fraction` and `FIXED_SIZE` are both hard-coded here.
- State its averaging method, and report a confidence interval, not one number.
- Add a rank-aware metric (MRR, NDCG) — Recall@K cannot see that q7's evidence
  chunk was 6th rather than 13th.
- Track latency and cost next to quality. Semantic chunking embeds every sentence
  at ingest; the incident note in this corpus records that as roughly 4× the
  previous reindex wall clock.
- Measure end to end. Better retrieval that does not improve answers is not better.

---

## What to take away

- **Recall@K measures coverage, not ranking.** If the evidence is not retrieved,
  no amount of prompt engineering recovers it.
- **Say how you averaged.** Macro `0.88` and micro `0.80` describe the same run.
- **Equal scores do not mean equal behaviour.** Two retrievers tie at `0.88`
  here; two chunkers tie at `0.90` while failing on completely different
  questions. Always look at the per-query breakdown.
- **Chunking is a retrieval decision, not a preprocessing detail.** It sets the
  ceiling the retriever works under.
- **Chunk-size variance is the hidden cost of semantic chunking.** Chunks too
  small rank on topic words and carry no answer; chunks too large are too diluted
  to rank. Floors, caps and overlap are not optional extras.
- **Evaluate the change you actually made.** One corpus, one retriever, one K,
  one variable — otherwise the number cannot be attributed to anything.
