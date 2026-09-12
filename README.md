# Understanding Recall@K for RAG Retrieval

This example explains how to evaluate a retriever using **Recall@K**. It compares a BM25 retriever with a dense retriever, but its main purpose is to explain the evaluation metricâ€”not to prove that one retriever is better.

## What problem does Recall@K solve?

Before an LLM can answer a question, the retriever must find the relevant evidence.

Recall@K answers this question:

> Of all the documents known to be relevant, how many did the retriever find within its top K results?

For example, `Recall@3` checks whether the relevant documents appeared anywhere within the first three retrieved results.

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

Each evaluation query has a set of documents that humans have identified as relevant. This is called the **ground truth**.

```python
{
    "query": "How can I reduce AWS Glue startup time?",
    "relevant_doc_ids": {"doc-2", "doc-5"},
}
```

Here, two documents are considered relevant. A retriever must find both to achieve perfect recall for this query.

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

Recall@K considers the entire top-K result set. It does not measure the exact position of a relevant document.

## Why does 1.00 not mean ranked first

The word **recall** refers to coverage, not ranking position.

Recall@3 asks:

> Did we retrieve all relevant documents somewhere in the top three?

It does not ask:

> Did we rank the best document first?

Use metrics such as **MRR** or **NDCG** when ranking position matters.

## Mean Recall@3

The sample execution produces these query-level scores:

```text
0.50, 1.00, 1.00, 1.00
```

Mean Recall@3 is the average across all evaluation queries:

```text
(0.50 + 1.00 + 1.00 + 1.00) / 4 = 0.875
```

Rounded to two decimal places:

```text
Mean Recall@3 = 0.88
```

means that on a typical query, the retriever found 88% of that query's relevant documents. Note this weights every query equally regardless of how many relevant documents it has. 
Pooling all relevant documents instead, micro-averaging: gives 4/5 = 0.80

## Why BM25 and dense retrieval can receive the same score

BM25 and dense retrieval may return different documents or rank them differently. However, they receive the same Recall@K score when they find the same number of relevant documents within the top K.

```text
Relevant: ['doc-2', 'doc-5']

BM25:  ['doc-2', 'doc-4', 'doc-3']
Dense: ['doc-2', 'doc-4', 'doc-1']
```

Both retrievers found only `doc-2`. Therefore, both receive:

```text
Recall@3 = 1 / 2 = 0.50
```

Recall@K ignores the non-relevant documents except for the fact that they may occupy positions that could have contained relevant documents.

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

## About this dataset

This repository uses a deliberately small dataset so that the Recall@K calculation is easy to follow manually. It is suitable for learning the fundamental, but it is not large enough to establish that BM25 is better or worse than dense retrieval.

In a production evaluation:

- Use a larger, representative document collection.
- Include realistic distractor documents.
- Create human-reviewed relevance labels.
- Evaluate many representative queries.
- Compare several values, such as Recall@1, Recall@3, Recall@5, and Recall@10.
- Review latency and cost alongside retrieval quality.

## Key takeaway

> Recall@K measures whether the retriever found the relevant evidence within its top K results. It measures coverage, not ranking position.

If the correct evidence is not retrieved, the LLM cannot reliably use itâ€”regardless of how well the prompt is written.
