"""Does semantic chunking help retrieval? Recall@K against a fixed size baseline.

Same corpus, same questions, same retriever. The only thing that changes is
where the chunk boundaries are, so any difference in Recall@K is the
chunking's doing.

Recall@K here is the fraction of a question's evidence sentences that appear
somewhere in the top K retrieved chunks. Each question has two evidence
sentences, so per question the score is 0.0, 0.5 or 1.0.
"""

import re
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chunking.semantic_chunking import MODEL_NAME, semantic_chunk
from data.corpus import DOCUMENTS, QUESTIONS

K = 3
FIXED_SIZE = 400


def fixed_chunk(text, size=FIXED_SIZE):
    """The baseline. Cuts every `size` characters and knows nothing else."""
    flat = " ".join(text.split())
    return [flat[i:i + size] for i in range(0, len(flat), size)]


def normalise(text):
    """Lowercase, strip punctuation, so a substring check ignores formatting."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def recall_at_k(retrieved_chunks, evidence):
    haystack = normalise(" ".join(retrieved_chunks))
    found = sum(1 for sentence in evidence if normalise(sentence) in haystack)
    return found / len(evidence)


def retrieve(query_embedding, chunk_embeddings, chunks, k=K):
    """Dense retrieval: cosine similarity between the query and every chunk."""
    scores = chunk_embeddings @ query_embedding
    top = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in top]


def evaluate(chunks, model):
    """Recall@K for every question, given one way of chunking the corpus."""
    chunk_embeddings = model.encode(chunks, normalize_embeddings=True)
    per_question = {}
    for question in QUESTIONS:
        query_embedding = model.encode(question["query"], normalize_embeddings=True)
        top = retrieve(query_embedding, chunk_embeddings, chunks)
        per_question[question["id"]] = recall_at_k(top, question["evidence"])
    return per_question


def main():
    model = SentenceTransformer(MODEL_NAME)

    fixed, semantic = [], []
    for document in DOCUMENTS:
        fixed.extend(fixed_chunk(document["text"]))
        semantic.extend(semantic_chunk(document["text"], model))

    results = {
        f"fixed {FIXED_SIZE}": (fixed, evaluate(fixed, model)),
        "semantic": (semantic, evaluate(semantic, model)),
    }

    # per question, so you can see which ones each strategy misses
    print(f"{'question':<10}" + "".join(f"{name:>14}" for name in results))
    print("-" * (10 + 14 * len(results)))
    for question in QUESTIONS:
        row = "".join(f"{scores[question['id']]:>14.2f}" for _, scores in results.values())
        print(f"{question['id']:<10}{row}")

    print()
    print(f"{'strategy':<20}{f'recall@{K}':>10}{'chunks':>9}")
    print("-" * 39)
    for name, (chunks, scores) in results.items():
        mean = sum(scores.values()) / len(scores)
        print(f"{name:<20}{mean:>10.2f}{len(chunks):>9}")


if __name__ == "__main__":
    main()
