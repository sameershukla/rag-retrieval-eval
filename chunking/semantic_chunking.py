"""Semantic chunking, one function per step.

    1. Split the document into sentences.
    2. Create an embedding for each sentence.
    3. Compare the meaning of neighbouring sentences.
    4. Keep similar sentences together.
    5. Start a new chunk when the meaning changes significantly.

Steps 4 and 5 are two halves of the same loop, so they share a function.
Run this file to see the output of every step on one document.
"""

import re

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


# 1. Split the document into sentences ---------------------------------------

# Whitespace that comes right after . ! or ? ends a sentence. The lookbehind
# keeps the punctuation on the sentence instead of throwing it away.
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_into_sentences(text):
    """The sentence is the unit of meaning we will compare."""
    flat = " ".join(text.split())  # newlines and double spaces become one space
    return [s.strip() for s in SENTENCE_END.split(flat) if s.strip()]


# 2. Create an embedding for each sentence -----------------------------------


def embed_sentences(sentences, model):
    """One vector per sentence. Similar meaning => vectors point the same way.

    normalize_embeddings=True scales every vector to length 1, so in step 3
    a plain dot product is the cosine similarity.
    """
    return model.encode(sentences, normalize_embeddings=True)


# 3. Compare the meaning of neighbouring sentences ---------------------------


def compare_neighbours(embeddings):
    """Cosine similarity between each sentence and the one before it.

    Returns len(sentences) - 1 numbers. similarities[i] is how close
    sentence i + 1 is to sentence i: near 1.0 means same topic, near 0.0
    means unrelated. Only neighbours are compared, which is what makes this
    a topic-change detector rather than a clustering step.
    """
    return [
        float(np.dot(embeddings[i], embeddings[i + 1]))
        for i in range(len(embeddings) - 1)
    ]


# 4 + 5. Keep similar sentences together, cut when the meaning changes -------


def group_sentences(sentences, similarities, threshold):
    """Walk the sentences in order and cut wherever similarity drops too low.

    A gap below `threshold` means the meaning changed significantly, so the
    current chunk is closed and the next sentence starts a fresh one.
    Otherwise the sentence joins the chunk it is already in.
    """
    chunks, current = [], [sentences[0]]
    for i, similarity in enumerate(similarities):
        if similarity < threshold:
            chunks.append(" ".join(current))
            current = []
        current.append(sentences[i + 1])
    chunks.append(" ".join(current))
    return chunks


def pick_threshold(similarities, cut_fraction=0.15):
    """What counts as 'significantly' is relative to this document.

    The weakest `cut_fraction` of neighbour similarities become cuts. A fixed
    number like 0.55 would need retuning for every new encoder or corpus;
    a fraction adapts on its own.
    """
    return float(np.percentile(similarities, cut_fraction * 100))


# The whole pipeline ---------------------------------------------------------


def semantic_chunk(text, model, cut_fraction=0.15):
    sentences = split_into_sentences(text)                    # 1
    if len(sentences) < 2:
        return sentences
    embeddings = embed_sentences(sentences, model)            # 2
    similarities = compare_neighbours(embeddings)             # 3
    threshold = pick_threshold(similarities, cut_fraction)
    return group_sentences(sentences, similarities, threshold)  # 4 + 5


def show_steps(text, model, cut_fraction=0.15):
    sentences = split_into_sentences(text)
    embeddings = embed_sentences(sentences, model)
    similarities = compare_neighbours(embeddings)
    threshold = pick_threshold(similarities, cut_fraction)

    print(f"step 1  {len(sentences)} sentences")
    print(f"step 2  {embeddings.shape[0]} embeddings of {embeddings.shape[1]} dimensions")
    print(f"step 3  {len(similarities)} neighbour similarities")
    print(f"step 4  keep together when similarity >= {threshold:.2f}")
    print(f"step 5  cut when similarity <  {threshold:.2f}\n")

    print(f"  {'':>6}       {sentences[0]}")
    for i, similarity in enumerate(similarities):
        marker = "CUT" if similarity < threshold else "   "
        print(f"  {similarity:>6.2f}  {marker}  {sentences[i + 1]}")

    print("\nchunks:\n")
    for i, chunk in enumerate(group_sentences(sentences, similarities, threshold), 1):
        print(f"  [{i}] {chunk}\n")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.corpus import DOCUMENTS

    model = SentenceTransformer(MODEL_NAME)
    document = next(d for d in DOCUMENTS if d["kind"] == "prose")
    print(f"{document['id']}\n")
    show_steps(document["text"], model)
