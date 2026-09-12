import re
import numpy as np
from rank_bm25 import BM25Okapi
from data.dataset import documents

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


# BM25 expects tokenized documents.
tokenized_documents = [
    tokenize(document["text"])
    for document in documents
]

bm25 = BM25Okapi(tokenized_documents)

def retrieve_bm25(query, k=3):
    query_tokens = tokenize(query)

    # Calculate the BM25 score between the query and every document.
    scores = bm25.get_scores(query_tokens)

    # Sort documents from highest to lowest score.
    ranked_indexes = np.argsort(scores)[::-1][:k]

    return [
        {
            "id": documents[index]["id"],
            "text": documents[index]["text"],
            "score": float(scores[index]),
        }
        for index in ranked_indexes
    ]

if __name__ == "__main__":
    query = "How can I reduce AWS Glue startup time?"

    results = retrieve_bm25(query, k=3)

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['id']} "
            f"| Score: {result['score']:.4f}"
        )
        print(f"   {result['text']}")