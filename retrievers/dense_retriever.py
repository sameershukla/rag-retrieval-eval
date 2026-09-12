import numpy as np
from sentence_transformers import SentenceTransformer
from data.dataset import documents


# Load a pretrained embedding model.
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert every document into an embedding.
document_embeddings = model.encode(
    [document["text"] for document in documents],
    normalize_embeddings=True,
)


def retrieve_dense(query, k=3):
    # Convert the query into an embedding.
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    # Since the embeddings are normalized,
    # the dot product gives cosine similarity.
    scores = document_embeddings @ query_embedding

    # Sort documents from highest to lowest similarity.
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

    results = retrieve_dense(query, k=3)

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['id']} "
            f"| Score: {result['score']:.4f}"
        )
        print(f"   {result['text']}")