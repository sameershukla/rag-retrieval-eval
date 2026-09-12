from data.dataset import evaluation_set
from evaluation.recall_at_k import recall_at_k
from retrievers.bm25_retriever import retrieve_bm25
from retrievers.dense_retriever import retrieve_dense


def evaluate(name, retrieve_function, k):
    recalls = []

    print(f"\n{name}")
    print("=" * 50)

    for example in evaluation_set:
        results = retrieve_function(example["query"], k)

        recall = recall_at_k(
            retrieved_documents=results,
            relevant_doc_ids=example["relevant_doc_ids"],
        )

        recalls.append(recall)

        print(f"\nQuery: {example['query']}")
        print(f"Relevant: {sorted(example['relevant_doc_ids'])}")
        print(f"Retrieved: {[result['id'] for result in results]}")
        print(f"Recall@{k}: {recall:.2f}")

    mean_recall = sum(recalls) / len(recalls)

    print(f"\nMean Recall@{k}: {mean_recall:.2f}")


if __name__ == "__main__":
    evaluate(
        name="BM25 Retriever",
        retrieve_function=retrieve_bm25,
        k=3,
    )

    evaluate(
        name="Dense Retriever",
        retrieve_function=retrieve_dense,
        k=3,
    )