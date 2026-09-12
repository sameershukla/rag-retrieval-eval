def recall_at_k(retrieved_documents, relevant_doc_ids):
    retrieved_ids = {
        document["id"]
        for document in retrieved_documents
    }

    relevant_retrieved = retrieved_ids.intersection(
        relevant_doc_ids
    )

    return len(relevant_retrieved) / len(relevant_doc_ids)