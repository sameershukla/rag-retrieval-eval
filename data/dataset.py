documents = [
    {
        "id": "doc-1",
        "text": "AWS Glue Flex execution reduces cost for non-urgent workloads."
    },
    {
        "id": "doc-2",
        "text": "Keeping AWS Glue resources warm can reduce job startup latency."
    },
    {
        "id": "doc-3",
        "text": "AWS Glue auto scaling adjusts compute resources during execution."
    },
    {
        "id": "doc-4",
        "text": "Insufficient capacity can delay the startup of an AWS Glue job."
    },
    {
        "id": "doc-5",
        "text": "Provisioned capacity provides dedicated resources for predictable workloads."
    },
    {
        "id": "doc-6",
        "text": "Amazon S3 stores files, logs, and data-lake datasets."
    },
]

evaluation_set = [
    {
        "query": "How can I reduce AWS Glue startup time?",
        "relevant_doc_ids": {"doc-2", "doc-5"},
    },
    {
        "query": "Why is my Glue job waiting before it starts?",
        "relevant_doc_ids": {"doc-4"},
    },
    {
        "query": "How can I lower the cost of a non-urgent Glue job?",
        "relevant_doc_ids": {"doc-1"},
    },
    {
        "query": "How does Glue change compute resources automatically?",
        "relevant_doc_ids": {"doc-3"},
    },
]