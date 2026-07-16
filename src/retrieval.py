from langchain_community.retrievers import BM25Retriever


def build_bm25_retriever(chunks, k=4):
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = k
    return retriever


def dense_retrieve(vectordb, query, k=4):
    docs = vectordb.similarity_search(query, k=k)

    for doc in docs:
        doc.metadata["retrieval_method"] = "similarity"

    return docs


def mmr_retrieve(vectordb, query, k=4, fetch_k=10):
    docs = vectordb.max_marginal_relevance_search(
        query=query,
        k=k,
        fetch_k=fetch_k
    )

    for doc in docs:
        doc.metadata["retrieval_method"] = "mmr"

    return docs


def bm25_retrieve(bm25_retriever, query):
    docs = bm25_retriever.invoke(query)

    for doc in docs:
        doc.metadata["retrieval_method"] = "bm25"

    return docs


def graph_hybrid_retrieve(
    query,
    vectordb,
    graph,
    chunk_lookup,
    seed_k=2,
    max_results=6
):
    seed_docs = vectordb.similarity_search(
        query,
        k=seed_k
    )

    retrieved = {}

    for doc in seed_docs:

        chunk_id = doc.metadata["chunk_id"]

        retrieved[chunk_id] = doc

        if graph.has_node(chunk_id):

            for neighbor_id in graph.neighbors(chunk_id):

                if neighbor_id in chunk_lookup:
                    retrieved[neighbor_id] = chunk_lookup[neighbor_id]

    final_docs = list(retrieved.values())[:max_results]

    for doc in final_docs:
        doc.metadata["retrieval_method"] = "graph_hybrid"

    return final_docs


def retrieve_context(
    query,
    retrieval_mode,
    vectordb,
    bm25_retriever=None,
    graph=None,
    chunk_lookup=None
):
    if retrieval_mode == "similarity":
        return dense_retrieve(vectordb, query)

    elif retrieval_mode == "mmr":
        return mmr_retrieve(vectordb, query)

    elif retrieval_mode == "bm25":
        return bm25_retrieve(bm25_retriever, query)

    elif retrieval_mode == "graph_hybrid":
        return graph_hybrid_retrieve(
            query,
            vectordb,
            graph,
            chunk_lookup
        )

    else:
        raise ValueError(
            f"Unsupported retrieval mode: {retrieval_mode}"
        )