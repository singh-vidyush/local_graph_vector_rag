# src/graph_builder.py

import os
import json
import networkx as nx

from networkx.readwrite import json_graph
from sklearn.metrics.pairwise import cosine_similarity


GRAPH_PATH = "graph_db/chunk_graph.json"


def build_chunk_graph(chunks, embeddings, similarity_threshold=0.75):
    """
    Build graph with:
    1. Chunk nodes
    2. Sequential edges
    3. Semantic similarity edges
    """

    graph = nx.Graph()

    # ---------------------------
    # Add nodes
    # ---------------------------
    for chunk in chunks:

        chunk_id = chunk.metadata["chunk_id"]

        graph.add_node(
            chunk_id,
            source=chunk.metadata.get("source"),
            chunk_index=chunk.metadata.get("chunk_index"),
            file_type=chunk.metadata.get("file_type"),
            text_preview=chunk.page_content[:150]
        )

    # ---------------------------
    # Sequential edges
    # ---------------------------
    sequential_edges = 0

    for i in range(len(chunks) - 1):

        current = chunks[i]
        next_chunk = chunks[i + 1]

        if (
            current.metadata.get("source")
            == next_chunk.metadata.get("source")
        ):

            graph.add_edge(
                current.metadata["chunk_id"],
                next_chunk.metadata["chunk_id"],
                relationship="next_chunk"
            )

            sequential_edges += 1

    # ---------------------------
    # Create embeddings
    # ---------------------------
    chunk_texts = [chunk.page_content for chunk in chunks]

    vectors = embeddings.embed_documents(chunk_texts)

    # ---------------------------
    # Semantic edges
    # ---------------------------
    similarities = cosine_similarity(vectors)

    semantic_edges = 0

    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):

            score = similarities[i][j]

            if score >= similarity_threshold:

                graph.add_edge(
                    chunks[i].metadata["chunk_id"],
                    chunks[j].metadata["chunk_id"],
                    relationship="semantic",
                    weight=float(score)
                )

                semantic_edges += 1

    print(f"Graph nodes: {graph.number_of_nodes()}")
    print(f"Sequential edges: {sequential_edges}")
    print(f"Semantic edges: {semantic_edges}")

    return graph


def save_graph(graph, graph_path=GRAPH_PATH):

    os.makedirs(os.path.dirname(graph_path), exist_ok=True)

    data = json_graph.node_link_data(graph)

    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("Graph saved successfully.")


def load_graph(graph_path=GRAPH_PATH):

    if not os.path.exists(graph_path):
        raise FileNotFoundError(
            "Graph database not found. Run python ingest.py first."
        )

    with open(graph_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return json_graph.node_link_graph(data)