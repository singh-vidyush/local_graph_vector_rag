# ingest.py

from src.document_loader import FileLoader
from src.chunking import Chunker
from src.storage import (
    get_embedding_model,
    create_vector_database
)
from src.graph_builder import (
    build_chunk_graph,
    save_graph
)


def main():

    print("=" * 50)
    print("Starting Ingestion Pipeline")
    print("=" * 50)

    # -----------------------------
    # Load Documents
    # -----------------------------
    loader = FileLoader("data")
    documents = loader.loader()

    if not documents:
        raise Exception(
            "No documents found in data directory."
        )

    print(f"\nOriginal documents: {len(documents)}")

    # -----------------------------
    # Chunk Documents
    # -----------------------------
    chunker = Chunker(documents)

    overlap_chunks = chunker.recursive_overlap()
    semantic_chunks = chunker.semantic_chunk()

    print(f"\nOverlapping chunks: {len(overlap_chunks)}")
    print(f"Semantic chunks: {len(semantic_chunks)}")

    # Assignment recommends selecting one method
    selected_chunks = semantic_chunks

    print("Selected method for indexing: semantic")

    # -----------------------------
    # Embedding Model
    # -----------------------------
    embeddings = get_embedding_model()

    # -----------------------------
    # Create ChromaDB
    # -----------------------------
    vectordb = create_vector_database(
        selected_chunks,
        embeddings
    )

    # -----------------------------
    # Build Graph
    # -----------------------------
    graph = build_chunk_graph(
        selected_chunks,
        embeddings,
        similarity_threshold=0.75
    )

    # -----------------------------
    # Save Graph
    # -----------------------------
    save_graph(graph)

    print("\nIngestion completed successfully.")


if __name__ == "__main__":
    main()