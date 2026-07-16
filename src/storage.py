import os
import shutil

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "training_institute_rag"


def get_embedding_model():
    """
    Load local Hugging Face embedding model.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Embedding model loaded.")

    return embeddings


def create_vector_database(chunks, embeddings):
    """
    Create and persist ChromaDB.
    """

    # Delete existing database if present
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        print("Existing ChromaDB removed.")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME
    )

    print(f"Chunks stored in ChromaDB: {len(chunks)}")
    print(f"ChromaDB location: {CHROMA_DIR}")

    return vectordb


def load_vector_database(embeddings):
    """
    Load existing ChromaDB.
    """

    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            "Vector database not found. Run python ingest.py first."
        )

    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    return vectordb
