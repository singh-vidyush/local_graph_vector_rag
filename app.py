from flask import Flask, render_template, request

from src.storage import (
    get_embedding_model,
    load_vector_database
)

from src.graph_builder import load_graph

from src.retrieval import build_bm25_retriever

from src.generation import answer_question

from src.document_loader import FileLoader
from src.chunking import Chunker


app = Flask(__name__)


# ---------------------------
# Load resources at startup
# ---------------------------

try:
    embeddings = get_embedding_model()

    vectordb = load_vector_database(
        embeddings
    )

    graph = load_graph()

    documents = FileLoader("data")

    chunks = Chunker(
        documents,
        method="semantic"
    )

    bm25_retriever = build_bm25_retriever(
        chunks
    )

    chunk_lookup = {
        chunk.metadata["chunk_id"]: chunk
        for chunk in chunks
    }

except Exception as e:
    print(e)

    vectordb = None
    graph = None
    bm25_retriever = None
    chunk_lookup = {}


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    error = None

    if request.method == "POST":

        question = request.form.get("question", "")

        retrieval_mode = request.form.get(
            "retrieval_mode",
            "similarity"
        )

        generation_mode = request.form.get(
            "generation_mode",
            "ollama"
        )

        try:

            if vectordb is None:
                raise Exception(
                    "Vector database not found. Run python ingest.py first."
                )

            result = answer_question(
                question=question,
                retrieval_mode=retrieval_mode,
                generation_mode=generation_mode,
                vectordb=vectordb,
                bm25_retriever=bm25_retriever,
                graph=graph,
                chunk_lookup=chunk_lookup,
            )

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        result=result,
        error=error
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )