import os

from langchain_ollama import ChatOllama
from google import genai

from src.retrieval import retrieve_context


def build_prompt(question, retrieved_documents):

    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_documents
    )

    return f"""
You are a document-grounded question-answering assistant.

Rules:
- Use only the retrieved context.
- Do not use outside knowledge.
- If the answer is not present in the context, respond:
  "The answer is not available in the provided documents."
- Give a concise answer.
- Mention source file names used.

Retrieved Context:
{context}

Question:
{question}
"""


def generate_with_ollama(
    prompt,
    model_name="llama3.2:3b"
):
    llm = ChatOllama(
        model=model_name,
        temperature=0
    )

    response = llm.invoke(prompt)

    return response.content


def generate_with_gemini(
    prompt,
    model_name="gemini-2.5-flash"
):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing."
        )

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    return response.text


def answer_question(
    question,
    retrieval_mode,
    generation_mode,
    vectordb,
    bm25_retriever=None,
    graph=None,
    chunk_lookup=None
):
    if not question.strip():
        raise ValueError(
            "Please enter a question."
        )

    docs = retrieve_context(
        query=question,
        retrieval_mode=retrieval_mode,
        vectordb=vectordb,
        bm25_retriever=bm25_retriever,
        graph=graph,
        chunk_lookup=chunk_lookup
    )

    if not docs:
        return {
            "answer":
            "The answer is not available in the provided documents.",
            "sources": [],
            "retrieved_chunks": [],
            "retrieval_mode": retrieval_mode,
            "generation_mode": generation_mode,
        }

    prompt = build_prompt(
        question,
        docs
    )

    if generation_mode == "ollama":

        answer = generate_with_ollama(prompt)

    elif generation_mode == "gemini":

        answer = generate_with_gemini(prompt)

    else:
        raise ValueError(
            f"Unsupported generation mode: {generation_mode}"
        )

    sources = list({
        doc.metadata.get("source", "Unknown")
        for doc in docs
    })

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": docs,
        "retrieval_mode": retrieval_mode,
        "generation_mode": generation_mode,
    }