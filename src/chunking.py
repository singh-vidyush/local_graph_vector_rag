from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


class Chunker:

    def __init__(self, documents):
        self.documents = documents

    def _add_metadata(self, chunks):

        for idx, chunk in enumerate(chunks, start=1):

            source = chunk.metadata.get("source", "unknown")

            filename = source.split("/")[-1].replace(".", "_")

            chunk.metadata["chunk_index"] = idx

            chunk.metadata["chunk_id"] = (
                f"{filename}_chunk_{idx:04d}"
            )

        return chunks

    def recursive_overlap(self):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(
            self.documents
        )

        chunks = self._add_metadata(chunks)

        print(
            f"Overlapping chunks: {len(chunks)}"
        )

        return chunks

    def semantic_chunk(self):

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile"
        )

        chunks = splitter.split_documents(
            self.documents
        )

        chunks = self._add_metadata(chunks)

        print(
            f"Semantic chunks: {len(chunks)}"
        )

        return chunks
