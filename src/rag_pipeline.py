"""Importable wrapper around the existing notebook RAG pipeline.

The retrieval and generation steps intentionally mirror ``rag_demo.ipynb``:
chunk -> embed -> FAISS IndexFlatIP -> top-k retrieval -> construct_prompt ->
local LM generation.  This module only packages those existing steps so the
same pipeline can be called by the evaluation runner.
"""
from pathlib import Path
from typing import List, Tuple

from .chunking import chunk_text
from .embedding import Embedder
from .lm import LM
from .prompt_constructor import construct_prompt
from .vector_store import FaissVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MARKDOWN_PATH = PROJECT_ROOT / "data" / "document.md"


def _extract_completion(decoded_output: str, prompt: str) -> str:
    """Return the generated completion from LM.generate_text's full decode.

    ``LM.generate_text`` decodes the complete sequence (prompt + completion).
    Removing that echoed prompt does not change retrieval or generation; it
    only exposes the answer that the notebook prints after the prompt.
    """
    text = decoded_output
    if text.startswith("<bos>"):
        text = text[len("<bos>") :]

    prompt_pos = text.find(prompt)
    if prompt_pos >= 0:
        text = text[prompt_pos + len(prompt) :]

    text = text.strip()
    for suffix in ("<end_of_turn>", "<eos>"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].rstrip()
    return text


class LocalRAGPipeline:
    """Stateful, importable form of the existing local RAG notebook."""

    def __init__(self, markdown_path: str | Path = DEFAULT_MARKDOWN_PATH, top_k: int = 3):
        self.markdown_path = str(markdown_path)
        self.top_k = top_k

        # Same offline preparation as the notebook.
        chunks = chunk_text(self.markdown_path)
        self.embedder = Embedder()
        self.embedded_chunks = self.embedder.embed_chunks(chunks)
        embeddings = [chunk["embedding"] for chunk in self.embedded_chunks]

        self.vector_store = FaissVectorStore(len(embeddings[0]))
        self.vector_store.add(embeddings)
        self.lm = LM()

    def run(self, query: str) -> Tuple[str, List[str]]:
        """Run retrieval + generation and return ``(answer, chunk_texts)``."""
        query_embedding = self.embedder.embed_query(query)
        retrieved = self.vector_store.search(query_embedding, top_k=self.top_k)
        prompt = construct_prompt(query, retrieved, self.embedded_chunks)
        raw_output = self.lm.generate_text(prompt)
        answer = _extract_completion(raw_output, prompt)

        retrieved_contexts = [
            self.embedded_chunks[index]["text"]
            for index, _score in retrieved
            if 0 <= index < len(self.embedded_chunks)
        ]
        return answer, retrieved_contexts


_default_pipeline: LocalRAGPipeline | None = None


def run(query: str) -> Tuple[str, List[str]]:
    """Convenience function requested by the evaluation layer.

    The index and local model are initialized once per process and reused for
    subsequent questions.
    """
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = LocalRAGPipeline()
    return _default_pipeline.run(query)
