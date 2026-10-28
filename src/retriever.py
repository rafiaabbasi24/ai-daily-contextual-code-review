import os
import json
import logging
import pathlib
import pickle
from typing import List, Tuple, Dict, Optional

import click
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Configure module level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _is_text_file(path: pathlib.Path) -> bool:
    """
    Heuristically decide whether a file is a text file based on its suffix.
    """
    text_suffixes = {
        ".py",
        ".java",
        ".js",
        ".ts",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".cs",
        ".go",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".rs",
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
    }
    return path.suffix.lower() in text_suffixes


def _read_file(path: pathlib.Path) -> str:
    """
    Read a file using UTF-8 encoding, falling back to latin-1 if needed.
    """
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


class CodeRetriever:
    """
    Embeds source code / documentation files and provides similarity search
    using a FAISS index.

    Parameters
    ----------
    model_name : str, optional
        Name of the SentenceTransformer model to use. Defaults to
        ``"all-MiniLM-L6-v2"``.
    device : str, optional
        Device identifier for the transformer model (e.g., ``"cpu"``,
        ``"cuda"``, ``"cuda:0"``). If ``None``, the model decides automatically.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.index: Optional[faiss.Index] = None
        # Mapping from internal integer IDs (FAISS) to document metadata
        self.id_to_meta: Dict[int, Dict] = {}
        self._next_id: int = 0

    # --------------------------------------------------------------------- #
    # Embedding utilities
    # --------------------------------------------------------------------- #

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Compute L2‑normalized embeddings for a list of texts.

        Parameters
        ----------
        texts : List[str]
            Texts to embed.

        Returns
        -------
        np.ndarray
            2‑D array of shape ``(len(texts), dim)`` with dtype ``float32``.
        """
        if not texts:
            return np.empty((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)

        logger.debug("Embedding %d texts with model %s", len(texts), self.model_name)
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device,
        )
        # Ensure dtype compatibility with FAISS
        return embeddings.astype(np.float32)

    # --------------------------------------------------------------------- #
    # Index management
    # --------------------------------------------------------------------- #

    def _ensure_index(self, dim: int):
        """
        Initialise a FAISS index if it does not exist yet.
        Uses inner product (IP) on L2‑normalized vectors which is equivalent to cosine similarity.
        """
        if self.index is None:
            logger.info("Creating new FAISS IndexFlatIP with dimension %d", dim)
            self.index = faiss.IndexFlatIP(dim)
        elif self.index.d != dim:
            raise ValueError(f"Existing index dimension ({self.index.d}) does not match embedding dimension ({dim}).")

    def add_documents(self, docs: List[str], meta: Optional[List[Dict]] = None):
        """
        Add a batch of documents to the index.

        Parameters
        ----------
        docs : List[str]
            Raw text of the documents.
        meta : List[Dict] | None
            Optional metadata for each document (e.g., file path). Must have the same length as ``docs``.
        """
        if meta is None:
            meta = [{} for _ in docs]
        if len(docs) != len(meta):
            raise ValueError("Length of docs and meta must be identical.")

        embeddings = self.embed_texts(docs)
        self._ensure_index(embeddings.shape[1])

        ids = np.arange(self._next_id, self._next_id + len(docs), dtype=np.int64)
        self.index.add_with_ids(embeddings, ids)

        for i, m in zip(ids, meta):
            self.id_to_meta[int(i)] = m

        self._next_id += len(docs)
        logger.info("Added %d documents to the index (total=%d).", len(docs), self._next_id)

    def search(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Retrieve the top‑k most similar documents for a query string.

        Parameters
        ----------
        query : str
            The search query.
        k : int, optional
            Number of results to return. Defaults to ``5``.

        Returns
        -------
        List[Tuple[Dict, float]]
            List of ``(metadata, score)`` tuples ordered by decreasing similarity.
        """
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("The FAISS index is empty. Add documents before searching.")

        query_emb = self.embed_texts([query])
        distances, indices = self.index.search(query_emb, k)
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            meta = self.id_to_meta.get(int(idx), {})
            results.append((meta, float(score)))
        return results

    # --------------------------------------------------------------------- #
    # Persistence helpers
    # --------------------------------------------------------------------- #

    def save(self, index_path: pathlib.Path, meta_path: pathlib.Path):
        """
        Serialize the FAISS index and metadata to disk.

        Parameters
        ----------
        index_path : pathlib.Path
            Destination for the FAISS index file.
        meta_path : pathlib.Path
            Destination for the metadata (pickle) file.
        """
        if self.index is None:
            raise RuntimeError("Cannot save an empty index.")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Saving FAISS index to %s", index_path)
        faiss.write_index(self.index, str(index_path))

        logger.info("Saving metadata to %s", meta_path)
        with meta_path.open("wb") as f:
            pickle.dump(
                {
                    "id_to_meta": self.id_to_meta,
                    "next_id": self._next_id,
                    "model_name": self.model_name,
                    "device": self.device,
                },
                f,
            )

    @classmethod
    def load(cls, index_path: pathlib.Path, meta_path: pathlib.Path) -> "CodeRetriever":
        """
        Load a previously saved retriever.

        Parameters
        ----------
        index_path : pathlib.Path
            Path to the FAISS index file.
        meta_path : pathlib.Path
            Path to the metadata pickle file.

        Returns
        -------
        CodeRetriever
            An instance populated with the saved index and metadata.
        """
        if not index_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError("Both index and metadata files must exist.")

        logger.info("Loading FAISS index from %s", index_path)
        index = faiss.read_index(str(index_path))

        logger.info("Loading metadata from %s", meta_path)
        with meta_path.open("rb") as f:
            data = pickle.load(f)

        retriever = cls(model_name=data["model_name"], device=data["device"])
        retriever.index = index
        retriever.id_to_meta = data["id_to_meta"]
        retriever._next_id = data["next_id"]
        return retriever


# ------------------------------------------------------------------------- #
# CLI utilities
# ------------------------------------------------------------------------- #

def _collect_files(root: pathlib.Path) -> List[Tuple[pathlib.Path, str]]:
    """
    Walk ``root`` recursively and return a list of (path, content) tuples
    for files that appear to be text based.
    """
    files = []
    for path in root.rglob("*"):
        if path.is_file() and _is_text_file(path):
            try:
                content = _read_file(path)
                files.append((path, content))
            except Exception as exc:
                logger.warning("Failed to read %s: %s", path, exc)
    logger.info("Collected %d text files from %s", len(files), root)
    return files


@click.group()
def cli():
    """Contextual Code Review Assistant – embedding and retrieval utilities."""
    pass


@cli.command()
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path))
@click.option(
    "--index-path",
    type=click.Path(dir_okay=False, path_type=pathlib.Path),
    required=True,
    help="File path where the FAISS index will be stored.",
)
@click.option(
    "--meta-path",
    type=click.Path(dir_okay=False, path_type=pathlib.Path),
    required=True,
    help="File path where metadata (pickle) will be stored.",
)
@click.option("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name.")
@click.option("--device", default=None, help="Device for the transformer (e.g., cpu, cuda).")
def index(repo_path: pathlib.Path, index_path: pathlib.Path, meta_path: pathlib.Path, model: str, device: Optional[str]):
    """
    Index all source files under REPO_PATH and store the FAISS index.
    """
    retriever = CodeRetriever(model_name=model, device=device)

    files = _collect_files(repo_path)
    batch_size = 128
    for i in tqdm(range(0, len(files), batch_size), desc="Indexing batches"):
        batch = files[i : i + batch_size]
        docs = [content for _, content in batch]
        metas = [{"path": str(path)} for path, _ in batch]
        retriever.add_documents(docs, metas)

    retriever.save(index_path, meta_path)
    click.echo(f"Indexing complete. Index saved to {index_path}, metadata saved to {meta_path}.")


@cli.command()
@click.argument("query", type=str)
@click.option(
    "--index-path",
    type=click.Path(dir_okay=False, path_type=pathlib.Path),
    required=True,
    help="Path to the saved FAISS index.",
)
@click.option(
    "--meta-path",
    type=click.Path(dir_okay=False, path_type=pathlib.Path),
    required=True,
    help="Path to the saved metadata (pickle).",
)
@click.option("--k", default=5, help="Number of results to return.")
@click.option("--model", default=None, help="Override model name (must match the one used for indexing).")
@click.option("--device", default=None, help="Device for the transformer (e.g., cpu, cuda).")
def search(query: str, index_path: pathlib.Path, meta_path: pathlib.Path, k: int, model: Optional[str], device: Optional[str]):
    """
    Search the previously built index for code similar to QUERY.
    """
    retriever = CodeRetriever.load(index_path, meta_path)

    # If the user forces a different model/device we re‑initialize the embedder.
    if model is not None and model != retriever.model_name:
        logger.info("Overriding model from %s to %s", retriever.model_name, model)
        retriever.model = SentenceTransformer(model, device=device)
        retriever.model_name = model
        retriever.device = device
    elif device is not None and device != retriever.device:
        logger.info("Moving model to device %s", device)
        retriever.model = SentenceTransformer(retriever.model_name, device=device)
        retriever.device = device

    results = retriever.search(query, k=k)
    if not results:
        click.echo("No results found.")
        return

    click.echo(f"Top {len(results)} results for query: \"{query}\"")
    for rank, (meta, score) in enumerate(results, start=1):
        path = meta.get("path", "<unknown>")
        click.echo(f"{rank}. {path} (score: {score:.4f})")


if __name__ == "__main__":
    cli()