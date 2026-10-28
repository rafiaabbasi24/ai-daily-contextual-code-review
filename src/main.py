import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import click
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline, set_seed


def _load_sentence_encoder(model_name: str) -> SentenceTransformer:
    """Load a SentenceTransformer model for embedding code snippets."""
    try:
        encoder = SentenceTransformer(model_name)
        encoder.eval()
        return encoder
    except Exception as exc:
        raise click.ClickException(f"Failed to load sentence‑transformer model '{model_name}': {exc}") from exc


def _embed_texts(encoder: SentenceTransformer, texts: List[str]) -> torch.Tensor:
    """Encode a list of strings into a single torch tensor (dtype=float32)."""
    with torch.no_grad():
        embeddings = encoder.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    return embeddings.cpu().numpy()


def _build_faiss_index(embeddings: torch.Tensor) -> faiss.Index:
    """Create a FAISS index (inner product) for normalized embeddings."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product works as cosine similarity for normalized vectors
    index.add(embeddings)
    return index


def _save_index(index: faiss.Index, id_to_path: Dict[int, str], index_path: Path) -> None:
    """Persist a FAISS index and its ID→path mapping."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    mapping_path = index_path.with_suffix(".pkl")
    with mapping_path.open("wb") as f:
        pickle.dump(id_to_path, f)


def _load_index(index_path: Path) -> Tuple[faiss.Index, Dict[int, str]]:
    """Load a FAISS index and its ID→path mapping."""
    if not index_path.is_file():
        raise click.ClickException(f"FAISS index not found at '{index_path}'. Did you run the 'index' command?")
    index = faiss.read_index(str(index_path))
    mapping_path = index_path.with_suffix(".pkl")
    if not mapping_path.is_file():
        raise click.ClickException(f"Mapping file missing: '{mapping_path}'.")
    with mapping_path.open("rb") as f:
        id_to_path = pickle.load(f)
    return index, id_to_path


def _gather_code_files(src_dir: Path) -> List[Path]:
    """Recursively collect source files with common code extensions."""
    extensions = {".py", ".js", ".ts", ".java", ".cpp", ".c", ".cs", ".go", ".rb", ".rs", ".php"}
    return [p for p in src_dir.rglob("*") if p.suffix.lower() in extensions and p.is_file()]


def _read_file(path: Path) -> str:
    """Read a file using UTF‑8, falling back to latin‑1 if needed."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _create_corpus(src_dir: Path) -> Tuple[List[str], List[Path]]:
    """Return a list of file contents and their corresponding paths."""
    files = _gather_code_files(src_dir)
    if not files:
        raise click.ClickException(f"No code files found in '{src_dir}'.")
    contents = [_read_file(p) for p in files]
    return contents, files


def _search_similar(
    index: faiss.Index,
    id_to_path: Dict[int, str],
    query_emb: torch.Tensor,
    k: int = 5,
) -> List[Path]:
    """Return the top‑k most similar file paths for a query embedding."""
    distances, indices = index.search(query_emb, k)
    similar_paths = []
    for idx in indices[0]:
        if idx == -1:
            continue
        similar_paths.append(Path(id_to_path[int(idx)]))
    return similar_paths


def _load_review_generator(model_name: str):
    """Create a text‑generation pipeline for producing review comments."""
    try:
        generator = pipeline("text-generation", model=model_name, device=0 if torch.cuda.is_available() else -1)
        set_seed(42)
        return generator
    except Exception as exc:
        raise click.ClickException(f"Failed to load review model '{model_name}': {exc}") from exc


def _generate_review(
    generator,
    code: str,
    context: str,
    max_new_tokens: int = 256,
) -> str:
    """Generate a review using the provided generator pipeline."""
    prompt = (
        "You are an expert software engineer performing a code review.\n"
        "Provide constructive feedback for the following code:\n\n"
        f"{code}\n\n"
        "Relevant context from similar code bases:\n"
        f"{context}\n\n"
        "Review:"
    )
    output = generator(prompt, max_new_tokens=max_new_tokens, num_return_sequences=1, do_sample=True)[0]["generated_text"]
    # Remove the prompt from the output
    review = output[len(prompt) :].strip()
    return review


@click.group()
def cli() -> None:
    """Contextual Code Review Assistant – CLI entry point."""
    pass


@cli.command()
@click.argument("src_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("index_path", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--model",
    "embed_model",
    default="sentence-transformers/all-MiniLM-L6-v2",
    show_default=True,
    help="SentenceTransformer model for embedding code.",
)
def index(src_dir: Path, index_path: Path, embed_model: str) -> None:
    """
    Build a FAISS index from all code files under SRC_DIR and store it at INDEX_PATH.
    """
    click.echo(f"Collecting code files from {src_dir} …")
    contents, files = _create_corpus(src_dir)

    click.echo(f"Loading embedding model '{embed_model}' …")
    encoder = _load_sentence_encoder(embed_model)

    click.echo("Embedding code snippets …")
    embeddings = _embed_texts(encoder, contents)

    click.echo("Creating FAISS index …")
    faiss_index = _build_faiss_index(embeddings)

    id_to_path = {i: str(p) for i, p in enumerate(files)}
    click.echo(f"Saving index to {index_path} …")
    _save_index(faiss_index, id_to_path, index_path)

    click.echo("Indexing complete.")


@cli.command()
@click.argument("code_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("index_path", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--embed-model",
    default="sentence-transformers/all-MiniLM-L6-v2",
    show_default=True,
    help="SentenceTransformer model for embedding the target code.",
)
@click.option(
    "--review-model",
    default="gpt2",
    show_default=True,
    help="Transformer model used to generate the review.",
)
@click.option(
    "--top-k",
    default=5,
    show_default=True,
    help="Number of similar snippets to retrieve for context.",
)
def review(code_path: Path, index_path: Path, embed_model: str, review_model: str, top_k: int) -> None:
    """
    Generate a contextual review for CODE_PATH using a pre‑built FAISS INDEX_PATH.
    """
    click.echo(f"Reading target code from {code_path} …")
    target_code = _read_file(code_path)

    click.echo(f"Loading FAISS index from {index_path} …")
    index, id_to_path = _load_index(index_path)

    click.echo(f"Loading embedding model '{embed_model}' …")
    encoder = _load_sentence_encoder(embed_model)

    click.echo("Embedding target code …")
    query_emb = _embed_texts(encoder, [target_code])

    click.echo(f"Searching for top‑{top_k} similar snippets …")
    similar_paths = _search_similar(index, id_to_path, query_emb, k=top_k)

    if similar_paths:
        click.echo("Gathering contextual snippets …")
        context_snippets = "\n---\n".join(_read_file(p) for p in similar_paths)
    else:
        click.echo("No similar snippets found; proceeding without additional context.")
        context_snippets = ""

    click.echo(f"Loading review generation model '{review_model}' …")
    generator = _load_review_generator(review_model)

    click.echo("Generating review …")
    review_text = _generate_review(generator, target_code, context_snippets)

    click.echo("\n--- Review ---\n")
    click.echo(review_text)


if __name__ == "__main__":
    # Entrypoint for `python -m src.main` usage.
    cli()