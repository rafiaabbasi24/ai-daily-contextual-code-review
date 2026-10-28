import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import click
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline


class Reviewer:
    """
    Reviewer wraps a language model to generate code review comments.
    It can optionally retrieve similar code examples from a FAISS index
    to provide contextual information to the LLM.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for the generation model.
    embed_model_name : str
        HuggingFace model identifier for the sentence‑transformers embedding model.
    faiss_index_path : Optional[Path]
        Path to a pre‑built FAISS index. If ``None``, retrieval is disabled.
    device : Optional[str]
        Torch device to use (e.g., ``"cpu"``, ``"cuda"``). If ``None``, the best
        available device is selected automatically.
    """

    def __init__(
        self,
        model_name: str,
        embed_model_name: str,
        faiss_index_path: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load generation model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.generator = pipeline(
            "text-generation",
            model=self.llm,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
        )

        # Load embedding model
        self.embedder = SentenceTransformer(embed_model_name, device=self.device)

        # Load FAISS index and associated metadata if provided
        self.index = None
        self.id_to_example: dict[int, str] = {}
        if faiss_index_path:
            self._load_faiss_index(faiss_index_path)

    # --------------------------------------------------------------------- #
    # Retrieval utilities
    # --------------------------------------------------------------------- #
    def _load_faiss_index(self, index_path: Path) -> None:
        """
        Load a FAISS index and its metadata.

        The metadata file must be a JSON dictionary where keys are stringified
        integer IDs (matching the FAISS index IDs) and values are the original
        example strings.

        Parameters
        ----------
        index_path : Path
            Path to the FAISS index file (e.g., ``index.faiss``). The metadata file
            is expected to be located alongside it with the same stem and a
            ``.json`` extension.
        """
        if not index_path.is_file():
            raise FileNotFoundError(f"FAISS index not found at {index_path}")

        # Load index
        self.index = faiss.read_index(str(index_path))

        # Load metadata
        meta_path = index_path.with_suffix(".json")
        if not meta_path.is_file():
            raise FileNotFoundError(f"FAISS metadata JSON not found at {meta_path}")

        with meta_path.open("r", encoding="utf-8") as f:
            raw_meta = json.load(f)
        # Convert keys to int for fast lookup
        self.id_to_example = {int(k): v for k, v in raw_meta.items()}

        if self.index.ntotal != len(self.id_to_example):
            raise ValueError(
                "Mismatch between number of vectors in FAISS index and metadata entries."
            )

    def retrieve_context(self, code: str, k: int = 5) -> List[Tuple[int, str, float]]:
        """
        Retrieve the *k* most similar code examples from the FAISS index.

        Parameters
        ----------
        code : str
            The source code snippet for which to find similar examples.
        k : int, default 5
            Number of nearest neighbours to return.

        Returns
        -------
        List[Tuple[int, str, float]]
            A list of tuples ``(doc_id, example_text, distance)`` ordered by
            increasing distance (i.e., most similar first). If the index is not
            loaded, an empty list is returned.
        """
        if self.index is None:
            return []

        embedding = self.embedder.encode([code], convert_to_numpy=True)
        distances, indices = self.index.search(embedding, k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue  # FAISS returns -1 for empty slots
            example = self.id_to_example.get(int(idx), "")
            results.append((int(idx), example, float(dist)))
        return results

    # --------------------------------------------------------------------- #
    # Generation utilities
    # --------------------------------------------------------------------- #
    def _build_prompt(
        self,
        code: str,
        retrieved_examples: Optional[List[Tuple[int, str, float]]] = None,
        max_context_examples: int = 3,
    ) -> str:
        """
        Construct a prompt for the LLM.

        The prompt includes optional retrieved examples followed by the target
        code snippet and an instruction to produce a review.

        Parameters
        ----------
        code : str
            The source code to be reviewed.
        retrieved_examples : Optional[List[Tuple[int, str, float]]]
            List of retrieved examples as returned by ``retrieve_context``.
        max_context_examples : int, default 3
            Upper bound on how many examples to embed in the prompt.

        Returns
        -------
        str
            The final prompt string.
        """
        parts: List[str] = ["You are an expert software engineer tasked with reviewing the following code."]

        if retrieved_examples:
            parts.append("\nRelevant examples from similar code bases (for context):")
            for _, example, _ in retrieved_examples[:max_context_examples]:
                parts.append("\n---\n" + example.strip() + "\n---")

        parts.append("\nCode to review:\n```python\n" + code.rstrip() + "\n```")
        parts.append(
            "\nProvide a concise, constructive review. "
            "List each comment as a bullet point, focusing on correctness, readability, "
            "performance, and best practices."
        )
        return "\n".join(parts)

    def generate_review(
        self,
        code: str,
        k: int = 5,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
    ) -> str:
        """
        Generate a code review for the supplied snippet.

        Parameters
        ----------
        code : str
            Source code to be reviewed.
        k : int, default 5
            Number of nearest neighbours to retrieve for context.
        max_new_tokens : int, default 256
            Maximum number of tokens to generate.
        temperature : float, default 0.2
            Sampling temperature.
        top_p : float, default 0.95
            Nucleus sampling probability.

        Returns
        -------
        str
            The generated review text.
        """
        retrieved = self.retrieve_context(code, k=k) if self.index else []
        prompt = self._build_prompt(code, retrieved_examples=retrieved)

        # Generation config mirrors the arguments for reproducibility
        gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # The pipeline returns a list of dicts; we take the first result
        output = self.generator(prompt, generation_config=gen_cfg, return_full_text=False)
        review = output[0]["generated_text"].strip()
        return review


# --------------------------------------------------------------------- #
# CLI implementation using click
# --------------------------------------------------------------------- #
@click.command()
@click.argument("code_path", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option(
    "--model",
    default="gpt2",
    show_default=True,
    help="HuggingFace model name for generation (e.g., 'gpt2', 'meta-llama/Llama-2-7b-chat-hf').",
)
@click.option(
    "--embed-model",
    default="sentence-transformers/all-MiniLM-L6-v2",
    show_default=True,
    help="HuggingFace model name for the sentence‑transformers embedding model.",
)
@click.option(
    "--faiss-index",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    default=None,
    help="Path to a FAISS index file (.faiss) with accompanying .json metadata.",
)
@click.option(
    "--k",
    default=5,
    show_default=True,
    type=int,
    help="Number of nearest neighbours to retrieve for context.",
)
@click.option(
    "--max-tokens",
    default=256,
    show_default=True,
    type=int,
    help="Maximum number of tokens to generate for the review.",
)
@click.option(
    "--temperature",
    default=0.2,
    show_default=True,
    type=float,
    help="Sampling temperature for generation.",
)
@click.option(
    "--top-p",
    default=0.95,
    show_default=True,
    type=float,
    help="Top‑p (nucleus) sampling parameter.",
)
def review(
    code_path: str,
    model: str,
    embed_model: str,
    faiss_index: Optional[str],
    k: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> None:
    """
    Generate a code review for the file at CODE_PATH.
    """
    code_path_obj = Path(code_path)
    with code_path_obj.open("r", encoding="utf-8") as f:
        code = f.read()

    try:
        reviewer = Reviewer(
            model_name=model,
            embed_model_name=embed_model,
            faiss_index_path=Path(faiss_index) if faiss_index else None,
        )
    except Exception as exc:
        click.echo(f"Error initializing Reviewer: {exc}", err=True)
        sys.exit(1)

    try:
        review_text = reviewer.generate_review(
            code,
            k=k,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        click.echo("\n=== Review ===\n")
        click.echo(review_text)
    except Exception as exc:
        click.echo(f"Error during review generation: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    # When executed directly, expose the CLI.
    review()