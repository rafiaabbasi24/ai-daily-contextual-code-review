# Contextual Code Review Assistant

## Project Overview

The **Contextual Code Review Assistant** is an AI-powered tool that helps developers perform more insightful code reviews by providing context-aware suggestions, best‑practice recommendations, and relevant documentation snippets. It leverages large language models (LLMs) and dense vector similarity search to retrieve related code fragments, documentation, and prior review comments, enabling reviewers to focus on higher‑level design and logic issues.

Key features:

- **Semantic code search** using sentence‑transformers and FAISS for fast nearest‑neighbor retrieval.
- **LLM‑driven suggestions** powered by Hugging Face Transformers (e.g., `facebook/opt-1.3b` or any compatible model).
- **Command‑line interface** built with Click for easy integration into CI pipelines or local workflows.
- **Extensible architecture**: plug in custom models, data sources, or post‑processing steps.

## Tech Stack

| Component | Library / Framework | Purpose |
|-----------|---------------------|---------|
| Language | Python (≥3.8) | Core implementation |
| Deep Learning | PyTorch | Tensor operations and model inference |
| Transformers | `transformers` | Access to pre‑trained LLMs |
| Sentence Embeddings | `sentence-transformers` | Generate dense embeddings for code snippets |
| Vector Store | `faiss-cpu` (or `faiss-gpu`) | Efficient similarity search |
| CLI | `click` | User‑friendly command‑line interface |

## Setup

### Prerequisites

- Python 3.8 or newer
- pip (>=21.0)
- (Optional) A GPU with CUDA for faster inference. If you have a GPU, install `faiss-gpu` instead of `faiss-cpu`.

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/contextual-code-review-assistant.git
cd contextual-code-review-assistant

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

The `requirements.txt` includes:

```
torch>=1.13.0
transformers>=4.30.0
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
click>=8.1.3
```

If you have a CUDA‑compatible GPU and want to use the GPU version of FAISS:

```bash
pip uninstall -y faiss-cpu
pip install faiss-gpu
```

### Model Download

The assistant uses a sentence‑transformer model for embedding code and a language model for generating suggestions. By default, it will download the following models on first run:

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- LLM: `facebook/opt-1.3b` (or any model you specify via CLI)

You can pre‑download them to avoid network latency:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('facebook/opt-1.3b'); AutoTokenizer.from_pretrained('facebook/opt-1.3b')"
```

## Usage

The tool is primarily accessed through the `crassistant` CLI.

### Basic Review

```bash
crassistant review path/to/your/file.py
```

The command will:

1. Load the target file.
2. Generate a dense embedding for each function/class.
3. Retrieve the top‑k most similar code snippets from the internal FAISS index (populated from a curated corpus of open‑source projects).
4. Pass the retrieved context and the target code to the LLM.
5. Print a markdown‑formatted review report with suggestions, potential bugs, and links to related documentation.

### Advanced Options

| Option | Description |
|--------|-------------|
| `--model <model_name>` | Specify a different LLM (e.g., `gpt2`, `bigscience/bloom-560m`). |
| `--embed-model <embed_name>` | Override the default embedding model. |
| `--top-k <int>` | Number of nearest neighbours to retrieve (default: 5). |
| `--output <file>` | Write the review report to a file instead of stdout. |
| `--no-cache` | Bypass the on‑disk embedding cache (useful for debugging). |

Example with custom models and output file:

```bash
crassistant review src/main.py \
    --model bigscience/bloom-560m \
    --embed-model sentence-transformers/all-MiniLM-L12-v2 \
    --top-k 10 \
    --output review_report.md
```

### Index Management

Before reviewing code you need an index of reference snippets. The `index` sub‑command helps you build and maintain this index.

#### Build a new index

```bash
crassistant index build \
    --source-dir path/to/open-source/codebase \
    --index-dir ./faiss_index \
    --embed-model sentence-transformers/all-MiniLM-L6-v2
```

#### Update an existing index

```bash
crassistant index update \
    --source-dir path/to/more/code \
    --index-dir ./faiss_index
```

#### Inspect index statistics

```bash
crassistant index stats ./faiss_index
```

### Integration with CI/CD

Add a step to your CI pipeline (e.g., GitHub Actions) to automatically run a review on changed Python files:

```yaml
name: Code Review Assistant
on: [pull_request]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          python -m venv .venv
          source .venv/bin/activate
          pip install -r requirements.txt
      - name: Build index (once per repo, cache if possible)
        run: |
          source .venv/bin/activate
          crassistant index build --source-dir ./reference_code --index-dir ./faiss_index
      - name: Run review on changed files
        run: |
          source .venv/bin/activate
          changed=$(git diff --name-only ${{ github.event.pull_request.base.sha }} ${{ github.sha }} | grep '\.py$' || true)
          for file in $changed; do
            crassistant review $file --output reviews/${{ github.sha }}_${{ github.run_number }}_${file//\//_}.md
          done
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/awesome-feature`).
3. Ensure code passes linting (`ruff check .`) and tests (`pytest -q`).
4. Submit a pull request with a clear description of the changes.

### Development Setup

```bash
# Install development extras
pip install -e .[dev]

# Run the test suite
pytest
```

The `pyproject.toml` defines optional dependencies for development:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "ruff>=0.0.260",
    "black>=23.0"
]
```

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

## Contact & Support

- **Maintainer**: Your Name <you@example.com>
- **Issues**: Please open a GitHub issue for bugs or feature requests.
- **Community**: Join our Discord channel (link in the repository README).