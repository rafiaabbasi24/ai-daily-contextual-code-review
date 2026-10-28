import pathlib
import textwrap
import tempfile

import pytest
from click.testing import CliRunner

# Import the click entry point. Adjust the import path if the project structure differs.
try:
    # Most projects expose a `cli` object in a `cli.py` module.
    from contextual_code_review_assistant.cli import cli  # type: ignore
except ImportError as exc:
    # Fallback to a `main` function if the project uses a different naming convention.
    try:
        from contextual_code_review_assistant.main import main as cli  # type: ignore
    except ImportError:
        raise ImportError(
            "Unable to locate the Click command entry point. "
            "Ensure that `contextual_code_review_assistant.cli.cli` or "
            "`contextual_code_review_assistant.main.main` exists."
        ) from exc


@pytest.fixture(scope="module")
def runner() -> CliRunner:
    """Provide a Click test runner for the entire test module."""
    return CliRunner()


def test_cli_help(runner: CliRunner) -> None:
    """The CLI should display a help message without error."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0, f"CLI exited with {result.exit_code}, output:\n{result.output}"
    # Basic sanity checks – the help text should contain the command name and a usage line.
    assert "Usage:" in result.output
    assert "Options:" in result.output or "Commands:" in result.output


def test_cli_review_on_simple_file(runner: CliRunner) -> None:
    """
    Run the CLI on a minimal Python file and verify that it completes successfully.

    The test does not assert on the exact review content (which may change with model
    updates) but ensures that the command finishes with exit code 0 and produces
    non‑empty output that mentions the processed file.
    """
    # Create a temporary Python source file.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp_file:
        tmp_file_path = pathlib.Path(tmp_file.name)
        tmp_file.write(
            textwrap.dedent(
                """
                def add(a, b):
                    \"\"\"Return the sum of *a* and *b*.\"\"\"
                    return a + b
                """
            )
        )
    try:
        # Invoke the CLI. Most CLI designs accept the file path as a positional argument.
        # If the project uses a sub‑command (e.g., `review`), the test will still work
        # because Click will treat the first unknown argument as a sub‑command and raise
        # a clear error, causing the test to fail and highlight the mismatch.
        result = runner.invoke(cli, [str(tmp_file_path)])

        # The command should exit cleanly.
        assert result.exit_code == 0, (
            f"CLI exited with {result.exit_code}. Output:\n{result.output}"
        )

        # The output should reference the file name and contain at least one line of text.
        output = result.output.strip()
        assert output, "CLI produced no output."
        assert tmp_file_path.name in output, "Output does not mention the processed file."

        # A very lightweight sanity check – the output typically contains a keyword like
        # 'review', 'analysis', or 'summary'. We check for at least one of them.
        keywords = ["review", "analysis", "summary", "suggestion"]
        assert any(kw.lower() in output.lower() for kw in keywords), (
            "Output does not contain any expected review‑related keywords."
        )
    finally:
        # Clean up the temporary file.
        tmp_file_path.unlink(missing_ok=True)