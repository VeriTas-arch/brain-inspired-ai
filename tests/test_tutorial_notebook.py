"""Contract tests for the thin teaching notebook."""

import json
from pathlib import Path


def _notebook() -> dict:
    return json.loads(Path("Atari_RL_Complete_Tutorial.ipynb").read_text(encoding="utf-8"))


def test_notebook_is_a_thin_interface_to_python_functionality() -> None:
    notebook = _notebook()
    code = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )

    assert len(code.splitlines()) <= 20
    assert "scripts.tutorial_examples" in code
    assert "scripts.run_experiments" in code
    assert "scripts.visualize_results" in code
    assert "pip install" not in code
    assert "subprocess" not in code
    assert "sys.path" not in code
    assert "def " not in code
    assert "class " not in code


def test_all_notebook_code_cells_compile_and_have_no_saved_outputs() -> None:
    notebook = _notebook()

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        compile(source, f"notebook-cell-{index}", "exec")
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
