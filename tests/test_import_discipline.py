"""Structural rules that are easy to state and easy to break by accident.

Both of these are promised in docs/contracts.md. A promise that nothing checks
drifts within days, and the drift only shows up somewhere expensive: a tool
calling `openai` directly bypasses every retry and every usage counter, and a
production module importing the fakes ships synthetic data into the demo.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "lakehouse_agent"


def _imported_modules(path: Path) -> set[str]:
    """Top-level module names imported by a file, from the AST rather than text.

    Reading the source with a regex would also match the word inside comments
    and docstrings, which several of these files legitimately contain.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


def _python_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)


def test_only_llm_module_imports_openai():
    """One door to MIA. Everything else goes through MiaLLM."""
    offenders = [
        path.relative_to(SRC).as_posix()
        for path in _python_files()
        if path.name != "llm.py" and "openai" in _imported_modules(path)
    ]
    assert offenders == [], (
        f"Bu dosyalar openai'yi dogrudan import ediyor: {offenders}. "
        "MIA'ya tek kapi llm.py; aksi halde retry ve kullanim logu atlanir."
    )


def test_llm_module_does_import_openai():
    """Guards the guard: if llm.py stopped importing openai the test above
    would pass for the wrong reason."""
    assert "openai" in _imported_modules(SRC / "llm.py")


def test_production_modules_do_not_import_the_testing_package():
    """The fakes ship inside the package so the CLI can reach them, which makes
    it that much easier for real code to pick them up by accident."""
    offenders = []
    for path in _python_files():
        if "testing" in path.relative_to(SRC).parts:
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            elif isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            if any(name.startswith("lakehouse_agent.testing") for name in names):
                offenders.append(path.relative_to(SRC).as_posix())
    assert offenders == [], (
        f"Bu uretim modulleri sahte katmani import ediyor: {sorted(set(offenders))}."
    )


def test_no_module_reads_settings_at_import_time():
    """Library code takes settings as a parameter.

    A module-level `get_agent_settings()` would make importing the package fail
    whenever MIA_API_KEY is absent, which is exactly the state every unit test
    runs in.
    """
    offenders = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:  # module level only, not inside functions
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Name)
                    and inner.func.id
                    in {"get_agent_settings", "get_settings", "Settings"}
                ):
                    if not isinstance(node, ast.FunctionDef | ast.ClassDef):
                        offenders.append(path.relative_to(SRC).as_posix())
    assert offenders == [], (
        f"Import aninda ayar okuyan moduller: {sorted(set(offenders))}."
    )


@pytest.mark.parametrize("name", ["contracts.py", "llm.py", "config.py"])
def test_core_modules_import_without_any_environment(name):
    """Importing the package must never require a key."""
    import importlib

    module = importlib.import_module(f"lakehouse_agent.{name[:-3]}")
    assert module is not None
