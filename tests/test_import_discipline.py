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


#: Functions allowed to reach into the fakes, each behind an explicit opt-in
#: flag. Anything not on this list importing them is a bug: the fakes ship in
#: the installed package, so a stray import is one demo away from putting
#: synthetic numbers on screen with nothing to mark them as synthetic.
FAKE_IMPORT_ALLOWLIST = {
    ("agent/loop.py", "_fake_script"),
    ("agent/loop.py", "_fake_deps"),
    ("agent/loop.py", "_live_deps"),
}


def _imports_the_fakes(node: ast.AST) -> bool:
    names: list[str] = []
    if isinstance(node, ast.ImportFrom) and node.module:
        names.append(node.module)
    elif isinstance(node, ast.Import):
        names.extend(alias.name for alias in node.names)
    return any(name.startswith("lakehouse_agent.testing") for name in names)


def test_no_production_module_imports_the_fakes_at_module_level():
    """A module-level import would make the fakes reachable from anywhere.

    The hazard is not the import statement, it is that importing the module at
    all would then pull the synthetic layer into the process. A function-level
    import behind a flag does not do that, so the next test handles those
    separately instead of banning them outright.
    """
    offenders = []
    for path in _python_files():
        if "testing" in path.relative_to(SRC).parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if _imports_the_fakes(node):
                offenders.append(path.relative_to(SRC).as_posix())
    assert offenders == [], (
        f"Bu uretim modulleri sahte katmani modul duzeyinde import ediyor: "
        f"{sorted(set(offenders))}."
    )


def test_lazy_fake_imports_stay_on_the_allowlist():
    """And the deferred ones stay countable, by file and by function name."""
    found = set()
    for path in _python_files():
        if "testing" in path.relative_to(SRC).parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if any(_imports_the_fakes(inner) for inner in ast.walk(node)):
                found.add((path.relative_to(SRC).as_posix(), node.name))
    assert found == FAKE_IMPORT_ALLOWLIST, (
        f"Sahte katmani import eden fonksiyonlar degisti: {sorted(found)}. "
        "Yeni bir giris noktasi eklendiyse listeyi bilerek guncelleyin."
    )


def test_agent_and_tools_do_not_import_chainlit():
    """The engine publishes events; the interface draws them.

    If the loop imported chainlit we could not run it headless, could not test
    it, and could not swap the interface later without touching the engine.
    """
    offenders = [
        path.relative_to(SRC).as_posix()
        for path in _python_files()
        if "chainlit" in _imported_modules(path)
    ]
    assert offenders == [], (
        f"Bu dosyalar chainlit import ediyor: {offenders}. "
        "Agent cekirdegi arayuzden bagimsiz kalmali."
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


def test_only_the_console_module_imports_rich():
    """The engine publishes events; the interface draws them.

    The same rule as chainlit, for the library the terminal interface is
    built on. A headless run -- the eval runner, CI, a piped demo -- must not
    load a presentation library, and the loop must stay swappable under a
    different screen.
    """
    offenders = [
        path.relative_to(SRC).as_posix()
        for path in _python_files()
        if "rich" in _imported_modules(path)
        and path.relative_to(SRC).as_posix() != "agent/console.py"
    ]
    assert offenders == [], (
        f"Bu dosyalar rich import ediyor: {offenders}. "
        "Sunum yalnizca agent/console.py icinde."
    )


def test_no_module_imports_plotly():
    """The engine builds a figure as a plain dict; only `app.py` draws it.

    The same rule as chainlit and rich, for the third presentation library.
    `tools/charts.py` produces plotly's wire format without the package, so
    a headless run never loads it and `Event.payload` stays plain JSON.
    """
    offenders = [
        path.relative_to(SRC).as_posix()
        for path in _python_files()
        if "plotly" in _imported_modules(path)
    ]
    assert offenders == [], (
        f"Bu dosyalar plotly import ediyor: {offenders}. "
        "Grafik motorda sozluk olarak kurulur, plotly yalnizca app.py'de."
    )
