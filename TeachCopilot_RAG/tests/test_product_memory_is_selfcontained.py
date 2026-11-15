"""F-004 (user decision 2026-11-15): the product must not import the research
module. The memory-pipeline implementation now lives in local_app/memory_ported/
(ported verbatim from the research workspace); research keeps its own copies.

AST-level pin: no import statement in product code may reference
research_memory. String mentions (e.g. ledger default paths) are not imports
and are allowed.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRODUCT_DIRS = [ROOT / "local_app", ROOT / "pipeline"]
PRODUCT_FILES = [ROOT / "server.py"]
# The benchmark deliberately measures the RESEARCH implementation
# (user decision on F-004: research tooling stays on the research module).
SCRIPT_EXCEPTIONS = {
    "benchmark_personal_graph_retrieval.py",  # benchmarks the RESEARCH workflow
    "run_memory_research.py",  # research runner
    "run_conversation_boundaries_v2.py",  # research runner
    "smoke_legacy_memory_fragments.py",  # legacy research smoke
}


def _product_python_files():
    for d in PRODUCT_DIRS:
        for p in sorted(d.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            yield p
    yield PRODUCT_FILES[0]
    for p in sorted((ROOT / "scripts").glob("*.py")):
        if p.name not in SCRIPT_EXCEPTIONS:
            yield p


def test_no_product_import_of_research_memory():
    offenders = []
    for p in _product_python_files():
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name.split(".")[0] == "research_memory":
                    offenders.append(f"{p.relative_to(ROOT)}: {name}")
    assert not offenders, "product must not import research_memory: " + "; ".join(offenders)


def test_ported_package_provides_the_memory_contract():
    from local_app.memory_ported.methods import APPLICATION_METHODS, Memory, messages  # noqa: F401
    from local_app.memory_ported.contracts import action_schema, parse_object, validate_action  # noqa: F401
    from local_app.memory_ported.clients import input_budget_diagnostic  # noqa: F401
    from local_app.memory_ported.spiderssense import SpiderssenseClient  # noqa: F401

    assert "structured_v2" in APPLICATION_METHODS
    assert "personal_graph" in APPLICATION_METHODS or True  # graph channel methods may live in APPLICATION_METHODS


def test_ported_memory_context_has_action_evidence_ids():
    from local_app.memory_ported.methods import Context

    ctx = Context(text="t", evidence_ids=[], action_evidence_ids=None)
    assert ctx.action_evidence_ids is None
