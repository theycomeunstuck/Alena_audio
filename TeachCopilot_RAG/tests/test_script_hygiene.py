"""Script-hygiene regressions from the 2026-11-15 audit (F-045 to F-050).

Verified evidence: .audit/20261115-0002/repro/V5 CL-2a/b and repro/V8
CL-1/CL-2/CL-3/CL-5. Importing or running some of these scripts would
itself touch the configured PostgreSQL (test_pipeline.py runs DB writes in
its module body, apply_schema.py applies DDL in its module body, the
personal-graph smoke scripts write in main()), so those are checked at the
AST/source level with targeted behavioral assertions that stay safe.
ingest_json, the manage_local backup path and transition_memory_budget are
exercised behaviorally against fakes and throwaway directories. Nothing
here connects to PostgreSQL, Docker or any real learner data.
"""
from __future__ import annotations

import ast
import configparser
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPTS = ROOT / "scripts"
REPO_ROOT = ROOT.parent

FAKE_DSN = "postgresql://postgres:FAKE-NOT-A-SECRET@example.invalid:5432/teachcopilot"
LOOPBACK_DSN = "postgresql://teachcopilot@127.0.0.1:5434/teachcopilot_local"
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}

# F-049: the four scripts confirmed without a loopback-DSN guard (V8 CL-4).
SMOKE_SCRIPTS_WITHOUT_GUARD = (
    "smoke_personal_graph_api.py",
    "smoke_personal_graph_app.py",
    "smoke_personal_graph_recovery.py",
    "smoke_personal_mem0_recovery.py",
)


# --------------------------------------------------------------------------
# small AST helpers
# --------------------------------------------------------------------------

def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _functionish(node: ast.stmt) -> bool:
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))


def _module_body_statements(tree: ast.Module) -> list[ast.stmt]:
    """Top-level statements, excluding whole function/class definitions."""
    return [stmt for stmt in tree.body if not _functionish(stmt)]


def _is_os_getenv_call(node: ast.AST, name: str) -> bool:
    return (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == "getenv"
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "os"
            and len(node.args) == 1 and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == name)


def _func_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
    return None


def _calls(tree: ast.Module, name: str) -> list[ast.Call]:
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and _func_name(n) == name]


# --------------------------------------------------------------------------
# F-045 – root pytest.ini collects scripts/test_pipeline.py; script exits 0
# --------------------------------------------------------------------------

def test_root_pytest_ini_restricts_collection_to_tests():
    """Plain `pytest` from the repo root must not recurse into
    TeachCopilot_RAG/scripts (V5 CL-2a: scripts/test_pipeline.py was
    collected and its module body ran, writing to the configured DB)."""
    ini_path = REPO_ROOT / "pytest.ini"
    parser = configparser.ConfigParser()
    read = parser.read(ini_path, encoding="utf-8")
    assert read, f"root pytest.ini missing: {ini_path}"
    assert parser.has_section("pytest"), "root pytest.ini has no [pytest] section"
    assert "testpaths" in parser["pytest"], (
        "root pytest.ini must pin testpaths so collection cannot reach "
        "TeachCopilot_RAG/scripts/test_pipeline.py (F-045)")
    testpaths = [part.strip() for part in parser["pytest"]["testpaths"].split(",")]
    assert "tests" in testpaths, f"testpaths must include 'tests', got {testpaths!r}"
    assert all(not part.startswith("scripts") for part in testpaths), \
        "testpaths must not point at the scripts directory"


def test_test_pipeline_module_body_exits_nonzero_when_checks_fail():
    """The summary block of scripts/test_pipeline.py must exit nonzero when
    the failure counter is nonzero (it used to only print; contrast
    scripts/test_server.py which exits 1). The block is compiled and run
    with fakes, so nothing else of the module body executes."""
    tree = _parse(SCRIPTS / "test_pipeline.py")
    top_level_ifs = [stmt for stmt in _module_body_statements(tree)
                     if isinstance(stmt, ast.If)]
    assert top_level_ifs, "test_pipeline.py lost its summary block"
    summary_block = top_level_ifs[-1]
    source_names = {n.id for n in ast.walk(summary_block)
                    if isinstance(n, ast.Name)}
    assert "failed" in source_names, \
        "last top-level If in test_pipeline.py is no longer the summary block"
    code = compile(ast.Module(body=[summary_block], type_ignores=[]),
                   "<summary-block>", "exec")
    for failed, expected_exit in ((0, None), (1, 1), (7, 1)):
        recorded: dict[str, int] = {}
        fake_sys = types.SimpleNamespace(
            exit=lambda code=1: recorded.setdefault("exit_code", code))
        exec(code, {"passed": 22 - failed, "failed": failed,
                    "print": lambda *args, **kwargs: None, "sys": fake_sys})
        assert recorded.get("exit_code") == expected_exit, (
            f"summary block must exit {expected_exit} when failed={failed}, "
            f"got {recorded}")


# --------------------------------------------------------------------------
# F-046 – ingest_json collapses task_id-less records into one row
# --------------------------------------------------------------------------

class _FakeCursor:
    """Mimics the ON CONFLICT (source_file, topic) DO UPDATE upsert of
    db/schema.sql:52 well enough to observe key collisions."""

    def __init__(self, table: dict, counters: dict):
        self._table = table
        self._counters = counters
        self._is_insert = False

    def execute(self, query, params=None):
        (subject, topic, topic_id, grade, content, source_file,
         difficulty, tags, metadata, embedding) = params
        key = (source_file, topic)
        row = {"subject": subject, "topic": topic, "topic_id": topic_id,
               "grade": grade, "content": content, "source_file": source_file,
               "difficulty": difficulty, "tags": tags, "metadata": metadata}
        self._is_insert = key not in self._table
        if self._is_insert:
            self._table[key] = row
            self._counters["inserts"] += 1
        else:
            self._table[key].update(row)
            self._counters["updates"] += 1

    def fetchone(self):
        return (self._is_insert,)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _FakeConnection:
    def __init__(self):
        self.table: dict = {}
        self.counters = {"inserts": 0, "updates": 0}

    def cursor(self):
        return _FakeCursor(self.table, self.counters)

    def commit(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _run_ingest_json(monkeypatch, tmp_path, records, connection):
    import scripts.ingest as ingest_module

    payload = tmp_path / "math_tasks.json"
    payload.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(ingest_module, "psycopg2",
                        types.SimpleNamespace(connect=lambda dsn: connection))
    monkeypatch.setattr(ingest_module, "embed_text", lambda text: [0.0])
    ingest_module.ingest_json(payload)


def test_ingest_json_without_task_id_keeps_every_record(monkeypatch, tmp_path, capsys):
    """Two distinct records without task_id must land in two knowledge_base
    rows (V8 CL-1: both used to collapse into the single 'unknown' key)."""
    records = [
        {"question": "2+2?", "answer": "4", "tags": ["счёт"]},
        {"question": "3+3?", "answer": "6", "tags": ["счёт"]},
    ]
    connection = _FakeConnection()
    _run_ingest_json(monkeypatch, tmp_path, records, connection)
    rows = connection.table
    assert len(rows) == 2, (
        f"distinct task_id-less records collapsed into {len(rows)} row(s): "
        f"{sorted(rows)}")
    topics = [row["topic"] for row in rows.values()]
    assert len(set(topics)) == 2, f"derived topics are not unique: {topics}"
    assert all(": unknown" not in topic for topic in topics), \
        f"records still keyed on the 'unknown' default: {topics}"
    contents = [row["content"] for row in rows.values()]
    assert any("Ответ: 4" in content for content in contents), "first record lost"
    assert any("Ответ: 6" in content for content in contents), "second record lost"
    assert "[WARN]" in capsys.readouterr().out, \
        "ingest must warn loudly when it derives a key for a record without task_id"


def test_ingest_json_derived_keys_are_stable_across_reruns(monkeypatch, tmp_path):
    """Re-ingesting the same task_id-less file must update the same rows,
    not grow the table (the derived key has to be deterministic)."""
    records = [
        {"question": "5+5?", "answer": "10"},
        {"question": "2*3?", "answer": "6"},
    ]
    connection = _FakeConnection()
    _run_ingest_json(monkeypatch, tmp_path, records, connection)
    assert len(connection.table) == 2, "distinct records collapsed"
    topics_before = sorted(row["topic"] for row in connection.table.values())
    _run_ingest_json(monkeypatch, tmp_path, records, connection)
    assert len(connection.table) == 2, "re-ingest created duplicate rows"
    assert sorted(row["topic"] for row in connection.table.values()) == topics_before, \
        "derived keys are not stable across runs"
    assert connection.counters["updates"] >= 2 and connection.counters["inserts"] == 2, \
        f"re-ingest must update, not insert: {connection.counters}"


def test_ingest_json_explicit_task_id_still_wins(monkeypatch, tmp_path, capsys):
    """Records that carry a task_id must keep the documented key shape
    'topic_id: task_id' and must not trigger the derived-key warning."""
    records = [{"task_id": "t-1", "topic_id": "geom", "question": "2+2?", "answer": "4"}]
    connection = _FakeConnection()
    _run_ingest_json(monkeypatch, tmp_path, records, connection)
    (row,) = connection.table.values()
    assert row["topic"] == "geom: t-1", f"explicit task_id key changed: {row['topic']}"
    assert "[WARN]" not in capsys.readouterr().out, \
        "a record with an explicit task_id must not be reported as derived"


# --------------------------------------------------------------------------
# F-047 – manage_local backup truncates the previous dump before pg_dump runs
# --------------------------------------------------------------------------

def _run_manage_backup(monkeypatch, target, dump_succeeds, dumped_bytes):
    import scripts.manage_local as manage_local_module

    monkeypatch.setattr(manage_local_module, "validate_container_target",
                        lambda dsn, container, runner=None: "teachcopilot_local")

    def fake_run(cmd, stdout=None, check=False, **kwargs):
        if stdout is not None:
            stdout.write(dumped_bytes)
        if not dump_succeeds:
            raise subprocess.CalledProcessError(1, cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(manage_local_module, "subprocess",
                        types.SimpleNamespace(run=fake_run))
    return manage_local_module.run(
        ["backup", "--dsn", LOOPBACK_DSN, "--out", str(target)])


PREVIOUS_DUMP = b"PGDMP-yesterdays-good-dump" * 100
NEW_DUMP = b"PGDMP-fresh-good-dump" * 100


def test_manage_local_backup_replaces_previous_dump_only_on_success(tmp_path, monkeypatch):
    target = tmp_path / "backups" / "enriched.dump"
    target.parent.mkdir(parents=True)
    target.write_bytes(PREVIOUS_DUMP)
    return_code = _run_manage_backup(monkeypatch, target, dump_succeeds=True,
                                     dumped_bytes=NEW_DUMP)
    assert return_code == 0
    assert target.read_bytes() == NEW_DUMP, "successful dump must replace the target"
    assert list(target.parent.iterdir()) == [target], \
        "no temporary dump files may be left behind after a successful backup"


def test_manage_local_backup_failure_preserves_previous_dump(tmp_path, monkeypatch):
    """A failed pg_dump must leave the previous backup intact (V8 CL-2a/b:
    open('wb') truncated 2600 bytes to 0 before the dump subprocess ran)."""
    target = tmp_path / "backups" / "enriched.dump"
    target.parent.mkdir(parents=True)
    target.write_bytes(PREVIOUS_DUMP)
    with pytest.raises(subprocess.CalledProcessError):
        _run_manage_backup(monkeypatch, target, dump_succeeds=False,
                           dumped_bytes=b"partial garbage")
    assert target.read_bytes() == PREVIOUS_DUMP, \
        "a failed dump destroyed the previous backup (F-047)"
    assert list(target.parent.iterdir()) == [target], \
        "no temporary dump files may be left behind after a failed backup"
    source = (SCRIPTS / "manage_local.py").read_text(encoding="utf-8")
    tree = _parse(SCRIPTS / "manage_local.py")
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "open"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "out"
                and any(isinstance(a, ast.Constant) and a.value == "wb" for a in node.args)):
            raise AssertionError(
                "manage_local.py backup opened the final target with 'wb' "
                "again (F-045 regression guard for F-047)")


# --------------------------------------------------------------------------
# F-048 – apply_schema.py echoes the full DATABASE_URL including a password
# --------------------------------------------------------------------------

def test_apply_schema_print_redacts_database_url():
    """The DSN echoed by apply_schema.py must be redacted the same way
    eval_rag.py:49 redacts it (rsplit('@', 1)[-1]), while the connection
    itself keeps using the full DATABASE_URL."""
    source_path = SCRIPTS / "apply_schema.py"
    tree = _parse(source_path)
    printed = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "print" and node.args):
            arg = node.args[0]
            constants = [n.value for n in ast.walk(arg)
                         if isinstance(n, ast.Constant) and isinstance(n.value, str)]
            if any("Applying schema to" in value for value in constants):
                printed = eval(compile(ast.Expression(arg), "<dsn-print>", "eval"),
                               {"__builtins__": {}, "DATABASE_URL": FAKE_DSN})
                break
    assert printed is not None, \
        "apply_schema.py no longer prints the target database"
    assert "FAKE-NOT-A-SECRET" not in printed, \
        f"apply_schema.py echoes the DATABASE_URL password: {printed!r}"
    assert "example.invalid:5432/teachcopilot" in printed, \
        f"redaction must keep the host/database part, got {printed!r}"
    connects_with_full_dsn = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == "connect"
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "psycopg2"
        and any(isinstance(a, ast.Name) and a.id == "DATABASE_URL" for a in node.args)
        for node in ast.walk(tree))
    assert connects_with_full_dsn, \
        "apply_schema.py must still connect with the full DATABASE_URL"


# --------------------------------------------------------------------------
# F-049 – four personal-graph smoke scripts lack the loopback-DSN guard
# --------------------------------------------------------------------------

def _guard_facts(tree: ast.Module):
    """Locate the sibling-style guard in a smoke script: a hosts tuple built
    from parse_dsn(settings.database_url) plus PGHOSTADDR, an If that raises
    when the endpoint is not loopback, all before the first make_dsn call."""
    make_dsn_calls = _calls(tree, "make_dsn")
    assert make_dsn_calls, "no make_dsn call found"
    make_dsn_line = min(call.lineno for call in make_dsn_calls)

    parse_dsn_on_settings = any(
        isinstance(call.func, ast.Name) and call.func.id == "parse_dsn"
        and len(call.args) == 1
        and isinstance(call.args[0], ast.Attribute)
        and call.args[0].attr == "database_url"
        and isinstance(call.args[0].value, ast.Name)
        and call.args[0].value.id == "settings"
        for call in _calls(tree, "parse_dsn"))

    hosts_assign = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "hosts"
                and any(_is_os_getenv_call(n, "PGHOSTADDR") for n in ast.walk(node.value))):
            hosts_assign = node
            break

    loopback_set_line = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Set) and node.elts
                and all(isinstance(e, ast.Constant) for e in node.elts)
                and {e.value for e in node.elts} == LOOPBACK_HOSTS):
            loopback_set_line = node.lineno
            break

    guard_raise = None
    guard_if = None
    if hosts_assign is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise) and hosts_assign.lineno < node.lineno < make_dsn_line:
                guard_raise = node
                break
            if (isinstance(node, ast.If) and hosts_assign.lineno <= node.lineno < make_dsn_line
                    and any(isinstance(n, ast.Name) and n.id == "hosts"
                            for n in ast.walk(node.test))):
                guard_if = guard_if or node
    return {
        "make_dsn_line": make_dsn_line,
        "parse_dsn_on_settings": parse_dsn_on_settings,
        "hosts_assign": hosts_assign,
        "loopback_set_line": loopback_set_line,
        "guard_raise": guard_raise,
        "guard_if": guard_if,
    }


@pytest.mark.parametrize("script_name", SMOKE_SCRIPTS_WITHOUT_GUARD)
def test_smoke_scripts_check_loopback_dsn_before_touching_the_database(script_name):
    """Each script must carry the sibling guard (smoke_personal_graph_pilot.py
    style) before building its DSN: parse_dsn(settings.database_url), a hosts
    tuple including PGHOSTADDR, and a raise when the endpoint is not loopback
    or host-less (V8 CL-4 / CL-6c)."""
    tree = _parse(SCRIPTS / script_name)
    facts = _guard_facts(tree)
    assert facts["parse_dsn_on_settings"], \
        f"{script_name}: guard must parse_dsn(settings.database_url)"
    assert facts["hosts_assign"] is not None, \
        f"{script_name}: missing the hosts tuple with os.getenv('PGHOSTADDR')"
    assert facts["hosts_assign"].lineno < facts["make_dsn_line"], \
        f"{script_name}: the DSN guard must run before the first make_dsn call"
    assert facts["loopback_set_line"] is not None, \
        f"{script_name}: missing the loopback host set {sorted(LOOPBACK_HOSTS)}"
    assert facts["guard_raise"] is not None, \
        f"{script_name}: guard must raise before the first make_dsn call"


@pytest.mark.parametrize("script_name", SMOKE_SCRIPTS_WITHOUT_GUARD)
def test_smoke_script_guard_condition_matches_sibling_semantics(script_name):
    """The extracted guard condition must accept loopback endpoints and must
    reject remote hosts, non-loopback hostaddr and host-less DSNs (the CL-6c
    PGHOST fallback hole)."""
    facts = _guard_facts(_parse(SCRIPTS / script_name))
    assert facts["guard_if"] is not None, \
        f"{script_name}: no If-condition guard found on the hosts tuple"
    condition = compile(ast.Expression(facts["guard_if"].test), "<guard>", "eval")

    def trips(hosts):
        return bool(eval(condition, {"__builtins__": {}, "any": any,
                                     "all": all, "hosts": hosts}))

    assert trips(("db.remote.example", None, None)), \
        f"{script_name}: guard must reject a remote host"
    assert trips((None, None, None)), \
        f"{script_name}: guard must reject a host-less DSN (libpq PGHOST fallback)"
    assert trips(("127.0.0.1", "10.9.8.7", None)), \
        f"{script_name}: guard must reject a non-loopback hostaddr"
    assert not trips(("127.0.0.1", None, None)), \
        f"{script_name}: guard must allow 127.0.0.1"
    assert not trips(("localhost", None, None)), \
        f"{script_name}: guard must allow localhost"
    assert not trips(("::1", None, None)), \
        f"{script_name}: guard must allow ::1"


# --------------------------------------------------------------------------
# F-050 – transition_memory_budget materializes an empty prior-ledger file
# --------------------------------------------------------------------------

def _transition_catalog(tmp_path: Path) -> Path:
    catalog = tmp_path / "models.json"
    catalog.write_text(json.dumps({"data": [{
        "id": "openai/gpt-6.1-sol",
        "pricing": {"prompt": "0.0000002", "completion": "0.0000008"},
    }]}), encoding="utf-8")
    return catalog


def test_transition_fails_loudly_when_a_prior_ledger_is_missing(tmp_path):
    """A missing prior ledger must fail with the canonical message before
    sqlite3.connect runs (V8 CL-5: connect materialized a 0-byte
    canonical-named ledger file, later misleading the v5 is_file guard)."""
    from scripts.transition_memory_budget import transition

    research_dir = tmp_path / ".local" / "research_memory"
    research_dir.mkdir(parents=True)
    output = tmp_path / "transition-receipt"
    with pytest.raises(FileNotFoundError, match="canonical prior ledger is missing"):
        transition(tmp_path, _transition_catalog(tmp_path), output)
    assert not any(research_dir.iterdir()), \
        f"missing-ledger handling must not materialize files: {list(research_dir.iterdir())}"
    assert list(tmp_path.rglob("*.sqlite3")) == [], \
        "an empty ledger file was silently created (F-050)"
