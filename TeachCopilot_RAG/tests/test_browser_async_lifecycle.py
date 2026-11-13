"""Playwright must release its event loop before unrelated asyncio tests run."""
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

import pytest


def test_two_browser_modules_then_async_graph_tests_in_one_process(tmp_path):
    root = Path(__file__).resolve().parents[1]
    report = tmp_path / 'lifecycle.xml'
    nodes = [
        'tests/test_session_end_context_ui.py::test_final_assistant_turn_has_its_own_safe_context_block_without_becoming_student_evidence',
        'tests/test_ui_acceptance.py::test_student_sees_only_own_profile_and_chat_renders_untrusted_text_safely',
        'tests/test_personal_graph_index.py::test_build_then_reuse_without_embedding_or_ingest',
        'tests/test_personal_graph_bridge.py::test_selected_context_chat_skips_graphiti_and_uses_source_scoped_search',
    ]
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', *nodes, f'--junitxml={report}'],
                            cwd=root, capture_output=True, text=True, encoding='utf-8',
                            errors='replace', timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    cases = ElementTree.parse(report).findall('.//testcase')
    assert len(cases) == len(nodes)
    if any(case.find('skipped') is not None for case in cases):
        pytest.skip('Playwright Chromium is unavailable for the lifecycle regression')
    assert all(case.find('failure') is None and case.find('error') is None for case in cases)
