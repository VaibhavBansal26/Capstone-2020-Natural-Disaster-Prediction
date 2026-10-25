"""Runs the node tests for src/ndp/static/voice.js (command interpreter, number/date parsing)."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
VOICE_TESTS = sorted((ROOT / "tests" / "voice").glob("*.test.mjs"))


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_voice_js_node_suite():
    assert VOICE_TESTS, "tests/voice/*.test.mjs is missing"
    script = ROOT / "src" / "ndp" / "static" / "voice.js"
    check = subprocess.run(
        ["node", "--check", str(script)], capture_output=True, text=True, timeout=30
    )
    assert check.returncode == 0, check.stderr
    # Pass the files explicitly: `node --test <dir>` is not expanded on newer node versions.
    proc = subprocess.run(
        ["node", "--test", *map(str, VOICE_TESTS)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
