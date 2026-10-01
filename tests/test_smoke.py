"""Smoke tests: the package imports and the CLI entry points work from the repo root."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "omega", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_package_imports() -> None:
    import omega

    assert omega.__version__


def test_cli_help_exits_zero() -> None:
    result = _run_cli("--help")
    assert result.returncode == 0, result.stderr
    assert "status" in result.stdout


def test_status_exits_zero_even_without_the_platform() -> None:
    result = _run_cli("status")
    assert result.returncode == 0, result.stderr
    assert "OMEGA" in result.stdout


def test_root_has_no_module_shadowing_the_package() -> None:
    assert not (REPO_ROOT / "omega.py").exists()
