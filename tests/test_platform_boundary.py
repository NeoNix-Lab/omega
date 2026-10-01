"""Architectural test: only ``omega.platform_link`` may import ``quant_platform``."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCANNED_ROOTS = (REPO_ROOT / "src" / "omega", REPO_ROOT / "studies")
ALLOWED_DIR = REPO_ROOT / "src" / "omega" / "platform_link"


def _imports_platform(source: str) -> list[int]:
    """Line numbers of statements importing ``quant_platform`` (any submodule)."""
    lines: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] == "quant_platform" for alias in node.names):
                lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and (node.module or "").split(".")[0] == "quant_platform":
                lines.append(node.lineno)
    return lines


def find_violations(roots: tuple[Path, ...], allowed_dir: Path) -> list[str]:
    violations: list[str] = []
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            if allowed_dir in path.parents:
                continue
            for lineno in _imports_platform(path.read_text(encoding="utf-8")):
                violations.append(f"{path.relative_to(root.parent)}:{lineno}")
    return violations


def test_only_platform_link_imports_quant_platform() -> None:
    assert find_violations(SCANNED_ROOTS, ALLOWED_DIR) == []


def test_scanner_detects_a_violation(tmp_path: Path) -> None:
    pkg = tmp_path / "omega"
    (pkg / "platform_link").mkdir(parents=True)
    (pkg / "platform_link" / "ok.py").write_text("import quant_platform\n")
    (pkg / "bad_plain.py").write_text("import quant_platform.access\n")
    (pkg / "bad_from.py").write_text("x = 1\nfrom quant_platform.replay import ReplaySpec\n")
    (pkg / "fine.py").write_text("from .platform_link import check_compat\nimport quant_platform_like\n")

    found = find_violations((pkg,), pkg / "platform_link")

    assert sorted(found) == ["omega/bad_from.py:2", "omega/bad_plain.py:1"]
