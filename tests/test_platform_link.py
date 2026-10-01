"""Tests for the platform link layer: compatibility check, lazy exports, pin consistency, CLI status."""

from __future__ import annotations

import argparse
import json
import tomllib
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

from omega import cli
from omega.platform_link import (
    PLATFORM_PIN,
    PLATFORM_SYMBOLS,
    CompatResult,
    CompatStatus,
    check_compat,
    compat,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class _FakeDist:
    def __init__(self, version: str, direct_url: Any) -> None:
        self.version = version
        self._direct_url = direct_url

    def read_text(self, filename: str) -> str | None:
        assert filename == "direct_url.json"
        if self._direct_url is None or isinstance(self._direct_url, str):
            return self._direct_url
        return json.dumps(self._direct_url)


def _install(monkeypatch: pytest.MonkeyPatch, dist: _FakeDist | None) -> None:
    def distribution(name: str) -> _FakeDist:
        assert name == "quant-platform"
        if dist is None:
            raise metadata.PackageNotFoundError(name)
        return dist

    monkeypatch.setattr(compat.metadata, "distribution", distribution)


def _vcs(commit: str) -> dict[str, Any]:
    return {"url": "https://example.invalid/qp", "vcs_info": {"vcs": "git", "commit_id": commit}}


def test_compat_ok_for_pinned_vcs_install(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _FakeDist(PLATFORM_PIN.version, _vcs(PLATFORM_PIN.commit)))
    result = check_compat()
    assert result.status is CompatStatus.OK
    assert result.installed_commit == PLATFORM_PIN.commit
    assert PLATFORM_PIN.tag in result.message


def test_compat_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, None)
    result = check_compat()
    assert result == CompatResult(CompatStatus.MISSING, None, None, result.message)
    assert "pip install" in result.message


def test_compat_mismatch_on_version(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _FakeDist("9.9.9", _vcs(PLATFORM_PIN.commit)))
    result = check_compat()
    assert result.status is CompatStatus.MISMATCH
    assert result.installed_version == "9.9.9"


def test_compat_mismatch_on_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _FakeDist(PLATFORM_PIN.version, _vcs("0" * 40)))
    result = check_compat()
    assert result.status is CompatStatus.MISMATCH
    assert result.installed_commit == "0" * 40


@pytest.mark.parametrize("direct_url", [None, "", "{not json", json.dumps({"dir_info": {"editable": True}})])
def test_compat_ok_but_unverifiable_commit(monkeypatch: pytest.MonkeyPatch, direct_url: Any) -> None:
    _install(monkeypatch, _FakeDist(PLATFORM_PIN.version, direct_url))
    result = check_compat()
    assert result.status is CompatStatus.OK
    assert result.installed_commit is None
    assert "not verifiable" in result.message


def test_pyproject_platform_extra_matches_the_pin() -> None:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extra = data["project"]["optional-dependencies"]["platform"]
    assert len(extra) == 1
    assert extra[0].endswith(f"@{PLATFORM_PIN.tag}")


def test_all_exported_symbols_resolve_from_platform_link() -> None:
    pytest.importorskip("quant_platform")
    import omega.platform_link as link

    missing = []
    for name in sorted(PLATFORM_SYMBOLS):
        try:
            getattr(link, name)
        except (AttributeError, ImportError) as exc:
            missing.append(f"{PLATFORM_SYMBOLS[name]}.{name} ({exc})")
    assert not missing, "not available at the pinned platform version: " + ", ".join(missing)


def test_unknown_attribute_raises_attribute_error() -> None:
    import omega.platform_link as link

    with pytest.raises(AttributeError):
        link.definitely_not_a_platform_symbol  # noqa: B018


def _status(monkeypatch: pytest.MonkeyPatch, status: CompatStatus, *, strict: bool) -> int:
    monkeypatch.setattr(
        cli, "check_compat", lambda: CompatResult(status, None, None, f"fake {status}")
    )
    return cli.cmd_status(argparse.Namespace(strict=strict))


def test_status_prints_platform_and_pin(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    assert _status(monkeypatch, CompatStatus.OK, strict=True) == 0
    out = capsys.readouterr().out
    assert "quant-platform Core" in out and "[OK]" in out
    assert PLATFORM_PIN.tag in out


@pytest.mark.parametrize("status", [CompatStatus.MISSING, CompatStatus.MISMATCH])
def test_status_strict_fails_when_platform_is_not_ok(monkeypatch: pytest.MonkeyPatch, status: CompatStatus) -> None:
    assert _status(monkeypatch, status, strict=True) == 1


@pytest.mark.parametrize("status", [CompatStatus.MISSING, CompatStatus.MISMATCH])
def test_status_non_strict_degrades_gracefully(monkeypatch: pytest.MonkeyPatch, status: CompatStatus) -> None:
    assert _status(monkeypatch, status, strict=False) == 0
