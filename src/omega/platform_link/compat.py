"""Compatibility check between the installed quant-platform and the version Omega is built against.

This module deliberately does **not** import ``quant_platform``: it only reads installed-package
metadata, so it works (and reports ``MISSING``) when the platform is not installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from importlib import metadata

DISTRIBUTION = "quant-platform"


@dataclass(frozen=True)
class PlatformPin:
    """The quant-platform release Omega is developed and tested against."""

    version: str
    commit: str
    tag: str


# Keep in sync with the ``platform`` extra in pyproject.toml (enforced by tests/test_platform_link.py).
PLATFORM_PIN = PlatformPin(
    version="0.1.0",
    commit="093f4f6fc267cd294cac31527f4557e55f30cf51",
    tag="wave-6-live-consumer-data-plane-storage-lifecycle-v1",
)


class CompatStatus(StrEnum):
    OK = "OK"
    MISSING = "MISSING"
    MISMATCH = "MISMATCH"


@dataclass(frozen=True)
class CompatResult:
    status: CompatStatus
    installed_version: str | None
    installed_commit: str | None
    message: str


def _installed_commit(dist: metadata.Distribution) -> str | None:
    """Commit of a VCS install, from PEP 610 ``direct_url.json``; ``None`` when not verifiable."""
    try:
        raw = dist.read_text("direct_url.json")
        if not raw:
            return None
        commit = json.loads(raw).get("vcs_info", {}).get("commit_id")
    except (OSError, ValueError, AttributeError):
        return None
    return commit if isinstance(commit, str) and commit else None


def check_compat(pin: PlatformPin = PLATFORM_PIN) -> CompatResult:
    """Compare the installed quant-platform against ``pin`` without importing it."""
    try:
        dist = metadata.distribution(DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return CompatResult(
            CompatStatus.MISSING,
            None,
            None,
            f"{DISTRIBUTION} is not installed; run: pip install -e \".[platform]\"",
        )

    version = dist.version
    commit = _installed_commit(dist)

    if version != pin.version:
        return CompatResult(
            CompatStatus.MISMATCH,
            version,
            commit,
            f"installed version {version} != pinned {pin.version} (tag {pin.tag})",
        )
    if commit is not None and commit != pin.commit:
        return CompatResult(
            CompatStatus.MISMATCH,
            version,
            commit,
            f"installed commit {commit[:12]} != pinned {pin.commit[:12]} (tag {pin.tag})",
        )

    note = "" if commit is not None else " (commit not verifiable: not a VCS install)"
    return CompatResult(
        CompatStatus.OK,
        version,
        commit,
        f"{DISTRIBUTION} {version} matches pin {pin.tag}{note}",
    )
