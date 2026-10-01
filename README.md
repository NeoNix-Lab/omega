# Omega (Ω) — Canonical Orchestrator for quant-platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Companion: quant-platform](https://img.shields.io/badge/companion-quant--platform-purple.svg)](https://github.com/NeoNix-Lab/quant-platform)

**Omega** is the dedicated user-facing command-line orchestrator and research workspace for [**quant-platform**](https://github.com/NeoNix-Lab/quant-platform).

---

## 🎯 Architectural Boundary & Zero Reinvention

In accordance with [**AGENTS.md**](https://github.com/NeoNix-Lab/quant-platform/blob/main/AGENTS.md) and [**ADR-0002**](https://neonix-lab.github.io/quant-platform/decisions/ADR-0002-api-first/), **Omega does not invent parallel quantitative logic, synthetic data, or duplicate backtest models**:

- 🛡️ **Zero Domain Duplication**: All quantitative business logic, temporal causal enforcement, footprint representations, walk-forward cross-validation, and double-entry portfolio accounting reside authoritatively in `quant-platform`.
- 🎛️ **Pure Orchestration (Capability J04)**: Omega acts as the thin, ergonomic client harness that directly composes the frozen application services of `quant_platform.application`.
- 📜 **Full Lineage & Provenance**: Every orchestrated run logs immutable Git commit SHAs, dataset partition signatures, and cryptographic artifact hashes (`sha256:...`) into `experiments/registry.jsonl`.

---

## 🏗️ Repository Layout

```text
omega/
├── experiments/        # Lightweight append-only experiment registry (registry.jsonl)
├── notebooks/          # Exploratory Jupyter notebooks & research prototypes
├── results/            # Figures, evaluation plots, and study logs (ignored by Git)
├── src/
│   └── omega/          # Orchestrator client package
│       ├── features/   # Canonical feature hub (interfacing quant_platform.features)
│       ├── strategies/ # Canonical strategy hub (interfacing quant_platform.strategy)
│       ├── orchestrator/# Application service composition (Wave 4 Replay & Wave 5 ML)
│       └── tracking/   # Immutable experiment registry & provenance capture
├── studies/            # Scripted research studies driving quant-platform
├── pyproject.toml      # Project metadata & pinned quant-platform extra
└── README.md
```

---

## ⚡ Quickstart Setup

### 1. Setup Virtual Environment

```bash
cd omega

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
.venv\Scripts\Activate.ps1       # On Windows (PowerShell)

# Install Omega + the quant-platform core (pinned tag) + dev tools
pip install -e ".[platform,dev]"

# `omega train` / `omega replay` run the platform's golden proofs, which read fixtures that are only
# present in a quant-platform checkout (they are not packaged). For those commands use an editable
# checkout instead of the pinned git install:
#   pip install -e ../quant-platform
```

---

## 🎛️ CLI Orchestrator Commands

Omega exposes the canonical capabilities of `quant-platform` directly from the command line:

### 1. Inspect Platform Status & Capabilities
```bash
omega status
```

### 2. Inspect Frozen Canonical Features & Representations
```bash
omega features list
```

### 3. Inspect Canonical Strategy Specifications
```bash
omega strategies list
```

### 4. Orchestrate Canonical Wave 5 Supervised Training
Executes `quant_platform.application.run_wave5_golden_supervised_proof` over canonical Bybit trade fixtures, verifies bitwise determinism, and logs formal artifact signatures:
```bash
omega train --name wave5_canonical_model
```

### 5. Orchestrate Canonical Wave 4 Deterministic Replay
Builds and verifies the canonical `ReplaySpec` against `DataGateway` and `HistoricalReplayRuntime`:
```bash
omega replay --capital 10000
```
*(To run against a live PostgreSQL catalog database, supply `--dsn postgresql://user:pass@host/db` or set `GOLDEN_REPLAY_E2E_DSN`).*

### 6. View Ranked Leaderboard of Tracked Runs
```bash
omega leaderboard
```

### 7. Inspect Full Provenance Manifest of a Specific Run
```bash
omega inspect <run_id_or_hash>
```

---

## 📜 Invariants to Preserve

1. **Keep Business Logic in quant-platform**: Never implement competing market-data parsers, fill engines, or fee schedules in Omega.
2. **Never Add Heavy Binaries to Git**: Model weights, parquet partitions, and databases belong in local storage or external object storage. Only commit lightweight metadata manifests to `experiments/`.
