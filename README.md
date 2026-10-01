# Omega (Ω) — Alpha Research & Experimentation Lab

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Companion: quant-platform](https://img.shields.io/badge/companion-quant--platform-purple.svg)](https://github.com/NeoNix-Lab/quant-platform)

**Omega** is the dedicated quantitative research, exploratory experimentation, and alpha discovery laboratory companion to [**quant-platform**](https://github.com/NeoNix-Lab/quant-platform).

---

## 🎯 Purpose & Philosophy

While `quant-platform` is the authoritative production engine designed with frozen data contracts, institutional double-entry accounting, and strict bitwise determinism, **Omega** is designed for **fast, unconstrained hypothesis exploration**:

- 🧪 **Rapid Feature & Signal Prototyping**: Test order flow imbalance, microstructural features, and deep learning ideas without the governance overhead of production releases.
- 📊 **Interactive Analysis & Notebooks**: Run Jupyter notebooks, event studies, and exploratory visualizations.
- 🛡️ **Zero Contamination**: Keeps `quant-platform` clean from heavy experiment artifacts, exploratory plots, model checkpoints, and transient scripts.
- 🚀 **Clear Promotion Path**: Validated alpha discoveries with robust Deflated Sharpe Ratios (DSR) and purged cross-validation are promoted into canonical `StrategySpec` policies within `quant-platform`.

---

## 🏗️ Repository Layout

```text
omega/
├── data/               # Local cache & temporary files (ignored by Git)
├── notebooks/          # Exploratory Jupyter notebooks & prototypes
├── results/            # Research logs, figures, and study outputs (ignored by Git)
├── src/
│   └── omega/          # Reusable research library
│       ├── evaluation/ # IC, t-stat, purged/embargoed evaluation helpers
│       ├── features/   # Experimental feature transformers
│       ├── signals/    # Factor combinations and directional filters
│       └── viz/        # Order flow, footprint, and PnL visualizers
├── studies/            # Parameterized, reproducible Python research scripts
├── pyproject.toml      # Project metadata and dependencies
├── requirements.txt    # Setup requirements with editable quant-platform link
└── README.md
```

---

## ⚡ Quickstart Setup

### 1. Clone & Setup Virtual Environment

```bash
# Navigate to the omega directory
cd omega

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
.venv\Scripts\Activate.ps1       # On Windows (PowerShell)

# Install research stack + editable quant-platform link
pip install -r requirements.txt
```

### 2. Run the First Alpha Study

```bash
python studies/quickstart_alpha_study.py
```

---

## 🔄 Research to Production Workflow

```text
┌────────────────────────────────────────────────────────┐
│                   OMEGA RESEARCH LAB                   │
│  1. Formulate Hypothesis (EventSpec, Feature Idea)     │
│  2. Exploratory Notebooks & Visual Inspection          │
│  3. Scripted Event Study in studies/                   │
│  4. Evaluate Information Coefficient (IC) & Stability  │
└───────────────────────────┬────────────────────────────┘
                            │ Validated Statistical Alpha
                            ▼
┌────────────────────────────────────────────────────────┐
│                     QUANT-PLATFORM                     │
│  1. python tools/workflow.py start <issue>             │
│  2. Formal FeatureDefinition / G01 StrategySpec        │
│  3. Purged & Embargoed Cross-Validation (ADR-0031)     │
│  4. Deterministic ReplayEngine Execution (ADR-0012)    │
│  5. Pull Request & Review into implement/wave-*        │
└────────────────────────────────────────────────────────┘
```

---

## 📜 Invariants to Preserve

Even in research scripts, always observe these core principles:
1. **Consume Market Data via DataGateway**: Never hardcode direct file paths to raw parquet partitions.
2. **Respect the Temporal Availability Floor**: Decisions at time $t$ must only observe data available strictly at or before $t$ ($t_{\text{available}} \le t_{\text{decision}}$).
3. **Keep Heavy Data out of Git**: Large `.parquet`, `.csv`, `.pt`, and `.pkl` files belong in `data/` or external object storage, never in version control.
