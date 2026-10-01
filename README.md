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
├── requirements.txt    # Convenience: editable install with platform + dev extras
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

# Install Omega + the quant-platform core (pinned tag) + dev tools
pip install -e ".[platform,dev]"
```

### 2. Run the First Alpha Study

```bash
python studies/quickstart_alpha_study.py
```

---

## 🎛️ CLI Orchestrator & Experiment Tracking

Omega includes an integrated command-line orchestrator that drives model training, purges walk-forward folds, evaluates metrics, seals artifacts cryptographically, and maintains a lightweight append-only experiment registry (`experiments/registry.jsonl`).

```bash
# 1. Inspect environment, quant-platform link, git revision, and artifact storage
omega status

# 2. List available derived features in the Feature Hub
omega features list

# 3. List available alpha strategies in the Strategy Hub
omega strategies list

# 4. Launch a causal alpha training run selecting specific features
omega train --name btc_alpha --features cvd_zscore,rsi,realized_vol --horizon 5

# 5. Simulate a strategy replay with portfolio accounting
omega replay --strategy volatility_breakout --capital 10000

# 6. View the ranked leaderboard of all tracked experiments
omega leaderboard --sort-by dsr

# 7. Inspect full JSON provenance and metrics for a specific run
omega inspect run_20261001_093514_custom_derived_features
```

---

## 🧩 Adding Custom Features & Strategies

### 1. Register a Derived Feature (in `src/omega/features/`)
```python
from omega.features.registry import register_feature
import pandas as pd

@register_feature(name="my_indicator", description="Custom price ratio", required_columns=("close",))
def compute_my_indicator(df: pd.DataFrame, window: int = 14) -> pd.Series:
    return (df["close"] / df["close"].rolling(window).mean() - 1.0).fillna(0.0)
```

### 2. Register a Strategy (in `src/omega/strategies/`)
```python
from omega.strategies.base import BaseStrategy, SignalDecision
from omega.strategies.registry import register_strategy
import pandas as pd

@register_strategy(name="my_strategy", description="Simple threshold strategy")
class MyStrategy(BaseStrategy):
    def generate_signal(self, row: pd.Series, context: dict) -> SignalDecision:
        if row.get("feat_my_indicator", 0.0) > 0.02:
            return SignalDecision.LONG(stop_loss=0.01, take_profit=0.02)
        return SignalDecision.HOLD()
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
