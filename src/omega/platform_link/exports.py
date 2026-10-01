"""Platform symbols Omega is allowed to use, and where they live in ``quant_platform``.

Add a name here (and only here) when Omega needs another platform symbol. If a name does not exist
at the pinned platform version, ``tests/test_platform_link.py`` fails naming it: report it in the
upstream-asks tracking issue instead of working around it.
"""

from __future__ import annotations

PLATFORM_SYMBOLS: dict[str, str] = {
    # Data access
    "DataGateway": "quant_platform.access",
    "Catalog": "quant_platform.access",
    "DataRequest": "quant_platform.access",
    "CoveragePolicy": "quant_platform.access",
    "LifecyclePolicy": "quant_platform.access",
    "DatasetIdentity": "quant_platform.data.models",
    "Instant": "quant_platform.data.models",
    # Replay
    "HistoricalReplayRuntime": "quant_platform.replay",
    "ReplaySpec": "quant_platform.replay",
    "ReplayContext": "quant_platform.replay",
    "ReplayResult": "quant_platform.replay",
    # Strategy
    "StrategySpec": "quant_platform.strategy",
    "StrategyInput": "quant_platform.strategy",
    "EntryPolicy": "quant_platform.strategy",
    "ExitPolicy": "quant_platform.strategy",
    "PositionPolicy": "quant_platform.strategy",
    "SignalCombinationPolicy": "quant_platform.strategy",
    "CapitalRiskPolicy": "quant_platform.strategy",
    "FixedFractionSizingPolicy": "quant_platform.strategy",
    # Execution costs
    "FeeSchedule": "quant_platform.execution",
    "SyntheticSlippageModel": "quant_platform.execution",
    # Validation
    "evaluate_dsr_v1": "quant_platform.validation.robustness",
    "evaluate_pbo_v1": "quant_platform.validation.robustness",
    "ComparableTrialPanel": "quant_platform.validation.robustness",
    "EffectiveTrialCountEvidence": "quant_platform.validation.robustness",
    "WalkForwardScheduleSpec": "quant_platform.validation.walk_forward",
    "build_walk_forward_folds": "quant_platform.validation.walk_forward",
}
