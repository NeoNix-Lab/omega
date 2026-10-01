"""Compile a strategy spec into a platform ``StrategySpec`` plus a signal provider factory."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .. import platform_link as qp
from ..features import FeatureSpec, compile_feature_provider
from .signals import ENTRY_SIGNAL_KEY, EXIT_SIGNAL_KEY, SignalLayerPolicy, SignalProvider, make_always_open_session
from .spec import StrategyDef, StrategySpecError


@dataclass(frozen=True)
class CompiledStrategy:
    """A strategy ready to replay: the platform spec, the features it needs, and a provider factory."""

    definition: StrategyDef
    feature_specs: tuple[FeatureSpec, ...]
    platform_spec: Any  # quant_platform.strategy.StrategySpec

    @property
    def identity(self) -> str:
        """The platform ``strategy_identity``: covers rules, thresholds, exits, sizing, risk and the
        identities of the feature specs (so any semantic change gives a new identity)."""
        return str(self.platform_spec.strategy_identity)

    def make_provider(self) -> SignalProvider:
        """A fresh provider (feature computation + signals). Use exactly one per replay."""
        return SignalProvider(
            self.definition,
            compile_feature_provider(self.feature_specs),
            provenance=self.definition_identity,
        )

    @property
    def definition_identity(self) -> str:
        return self.definition.identity


def compile_strategy(definition: StrategyDef, feature_specs: Sequence[FeatureSpec]) -> CompiledStrategy:
    """Validate that every required feature is provided, then build the platform ``StrategySpec``.

    Only the feature specs listed in ``definition.features`` are used (extra ones are ignored). A listed
    feature that is not provided fails here, at compile time: there is no silent fallback.
    """
    by_name = {spec.name: spec for spec in feature_specs}
    missing = [name for name in definition.features if name not in by_name]
    if missing:
        raise StrategySpecError(
            f"strategy {definition.name!r} needs feature spec(s) {missing}, but only {sorted(by_name)} were provided"
        )
    selected = tuple(by_name[name] for name in definition.features)

    name = definition.name
    risk = definition.risk_dict
    cooldown = definition.cooldown_dict
    direction = qp.Direction.LONG if definition.entry.direction == "long" else qp.Direction.SHORT
    try:
        platform_spec = qp.StrategySpec(
            strategy_key=f"omega.{name}",
            semantic_version=definition.version,
            entry_policy=qp.EntryPolicy(
                policy_key=f"omega.{name}.entry", direction=direction, signal_key=ENTRY_SIGNAL_KEY, confidence="1"
            ),
            exit_policy=qp.ExitPolicy(policy_key=f"omega.{name}.exit", signal_key=EXIT_SIGNAL_KEY),
            position_policy=qp.PositionPolicy(
                policy_key=f"omega.{name}.position",
                long_target_position=definition.position_size,
                short_target_position=definition.position_size,
            ),
            sizing_policy=qp.FixedFractionSizingPolicy(
                policy_key=f"omega.{name}.sizing",
                lot_size=definition.lot_size,
                min_size=definition.min_size,
                max_size=definition.max_size,
            ),
            risk_policy=qp.CapitalRiskPolicy(policy_key=f"omega.{name}.risk", **risk),
            session_policy=make_always_open_session(f"omega.{name}.session"),
            cooldown_policy=qp.CooldownPolicyDefinition(policy_key=f"omega.{name}.cooldown", **cooldown),
            signal_combination_policy=qp.SignalCombinationPolicy(
                policy_key=f"omega.{name}.signals", mode=qp.SignalCombinationMode.ALL, signal_keys=(ENTRY_SIGNAL_KEY,)
            ),
            execution_policy=SignalLayerPolicy(definition, {spec.name: spec.identity for spec in selected}),
        )
    except qp.StrategyError as exc:
        raise StrategySpecError(f"strategy {name!r} is not a valid platform StrategySpec: {exc}") from exc
    return CompiledStrategy(definition=definition, feature_specs=selected, platform_spec=platform_spec)
