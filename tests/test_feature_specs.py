"""Feature spec model: validation, strict YAML, canonical identity, kernel registry."""

from __future__ import annotations

import glob
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("quant_platform")

from omega import platform_link as qp  # noqa: E402
from omega.features import (  # noqa: E402
    FeatureSpecError,
    KernelDef,
    KernelError,
    list_kernels,
    load_feature_spec,
    parse_feature_spec_text,
    register_kernel,
)
from omega.features import kernels as kernels_module  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# fixtures/candle-definition-v1/golden-5m.json in quant-platform (expected_definition_identity)
GOLDEN_5M_DEFINITION_IDENTITY = (
    "candle-definition-v1:sha256:20dc1c641768e98d05d75db0a2e57c1a948156bb5f2302bd0c0c321c735c2d0e"
)

MOMENTUM = """
schema: omega.feature-spec/v1
name: momentum_5m_12
version: 1
inputs: {representation: candles, bar: 5m}
kernel: price_momentum
params: {window: 12}
"""


def _spec(**overrides: object) -> str:
    base = textwrap.dedent(MOMENTUM)
    for key, value in overrides.items():
        base += f"{key}: {value}\n"
    return base


def test_valid_candle_spec_fields() -> None:
    spec = parse_feature_spec_text(MOMENTUM)
    assert spec.key == "feature.momentum_5m_12"
    assert spec.representation == "candles"
    assert spec.bar_ns == 300_000_000_000
    assert spec.warmup == 13  # window + 1 candles
    assert spec.params_dict == {"window": 12}
    assert spec.identity.startswith("omega-feature-spec-v1:sha256:")


def test_bar_5m_matches_the_platform_golden_candle_definition() -> None:
    spec = parse_feature_spec_text(MOMENTUM)
    definition = qp.CandleDefinitionV1.from_duration(spec.bar_ns)
    assert definition.definition_identity == GOLDEN_5M_DEFINITION_IDENTITY


def test_identity_is_stable_across_formatting_key_order_and_duration_spelling() -> None:
    a = parse_feature_spec_text(MOMENTUM)
    b = parse_feature_spec_text(
        textwrap.dedent(
            """
            kernel: price_momentum
            params:
              window: 12
            inputs:
              bar: 300s
              representation: candles
            version: 1
            name: momentum_5m_12
            schema: omega.feature-spec/v1
            """
        )
    )
    c = parse_feature_spec_text(MOMENTUM.replace("bar: 5m", "bar: 300000000000"))
    assert a.identity == b.identity == c.identity


def test_explicit_default_equals_omitted_default() -> None:
    omitted = parse_feature_spec_text(MOMENTUM.replace("params: {window: 12}", ""))
    explicit = parse_feature_spec_text(MOMENTUM)
    assert omitted.params_dict == {"window": 12} == explicit.params_dict  # 12 is the kernel default
    assert omitted.identity == explicit.identity


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("window: 12", "window: 13"),
        ("bar: 5m", "bar: 1m"),
        ("version: 1", "version: 2"),
        ("name: momentum_5m_12", "name: momentum_other"),
    ],
)
def test_identity_changes_with_any_semantic_change(old: str, new: str) -> None:
    assert parse_feature_spec_text(MOMENTUM).identity != parse_feature_spec_text(MOMENTUM.replace(old, new)).identity


def test_extra_warmup_is_allowed_and_part_of_identity() -> None:
    longer = parse_feature_spec_text(_spec(warmup=50))
    assert longer.warmup == 50
    assert longer.identity != parse_feature_spec_text(MOMENTUM).identity


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        (MOMENTUM.replace("window: 12", "window: 0"), "must be >= 1"),
        (MOMENTUM.replace("window: 12", "window: -3"), "must be >= 1"),
        (MOMENTUM.replace("window: 12", "window: true"), "got a boolean"),
        (MOMENTUM.replace("window: 12", "window: 2.5"), "must be an integer"),
        (MOMENTUM.replace("window: 12", "window: 12, nonsense: 1"), "unknown param"),
        (MOMENTUM.replace("kernel: price_momentum", "kernel: does_not_exist"), "available kernels"),
        (MOMENTUM.replace("schema: omega.feature-spec/v1", "schema: other/v9"), "schema must be"),
        (MOMENTUM.replace("name: momentum_5m_12", "name: Bad-Name"), "name must match"),
        (MOMENTUM.replace("version: 1", "version: 0"), "version must be"),
        (MOMENTUM.replace("bar: 5m", "bar: 5x"), "inputs.bar"),
        (MOMENTUM.replace("{representation: candles, bar: 5m}", "{representation: candles}"), "inputs.bar"),
        (MOMENTUM.replace("representation: candles", "representation: trades"), "works on candles"),
        (MOMENTUM + "surprise: 1\n", "unknown key"),
        (_spec(warmup=3), "warmup must be an integer >= 13"),
        (_spec(output="int"), "output must be"),
        ("[1, 2, 3]", "must be a mapping"),
    ],
)
def test_invalid_specs_fail_at_load_with_actionable_messages(text: str, fragment: str) -> None:
    with pytest.raises(FeatureSpecError, match=fragment):
        parse_feature_spec_text(text, source="spec.yaml")


def test_trades_spec_rejects_a_bar() -> None:
    text = """
schema: omega.feature-spec/v1
name: imb
version: 1
inputs: {representation: trades, bar: 1m}
kernel: imbalance_ratio
"""
    with pytest.raises(FeatureSpecError, match="only valid for candle"):
        parse_feature_spec_text(text)


def test_yaml_duplicate_keys_are_rejected() -> None:
    with pytest.raises(FeatureSpecError, match="duplicate key"):
        parse_feature_spec_text(MOMENTUM.replace("params: {window: 12}", "params: {window: 12, window: 20}"))


def test_error_messages_name_the_source_file() -> None:
    with pytest.raises(FeatureSpecError, match=r"my_spec\.yaml: unknown key"):
        parse_feature_spec_text(MOMENTUM + "x: 1\n", source="my_spec.yaml")


def test_list_of_builtin_kernels() -> None:
    assert [k.kernel_id for k in list_kernels()] == [
        "atr_pct",
        "cvd_zscore",
        "imbalance_ratio",
        "price_momentum",
        "realized_vol",
    ]


def test_registering_a_duplicate_kernel_raises() -> None:
    existing = next(k for k in list_kernels() if k.kernel_id == "cvd_zscore")
    with pytest.raises(KernelError, match="already registered"):
        register_kernel(existing)


def test_unknown_kernel_error_lists_the_available_ones() -> None:
    with pytest.raises(FeatureSpecError) as excinfo:
        parse_feature_spec_text(MOMENTUM.replace("price_momentum", "nope"))
    message = str(excinfo.value)
    assert all(name in message for name in ("atr_pct", "cvd_zscore", "price_momentum"))


@pytest.fixture
def custom_kernel():
    """Register a throw-away kernel, remove it afterwards."""

    class _LastPrice:
        def __init__(self, scale: int) -> None:
            self._scale = scale

        def update(self, price, size, side):  # noqa: ANN001
            return price * self._scale

    definition = KernelDef(
        kernel_id="test_scaled_price",
        representation="trades",
        description="price * scale (test only)",
        params=(kernels_module.ParamDef("scale", "int", 1, 1, 1000),),
        min_warmup=lambda p: 1,
        factory=lambda p, bar_ns: _LastPrice(p["scale"]),
    )
    register_kernel(definition)
    yield definition
    kernels_module._REGISTRY.pop("test_scaled_price", None)


def test_new_kernels_can_be_registered_and_used_from_a_spec(custom_kernel: KernelDef) -> None:
    spec = parse_feature_spec_text(
        "schema: omega.feature-spec/v1\nname: scaled\nversion: 1\n"
        "inputs: {representation: trades}\nkernel: test_scaled_price\nparams: {scale: 10}\n"
    )
    assert spec.kernel == "test_scaled_price"
    assert spec.warmup == 1


@pytest.mark.parametrize("path", sorted(glob.glob(str(REPO_ROOT / "studies" / "specs" / "features" / "*.yaml"))))
def test_shipped_example_specs_are_valid(path: str) -> None:
    spec = load_feature_spec(path)
    assert Path(path).stem == spec.name


def test_shipped_example_specs_have_unique_names_and_identities() -> None:
    specs = [load_feature_spec(p) for p in glob.glob(str(REPO_ROOT / "studies" / "specs" / "features" / "*.yaml"))]
    assert len(specs) >= 5
    assert len({s.name for s in specs}) == len(specs)
    assert len({s.identity for s in specs}) == len(specs)
