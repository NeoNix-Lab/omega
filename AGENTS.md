# AGENTS.md

Rules for any coding agent (and human) working in this repository. They come from epic #1
("Omega as first consumer of the quant-platform core"); keep both in sync.

## What Omega is

Alpha research lab built on top of `quant-platform`. Omega authors feature/strategy specs and
orchestrates experiments; the platform provides data access, deterministic replay and statistical
validation. Omega consumes the platform **in-process (Python import)** only.

## Alpha-integrity guardrails

- G1. Nothing computed on synthetic data may be labelled "alpha". Synthetic data is only a test
  fixture and must be marked as such in outputs.
- G2. A candidate is "promotable" only if: DSR is evaluable with `k_eff` derived from the real trial
  count in the registry (abandoned trials included), PBO is evaluable and below the configured
  threshold, costs (fees + slippage) are on, it was confirmed at tick level, and the lockbox was
  consumed once.
- G3. Features must be causal: every feature value carries `available_at <= decision time`.
- G4. Every tracked run stores platform identities (dataset, replay spec, strategy, trial
  population), not only metrics.
- G5. Money and quantity are `Decimal` on the platform side; no float accounting in Omega paths
  that feed the ledger or replay.
- G6. Bar-level (screening) results are never reported as final performance.

## Agent contract

- One issue = one PR (draft). Stage explicit paths only (never `git add -A` / `git add .`).
- Never modify `quant-platform`. If the platform blocks you, stop and comment on issue #15
  (upstream asks) with the exact finding (file/line, command, error).
- No bulk data, models or secrets in git (`data/`, `results/` stay ignored).
- Tests are part of the issue: `pytest` must pass and new behaviour needs tests. Verify
  determinism by running twice and comparing identities/hashes.
- Before pushing run `ruff check .` and `pytest -q`; quote the output in the PR description.
- All `quant_platform` imports go through `omega.platform_link` (once it exists).
- Do not widen scope. Out-of-scope ideas go in an issue comment.
- Definition of Done = every acceptance criterion checked with evidence, docs updated where listed.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[platform,dev]"
omega status
```

Entry points are the console script `omega` and `python -m omega` only (there is no root `omega.py`:
it would shadow the `omega` package).
