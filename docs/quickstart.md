# Quickstart

## Requirements

The package requires Python 3.14 and uses [uv](https://docs.astral.sh/uv/). From the repository root:

```console
uv sync
uv run pytest
uv run python examples/synthetic_season.py
```

The optional `esa_model._motivation_c` extension accelerates conditioned motivation simulations. Installation falls back to Python when a C compiler is unavailable.

## A safe first run

`examples/synthetic_season.py` is the supported source-only introduction. It performs no network access and reads no data files. The example:

1. defines four fictional teams and a dated synthetic schedule;
2. implements the small model protocol needed by the static simulator;
3. samples one coherent four-team latent state for each season path;
4. explicitly supplies an empty relocation policy; and
5. prints expected points, 80% point intervals, and first-place percentages.

Run it with:

```console
uv run python examples/synthetic_season.py
```

The important boundary is explicit in the call:

```python
from esa_model.posterior_simulation import (
    simulate_seasons_from_posterior,
)
from esa_model.relocated_home_policy import RelocatedHomeHfaPolicy

expected, places, intervals = simulate_seasons_from_posterior(
    model,
    team_ids,
    fixtures,
    simulations=1_000,
    seed=42,
    relocation_policy=RelocatedHomeHfaPolicy(()),
)
```

`expected` has one expected final-points value per team. `places[team, place]` is a **percentage**, not a fraction. `intervals` contains the 10th and 90th percentiles of final points.

For a single fixture rather than a season, [Worked match](worked-match.md) explains and reproduces every synthetic input and displayed number.

## Why the empty policy matters

The simulators retain compatibility with the larger application: if `relocation_policy` is omitted, they try to load an adopted period registry from `data/relocated_home_periods.csv`. That private input is intentionally not exported. Public and synthetic callers should pass either their own checked `RelocatedHomeHfaPolicy` or `RelocatedHomeHfaPolicy(())`.

## Moving to a real model

A real run requires more than replacing the team names. The caller must supply:

- fitted or justified `FrozenGlobals` values;
- a populated `JointCovarianceDCModel` or an object implementing the static simulator protocol;
- stable team IDs and dated `Match` objects;
- an existing table when simulating a season in progress;
- explicit relocation periods and, if wanted, motivation coefficients and point adjustments; and
- enough Monte Carlo paths for the required precision.

The export has loaders and fitting utilities for some of these tasks, but it does not provide the source files or one command that assembles an operational model. Do not infer operational reproduction from a successful synthetic run.

Very small path counts are useful for wiring only. A fixed seed makes a given configuration reproducible; it does not remove Monte Carlo error. See [Scale and performance](scale-and-performance.md) before comparing the example count, core API defaults, and the larger application's production settings.

## Build this documentation

Documentation dependencies are isolated from model and development dependencies:

```console
uv sync --locked --only-group docs --no-install-project
uv run --no-sync mkdocs serve
```

For the same strict build used in CI:

```console
uv run --no-sync mkdocs build --strict
```
