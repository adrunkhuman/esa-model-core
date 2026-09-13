# Data and source boundary

## What this repository is

This repository is generated from an explicit allowlist in a separate source tree. It contains computational modules plus hand-curated packaging, documentation, a synthetic example, and synthetic tests. Python imports remain under `esa_model`; the distribution is named `esa-model-core`.

It is a useful reference implementation, but not a complete export of the private operational application.

## Included

- Dixon–Coles baseline and sequential Gaussian team-strength updates.
- Joint-covariance attack, defence, and home-advantage state.
- Exact-score probabilities and Ekstraklasa table ranking.
- Static-posterior and chronological league simulation, including process workers.
- Motivation, attendance, relocation, derby, point-adjustment, season-transition, and mechanisms for clubs promoted from I liga.
- Causal xG calibration functions for caller-supplied observations.
- Lower-tier bridge and Polish Cup probability/simulation machinery.
- Immutable forecast contracts and probability-attribution helpers.
- Optional native motivation-kernel source.

Stage 2, Stage 3, xG, promotion-related, and Cup modules are included because their algorithms are informative. Their loaders require files that are not distributed.

## Excluded

- Match, score, xG, attendance, squad, venue, schedule, Cup, and identity-mapping datasets.
- Fitted team states, fitted coefficient posteriors, generated predictions, plots, diagnostics, and databases.
- Operational model assembly, provider refreshers, frontend storage, release identifiers, deployment code, benchmark logs, and research history.
- The original repository's Git history.

No exported function downloads data automatically. Some loaders and command-line `main` functions nevertheless retain paths expected by the larger application. A missing default file is a boundary signal, not an invitation to invent a replacement silently.

## Caller responsibilities

For a meaningful forecast, callers must provide and validate:

- **Stable team and canonical club IDs** for state continuity and derby matching.
- **Historical dated results** for causal fitting and state advancement.
- **Fitted global scoring parameters** covering the intercept, home advantage, Dixon–Coles correlation, and initial uncertainty.
- **Season priors and transition settings** for opening means, variances, persistence, and covariance.
- **A complete dated schedule** for chronological batching and table objectives.
- **An existing table or completed matches** when starting during a season.
- **An explicit relocation policy** to avoid the absent compatibility default.
- **Optional xG values and variances** for the observation blend.
- **An optional motivation posterior** for late-season effects.
- **Enough outer and inner simulation paths** for the required Monte Carlo precision.

Inputs at a prediction cutoff must contain only information available by that cutoff. Matches sharing a date should be predicted before any result from that date is observed. Opening roster features should share an explicit cutoff and complete identity coverage.

## Embedded assumptions and defaults

This is not a coefficient-free framework. Source modules retain numeric defaults for scoring support, quadrature, uncertainty candidates, state transitions, transforms for clubs promoted from I liga, lower-tier/Cup mechanics, relocation distance, motivation-policy fields, and the derby shift. Some values were selected in work whose datasets and experiment records are not public here.

[Scale and performance](scale-and-performance.md) also records selected operational path counts and a concise timing summary. Those values are published to explain realistic compute scale. They do not expose the underlying data copies, full benchmark logs, release lineage, or deployment procedure, and they do not make the omitted application assembly part of this export.

The three derby pairs are executable policy configuration and are named in [Priors and policies](priors-and-policies.md). The general identity map is absent.

Several functions intentionally retain Ekstraklasa conventions: season strings such as `2025/26`, three-point wins, an 18-team motivation context, and the modeled league tie-break order. Polish Cup code similarly retains tournament concepts from its source setting. Adapt these assumptions before using another competition.

## No operational reproduction claim

A caller can fit a new model with public or synthetic data, but that creates a new lineage. It does not reproduce the private forecast merely because it uses the same source modules. Equivalent reproduction would require independently sourced inputs, fitted states, selected policies, cutoff discipline, and assembly logic, none of which this repository certifies.

## Data safety

Treat data refresh and model fitting as separate from package exploration. The documented quickstart uses only generated fixtures. Do not add fitted artifacts or datasets to this repository merely to make a private-path loader succeed; pass explicit caller-owned inputs or build a separate data layer instead.
