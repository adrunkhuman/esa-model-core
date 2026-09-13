# esa-model-core

`esa-model-core` is a source-only reference implementation of a probabilistic Ekstraklasa forecasting model. It estimates an evolving joint distribution of club attack, club defence, and home advantage; converts that state into score probabilities; and simulates a remaining season.

## Mental model

Think of the code as four layers:

1. **State.** Each club has latent attack and defence ratings. Home advantage can also be latent. One Gaussian covariance matrix represents uncertainty and correlations across the active parameters.
2. **Match distribution.** The state determines two log scoring rates. A Dixon–Coles correction reshapes the four lowest score cells, and numerical integration carries rating uncertainty into result probabilities.
3. **Season path.** A static simulation holds one sampled state fixed. A chronological simulation instead advances one fixture date at a time, samples scores and synthetic xG, and learns from each completed date batch.
4. **Output contracts.** Simulated paths become expected points, finish distributions, point intervals, and optional immutable forecast payloads.

The code also contains season-transition functions, a bridge between Ekstraklasa and lower tiers including I liga, Polish Cup machinery, and narrowly scoped policies for temporary venues, three derbies, and late-season motivation.

!!! important
    This export supplies algorithms and type contracts, not an assembled forecast. It contains no datasets, fitted team states, fitted policy posterior, schedules, generated forecasts, or operational orchestration. Callers must construct and validate those inputs. See [Data and source boundary](data-and-source-boundary.md).

## What to read

- [Quickstart](quickstart.md) runs a complete synthetic static-posterior season without external data.
- [Statistical model](statistical-model.md) gives the scoring equations, uncertainty integration, and update mechanics.
- [Worked match](worked-match.md) calculates one fully synthetic forecast from ratings through uncertain result probabilities.
- [Priors and policies](priors-and-policies.md) explains the supplied transition and fixture-policy machinery.
- [Season simulation](simulation.md) distinguishes static and chronological paths.
- [Polish Cup model](polish-cup.md) explains knockout progression, lower-tier ratings, and the league boundary.
- [Scale and performance](scale-and-performance.md) documents API defaults, larger-application path counts, and a measured benchmark.
- [API guide](api.md) identifies practical entry points and calls out interfaces whose private defaults are absent.
- [Limitations and glossary](limitations-and-glossary.md) defines terms and interpretation limits.

## Intended use

This was designed as a personal, low-consequence forecasting model. Closing odds may be used by evaluation code as a benchmark, but they are not model inputs. Forecasts describe distributions, not certainties or betting advice. Reassess the assumptions and validation before using the implementation for material financial decisions, automation, or public claims.
