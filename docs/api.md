# API guide

## Import boundary

`esa_model.__init__` intentionally exports nothing. Import from the module that owns a type or function. The project does not promise a separately versioned stable public API; these are the practical source boundaries in the current export.

The export has no operational snapshot builder. `LiveSnapshot` is an output contract only. Modules such as `live_snapshot`, `live_inputs`, frontend history storage, and a joint league/Cup calendar orchestrator are not present.

## Core data types

### `baseline.Match`

```python
Match(match_id, date, season, home, away, home_goals, away_goals)
```

A dated result or fixture record. Simulators use the same type for future fixtures, commonly with placeholder goal values until a score is sampled. Team IDs must match the simulator's `team_ids` exactly.

### `stage1.FrozenGlobals`

Holds the scoring intercept, home advantage, Dixon–Coles `rho`, initial attack/defence uncertainty, and optional dynamic scoring/home-advantage covariance terms. The export does not supply fitted values.

### `stage1.TeamState`

Stores attack and defence means and marginal variances. `JointCovarianceDCModel` also owns a full covariance matrix; changing only a detached `TeamState` is not a substitute for a coherent joint update.

### `stage1.GoalRateForecast`

Contains home and away log-rate means, their variances, and their covariance. Variances are in log-rate space.

## Match probability functions

### `stage1.dixon_coles_score_matrix`

```python
dixon_coles_score_matrix(
    home_rate,
    away_rate,
    rho,
) -> numpy.ndarray
```

Returns the normalized fixed-rate score matrix over 0–10 goals per side.

### `stage1.uncertain_probabilities`

```python
uncertain_probabilities(
    home_log_rate_mean,
    home_log_rate_variance,
    away_log_rate_mean,
    away_log_rate_variance,
    rho,
    quadrature_points=7,
    home_away_log_rate_covariance=0.0,
) -> tuple[float, float, float, float, float]
```

Integrates Gaussian log-rate uncertainty. Return order is home, draw, away, expected home goals, expected away goals.

### `live_probability.live_uncertain_score_matrix`

The live numerical implementation returns a normalized 0–10 matrix followed by expected home and away goals. Those expected-goal values are integrated directly from the rates, not calculated from the truncated matrix. `live_uncertain_probabilities` returns the same five-value order as above; `live_uncertain_probabilities_batch` handles arrays. These are data-free numerical functions.

## Joint state model

### `joint_covariance.JointCovarianceDCModel`

Construct with caller-selected drift and summer variances plus `FrozenGlobals`. Optional arguments configure canonical identities, season priors and covariance, dynamic home advantage, crowd effects, xG observations and variances, mean/variance adjustments, and transition persistence.

Important methods include:

**State progression.** `advance_to(date, season, active_teams)` advances time and initializes or transitions active state.

**Rates and predictions.** `goal_rates(match)` returns uncertain core rates. `goal_rates_with_home_advantage_scale(match, scale)` recomputes them with scaled home advantage. `predict(match)` returns expected goals and result probabilities without fixture policies.

**Observation.** `observe(matches)` handles one score batch. `observe_with_xg(...)` handles a completely covered batch with transient xG. `observe_with_log_rate_offsets(...)` removes known temporary effects during updating.

**Sampling and prior changes.** `sample_parameter_state(generator)` draws one coherent state. `adjust_team_means(...)` and `add_team_variance(...)` apply explicit prior changes.

`observe_with_xg` requires xG and variance keys to exactly cover the batch. `observe_with_log_rate_offsets` rejects unknown match IDs and a team appearing twice on the same date.

## Season simulation

### `posterior_simulation.simulate_seasons_from_posterior`

```python
simulate_seasons_from_posterior(
    model,
    team_ids,
    fixtures,
    starting_table=None,
    simulations=50_000,
    seed=20260716,
    relocation_policy=None,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
```

Samples one latent state per path and holds it fixed. Its actual default is 50,000 paths, inherited from `league_table.SIMULATIONS`; vectorized exact-covariance work is batched in groups of at most 5,000. Returns expected points, finish-position percentages, and 10th/90th point intervals. See [Scale and performance](scale-and-performance.md).

For standalone use, pass `RelocatedHomeHfaPolicy(())` or a caller-built policy. `None` asks for a private compatibility CSV that is not included.

A custom model can support this function by providing `globals`, `sample_parameter_state`, `home_advantage_from_state`, and `goal_rates_from_state`; see `examples/synthetic_season.py`.

### `chronological_simulation.simulate_seasons_chronologically`

```python
simulate_seasons_chronologically(
    model,
    team_ids,
    fixtures,
    season,
    policy=None,
    point_adjustments=None,
    completed_matches=(),
    simulations=1,
    seed=20260716,
    motivation_simulations=500,
    future_xg_variance=(0.0, 0.0),
    workers=1,
    relocation_policy=None,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
```

Requires a populated `JointCovarianceDCModel` and non-empty dated remaining fixtures. The core default is one outer path, 500 inner motivation paths, and one worker. Each outer path copies the model and updates it after each fixture date. The return units match the static simulator. Again, pass relocation policy explicitly.

## Fixture policies

### `relocated_home_policy.RelocatedHomeHfaPolicy`

`RelocatedHomeHfaPolicy(periods)` owns immutable `RelocatedHomePeriod` records. `adjustment` returns a home-log-rate offset and `adjust_rates` applies it. `distance_dose` is usable independently.

`load_relocated_home_policy(path)` is safe when given a caller-owned checked CSV. `adopted_relocated_home_policy()` and defaults that call it expect absent private data.

### `derby_scoring_policy.adjust_derby_scoring_rates`

Applies the embedded three-pair policy to a `Match` and `GoalRateForecast`. Supply canonical IDs through the model when season-local IDs differ.

### `motivation_policy.MotivationPolicy`

Stores coefficient grids and normalized weights. `MotivationScores` contains home/away settled and urgency features. The loader requires a caller-owned policy file; no fitted posterior is included.

### `live_forecast`

`operational_goal_rates`, `operational_forecast`, and `match_probability_breakdown` reproduce a larger application's policy ordering and attribution. They are **not** clean source-only entry points because relocation defaults resolve through the absent adopted CSV and there is no policy argument to override it. They also do not compute motivation scores. Prefer core model methods and explicit policy functions unless you intentionally supply the compatible file.

`match_probability_breakdown` returns final home/draw/away fractions and an additive attribution around a neutral certain-rate base. It uses Shapley-style contributions for home advantage, uncertainty, and attack/defence matchups, followed by venue, derby, and motivation deltas.

## Output contracts

`live_contracts` contains frozen, slotted dataclasses shared by forecast, storage, and presentation layers in the larger system.

**Identity and ratings.** `SeasonTeam` contains local and canonical identity plus presentation metadata. `TeamRating` contains rating indexes, points, finish distributions, and optional qualification fields.

**Matches and attribution.** `PanelMatch` represents a completed or future match with nullable outcome probabilities. `ProbabilityBreakdown` contains base probabilities and additive labeled adjustments.

**Aggregate payloads.** `CupProjection` contains named entrant fractions plus an “Others” fraction. `LiveSnapshot` is the immutable top-level payload and provides `to_json()` serialization.

`CupProjection` validates unique entrants and requires named probabilities plus “Others” to total one within `1e-9`.

Probability units are deliberately mixed at this boundary: `TeamRating.place_probabilities` contains percentages, while match, Cup, and named qualification probabilities are fractions in `[0, 1]`. Point interval fields represent the 10th and 90th percentiles.

Constructing these dataclasses does not calculate a forecast. The assembly layer that populates a complete `LiveSnapshot` is excluded.

## Fitting and data utilities

- `baseline` fits and evaluates a fixed-strength Dixon–Coles model. Its odds helpers are evaluation utilities, not forecast inputs.
- `stage1` contains sequential marginal-state fitting and evaluation.
- `stage2` fits covariate priors from caller data.
- `stage3` constructs cross-tier bridge histories and priors for clubs promoted from I liga.
- `xg_model` loads source-linked xG and fits causal source-to-goal calibration.
- `continuity_transition_policy` handles continuity for clubs remaining in Ekstraklasa. `promoted_squad_policy` and `promoted_uncertainty_policy` handle clubs promoted from I liga; their module names remain unchanged.
- `cup_side` and `cup_projection` provide the calculations described in the [Polish Cup model](polish-cup.md).

Several `main()` functions and default paths belong to the original research workflow. They cannot run from this export without datasets and are not supported as a one-command public pipeline.
