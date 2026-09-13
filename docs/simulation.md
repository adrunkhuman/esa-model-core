# Season simulation

## Two season engines

`posterior_simulation` and `chronological_simulation` answer different questions.

**Static posterior.** `simulate_seasons_from_posterior` samples one coherent strength state for each season path and holds it fixed. This is useful for synthetic examples, comparisons, and lower-level analysis.

**Chronological.** `simulate_seasons_chronologically` advances and updates strength after each fixture date. Use it when simulated future results should change later strength estimates.

Do not describe a static-posterior output as a chronological forecast. See [Scale and performance](scale-and-performance.md) for API defaults, operational path counts, and timing evidence.

## One chronological path

For each future date, the chronological engine:

1. copies and advances the model's uncertainty;
2. samples one coherent latent league state;
3. computes motivation from that path's table and remaining calendar when a policy is supplied and its loaded gates apply;
4. applies relocation, derby, and motivation offsets;
5. samples every score on the date from the shared state;
6. samples synthetic Gamma xG around the scoring rates, including caller-configured future observation variance;
7. updates the copied model with the complete date batch; and
8. updates the table before proceeding to the next date.

A path in which a club is genuinely stronger can therefore affect all later fixtures, rather than producing unrelated match-by-match perturbations. The source model is copied, so simulation paths do not mutate it or one another.

Fixtures are sorted by date and match ID. All matches on a date use the same pre-date state. Teams may not appear twice in one observed batch.

## Outputs and reproducibility

Both season APIs return:

1. expected final points, one value per team;
2. a team-by-place matrix of finish **percentages**; and
3. 10th/90th percentile final-point intervals.

Expected points can be fractional because they are means across paths. With one simulation, place values are still encoded as `0` or `100`.

Chronological multi-path seeds follow `seed + path * 1_000_003` and remain stable across worker counts. A stable seed supports debugging and comparison; it does not establish adequate Monte Carlo precision. Small path counts are smoke tests only.

## Table ranking

The supplied Ekstraklasa ranking sequence uses:

1. points;
2. head-to-head points;
3. head-to-head goal difference;
4. overall goal difference;
5. goals scored;
6. wins; and
7. away wins.

A random draw resolves a remaining tie because discipline and fair-play data are outside the model. Adapt `league_table` before applying the package to competitions with different rules.

## Starting partway through a season

The static simulator accepts a `starting_table`. Chronological simulation instead accepts completed matches and reconstructs the table used by each path. Remaining and completed match IDs must not overlap, and all matches must belong to the requested season and team set.

Known point adjustments are filtered by season and applied causally. Callers are responsible for supplying a complete, internally consistent schedule and observation cutoff.

## Polish Cup machinery

The export can calculate regulation and advancement probabilities and simulate a caller-defined staged knockout tournament. It models extra time, a supplied shootout probability, lower-tier strength, unknown draws, and persistent uncertainty through a Cup path.

See [Polish Cup model](polish-cup.md) for the full mechanics and limitations. The private joint league–Cup calendar is not exported. In that larger orchestrator, an Ekstraklasa entrant reads its current league-path strength, but a simulated Cup result advances only the bracket and never updates league ratings, table state, motivation, or the cross-tier bridge.

## Monte Carlo interpretation

Simulation error is additional to model uncertainty. Increase paths until relevant reported probabilities are stable enough for the intended decision. Report outer paths and, when motivation is enabled, inner paths separately. The resulting interval covers variation represented by the model and simulator; it does not include every source of football uncertainty.
