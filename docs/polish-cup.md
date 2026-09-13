# Polish Cup model

The export includes useful Polish Cup match and tournament mechanics. It does not include the entrants, completed bracket, draw dates, lower-tier match history, identity mapping, or fitted cross-tier state needed to recreate the private forecast.

## The model's knockout representation

`cup_side.TournamentRules` represents a staged single-elimination tournament. It contains opening pairings, round metadata, optional clubs entering before later rounds, hosting behavior, and a shootout probability.

Within each simulated tournament path:

1. use a known opening draw when it is marked fixed; otherwise shuffle the opening field;
2. play every pairing and retain one advancing club;
3. add any clubs scheduled to enter before the next round;
4. shuffle survivors and new entrants for an unknown next draw; and
5. continue until the round marked as the final produces one winner.

Validation requires all entrants to appear exactly once in the opening pairings or a later entry batch. Every stage must have an even field and the configured rounds must reduce it to one champion.

The projection layer can freeze completed winners and use a known draw only after its caller-supplied availability date. This avoids treating a future draw as if it had been known at an earlier forecast cutoff.

## Regulation time, extra time, and penalties

`cup_match_probabilities` first returns regulation-time home-win, draw, and away-win probabilities. It uses the same uncertain Dixon–Coles scoring machinery as league matches, with Cup-specific cross-tier terms.

`advancement_probabilities` then turns those three outcomes into two advancement probabilities:

- a regulation winner advances directly;
- after a regulation draw, extra time uses the same log-rate moments with both mean rates scaled to one third of 90-minute rates;
- an extra-time winner advances; and
- after an extra-time draw, the home side advances with `shootout_probability`, which defaults to 0.5.

The one-third scale is implemented by adding \(\log(1/3)\) to each mean log rate. Here \(\log\) is the natural logarithm; 1/3 represents 30 minutes relative to 90 minutes.

This is an approximation. Extra time is not conditioned on the exact score that produced the regulation draw. Penalties are represented by one supplied probability rather than kick-by-kick or team-specific shootout strength. The code calculates advancement probability; it does not simulate a detailed extra-time scoreline.

## Hosting and the final

The generic tournament simulator can place the lower-ranked tier at home when `lower_tier_hosts` is true and the clubs come from different tiers. Same-tier hosting is randomized when the pairing is not fixed. A round marked as the final is modeled as neutral.

These are explicit software rules, not a claim that they describe every historical Polish Cup tie. Callers must build `TournamentRules` that match the competition state they intend to model.

## Putting I liga and other tiers on one scale

Ekstraklasa is the top tier in this model and receives canonical tier rank 1. I liga—the Polish second tier—receives rank 2. The Cup code accepts ranks 1 through 5 so lower divisions can be represented too.

`CupTeamRating` supplies each club's attack and defence means and variances. A seven-parameter `BridgeState` places tier effects on the Ekstraklasa scoring scale and carries covariance between those effects.

The retained tier coefficient is:

- 0 for Ekstraklasa;
- 1 for I liga; and
- \(1+(r-2)s\) for lower tier rank \(r\), using step scale \(s\).

Here \(r\) is a canonical tier rank from 3 through 5 and \(s\) defaults to 0.75. Lower tiers also receive extra marginal variance

\[
((r-2)u)^2,
\]

where \(u\) is the extra-tier standard-deviation setting and defaults to 0.3. This expresses increasing uncertainty as the model extrapolates farther from the top-two-tier bridge. It is a selected approximation, not a fitted rating for an otherwise unknown club.

In the larger projection context, tier-rank-1 clubs read their current Ekstraklasa state. Tier-rank-2 clubs can read a separately replayed I liga state. A club below I liga has no supplied team-state mean in `_rating`; its differences come from bridge terms and extra uncertainty unless the caller constructs richer ratings directly.

The default `early_esa_penalty` hook exists for selected early-round adjustments but is zero in `ADOPTED_CUP_POLICY`.

## One coherent tournament path

`tournament_final_win_probabilities` samples each club's latent Cup rating and the cross-tier bridge once per tournament path. That sampled strength persists through all of the club's ties in that path. This preserves a simple form of tournament-wide uncertainty: a path where a club is stronger than its mean affects more than one round.

Unknown draws and match advancement are then sampled round by round. Repeating paths produces title probabilities, win counts, Monte Carlo standard errors, and Wilson 95% intervals.

## Relationship to league state

The exported standalone Cup projection reads a caller-constructed `CupProjectionContext` and supplied league state but does not update either. Future simulated Cup outcomes also do not update the cross-tier bridge. A supplied bridge may already reflect historical matches between Ekstraklasa and I liga clubs; preparing operational source data is outside the export.

The private application has a non-exported joint-calendar orchestrator. It interleaves league and Cup event dates inside each outer path. At a Cup date, an Ekstraklasa entrant reads that path's currently sampled league strength, so league form uncertainty and Cup prospects co-vary.

The information flow is deliberately one-way. A simulated Cup result advances the local bracket only. It does not update league attack or defence, the league table, motivation state, or the bridge. A separate Cup random stream also prevents unknown Cup draws from perturbing the league path's random sequence.

## European-place interaction

The private joint-calendar orchestrator combines each simulated Cup winner with that path's final league order. Its season-specific allocation code produces overall European qualification, league-route, Cup-route, reallocation, and named-competition probabilities.

That allocation is not part of `cup_side` or the exported chronological league simulator. The orchestrator is not exported, so this repository cannot reproduce those combined league–Cup outputs by itself. Callers should not infer a European allocation rule merely from standalone Cup title probabilities.

## What can run from this export?

With caller-created ratings, bridge moments, and tournament rules, these exported functions are directly usable:

- `cup_match_probabilities` for regulation outcomes;
- `advancement_probabilities` for one-tie advancement; and
- `tournament_final_win_probabilities` for a complete staged knockout simulation.

`cup_projection.project_cup` accepts a caller-constructed `CupProjectionContext` and compatible league model. The private JSON loaders, identity mapping, lower-tier replay, and `prepare_cup_projection_context` assembly are not exported. The safest source-only introduction is to construct `CupTeamRating`, `BridgeState`, and `TournamentRules` explicitly with synthetic inputs.

## Main approximations

- Goal probabilities use the same finite 0–10 score support and quadrature approximation as the league model.
- Extra time scales 90-minute rates by one third without conditioning on the regulation score.
- Shootouts use a fixed supplied probability.
- Unknown future draws are random shuffles, subject to staged entry and hosting rules.
- One latent Cup state persists through a path; simulated Cup results do not teach that state.
- Lower tiers beyond I liga are extrapolated through fixed coefficient and uncertainty steps.
- Input dates, entrants, tier ranks, and known draws are caller responsibilities.
