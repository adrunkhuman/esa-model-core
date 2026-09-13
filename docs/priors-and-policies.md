# Season priors and fixture policies

This export contains mechanisms for constructing priors and applying prediction policies. It does **not** contain the fitted priors, input snapshots, or research artifacts used by the larger private application. Numeric defaults retained in source should be treated as selected policy values, not universal football constants.

## Clubs remaining in Ekstraklasa

A club that stays in Ekstraklasa from one season to the next can carry its previous state across the summer. In this documentation, “remaining” distinguishes these clubs from clubs promoted into Ekstraklasa from I liga, Poland's second tier. `JointCovarianceDCModel` supports `reset`, `budgeted`, `full`, and `diagonal` prior-transition modes, plus season- and team-specific persistence. The original adopted stack used diagonal carry: attack and defence persist separately rather than carrying their off-diagonal covariance unchanged.

The supplied Stage 2 and continuity functions can fit or apply opening-mean and persistence adjustments from caller data such as squad value and retained minutes. These are opening priors, not match-day covariates. Inputs should use one explicit cutoff and complete identity coverage; otherwise an apparently causal prior may contain future roster information.

## Clubs promoted from I liga

Clubs promoted from I liga enter Ekstraklasa without a previous Ekstraklasa state for the new season. Clubs moving the other way are relegated from Ekstraklasa to I liga. The source includes broad opening priors for these promoted clubs and functions for:

- adjusting a promoted club's means from squad-value parity with clubs remaining in Ekstraklasa;
- increasing marginal uncertainty as retained prior-season minutes fall;
- blending toward a caller-selected promoted baseline; and
- bridging lower-tier and Ekstraklasa strength.

One retained turnover transform has the form

\[
v(r)=1+2(1-r),
\]

where \(r\) is the retained-minutes share, from zero to one, and \(v(r)\) is the resulting multiplier on marginal attack and defence variance. The constants 1 and 2 set the retained baseline and the rate at which uncertainty rises. The code also contains selected promoted-squad and uncertainty defaults. Their supporting datasets and model-selection history are not exported, so users should validate or replace them rather than present them as newly established evidence.

## Relocated home matches

`RelocatedHomeHfaPolicy` accepts caller-supplied periods in which a nominal home club plays away from its ordinary ground. A period includes a stable episode ID, home-team ID, inclusive dates, classification, distance, a resilient-exclusion flag, and explicit match exclusions.

Distances at or below 20 km receive no adjustment. Above that threshold, the removed fraction of home advantage is

\[
q(D)=1-2^{-(D-20)/20}.
\]

Here \(D\) is venue distance in kilometres and \(q(D)\) is the fraction of home advantage removed. The 20 km terms set both the no-adjustment threshold and the distance over which the remaining advantage halves. The remaining home-advantage scale is \(1-q(D)\). `uncertain` periods, resilient exclusions, nearby venues, explicit exclusions, and unmatched fixtures are unchanged. This is a prediction-only effect.

No period registry is exported. Construct a policy directly:

```python
from esa_model.relocated_home_policy import RelocatedHomeHfaPolicy

no_relocations = RelocatedHomeHfaPolicy(())
```

Pass it to either season simulator. Omitting it invokes a compatibility default that expects a non-exported CSV.

## Three low-scoring derbies

The embedded derby policy applies to:

- Legia Warszawa–Lech Poznań;
- Wisła Kraków–Cracovia; and
- Górnik Zabrze–Piast Gliwice.

It adds `-0.20` to **both** log scoring rates, a multiplicative rate factor of

\[
e^{-0.2}\approx0.819.
\]

Here \(e\) is the base of the natural logarithm, so the shift leaves about 81.9% of each original scoring rate. Canonical club IDs make the rule independent of season-local IDs when a mapping is supplied. The policy changes predictions and simulated observations; it does not rewrite historical scores. This is a narrow selected rule, not a general causal estimate for all derbies.

## Late-season motivation

The motivation machinery produces two prediction-only features from a current table and remaining schedule:

- **settled score:** low objective-zone leverage; and
- **urgency score:** the largest conditional win-versus-loss change in title, European-place, or survival probability.

Both lie in `[0, 1]` and should be computed from an unadjusted parent forecast so the policy does not feed back into its own feature. The remaining-match gates and score scales are fields in the caller-loaded `MotivationPolicy`; they are not hardcoded by the scoring machinery. This export does not publish a fitted policy snapshot.

For settled contrast \(s_A-s_H\), urgency contrast \(u_H-u_A\), and coefficient draw \((\beta,\gamma)\), the home log-rate shift is

\[
\Delta=\beta(s_A-s_H)+\gamma(u_H-u_A),
\]

Here \(s_H\) and \(s_A\) are the home and away settled scores; \(u_H\) and \(u_A\) are their urgency scores; \(H\) and \(A\) label home and away; \(\beta\) and \(\gamma\) are policy coefficients; and \(\Delta\) is the home log-rate adjustment. The away adjustment is \(-\Delta\). `MotivationPolicy` stores coefficient grids and weights supplied by the caller. Its marginal adjustment propagates coefficient uncertainty into both rate variances and their covariance. Chronological simulation can sample coefficients and recompute table-dependent scores inside each path.

The optional native extension accelerates conditioned zone-count simulation. The export ships neither fitted motivation coefficients nor precomputed feature snapshots.

!!! warning
    `live_forecast.operational_forecast` layers relocation and derby rules but does not calculate motivation. It also uses the non-exported adopted relocation registry. The lower-level simulator accepts an explicit relocation policy and is the safer source-only boundary.

## Point adjustments

Announcement-dated point adjustments alter a simulated table only from their effective date. They do not change latent attack or defence. This preserves the information set available to earlier fixture forecasts.
