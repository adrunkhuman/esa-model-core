# Limitations and glossary

## Interpretation limits

- Football scores are sparse, and team strength changes for reasons the state model cannot directly observe. Intervals cover modeled uncertainty, not every real-world uncertainty.
- Score matrices stop at 10 goals per side and renormalize the retained cells. Result probabilities therefore use a finite-support approximation; expected goals are integrated from the rates separately and are not truncated at ten.
- Seven-point Gauss–Hermite quadrature is the default approximation to the Gaussian uncertainty integral.
- Monte Carlo outputs contain sampling error. Small development runs are not publication precision.
- Motivation represents modeled table-objective leverage, not player psychology. It activates only through remaining-match gates supplied by the loaded policy.
- Venue and three-derby adjustments are narrow selected policies, not universal causal estimates.
- Defaults for clubs promoted from I liga and for summer transitions require independent validation because their supporting datasets and selection history are not exported.
- Discipline and fair-play tie-break data are absent, so random noise can decide an otherwise unresolved final table tie.
- Cup projections require caller-supplied lower-tier ratings, bridge state, entrants, and bracket assumptions. The export does not provide a joint live calendar.
- Market odds are excluded from forecasts by design. This is not a market-consensus model and is not wagering advice.
- A synthetic or newly fitted output from this code is not a reproduction of the private operational forecast.

## Glossary

**attack rating**
: A latent additive effect on a club's own log scoring rate. Higher is stronger.

**defence rating**
: A latent quantity subtracted from the opponent's log scoring rate. Higher is stronger defence.

**home-field advantage (HFA)**
: An additive home log-rate effect. It may be fixed or dynamic and can be attenuated for a supplied relocated-venue period.

**joint covariance**
: One matrix describing uncertainty and correlation among all active attack, defence, and dynamic HFA parameters.

**Dixon–Coles correction**
: Multipliers applied to the four lowest cells of an independent Poisson score matrix to model low-score dependence.

**Gauss–Hermite quadrature**
: Deterministic weighted nodes used to integrate match probabilities over Gaussian log-rate uncertainty.

**causal date batch**
: All matches on one calendar date are predicted before any result from that date updates the state.

**xG (expected goals)**
: A continuous description of chance quality. Caller-supplied calibrated xG can be blended with goals; simulated future xG supports chronologically consistent path updates.

**club remaining in Ekstraklasa**
: A club that competes in Ekstraklasa in consecutive seasons. It can carry an Ekstraklasa state through the summer transition.

**club promoted from I liga**
: A club moving from I liga, Poland's second tier, into Ekstraklasa. It needs a new Ekstraklasa opening prior rather than a carried top-tier state.

**club relegated to I liga**
: A club moving out of Ekstraklasa into I liga after the previous season.

**season prior**
: The opening distribution for team strength. The code can incorporate previous state for clubs remaining in Ekstraklasa, caller covariates, retention, and policies for clubs promoted from I liga.

**retained-minutes share**
: The share of prior-season player minutes belonging to players retained at a chosen cutoff; one possible input to continuity and uncertainty for clubs promoted from I liga.

**settled score**
: A late-season probability-like measure of low table-objective leverage.

**urgency score**
: A late-season measure based on the largest modeled win-versus-loss probability swing for title, European-place, or relegation objectives.

**static posterior simulation**
: A season path in which one sampled latent team-strength state remains fixed through future fixtures.

**chronological simulation**
: A season path that advances dates, samples shared latent state, observes sampled goals and optional synthetic xG in batches, and updates ratings.

**prediction-only policy**
: A temporary fixture-rate adjustment that should be accounted for during observation but should not become persistent team strength.

**canonical club ID**
: An identity stable across season-local team IDs. The derby policy uses canonical pairs; the mapping dataset is not exported.

**finish distribution**
: The simulated probability of each team ending in each league place. Season APIs return these values as percentages.

**point interval**
: The simulator's 10th and 90th percentiles of final points, often described as an 80% model interval.

**LiveSnapshot**
: An immutable output dataclass with JSON serialization. This export supplies the contract but not the operational builder that populates it.

**source-only reference**
: Algorithms and interfaces without the datasets, fitted artifacts, orchestration, and lineage required to recreate the source application's forecasts.
