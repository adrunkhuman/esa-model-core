# Statistical scoring model

## Start with the football idea

The model gives each club two hidden strengths: attack and defence. “Hidden” means they are not observed directly; the model estimates them from results and, when supplied, expected-goals observations. It also estimates a league scoring level and home advantage.

For one fixture, those quantities produce an expected scoring rate for each side. A rate of 1.4 means 1.4 goals on average across many comparable matches, not that the match will literally contain 1.4 goals. The rates feed a score distribution, which produces probabilities for results such as 0–0, 2–1, or 1–3.

## Scoring-rate equations

For a match with club \(i\) at home and club \(j\) away, the core rates are

\[
\log \lambda_{ij}^{H}
= \alpha + a_i - d_j + h,
\]

\[
\log \lambda_{ij}^{A}
= \alpha + a_j - d_i.
\]

Here \(\lambda^H\) and \(\lambda^A\) are the home and away scoring rates. The ratings combine four model quantities:

- \(\alpha\): the league-wide baseline log scoring rate;
- \(a_i\), \(a_j\): the two clubs' attack ratings;
- \(d_i\), \(d_j\): their defence ratings; and
- \(h\): the home-advantage contribution to the home log rate.

Taking the exponential converts a log rate back to goals per match. A larger attack rating raises that club's rate. A larger defence rating lowers its opponent's rate because defence is subtracted. Crowd effects and prediction-only policies can add temporary offsets to these log rates.

## A distribution, not one best rating

Suppose the model estimates Amber's attack rating as \(0.20\). That is a rating on the log scoring scale—not an average of 0.20 goals per match. Nor does the model claim that the rating is known exactly: \(0.10\) or \(0.30\) might also be plausible given the available evidence.

That evidence consists of **observations**: goals scored in completed matches and, when supplied, calibrated expected-goals (xG) values. The fixture records tell the model who played whom, when, and at which side's home ground. Results do not directly reveal attack or defence strength; the model infers those hidden ratings from them, starting from its season-opening assumptions and allowing strength to change over time.

### Central estimates: one number per rating

The model maintains a probability distribution over plausible ratings. A rating's **mean** is its probability-weighted average within that distribution—not an average of observed goals or a historical average of past ratings.

Imagine looking at just Amber's attack and Blue's defence. Their current estimated means might be

\[
m =
\begin{bmatrix}
0.20 \\
0.10
\end{bmatrix}.
\]

This is a **vector**, a list with one entry per rating: Amber attack first, Blue defence second. The full model has entries for every active club's attack and defence, plus home advantage when it is estimated as part of the changing state.

### Uncertainty: how much each estimate can vary

The means alone cannot distinguish a well-established rating from one based on little evidence. For that, the model also stores a **covariance matrix**. For our two-rating illustration, it might be

\[
C =
\begin{bmatrix}
0.0400 & 0.0100 \\
0.0100 & 0.0225
\end{bmatrix}.
\]

The row and column order matches the mean vector. The diagonal entries describe each rating's own uncertainty:

- Amber attack has variance \(0.04\), hence standard deviation \(0.20\).
- Blue defence has variance \(0.0225\), hence standard deviation \(0.15\).

Under the model's bell-shaped, or Gaussian, distribution, roughly 68% of the plausible Amber attack values lie within one standard deviation of its mean: from \(0.00\) to \(0.40\). These are possible **rating values**, not a range of match scores.

### Connections: which ratings can vary together

The off-diagonal value, \(0.01\), is the covariance between these two ratings. Its positive sign says that a higher plausible Amber attack tends to accompany a higher plausible Blue defence.

Why might that happen? The match scoring equation contains attack **minus** opposing defence. Attack \(0.20\) against defence \(0.10\) gives the same difference as attack \(0.30\) against defence \(0.20\). Evidence from their meeting cannot fully distinguish those explanations. Other matches help, but the remaining uncertainty can still connect the two estimates.

That connection matters for forecasting. For this illustration, the variance of attack minus defence is

\[
0.04 + 0.0225 - 2(0.01) = 0.0425.
\]

Ignoring the connection would give \(0.0625\), overstating uncertainty in this particular matchup. A negative covariance would have the opposite effect. This is why the model keeps a joint matrix rather than just a separate uncertainty value for each club.

### The compact notation

`JointCovarianceDCModel` represents all of this as

\[
\theta \mid \mathcal D_t
\sim \mathcal N(m_t, C_t).
\]

Here \(\theta\) collects the unknown ratings. After using the observations \(\mathcal D_t\) available by prediction time \(t\), the model approximates their joint distribution as Gaussian, with mean vector \(m_t\) and covariance matrix \(C_t\). The two-rating numbers above are invented to illustrate that structure; they are not a fitted model or the full league state.

## From rating uncertainty to rate uncertainty

For the home scoring rate, we need the home club's attack, minus the away club's defence, plus home advantage. A **design vector** \(x\) expresses that selection: \(+1\) for a term to add, \(-1\) for a term to subtract, and zero for unused ratings.

When home advantage is part of the uncertain state, the mean and variance of the log scoring rate are

\[
\operatorname{E}[\log \lambda]
= \alpha + x^T m_t,
\]

\[
\operatorname{Var}[\log \lambda]
= x^T C_t x.
\]

Here \(\lambda\) is the scoring rate for whichever side we are calculating. The first equation combines the relevant mean ratings; the second combines their uncertainty, including correlations rather than just adding individual variances.

If home advantage is fixed instead of estimated within the state, add \(h\) separately to the home log-rate mean. Fixed crowd or policy offsets are separate too.

Let \(x_H\) be the home-rate design vector and \(x_A\) the away-rate design vector. Their log-rate covariance is

\[
\operatorname{Cov}
(\log\lambda^H,\log\lambda^A)
= x_H^T C_t x_A.
\]

A positive covariance means higher plausible home rates tend to accompany higher away rates; a negative covariance means they tend to move in opposite directions. This matters for the resulting score and draw probabilities.

## Symmetric log uncertainty, skewed goal rates

A Gaussian distribution is symmetric in log-rate space, but the exponential transformation to goals is not. If a log rate has mean \(\mu\) and standard deviation \(\sigma\), equally likely one-standard-deviation points become

\[
\exp(\mu-\sigma)
\quad\text{and}\quad
\exp(\mu+\sigma).
\]

Holding \(\mu\) fixed at zero and setting \(\sigma=0.20\) gives rates 0.8187 and 1.2214. Their deviations from \(\exp(0)=1\) are −0.1813 and +0.2214: symmetric log movements become asymmetric rate movements.

For an uncapped log-normal rate, its mean is

\[
\operatorname{E}[\lambda]
= \exp(\mu + \sigma^2/2).
\]

Here \(\sigma^2\) is the log-rate variance. The formula assumes that \(\mu\) stays fixed while variance changes. With \(\mu=0\) and \(\sigma=0.20\), the mean rate is 1.0202 rather than 1.

This skew comes from exponentiating the symmetric Gaussian. The Poisson distribution then maps each non-negative rate to discrete goal counts, and mixing over uncertain rates changes both zero probability and tail probability. Home/away log-rate covariance changes which pairs of rates are likely together. Consequently, there is no universal rule that more uncertainty must help the favourite, hurt the favourite, or reduce draws; the full matchup and covariance determine the result.

The [worked match](worked-match.md) compares one identical set of log-rate means with and without uncertainty.

## Turning rates into score probabilities

At fixed rates, the starting assumption is that each side's goals follow a Poisson distribution. A Poisson distribution converts a non-negative expected rate into probabilities for 0, 1, 2, and more goals.

Football scores contain low-score dependence that two independent Poisson distributions do not fully capture. The Dixon–Coles correction therefore changes four cells—0–0, 0–1, 1–0, and 1–1—using the parameter \(\rho\), which controls the direction and size of that low-score correction. Other score cells keep their independent-Poisson values.

The code retains score cells from 0 through 10 goals per side. It discards mass outside that grid and renormalizes the retained cells so their sum is one. Renormalization conditions the displayed score distribution on the retained grid; it does **not** move omitted tail probability into the ten-goal cells.

## Expected goals are not truncated at ten

The result probabilities come from the normalized 0–10 score matrix. Expected home and away goals are calculated separately from the uncertainty-integrated scoring rates. They are not averages of that truncated matrix. This distinction prevents the display grid from artificially capping expected goals.

## Integrating over uncertainty

A shortcut would plug the mean ratings into the scoring equations. Because exponentials and score probabilities are nonlinear, that is not generally the same as averaging over uncertain ratings.

The implementation uses Gauss–Hermite quadrature. This is deterministic numerical integration: it evaluates the scoring model at a small set of carefully chosen points in the Gaussian distribution, weights the answers, and adds them. The default uses seven points for each log-rate dimension and includes home/away covariance.

The common probability return order is:

```text
home win, draw, away win, expected home goals, expected away goals
```

The first three values are fractions between zero and one.

## Learning in causal date batches

Uncertainty grows as calendar time passes without observations. `annual_drift_variance` and `home_advantage_drift_variance` are rates of variance growth, not standard deviations. Summer transition uncertainty and season-opening priors are separate additions.

Matches on the same date form one atomic batch:

1. advance the state to that date;
2. predict every match from the same pre-date state;
3. collect goals and optional calibrated xG observations; and
4. update the joint Gaussian state once with the full batch.

This prevents file order from letting an earlier-listed result leak into another match's pre-match forecast. Known venue, derby, or motivation offsets can also be supplied during observation so temporary effects are not learned as permanent team strength.

## Goals and expected-goals observations

The model can blend goals with caller-supplied xG through `xg_weight`. Callers also provide source-calibration and measurement variances. Causal calibration means that a prediction at time \(t\) uses only calibration evidence already available in \(\mathcal D_t\).

The export contains calibration functions but no xG dataset, fitted transform, selected operational weight, or provider bridge. Closing odds do not enter the state update.

## Keeping ratings on a consistent scale

Only differences such as attack minus opposing defence enter the scoring equations. Without a convention, adding the same constant to every attack and defence rating would leave every rate unchanged, so infinitely many coordinate values would describe the same model.

A centering convention, called a **gauge**, fixes that ambiguity. The implementation computes the mean active-team attack value, then subtracts that same shift from every active attack **and** defence coordinate. Each difference \(a_i-d_j\) is preserved. It applies the equivalent linear projection to the covariance matrix, so means and uncertainty remain consistent. Display indexes centered around a chosen value are a separate presentation transform.

## Persistent state versus fixture policy

The statistical state estimates persistent scoring strength. Relocation, derby, and motivation rules are temporary fixture-level adjustments layered onto rates. Keeping these layers separate means that a temporary venue or table situation need not permanently weaken or strengthen a club's latent rating.
