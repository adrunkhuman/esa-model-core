# Worked synthetic match

This example follows one fictional match from ratings to a result forecast. Every input is invented; no real club, fitted state, or private file is used. Displayed results are rounded for readability. The calculation retains full precision throughout.

Run it with:

```console
uv run python examples/worked_match.py
```

## 1. Choose transparent inputs

Amber is at home to Blue. Use:

- a baseline scoring rate of \(1.25\) goals, giving intercept \(\alpha=\log(1.25)\approx0.223\);
- Amber attack \(a_H=0.20\) and defence \(d_H=0.05\);
- Blue attack \(a_A=0.05\) and defence \(d_A=0.10\);
- fixed home advantage \(h=0.15\); and
- Dixon–Coles low-score correction parameter \(\rho=-0.08\).

Attack, defence, and home advantage are contributions to the **log scoring rate**, not goals to add directly. The subscripts distinguish home and away.

For uncertainty, choose home log-rate variance \(v_H=0.04\), away log-rate variance \(v_A=0.0225\), and covariance \(c_{HA}=0.01\). These variances correspond to standard deviations of \(0.20\) and \(0.15\). The positive covariance means higher plausible home rates tend to accompany higher away rates.

The example supplies an empty relocation policy, uses fictional IDs outside the derby rules, and applies no motivation adjustment. We can therefore follow the core scoring calculation without additional policy shifts.

## 2. Calculate the log rates

The home mean log rate is

\[
\mu_H = \alpha + a_H - d_A + h
\]

\[
= \log(1.25) + 0.20 - 0.10 + 0.15
\approx 0.473.
\]

The away mean log rate is

\[
\mu_A = \alpha + a_A - d_H
\]

\[
= \log(1.25) + 0.05 - 0.05
\approx 0.223.
\]

Converting these log rates to goals gives

\[
\lambda_H = \exp(\mu_H) \approx 1.605,
\]

\[
\lambda_A = \exp(\mu_A) = 1.25.
\]

At these fixed rates, Amber would average about 1.61 goals and Blue 1.25 across many comparable matches. That does not yet tell us the chance of a particular score or result.

## 3. Build ordinary Poisson score cells

For a scoring rate \(\lambda\), the Poisson probability of exactly \(k\) goals is

\[
P(k\mid\lambda)
= \frac{e^{-\lambda}\lambda^k}{k!}.
\]

Before the Dixon–Coles correction, the two teams' probabilities are multiplied to get a score probability. For example, the uncorrected chance of 0–0 is

\[
P(0,0) = e^{-\lambda_H}e^{-\lambda_A}
\approx 0.0576,
\]

or about 5.76%.

## 4. Correct the four lowest scores

Dixon–Coles adjusts only 0–0, 0–1, 1–0, and 1–1. For 0–0, the multiplier is

\[
\tau_{00}=1-\lambda_H\lambda_A\rho
\approx 1.161.
\]

Multiplying the original probability by this factor gives

\[
P_{\mathrm{corrected}}(0,0)
= P(0,0)\,\tau_{00}
\approx 0.0668,
\]

or about 6.68%. Here \(\tau_{00}\) is the correction factor for the 0–0 cell. The other factors are \(1+\lambda_H\rho\) for 0–1, \(1+\lambda_A\rho\) for 1–0, and \(1-\rho\) for 1–1.

| Score | Before | After |
| --- | ---: | ---: |
| 0–0 | 5.76% | 6.68% |
| 0–1 | 7.19% | 6.27% |
| 1–0 | 9.24% | 8.31% |
| 1–1 | 11.55% | 12.47% |

With this negative value of \(\rho\), the correction increases the two low-score draws at the expense of the two one-goal wins.

The code builds a grid from 0 through 10 goals per side, then normalizes the retained cells to sum to one. The table shows probabilities before that final normalization; its effect here is smaller than the displayed precision.

## 5. Aggregate fixed-rate outcomes

Add every score cell in which Amber scores more than Blue to obtain the home-win probability. Equal scores contribute to the draw probability; the remaining cells give an away win.

At the fixed rates above:

- home win: 44.74%;
- draw: 26.41%; and
- away win: 28.85%.

These are sums over the score grid, not estimates from Monte Carlo paths.

## 6. Average over rating uncertainty

Now keep the same mean log rates but include the chosen variances and covariance. The model evaluates plausible pairs of rates, calculates score probabilities for each pair, and takes a weighted average. This numerical integration is called **Gauss–Hermite quadrature**.

The example uses `live_uncertain_probabilities` to perform the integration over \(7\times7\) pairs of log rates. There is no need to calculate every point and score cell by hand to understand the result:

| Outcome | Fixed rates | With uncertainty |
| --- | ---: | ---: |
| Home win | 44.74% | 44.96% |
| Draw | 26.41% | 26.09% |
| Away win | 28.85% | 28.95% |

Home expected goals rise from about 1.61 to 1.64, and away expected goals from 1.25 to 1.26.

## 7. Why symmetric uncertainty changes the averages

The uncertainty is symmetric around each **log rate**, not around each goal rate. Exponentiation gives a positive log-rate change a larger effect than an equally sized negative change. For example, around a log rate of zero:

\[
\exp(-0.20) \approx 0.819,
\qquad
\exp(0.20) \approx 1.221.
\]

Relative to a goal rate of 1, those changes are roughly −0.18 and +0.22—not equal opposites.

For a Gaussian log rate with mean \(\mu\) and variance \(v\), the average goal rate is

\[
\operatorname{E}[\lambda]=\exp(\mu+v/2).
\]

For Amber, this gives

\[
\exp\bigl(\log(1.25)+0.25+0.04/2\bigr)
\approx 1.637.
\]

For Blue, it gives

\[
\exp\bigl(\log(1.25)+0.0225/2\bigr)
\approx 1.264.
\]

These expected goals are calculated from the rates, not from the truncated score grid. Covariance does not change either team's individual average, but it does affect which pairs of rates are likely together—and therefore the result probabilities.

For this matchup, uncertainty raises both expected-goal values and slightly lowers the draw probability. That is not a universal rule about draws or favourites. The Poisson distribution maps each rate to non-negative, whole-number goals; averaging these distributions changes the chance of zero goals and high scores as well as the mean. The final effect depends on both teams' means, variances, covariance, and the low-score correction.

## What was manual and what was numerical?

The rating arithmetic, exponentials, individual Poisson cells, and correction factors can be reproduced with a calculator. The final uncertain result probabilities use deterministic numerical integration from the exported API. No season simulation or random sampling is involved.
