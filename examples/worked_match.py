"""Print the synthetic numbers used by the worked-match documentation."""

# ruff: noqa: I001 -- imports resolve as third-party only after export.

import math
from dataclasses import dataclass
from datetime import date

import numpy as np

from esa_model.baseline import Match
from esa_model.live_probability import live_uncertain_probabilities
from esa_model.relocated_home_policy import RelocatedHomeHfaPolicy
from esa_model.stage1 import GoalRateForecast, dixon_coles_score_matrix

SCORING_INTERCEPT = math.log(1.25)
HOME_ATTACK = 0.20
HOME_DEFENSE = 0.05
AWAY_ATTACK = 0.05
AWAY_DEFENSE = 0.10
HOME_ADVANTAGE = 0.15
RHO = -0.08
HOME_VARIANCE = 0.04
AWAY_VARIANCE = 0.0225
RATE_COVARIANCE = 0.01


@dataclass(frozen=True, slots=True)
class WorkedValues:
    home_log_rate: float
    away_log_rate: float
    home_rate: float
    away_rate: float
    low_score_cells: dict[str, tuple[float, float, float]]
    fixed_outcomes: tuple[float, float, float]
    uncertain_outcomes: tuple[float, float, float, float, float]
    analytic_uncertain_expected_goals: tuple[float, float]


def poisson_probability(goals: int, rate: float) -> float:
    """Return one Poisson probability."""
    return math.exp(-rate) * rate**goals / math.factorial(goals)


def worked_values() -> WorkedValues:
    """Calculate the documented rates, low-score cells, and outcomes."""
    home_log_rate = SCORING_INTERCEPT + HOME_ATTACK - AWAY_DEFENSE + HOME_ADVANTAGE
    away_log_rate = SCORING_INTERCEPT + AWAY_ATTACK - HOME_DEFENSE
    home_rate = math.exp(home_log_rate)
    away_rate = math.exp(away_log_rate)

    match = Match("synthetic-worked", date(2030, 8, 1), "2030/31", "amber", "blue", 0, 0)
    uncertain_rates = GoalRateForecast(
        home_log_rate,
        HOME_VARIANCE,
        away_log_rate,
        AWAY_VARIANCE,
        RATE_COVARIANCE,
    )
    adjusted_rates = RelocatedHomeHfaPolicy(()).adjust_rates(match, uncertain_rates, HOME_ADVANTAGE)
    assert adjusted_rates == uncertain_rates

    score_matrix = dixon_coles_score_matrix(home_rate, away_rate, RHO)
    fixed_outcomes = (
        float(np.tril(score_matrix, -1).sum()),
        float(np.trace(score_matrix)),
        float(np.triu(score_matrix, 1).sum()),
    )
    uncertain_outcomes = live_uncertain_probabilities(
        home_log_rate,
        HOME_VARIANCE,
        away_log_rate,
        AWAY_VARIANCE,
        RHO,
        home_away_log_rate_covariance=RATE_COVARIANCE,
    )

    low_score_cells = {}
    for home_goals, away_goals, correction in (
        (0, 0, 1.0 - home_rate * away_rate * RHO),
        (0, 1, 1.0 + home_rate * RHO),
        (1, 0, 1.0 + away_rate * RHO),
        (1, 1, 1.0 - RHO),
    ):
        poisson = poisson_probability(home_goals, home_rate) * poisson_probability(away_goals, away_rate)
        low_score_cells[f"{home_goals}-{away_goals}"] = (poisson, correction, poisson * correction)

    return WorkedValues(
        home_log_rate,
        away_log_rate,
        home_rate,
        away_rate,
        low_score_cells,
        fixed_outcomes,
        uncertain_outcomes,
        (
            math.exp(home_log_rate + HOME_VARIANCE / 2.0),
            math.exp(away_log_rate + AWAY_VARIANCE / 2.0),
        ),
    )


def main() -> None:
    values = worked_values()
    print(f"log rates: {values.home_log_rate:.3f}, {values.away_log_rate:.3f}")
    print(f"fixed rates: {values.home_rate:.3f}, {values.away_rate:.3f}")
    for score, (poisson, correction, corrected) in values.low_score_cells.items():
        print(f"{score}: poisson={poisson:.2%} factor={correction:.3f} corrected={corrected:.2%}")
    print("fixed H/D/A: " + ", ".join(f"{value:.2%}" for value in values.fixed_outcomes))
    print("uncertain H/D/A: " + ", ".join(f"{value:.2%}" for value in values.uncertain_outcomes[:3]))
    print("uncertain xG: " + ", ".join(f"{value:.3f}" for value in values.uncertain_outcomes[3:]))


if __name__ == "__main__":
    main()
