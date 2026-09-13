"""Coarse Polish stadium-capacity regimes and crowd-effect estimation."""

from datetime import date, timedelta

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from esa_model.baseline import RIDGE, Match


def crowd_capacity(match_date: date) -> float:
    # Coarse national COVID-era limits; local-zone exceptions are intentionally ignored.
    if match_date < date(2020, 5, 29):
        return 1.0
    if match_date < date(2020, 6, 19):
        return 0.0
    if match_date < date(2020, 10, 17):
        return 0.25 if match_date < date(2020, 7, 25) else 0.5
    if match_date < date(2021, 5, 15):
        return 0.0
    if match_date < date(2021, 6, 26):
        return 0.25
    if match_date < date(2021, 12, 15):
        return 0.5
    if match_date < date(2022, 3, 1):
        return 0.25
    return 1.0


def crowd_absence(match_date: date) -> float:
    return 1.0 - crowd_capacity(match_date)


def fit_crowd_effect(matches: list[Match], cutoff_date: date, window_days: int = 1_096) -> float:
    """Fit the crowd offset on a fixed three-year trailing window by default."""
    training = [match for match in matches if cutoff_date - timedelta(days=window_days) <= match.date < cutoff_date]
    absence = np.asarray([crowd_absence(match.date) for match in training])
    if not np.any(absence > 0.0) or not np.any(absence == 0.0):
        return 0.0
    teams = tuple(sorted({team for match in training for team in (match.home, match.away)}))
    indices = {team: index for index, team in enumerate(teams)}
    home_teams = np.asarray([indices[match.home] for match in training], dtype=np.int64)
    away_teams = np.asarray([indices[match.away] for match in training], dtype=np.int64)
    goals = np.asarray([match.home_goals for match in training], dtype=float)
    team_count = len(teams)
    initial = np.zeros(2 * team_count + 1)
    initial[2 * team_count - 1] = np.log(max(float(goals.mean()), 0.1))

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        attack = np.empty(team_count)
        attack[:-1] = parameters[: team_count - 1]
        attack[-1] = -attack[:-1].sum()
        start = team_count - 1
        defense = parameters[start : start + team_count]
        home_advantage = parameters[-2]
        crowd_effect = parameters[-1]
        log_rates = attack[home_teams] + defense[away_teams] + home_advantage + crowd_effect * absence
        rates = np.exp(log_rates)
        residual = goals - rates
        attack_gradient = np.zeros(team_count)
        defense_gradient = np.zeros(team_count)
        np.add.at(attack_gradient, home_teams, residual)
        np.add.at(defense_gradient, away_teams, residual)
        centered_defense = defense - defense.mean()
        penalty = RIDGE * (np.dot(attack, attack) + np.dot(centered_defense, centered_defense))
        attack_penalty = 2.0 * RIDGE * attack
        gradient = np.concatenate(
            (
                attack_gradient[:-1] - attack_gradient[-1],
                defense_gradient,
                [residual.sum(), np.dot(residual, absence)],
            )
        )
        penalty_gradient = np.concatenate(
            (
                attack_penalty[:-1] - attack_penalty[-1],
                2.0 * RIDGE * centered_defense,
                [0.0, 0.0],
            )
        )
        likelihood = np.sum(goals * log_rates - rates - gammaln(goals + 1.0))
        return -float(likelihood) + float(penalty), -gradient + penalty_gradient

    bounds = [(None, None)] * (len(initial) - 2) + [(-1.0, 1.0), (-1.0, 1.0)]
    result = minimize(objective, initial, method="L-BFGS-B", jac=True, bounds=bounds)
    if not result.success:
        raise RuntimeError(f"Crowd-effect fit failed before {cutoff_date}: {result.message}")
    return float(result.x[-1])
