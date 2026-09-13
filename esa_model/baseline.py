"""Naive Dixon-Coles baseline for Ekstraklasa match outcomes."""

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
MAX_GOALS = 10
RIDGE = 0.1
BURN_IN_SEASONS = 3


@dataclass(frozen=True, slots=True)
class Match:
    match_id: str
    date: date
    season: str
    home: str
    away: str
    home_goals: int
    away_goals: int

    @property
    def outcome(self) -> int:
        if self.home_goals > self.away_goals:
            return 0
        if self.home_goals == self.away_goals:
            return 1
        return 2


@dataclass(slots=True)
class Model:
    teams: tuple[str, ...]
    attack: FloatArray
    defense: FloatArray
    home_advantage: float
    rho: float
    converged: bool


def load_matches(path: Path) -> list[Match]:
    with path.open(encoding="utf-8", newline="") as source:
        rows = csv.DictReader(source)
        required = {"match_id", "date", "season", "home_team_id", "away_team_id", "home_goals", "away_goals"}
        missing = required - set(rows.fieldnames or ())
        if missing:
            raise ValueError(f"Missing match columns: {', '.join(sorted(missing))}")
        matches = []
        match_ids = set()
        for row in rows:
            match_id = row["match_id"]
            if not match_id:
                continue
            if match_id in match_ids:
                raise ValueError(f"Duplicate match ID: {match_id}")
            home = row["home_team_id"]
            away = row["away_team_id"]
            if not row["season"] or not home or not away:
                raise ValueError(f"Match {match_id} has an empty season or team ID")
            if home == away:
                raise ValueError(f"Match {match_id} has the same home and away team")
            home_goals = int(row["home_goals"])
            away_goals = int(row["away_goals"])
            if home_goals < 0 or away_goals < 0:
                raise ValueError(f"Match {match_id} has negative goals")
            matches.append(
                Match(
                    match_id,
                    date.fromisoformat(row["date"]),
                    row["season"],
                    home,
                    away,
                    home_goals,
                    away_goals,
                )
            )
            match_ids.add(match_id)
    return sorted(matches, key=lambda match: (match.date, match.match_id))


def unpack(parameters: FloatArray, team_count: int) -> tuple[FloatArray, FloatArray, float, float]:
    attack = np.empty(team_count)
    attack[:-1] = parameters[: team_count - 1]
    attack[-1] = -attack[:-1].sum()
    start = team_count - 1
    return attack, parameters[start : start + team_count], float(parameters[-2]), float(parameters[-1])


def initial_parameters(
    teams: tuple[str, ...],
    home_goals: IntArray,
    away_goals: IntArray,
    weights: FloatArray,
    warm: Model | None,
) -> FloatArray:
    away_mean = max(float(np.average(away_goals, weights=weights)), 0.1)
    home_mean = max(float(np.average(home_goals, weights=weights)), 0.1)
    attack = np.zeros(len(teams))
    defense = np.full(len(teams), np.log(away_mean))
    home_advantage = np.log(home_mean / away_mean)
    rho = 0.0
    if warm is not None:
        old = {team: index for index, team in enumerate(warm.teams)}
        for index, team in enumerate(teams):
            if team in old:
                attack[index] = warm.attack[old[team]]
                defense[index] = warm.defense[old[team]]
        shift = float(attack.mean())
        attack -= shift
        defense += shift
        home_advantage = warm.home_advantage
        rho = warm.rho
    return np.concatenate((attack[:-1], defense, [home_advantage, rho]))


def dc_terms(
    home_goals: IntArray,
    away_goals: IntArray,
    home_rates: FloatArray,
    away_rates: FloatArray,
    rho: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    tau = np.ones_like(home_rates)
    d_home = np.zeros_like(home_rates)
    d_away = np.zeros_like(home_rates)
    d_rho = np.zeros_like(home_rates)
    score_00 = (home_goals == 0) & (away_goals == 0)
    score_01 = (home_goals == 0) & (away_goals == 1)
    score_10 = (home_goals == 1) & (away_goals == 0)
    score_11 = (home_goals == 1) & (away_goals == 1)
    tau[score_00] = 1 - home_rates[score_00] * away_rates[score_00] * rho
    d_home[score_00] = -away_rates[score_00] * rho
    d_away[score_00] = -home_rates[score_00] * rho
    d_rho[score_00] = -home_rates[score_00] * away_rates[score_00]
    tau[score_01] = 1 + home_rates[score_01] * rho
    d_home[score_01] = rho
    d_rho[score_01] = home_rates[score_01]
    tau[score_10] = 1 + away_rates[score_10] * rho
    d_away[score_10] = rho
    d_rho[score_10] = away_rates[score_10]
    tau[score_11] = 1 - rho
    d_rho[score_11] = -1
    return tau, d_home, d_away, d_rho


def fit(matches: list[Match], evaluation_date: date, xi: float, warm: Model | None = None) -> Model:
    teams = tuple(sorted({team for match in matches for team in (match.home, match.away)}))
    indices = {team: index for index, team in enumerate(teams)}
    home_teams = np.array([indices[match.home] for match in matches], dtype=np.int64)
    away_teams = np.array([indices[match.away] for match in matches], dtype=np.int64)
    home_goals = np.array([match.home_goals for match in matches], dtype=np.int64)
    away_goals = np.array([match.away_goals for match in matches], dtype=np.int64)
    days_old = np.array([(evaluation_date - match.date).days for match in matches])
    weights = np.exp(-xi * days_old / 365)
    initial = initial_parameters(teams, home_goals, away_goals, weights, warm)
    team_count = len(teams)

    def objective(parameters: FloatArray) -> tuple[float, FloatArray]:
        attack, defense, home_advantage, rho = unpack(parameters, team_count)
        home_log_rates = attack[home_teams] + defense[away_teams] + home_advantage
        away_log_rates = attack[away_teams] + defense[home_teams]
        home_rates = np.exp(home_log_rates)
        away_rates = np.exp(away_log_rates)
        tau, d_home, d_away, d_rho = dc_terms(home_goals, away_goals, home_rates, away_rates, rho)
        if np.any(tau <= 0) or not np.all(np.isfinite(tau)):
            return 1e100, np.zeros_like(parameters)
        log_likelihood = weights * (
            home_goals * home_log_rates
            - home_rates
            - gammaln(home_goals + 1)
            + away_goals * away_log_rates
            - away_rates
            - gammaln(away_goals + 1)
            + np.log(tau)
        )
        home_gradient = weights * (home_goals - home_rates + d_home * home_rates / tau)
        away_gradient = weights * (away_goals - away_rates + d_away * away_rates / tau)
        attack_gradient = np.zeros(team_count)
        defense_gradient = np.zeros(team_count)
        np.add.at(attack_gradient, home_teams, home_gradient)
        np.add.at(attack_gradient, away_teams, away_gradient)
        np.add.at(defense_gradient, away_teams, home_gradient)
        np.add.at(defense_gradient, home_teams, away_gradient)
        likelihood_gradient = np.concatenate(
            (
                attack_gradient[:-1] - attack_gradient[-1],
                defense_gradient,
                [home_gradient.sum(), np.sum(weights * d_rho / tau)],
            )
        )
        centered_defense = defense - defense.mean()
        penalty = RIDGE * (np.dot(attack, attack) + np.dot(centered_defense, centered_defense))
        attack_penalty = 2 * RIDGE * attack
        penalty_gradient = np.concatenate(
            (
                attack_penalty[:-1] - attack_penalty[-1],
                2 * RIDGE * centered_defense,
                [0.0, 0.0],
            )
        )
        return -float(log_likelihood.sum()) + float(penalty), -likelihood_gradient + penalty_gradient

    bounds = [(None, None)] * (len(initial) - 2) + [(-2.0, 2.0), (-0.15, 0.15)]
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 200, "ftol": 1e-9, "gtol": 1e-6},
    )
    if not result.success:
        raise RuntimeError(f"Dixon-Coles fit failed: {result.message}")
    attack, defense, home_advantage, rho = unpack(result.x, team_count)
    return Model(teams, attack, defense.copy(), home_advantage, rho, True)


def probabilities(home_rate: float, away_rate: float, rho: float) -> tuple[float, float, float]:
    goals = np.arange(MAX_GOALS + 1)
    matrix = np.outer(poisson.pmf(goals, home_rate), poisson.pmf(goals, away_rate))
    matrix[0, 0] *= 1 - home_rate * away_rate * rho
    matrix[0, 1] *= 1 + home_rate * rho
    matrix[1, 0] *= 1 + away_rate * rho
    matrix[1, 1] *= 1 - rho
    matrix /= matrix.sum()
    return float(np.tril(matrix, -1).sum()), float(np.trace(matrix)), float(np.triu(matrix, 1).sum())


def rps(probability: tuple[float, float, float], outcome: int) -> float:
    cumulative = (probability[0], probability[0] + probability[1])
    actual = (float(outcome == 0), float(outcome <= 1))
    return sum((predicted - observed) ** 2 for predicted, observed in zip(cumulative, actual, strict=True)) / 2
