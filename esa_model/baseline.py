"""Naive Dixon-Coles baseline for Ekstraklasa match outcomes."""

import argparse
import bisect
import csv
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
WINDOW_DAYS = 1_096
MAX_GOALS = 10
RIDGE = 0.1
NAIVE = (0.44, 0.27, 0.29)
HALF_LIVES = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0)
HOLDOUT_SEASONS = 4
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


@dataclass(frozen=True, slots=True)
class Prediction:
    match_id: str
    date: date
    season: str
    home_team_id: str
    away_team_id: str
    home_goals: int
    away_goals: int
    home_xg: float
    away_xg: float
    p_home: float
    p_draw: float
    p_away: float
    outcome: int
    rps: float
    cold_start: bool


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


def load_odds(path: Path) -> dict[str, tuple[float, float, float]]:
    with path.open(encoding="utf-8", newline="") as source:
        rows = csv.DictReader(source)
        required = {"match_id", "home_odds", "draw_odds", "away_odds"}
        missing = required - set(rows.fieldnames or ())
        if missing:
            raise ValueError(f"Missing odds columns: {', '.join(sorted(missing))}")
        odds = {}
        for row in rows:
            match_id = row["match_id"]
            if not match_id:
                raise ValueError("Odds row has an empty match ID")
            if match_id in odds:
                raise ValueError(f"Duplicate odds match ID: {match_id}")
            values = (float(row["home_odds"]), float(row["draw_odds"]), float(row["away_odds"]))
            if not all(math.isfinite(value) and value > 1.0 for value in values):
                raise ValueError(f"Match {match_id} odds must be finite and greater than one")
            odds[match_id] = values
        return odds


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


def walk_forward(matches: list[Match], xi: float, evaluation_seasons: set[str]) -> list[Prediction]:
    dates = [match.date for match in matches]
    teams_by_season: dict[str, set[str]] = defaultdict(set)
    for match in matches:
        teams_by_season[match.season].update((match.home, match.away))
    seasons = sorted(teams_by_season)
    predictions = []
    warm = None
    for evaluation_date in sorted({match.date for match in matches if match.season in evaluation_seasons}):
        start = bisect.bisect_left(dates, evaluation_date - timedelta(days=WINDOW_DAYS))
        end = bisect.bisect_left(dates, evaluation_date)
        model = fit(matches[start:end], evaluation_date, xi, warm)
        warm = model
        model_index = {team: index for index, team in enumerate(model.teams)}
        tests = matches[end : bisect.bisect_right(dates, evaluation_date)]
        for match in tests:
            if match.season not in evaluation_seasons:
                continue
            season_index = seasons.index(match.season)
            relegated = teams_by_season[seasons[season_index - 1]] - teams_by_season[match.season]
            fallback_indices = [model_index[team] for team in relegated if team in model_index]
            if fallback_indices:
                fallback = (
                    float(model.attack[fallback_indices].mean()),
                    float(model.defense[fallback_indices].mean()),
                )
            else:
                fallback = float(model.attack.mean()), float(model.defense.mean())
            home_attack, home_defense = (
                (model.attack[model_index[match.home]], model.defense[model_index[match.home]])
                if match.home in model_index
                else fallback
            )
            away_attack, away_defense = (
                (model.attack[model_index[match.away]], model.defense[model_index[match.away]])
                if match.away in model_index
                else fallback
            )
            home_rate = float(np.exp(home_attack + away_defense + model.home_advantage))
            away_rate = float(np.exp(away_attack + home_defense))
            outcome_probability = probabilities(home_rate, away_rate, model.rho)
            predictions.append(
                Prediction(
                    match.match_id,
                    match.date,
                    match.season,
                    match.home,
                    match.away,
                    match.home_goals,
                    match.away_goals,
                    home_rate,
                    away_rate,
                    *outcome_probability,
                    match.outcome,
                    rps(outcome_probability, match.outcome),
                    match.home not in model_index or match.away not in model_index,
                )
            )
    return predictions


def evaluate_half_life(arguments: tuple[Path, float, set[str]]) -> tuple[float, list[Prediction]]:
    path, half_life, seasons = arguments
    return half_life, walk_forward(load_matches(path), math.log(2) / half_life, seasons)


def mean_rps(predictions: list[Prediction]) -> dict[str, float | int]:
    return {
        "matches": len(predictions),
        "model_rps": sum(prediction.rps for prediction in predictions) / len(predictions),
        "naive_rps": sum(rps(NAIVE, prediction.outcome) for prediction in predictions) / len(predictions),
        "cold_starts": sum(prediction.cold_start for prediction in predictions),
    }


def odds_probability(odds: tuple[float, float, float]) -> tuple[float, float, float]:
    inverted = (1 / odds[0], 1 / odds[1], 1 / odds[2])
    total = sum(inverted)
    return inverted[0] / total, inverted[1] / total, inverted[2] / total


def add_market(
    predictions: list[Prediction], odds: dict[str, tuple[float, float, float]], holdout_seasons: set[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for prediction in predictions:
        row = asdict(prediction)
        row["date"] = prediction.date.isoformat()
        if prediction.match_id in odds:
            market_probability = odds_probability(odds[prediction.match_id])
            row.update(
                {
                    "market_p_home": market_probability[0],
                    "market_p_draw": market_probability[1],
                    "market_p_away": market_probability[2],
                    "market_rps": rps(market_probability, prediction.outcome),
                }
            )
        else:
            row.update({"market_p_home": "", "market_p_draw": "", "market_p_away": "", "market_rps": ""})
        rows.append(row)

    common = [row for row in rows if row["market_rps"] != ""]
    holdout = [row for row in common if row["season"] in holdout_seasons]

    def metrics(selected: list[dict[str, Any]]) -> dict[str, float | int]:
        return {
            "matches": len(selected),
            "naive_rps": sum(rps(NAIVE, row["outcome"]) for row in selected) / len(selected),
            "model_rps": sum(row["rps"] for row in selected) / len(selected),
            "market_rps": sum(row["market_rps"] for row in selected) / len(selected),
        }

    comparison = {
        "all_common_matches": metrics(common),
        "holdout": metrics(holdout),
    }
    return rows, comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    data_path = Path("data/matches.csv")
    matches = load_matches(data_path)
    odds = load_odds(Path("data/odds.csv"))
    seasons = sorted({match.season for match in matches})
    evaluation_seasons = seasons[BURN_IN_SEASONS:]
    holdout_seasons = set(evaluation_seasons[-HOLDOUT_SEASONS:])
    tuning_seasons = set(evaluation_seasons) - holdout_seasons
    tasks = [(data_path, half_life, set(evaluation_seasons)) for half_life in HALF_LIVES]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(evaluate_half_life, tasks))

    decay_search = []
    by_half_life = {}
    for half_life, predictions in results:
        by_half_life[half_life] = predictions
        tuning = [prediction for prediction in predictions if prediction.season in tuning_seasons]
        decay_search.append({"half_life": half_life, **mean_rps(tuning)})
    selected = min(decay_search, key=lambda row: row["model_rps"])
    selected_half_life = float(selected["half_life"])
    predictions = by_half_life[selected_half_life]
    tuning = [prediction for prediction in predictions if prediction.season in tuning_seasons]
    holdout = [prediction for prediction in predictions if prediction.season in holdout_seasons]
    rows, market_comparison = add_market(predictions, odds, holdout_seasons)
    with Path("artifacts/baseline_predictions.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "model": {
            "window_days": WINDOW_DAYS,
            "ridge": RIDGE,
            "score_matrix": "0..10, normalized",
            "selected_half_life": selected_half_life,
        },
        "tuning": mean_rps(tuning),
        "holdout": mean_rps(holdout),
        "decay_search": decay_search,
        "market_comparison": market_comparison,
    }
    Path("artifacts/baseline_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
