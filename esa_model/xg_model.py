"""Causal source-specific xG calibration for model state updates."""

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from esa_model.baseline import Match

CALIBRATION_WINDOW_SEASONS = 3


@dataclass(frozen=True, slots=True)
class SourceGoalCalibration:
    source: str
    target_season: str
    training_start_season: str
    training_end_season: str
    training_matches: int
    home_xg_mean: float
    away_xg_mean: float
    home_log_goal_rate: float
    away_log_goal_rate: float
    xg_slope: float
    home_excess_variance: float
    away_excess_variance: float
    converged: bool


def load_source_xg(path: Path) -> dict[str, tuple[str, float, float, str]]:
    observations = {}
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            match_id = row["match_id"]
            if match_id in observations:
                raise ValueError(f"Duplicate selected xG match ID: {match_id}")
            observations[match_id] = (
                row["source"],
                float(row["home_xg"]),
                float(row["away_xg"]),
                row["season"],
            )
    return observations


def fit_goal_calibration(
    source: str,
    target_season: str,
    training_seasons: list[str],
    matches: list[Match],
    raw_xg: dict[str, tuple[str, float, float, str]],
) -> SourceGoalCalibration:
    training_set = set(training_seasons)
    rows = [
        (match, raw_xg[match.match_id])
        for match in matches
        if match.season in training_set and match.match_id in raw_xg and raw_xg[match.match_id][0] == source
    ]
    if not rows:
        raise ValueError(f"No {source} observations in calibration seasons {training_seasons}")
    home_xg = np.asarray([values[1] for _, values in rows])
    away_xg = np.asarray([values[2] for _, values in rows])
    xg = np.concatenate((home_xg, away_xg))
    goals = np.asarray([match.home_goals for match, _ in rows] + [match.away_goals for match, _ in rows], dtype=float)
    home = np.concatenate((np.ones(len(rows), dtype=bool), np.zeros(len(rows), dtype=bool)))
    home_mean = float(home_xg.mean())
    away_mean = float(away_xg.mean())
    centered_xg = xg - np.where(home, home_mean, away_mean)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        eta = np.where(home, parameters[0], parameters[1]) + parameters[2] * centered_xg
        rates = np.exp(eta)
        residual = rates - goals
        gradient = np.asarray([residual[home].sum(), residual[~home].sum(), float(residual @ centered_xg)])
        likelihood = np.sum(rates - goals * eta + gammaln(goals + 1.0))
        return float(likelihood), gradient

    initial = np.asarray(
        [
            np.log(max(float(goals[home].mean()), 1e-6)),
            np.log(max(float(goals[~home].mean()), 1e-6)),
            0.5,
        ]
    )
    result = minimize(
        lambda parameters: objective(parameters)[0],
        initial,
        jac=lambda parameters: objective(parameters)[1],
        method="L-BFGS-B",
        bounds=((None, None), (None, None), (0.0, 3.0)),
    )
    parameters = np.asarray(result.x)
    rates = np.exp(np.where(home, parameters[0], parameters[1]) + parameters[2] * centered_xg)
    raw_excess = (goals - rates) ** 2 - rates
    return SourceGoalCalibration(
        source,
        target_season,
        training_seasons[0],
        training_seasons[-1],
        len(rows),
        home_mean,
        away_mean,
        float(parameters[0]),
        float(parameters[1]),
        float(parameters[2]),
        max(float(raw_excess[home].mean()), 0.0),
        max(float(raw_excess[~home].mean()), 0.0),
        bool(result.success),
    )


def build_causal_goal_scale(
    matches: list[Match], raw_xg: dict[str, tuple[str, float, float, str]]
) -> tuple[
    dict[str, tuple[float, float]],
    dict[str, tuple[float, float]],
    list[SourceGoalCalibration],
]:
    source_seasons: dict[str, list[str]] = defaultdict(list)
    for source, _, _, season in raw_xg.values():
        if season not in source_seasons[source]:
            source_seasons[source].append(season)
    for seasons in source_seasons.values():
        seasons.sort()

    calibrated: dict[str, tuple[float, float]] = {}
    variances: dict[str, tuple[float, float]] = {}
    fits = []
    for source, seasons in sorted(source_seasons.items()):
        for index, target_season in enumerate(seasons):
            training_seasons = seasons[max(0, index - CALIBRATION_WINDOW_SEASONS) : index]
            if not training_seasons:
                continue
            fit = fit_goal_calibration(source, target_season, training_seasons, matches, raw_xg)
            if not fit.converged:
                raise RuntimeError(f"Goal calibration failed for {source} {target_season}")
            fits.append(fit)
            for match_id, (provider, home_xg, away_xg, season) in raw_xg.items():
                if provider != source or season != target_season:
                    continue
                calibrated[match_id] = (
                    float(np.exp(fit.home_log_goal_rate + fit.xg_slope * (home_xg - fit.home_xg_mean))),
                    float(np.exp(fit.away_log_goal_rate + fit.xg_slope * (away_xg - fit.away_xg_mean))),
                )
                variances[match_id] = (fit.home_excess_variance, fit.away_excess_variance)
    return calibrated, variances, fits
