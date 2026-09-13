"""Adopted retained-minutes defensive season-transition policy."""

import csv
import math
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np

from esa_model.joint_covariance import SeasonPriorPersistence, SeasonTeamPriorPersistence
from esa_model.stage1 import TeamState
from esa_model.stage2 import CovariateTarget, PriorFit

RIDGE_ALPHA = 1.0


@dataclass(frozen=True, slots=True)
class DefenseContinuityFit:
    coefficients: tuple[float, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    retention_center: float
    intercept: float

    def features(self, row: CovariateTarget, retention: float) -> tuple[float, ...]:
        return (*row.features("defense"), row.last_defense_mean * (retention - self.retention_center))

    def predict(self, row: CovariateTarget, retention: float) -> float:
        standardized = (np.asarray(self.features(row, retention)) - np.asarray(self.feature_means)) / np.asarray(
            self.feature_scales
        )
        return self.intercept + float(standardized @ np.asarray(self.coefficients))

    def persistence(self, retention: float) -> float:
        return self.coefficients[3] / self.feature_scales[3] + (
            self.coefficients[4] / self.feature_scales[4] * (retention - self.retention_center)
        )


def load_retained_minutes(path: Path, *, period: str | None = None) -> dict[tuple[str, str], float]:
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    output = {}
    for row in rows:
        if period is not None and row.get("period") != period:
            continue
        if row.get("status") != "ok":
            continue
        value = float(row["minutes_retained_share"])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Invalid retained-minutes share for {row['season']} {row['local_team_id']}")
        key = (row["season"], row["local_team_id"])
        if key in output:
            raise ValueError(f"Duplicate retained-minutes row: {key}")
        output[key] = value
    if not output:
        raise ValueError(f"No valid retained-minutes rows in {path}")
    return output


def load_prospective_retained_minutes(
    path: Path, season: str, expected_teams: set[str]
) -> tuple[date, dict[tuple[str, str], float]]:
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    retained = load_retained_minutes(path)
    if {row["season"] for row in rows} != {season} or {row["local_team_id"] for row in rows} != expected_teams:
        raise ValueError("Prospective retained-minutes snapshot does not match the returning league teams")
    cutoffs = {row["cutoff_date"] for row in rows}
    if len(cutoffs) != 1 or len(retained) != len(expected_teams):
        raise ValueError("Prospective retained-minutes snapshot must have one complete dated row per returning team")
    return date.fromisoformat(cutoffs.pop()), retained


def fit_defense_continuity(
    rows: list[CovariateTarget], retained_minutes: dict[tuple[str, str], float]
) -> DefenseContinuityFit:
    training = [row for row in rows if not row.promoted and (row.season, row.team_id) in retained_minutes]
    if len(training) < 16:
        raise ValueError(f"Need at least 16 returning continuity rows, got {len(training)}")
    retention_center = float(np.mean([retained_minutes[row.season, row.team_id] for row in training]))
    matrix = np.asarray(
        [
            (
                *row.features("defense"),
                row.last_defense_mean * (retained_minutes[row.season, row.team_id] - retention_center),
            )
            for row in training
        ]
    )
    targets = np.asarray([row.target_defense_mean for row in training])
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales == 0.0] = 1.0
    design = np.column_stack([np.ones(len(matrix)), (matrix - means) / scales])
    penalty = np.diag([0.0] + [RIDGE_ALPHA] * matrix.shape[1])
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return DefenseContinuityFit(
        tuple(float(value) for value in coefficients[1:]),
        tuple(float(value) for value in means),
        tuple(float(value) for value in scales),
        retention_center,
        float(coefficients[0]),
    )


def role_persistence(fit: PriorFit) -> tuple[float, float]:
    return (
        fit.attack.coefficients[-1] / fit.attack.feature_scales[-1],
        fit.defense.coefficients[-1] / fit.defense.feature_scales[-1],
    )


def build_defense_continuity_transitions(
    rows: list[CovariateTarget],
    base_priors: dict[tuple[str, str], TeamState],
    base_fits: dict[str, PriorFit],
    retained_minutes: dict[tuple[str, str], float],
) -> tuple[
    dict[tuple[str, str], TeamState],
    SeasonPriorPersistence,
    SeasonTeamPriorPersistence,
]:
    priors = dict(base_priors)
    global_persistence = {season: role_persistence(fit) for season, fit in base_fits.items()}
    team_persistence: SeasonTeamPriorPersistence = {}
    seasons = sorted({row.season for row in rows})
    for index, season in enumerate(seasons):
        if season not in base_fits:
            continue
        training = [row for row in rows if row.season in set(seasons[:index]) and not row.promoted]
        if len(training) < 16:
            continue
        fit = fit_defense_continuity(training, retained_minutes)
        current = [
            row
            for row in rows
            if row.season == season
            and not row.promoted
            and (season, row.team_id) in priors
            and (season, row.team_id) in retained_minutes
        ]
        team_persistence[season] = {}
        for row in current:
            retention = retained_minutes[season, row.team_id]
            priors[season, row.team_id] = replace(priors[season, row.team_id], defense_mean=fit.predict(row, retention))
            team_persistence[season][row.team_id] = (
                global_persistence[season][0],
                fit.persistence(retention),
            )
    return priors, global_persistence, team_persistence
