"""Adopted prediction-only table-motivation policy."""

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from esa_model.stage1 import GoalRateForecast


@dataclass(frozen=True, slots=True)
class MotivationScores:
    home_settled: float = 0.0
    away_settled: float = 0.0
    home_urgency: float = 0.0
    away_urgency: float = 0.0

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in (self.home_settled, self.away_settled, self.home_urgency, self.away_urgency)
        ):
            raise ValueError("Motivation scores must be finite probabilities")


@dataclass(frozen=True, slots=True)
class MotivationPolicy:
    grid: np.ndarray
    beta_weights: np.ndarray
    gamma_weights: np.ndarray
    settled_scale: float
    settled_remaining_gate: int
    urgency_scale: float
    urgency_remaining_gate: int
    trained_through_season: str

    @property
    def beta_mean(self) -> float:
        return float(np.dot(self.grid, self.beta_weights))

    @property
    def beta_variance(self) -> float:
        return float(np.dot(self.beta_weights, np.square(self.grid - self.beta_mean)))

    @property
    def gamma_mean(self) -> float:
        return float(np.dot(self.grid, self.gamma_weights))

    @property
    def gamma_variance(self) -> float:
        return float(np.dot(self.gamma_weights, np.square(self.grid - self.gamma_mean)))

    def adjust_rates(self, rates: GoalRateForecast, scores: MotivationScores) -> GoalRateForecast:
        settled_contrast = scores.away_settled - scores.home_settled
        urgency_contrast = scores.home_urgency - scores.away_urgency
        shift = self.beta_mean * settled_contrast + self.gamma_mean * urgency_contrast
        shift_variance = self.beta_variance * settled_contrast**2 + self.gamma_variance * urgency_contrast**2
        return GoalRateForecast(
            rates.home_log_rate_mean + shift,
            rates.home_log_rate_variance + shift_variance,
            rates.away_log_rate_mean - shift,
            rates.away_log_rate_variance + shift_variance,
            rates.home_away_log_rate_covariance - shift_variance,
        )


def load_motivation_policy(path: Path) -> MotivationPolicy:
    with path.open(encoding="utf-8") as source:
        payload = json.load(source)
    required = {
        "version",
        "trained_through_season",
        "grid",
        "beta_weights",
        "gamma_weights",
        "settled_scale",
        "settled_remaining_gate",
        "urgency_scale",
        "urgency_remaining_gate",
        "observation_updates",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Missing motivation-posterior fields: {', '.join(sorted(missing))}")
    if payload["version"] != 1 or payload["observation_updates"] is not False:
        raise ValueError("Unsupported motivation-posterior policy")
    grid = np.asarray(payload["grid"], dtype=float)
    beta = np.asarray(payload["beta_weights"], dtype=float)
    gamma = np.asarray(payload["gamma_weights"], dtype=float)
    if (
        grid.ndim != 1
        or len(grid) < 3
        or beta.shape != grid.shape
        or gamma.shape != grid.shape
        or not np.all(np.isfinite(grid))
        or not np.all(np.diff(grid) > 0.0)
        or not np.all(np.isfinite(beta))
        or not np.all(np.isfinite(gamma))
        or np.any(beta < 0.0)
        or np.any(gamma < 0.0)
        or not math.isclose(float(beta.sum()), 1.0, abs_tol=1e-9)
        or not math.isclose(float(gamma.sum()), 1.0, abs_tol=1e-9)
    ):
        raise ValueError("Invalid motivation-posterior grid or weights")
    for field in ("settled_scale", "urgency_scale"):
        if not math.isfinite(payload[field]) or payload[field] <= 0.0:
            raise ValueError(f"Motivation {field} must be positive")
    for field in ("settled_remaining_gate", "urgency_remaining_gate"):
        if not isinstance(payload[field], int) or payload[field] <= 0:
            raise ValueError(f"Motivation {field} must be a positive integer")
    return MotivationPolicy(
        grid,
        beta / beta.sum(),
        gamma / gamma.sum(),
        float(payload["settled_scale"]),
        payload["settled_remaining_gate"],
        float(payload["urgency_scale"]),
        payload["urgency_remaining_gate"],
        str(payload["trained_through_season"]),
    )


def load_historical_motivation_scores(path: Path, policy: MotivationPolicy) -> dict[str, MotivationScores]:
    scores = {}
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):

            def team_score(source: dict[str, str], side: str) -> tuple[float, float]:
                remaining = int(source[f"{side}_remaining"])
                settled = (
                    max(0.0, 1.0 - float(source[f"{side}_leverage"]) / policy.settled_scale)
                    if remaining <= policy.settled_remaining_gate
                    else 0.0
                )
                urgency = (
                    min(1.0, float(source[f"{side}_objective_swing"]) / policy.urgency_scale)
                    if remaining <= policy.urgency_remaining_gate
                    else 0.0
                )
                return settled, urgency

            home_settled, home_urgency = team_score(row, "home")
            away_settled, away_urgency = team_score(row, "away")
            scores[row["match_id"]] = MotivationScores(home_settled, away_settled, home_urgency, away_urgency)
    return scores
