"""Adopted prediction-only HFA policy for sustained temporary home grounds."""

import csv
import math
from dataclasses import dataclass
from datetime import date
from functools import cache
from pathlib import Path
from typing import Protocol

from esa_model.baseline import Match
from esa_model.evaluation import MatchForecast
from esa_model.stage1 import FrozenGlobals, GoalRateForecast, uncertain_probabilities

DISTANCE_THRESHOLD_KM = 20.0
DISTANCE_HALF_KM = 20.0
DEFAULT_PERIODS_PATH = Path("data/relocated_home_periods.csv")


@dataclass(frozen=True, slots=True)
class RelocatedHomePeriod:
    episode_id: str
    home_team_id: str
    start_date: date
    end_date: date
    classification: str
    distance_km: float
    resilient_exclusion: bool
    excluded_match_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class RelocatedHomeHfaPolicy:
    periods: tuple[RelocatedHomePeriod, ...]

    def period_for(self, match: Match) -> RelocatedHomePeriod | None:
        matches = [
            period
            for period in self.periods
            if period.home_team_id == match.home
            and period.start_date <= match.date <= period.end_date
            and match.match_id not in period.excluded_match_ids
        ]
        if len(matches) > 1:
            raise ValueError(f"Multiple relocated-home periods cover match {match.match_id}")
        return matches[0] if matches else None

    def adjustment(self, match: Match, home_advantage_mean: float) -> float:
        period = self.period_for(match)
        if period is None or period.classification == "uncertain" or period.resilient_exclusion:
            return 0.0
        return -distance_dose(period.distance_km) * home_advantage_mean

    def adjust_rates(self, match: Match, rates: GoalRateForecast, home_advantage_mean: float) -> GoalRateForecast:
        adjustment = self.adjustment(match, home_advantage_mean)
        if adjustment == 0.0:
            return rates
        return GoalRateForecast(
            rates.home_log_rate_mean + adjustment,
            rates.home_log_rate_variance,
            rates.away_log_rate_mean,
            rates.away_log_rate_variance,
            rates.home_away_log_rate_covariance,
        )


class RelocatedHomeRateModel(Protocol):
    @property
    def globals(self) -> FrozenGlobals: ...

    @property
    def home_advantage_mean(self) -> float: ...

    def goal_rates(self, match: Match) -> GoalRateForecast: ...


def distance_dose(distance_km: float) -> float:
    if not math.isfinite(distance_km) or distance_km < 0.0:
        raise ValueError("Stadium distance must be finite and nonnegative")
    if distance_km <= DISTANCE_THRESHOLD_KM:
        return 0.0
    return 1.0 - 2.0 ** (-(distance_km - DISTANCE_THRESHOLD_KM) / DISTANCE_HALF_KM)


def load_relocated_home_policy(path: Path = DEFAULT_PERIODS_PATH) -> RelocatedHomeHfaPolicy:
    periods = []
    identities = set()
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            episode_id = row["episode_id"]
            if episode_id in identities:
                raise ValueError(f"Duplicate relocated-home episode: {episode_id}")
            identities.add(episode_id)
            classification = row["classification"]
            if classification not in {"strong", "secondary", "uncertain"}:
                raise ValueError(f"Invalid relocated-home classification: {episode_id}")
            resilient_text = row["resilient_exclusion"].lower()
            if resilient_text not in {"true", "false"}:
                raise ValueError(f"Invalid resilient exclusion: {episode_id}")
            start_date = date.fromisoformat(row["start_date"])
            end_date = date.fromisoformat(row["end_date"])
            if end_date < start_date:
                raise ValueError(f"Relocated-home episode ends before it starts: {episode_id}")
            distance_km = float(row["distance_km"])
            distance_dose(distance_km)
            periods.append(
                RelocatedHomePeriod(
                    episode_id,
                    row["home_team_id"],
                    start_date,
                    end_date,
                    classification,
                    distance_km,
                    resilient_text == "true",
                    frozenset(filter(None, row["excluded_match_ids"].split(";"))),
                )
            )
    return RelocatedHomeHfaPolicy(tuple(periods))


@cache
def adopted_relocated_home_policy() -> RelocatedHomeHfaPolicy:
    return load_relocated_home_policy()


def relocated_home_goal_rates(
    model: RelocatedHomeRateModel,
    match: Match,
    policy: RelocatedHomeHfaPolicy | None = None,
) -> GoalRateForecast:
    selected_policy = policy or adopted_relocated_home_policy()
    home_advantage_scale = 1.0 + selected_policy.adjustment(match, 1.0)
    scaled_rates = getattr(model, "goal_rates_with_home_advantage_scale", None)
    if scaled_rates is not None:
        return scaled_rates(match, home_advantage_scale)
    return selected_policy.adjust_rates(match, model.goal_rates(match), getattr(model, "home_advantage_mean", 0.0))


def relocated_home_forecast(
    model: RelocatedHomeRateModel,
    match: Match,
    policy: RelocatedHomeHfaPolicy | None = None,
) -> MatchForecast:
    rates = relocated_home_goal_rates(model, match, policy)
    probability = uncertain_probabilities(
        rates.home_log_rate_mean,
        rates.home_log_rate_variance,
        rates.away_log_rate_mean,
        rates.away_log_rate_variance,
        model.globals.rho,
        home_away_log_rate_covariance=rates.home_away_log_rate_covariance,
    )
    return MatchForecast(probability[3], probability[4], *probability[:3])
