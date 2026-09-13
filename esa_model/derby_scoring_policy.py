"""Owner-selected prediction-only scoring policy for three major derbies."""

from collections.abc import Mapping

from esa_model.baseline import Match
from esa_model.stage1 import GoalRateForecast

DERBY_SCORING_LOG_RATE_SHIFT = -0.2
DERBY_SCORING_PAIRS = frozenset(
    {
        frozenset(("255", "238")),  # Legia Warszawa - Lech Poznan
        frozenset(("422", "5689")),  # Wisla Krakow - Cracovia
        frozenset(("428", "6112")),  # Gornik Zabrze - Piast Gliwice
    }
)


def canonical_derby_pair(
    match: Match,
    canonical_club_ids: Mapping[tuple[str, str], str] | None,
) -> frozenset[str]:
    mapping = canonical_club_ids or {}
    return frozenset(
        (
            mapping.get((match.season, match.home), match.home),
            mapping.get((match.season, match.away), match.away),
        )
    )


def is_adopted_low_scoring_derby(
    match: Match,
    canonical_club_ids: Mapping[tuple[str, str], str] | None,
) -> bool:
    return canonical_derby_pair(match, canonical_club_ids) in DERBY_SCORING_PAIRS


def adjust_derby_scoring_rates(
    match: Match,
    rates: GoalRateForecast,
    canonical_club_ids: Mapping[tuple[str, str], str] | None,
) -> GoalRateForecast:
    if not is_adopted_low_scoring_derby(match, canonical_club_ids):
        return rates
    return GoalRateForecast(
        rates.home_log_rate_mean + DERBY_SCORING_LOG_RATE_SHIFT,
        rates.home_log_rate_variance,
        rates.away_log_rate_mean + DERBY_SCORING_LOG_RATE_SHIFT,
        rates.away_log_rate_variance,
        rates.home_away_log_rate_covariance,
    )
