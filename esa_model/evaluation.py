"""Model-independent chronological evaluation contracts and runner."""

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from esa_model.baseline import Match, rps

PROMOTED_FIRST_17 = "Promoted first 17"
OTHER_FIRST_FIVE = "Other first five"
REST = "Rest"
SEGMENT_NAMES = (PROMOTED_FIRST_17, OTHER_FIRST_FIVE, REST)


@dataclass(frozen=True, slots=True)
class MatchForecast:
    home_xg: float
    away_xg: float
    p_home: float
    p_draw: float
    p_away: float


@dataclass(frozen=True, slots=True)
class StagePrediction:
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


class MatchModel(Protocol):
    def advance_to(
        self,
        match_date: date,
        season: str,
        active_teams: set[str],
    ) -> None: ...

    def predict(self, match: Match) -> MatchForecast: ...

    def observe(self, matches: list[Match]) -> None: ...


def validate_forecast(forecast: MatchForecast) -> None:
    expected_goals = (forecast.home_xg, forecast.away_xg)
    probabilities = (forecast.p_home, forecast.p_draw, forecast.p_away)
    if not all(math.isfinite(value) and value >= 0.0 for value in expected_goals):
        raise ValueError("Expected goals must be finite and nonnegative")
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError("Outcome probabilities must be finite and between zero and one")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Outcome probabilities must sum to one")


def classify_segments(
    matches: list[Match],
    canonical_club_ids: dict[tuple[str, str], str] | None = None,
) -> dict[str, str]:
    """Assign mutually exclusive pre-registered segments to chronological matches."""
    teams_by_season: dict[str, set[str]] = defaultdict(set)
    for match in matches:
        teams_by_season[match.season].update((match.home, match.away))
    seasons = sorted(teams_by_season)
    previous_teams = {season: teams_by_season[seasons[index - 1]] for index, season in enumerate(seasons) if index}
    appearances: dict[tuple[str, str], int] = defaultdict(int)
    segments = {}
    for match in matches:
        previous = previous_teams.get(match.season, set())
        if canonical_club_ids is None:
            promoted = teams_by_season[match.season] - previous
        else:
            previous_clubs = {canonical_club_ids[seasons[seasons.index(match.season) - 1], team] for team in previous}
            promoted = {
                team
                for team in teams_by_season[match.season]
                if canonical_club_ids[match.season, team] not in previous_clubs
            }
        home_games = appearances[(match.season, match.home)]
        away_games = appearances[(match.season, match.away)]
        promoted_first_half = (match.home in promoted and home_games < 17) or (
            match.away in promoted and away_games < 17
        )
        if promoted_first_half:
            segment = PROMOTED_FIRST_17
        elif home_games < 5 or away_games < 5:
            segment = OTHER_FIRST_FIVE
        else:
            segment = REST
        segments[match.match_id] = segment
        appearances[(match.season, match.home)] += 1
        appearances[(match.season, match.away)] += 1
    return segments


def run_walk_forward(
    matches: list[Match],
    model: MatchModel,
    evaluation_seasons: set[str],
) -> list[StagePrediction]:
    teams_by_season: dict[str, set[str]] = defaultdict(set)
    matches_by_date: dict[date, list[Match]] = defaultdict(list)
    for match in matches:
        teams_by_season[match.season].update((match.home, match.away))
        matches_by_date[match.date].append(match)
    predictions = []
    for match_date, date_matches in sorted(matches_by_date.items()):
        season = date_matches[0].season
        if any(match.season != season for match in date_matches):
            raise ValueError(f"Matches on {match_date} span multiple seasons")
        active_teams = teams_by_season[season]
        model.advance_to(match_date, season, active_teams)
        if season in evaluation_seasons:
            for match in date_matches:
                forecast = model.predict(match)
                validate_forecast(forecast)
                outcome_probability = (forecast.p_home, forecast.p_draw, forecast.p_away)
                predictions.append(
                    StagePrediction(
                        match.match_id,
                        match.date,
                        match.season,
                        match.home,
                        match.away,
                        match.home_goals,
                        match.away_goals,
                        forecast.home_xg,
                        forecast.away_xg,
                        *outcome_probability,
                        match.outcome,
                        rps(outcome_probability, match.outcome),
                    )
                )
        model.observe(date_matches)
    return predictions
