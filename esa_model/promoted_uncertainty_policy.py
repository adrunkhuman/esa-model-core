"""Operational T2 promoted-team turnover variance policy and input contracts."""

import csv
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from esa_model.baseline import Match
from esa_model.stage2 import CovariateTarget

T2_SLOPE = 2.0
COMPLETE_MINUTES = 20_000
HISTORICAL_START_SEASON = "2018/19"
SEASON_START_PERIOD = "season_start_prior"
PROSPECTIVE_PATH = Path("data/promoted_turnover_2026_27.csv")
MEMBERSHIP_PROVENANCE = "season_roster_joined_on_or_before_cutoff"
HISTORICAL_COLUMNS = frozenset(
    {
        "season",
        "period",
        "cutoff_date",
        "local_team_id",
        "local_team_name",
        "transfermarkt_club_id",
        "source_season",
        "source_competition_id",
        "source_players",
        "retained_source_players",
        "source_minutes",
        "retained_minutes",
        "minutes_retained_share",
        "status",
    }
)
PROSPECTIVE_COLUMNS = frozenset(
    {
        "season",
        "cutoff_date",
        "local_team_id",
        "canonical_club_id",
        "local_team_name",
        "transfermarkt_club_id",
        "source_season",
        "source_competition_id",
        "source_players",
        "retained_source_players",
        "source_minutes",
        "retained_minutes",
        "minutes_retained_share",
        "raw_turnover",
        "t2_variance_scale",
        "status",
        "roster_source_url",
        "performance_source_url",
        "membership_provenance",
    }
)


@dataclass(frozen=True, slots=True)
class PromotedTurnoverScale:
    season: str
    team_id: str
    canonical_club_id: str
    cutoff_date: date
    minutes_retained_share: float
    raw_turnover: float
    variance_scale: float
    source: str


def turnover_scale(minutes_retained_share: float) -> float:
    """Return the T2 diagonal-variance multiplier for retained prior-season minutes."""
    if not math.isfinite(minutes_retained_share) or not 0.0 <= minutes_retained_share <= 1.0:
        raise ValueError(f"Minutes retained share must be finite and within [0, 1]: {minutes_retained_share}")
    return 1.0 + T2_SLOPE * (1.0 - minutes_retained_share)


def season_opening_dates(matches: list[Match]) -> dict[str, date]:
    openings: dict[str, date] = {}
    for match in matches:
        openings[match.season] = min(openings.get(match.season, match.date), match.date)
    return openings


def _previous_season(season: str) -> str:
    year = int(season[:4]) - 1
    return f"{year}/{str(year + 1)[-2:]}"


def _read_rows(path: Path, required: frozenset[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        if fields is None or len(fields) != len(set(fields)) or required - set(fields):
            raise ValueError(f"Missing or duplicate required promoted turnover columns in {path}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed promoted turnover row in {path}")
    return rows  # type: ignore[return-value]


def _text(row: dict[str, str], field: str, context: object) -> str:
    value = row[field].strip()
    if not value:
        raise ValueError(f"Missing {field} for {context}")
    return value


def _integer(row: dict[str, str], field: str, context: object) -> int:
    try:
        value = int(_text(row, field, context))
    except ValueError as error:
        raise ValueError(f"Invalid {field} for {context}") from error
    if value < 0:
        raise ValueError(f"Invalid {field} for {context}")
    return value


def _number(row: dict[str, str], field: str, context: object) -> float:
    try:
        value = float(_text(row, field, context))
    except ValueError as error:
        raise ValueError(f"Invalid {field} for {context}") from error
    if not math.isfinite(value):
        raise ValueError(f"Invalid {field} for {context}")
    return value


def _cutoff(row: dict[str, str], context: object) -> date:
    try:
        return date.fromisoformat(_text(row, "cutoff_date", context))
    except ValueError as error:
        raise ValueError(f"Invalid cutoff_date for {context}") from error


def _validate_counts_and_minutes(row: dict[str, str], context: object) -> tuple[int, int, int, int, float]:
    source_players = _integer(row, "source_players", context)
    retained_players = _integer(row, "retained_source_players", context)
    source_minutes = _integer(row, "source_minutes", context)
    retained_minutes = _integer(row, "retained_minutes", context)
    retained_share = _number(row, "minutes_retained_share", context)
    if (
        retained_players > source_players
        or source_minutes < COMPLETE_MINUTES
        or retained_minutes > source_minutes
        or not 0.0 <= retained_share <= 1.0
        or not math.isclose(retained_share, retained_minutes / source_minutes, abs_tol=1e-12)
    ):
        raise ValueError(f"Invalid promoted turnover counts or minutes for {context}")
    return source_players, retained_players, source_minutes, retained_minutes, retained_share


def _scales(rows: list[PromotedTurnoverScale]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for row in rows:
        season = output.setdefault(row.season, {})
        if row.team_id in season:
            raise ValueError(f"Duplicate promoted turnover scale: {(row.season, row.team_id)}")
        season[row.team_id] = row.variance_scale
    return output


def historical_promoted_turnover_scales(
    covariates: list[CovariateTarget],
    canonical_ids: dict[tuple[str, str], str],
    opening_dates: dict[str, date],
    path: Path = Path("data/squad_continuity.csv"),
) -> tuple[dict[str, dict[str, float]], list[PromotedTurnoverScale]]:
    """Load every covered promoted transition from the historical continuity snapshot.

    The production model fails closed: a promoted target cannot receive an inferred
    turnover value when its exact season-start continuity source is absent or invalid.
    """
    targets: dict[tuple[str, str], CovariateTarget] = {}
    for row in covariates:
        if not row.promoted or row.season < HISTORICAL_START_SEASON:
            continue
        key = row.season, row.team_id
        if key in targets:
            raise ValueError(f"Duplicate promoted covariate: {key}")
        if canonical_ids.get(key) != row.canonical_club_id:
            raise ValueError(f"Promoted covariate canonical identity mismatch: {key}")
        targets[key] = row
    continuity: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in _read_rows(path, HISTORICAL_COLUMNS):
        if row["period"] != SEASON_START_PERIOD:
            continue
        key = row["season"], row["transfermarkt_club_id"]
        continuity.setdefault(key, []).append(row)
    output = []
    for _, target in sorted(targets.items()):
        snapshot_key = target.season, target.canonical_club_id
        rows = continuity.get(snapshot_key, [])
        if len(rows) != 1:
            raise ValueError(f"Expected one promoted continuity snapshot for {snapshot_key}; found {len(rows)}")
        row = rows[0]
        if (
            _text(row, "season", snapshot_key) != target.season
            or _text(row, "period", snapshot_key) != SEASON_START_PERIOD
            or _text(row, "local_team_id", snapshot_key) != target.team_id
            or _text(row, "transfermarkt_club_id", snapshot_key) != target.canonical_club_id
            or _text(row, "source_season", snapshot_key) != _previous_season(target.season)
            or _text(row, "source_competition_id", snapshot_key) != "PL2"
            or _text(row, "status", snapshot_key) != "ok"
        ):
            raise ValueError(f"Invalid promoted continuity identity or source for {snapshot_key}")
        _text(row, "local_team_name", snapshot_key)
        cutoff = _cutoff(row, snapshot_key)
        if cutoff != opening_dates.get(target.season):
            raise ValueError(f"Promoted continuity cutoff does not equal the season opener for {snapshot_key}")
        _, _, _, _, retained = _validate_counts_and_minutes(row, snapshot_key)
        scale = turnover_scale(retained)
        output.append(
            PromotedTurnoverScale(
                target.season,
                target.team_id,
                target.canonical_club_id,
                cutoff,
                retained,
                1.0 - retained,
                scale,
                str(path),
            )
        )
    return _scales(output), output


def prospective_promoted_turnover_scales(
    season_path: Path,
    path: Path = PROSPECTIVE_PATH,
) -> tuple[dict[str, dict[str, float]], list[PromotedTurnoverScale]]:
    """Load the checked prospective T2 snapshot for exactly the configured promoted clubs."""
    with season_path.open(encoding="utf-8") as source:
        season_config = json.load(source)
    season = str(season_config["season"])
    opener = date.fromisoformat(season_config["start_date"])
    promoted = {str(team["team_id"]): team for team in season_config["teams"] if team["promoted"]}
    rows = _read_rows(path, PROSPECTIVE_COLUMNS)
    by_team: dict[str, dict[str, str]] = {}
    for row in rows:
        team_id = row.get("local_team_id", "")
        if team_id in by_team:
            raise ValueError(f"Duplicate prospective promoted turnover row: {team_id}")
        by_team[team_id] = row
    if by_team.keys() != promoted.keys():
        raise ValueError(f"Prospective promoted turnover teams do not match frontend: {sorted(by_team)}")
    output = []
    for team_id, team in sorted(promoted.items()):
        row = by_team[team_id]
        context = season, team_id
        canonical = str(team["canonical_club_id"])
        source_season = _previous_season(season)
        if (
            _text(row, "season", context) != season
            or _text(row, "canonical_club_id", context) != canonical
            or _text(row, "transfermarkt_club_id", context) != canonical
            or _text(row, "local_team_name", context) != str(team["name"])
            or _text(row, "source_season", context) != source_season
            or _text(row, "source_competition_id", context) != "PL2"
            or _text(row, "status", context) != "ok"
            or _text(row, "membership_provenance", context) != MEMBERSHIP_PROVENANCE
        ):
            raise ValueError(f"Invalid prospective promoted turnover identity or source for {team_id}")
        cutoff = _cutoff(row, context)
        _, _, _, _, retained = _validate_counts_and_minutes(row, context)
        turnover = _number(row, "raw_turnover", context)
        scale = _number(row, "t2_variance_scale", context)
        roster_url = _text(row, "roster_source_url", context)
        performance_url = _text(row, "performance_source_url", context)
        expected_roster_url = f"https://www.transfermarkt.com/-/kader/verein/{canonical}/saison_id/{season[:4]}/plus/1"
        expected_performance_url = (
            "https://www.transfermarkt.com/-/leistungsdaten/verein/"
            f"{canonical}/reldata/PL2%26{source_season[:4]}/plus/1"
        )
        expected_scale = turnover_scale(retained)
        if (
            cutoff > opener
            or roster_url != expected_roster_url
            or performance_url != expected_performance_url
            or not math.isclose(turnover, 1.0 - retained, abs_tol=1e-12)
            or not math.isclose(scale, expected_scale, abs_tol=1e-12)
        ):
            raise ValueError(f"Prospective promoted turnover formula or cutoff is invalid for {team_id}")
        output.append(
            PromotedTurnoverScale(
                season, team_id, str(team["canonical_club_id"]), cutoff, retained, turnover, scale, str(path)
            )
        )
    return _scales(output), output


def merge_season_variance_scales(
    *sources: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Merge policy sources without allowing a target season/team to be overwritten."""
    merged: dict[str, dict[str, float]] = {}
    for source in sources:
        for season, teams in source.items():
            destination = merged.setdefault(season, {})
            for team, scale in teams.items():
                if team in destination:
                    raise ValueError(f"Duplicate merged season variance scale: {(season, team)}")
                destination[team] = scale
    return merged
