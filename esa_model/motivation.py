"""Leakage-safe late-season table leverage features."""

import csv
import zlib
from collections import Counter
from dataclasses import dataclass
from datetime import date
from importlib import import_module
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np

from esa_model.baseline import Match


class _MotivationCExtension(Protocol):
    def simulate_conditioned_zone_counts(self, *arrays: np.ndarray) -> np.ndarray: ...


try:
    _motivation_c_module = import_module("esa_model._motivation_c")
    if not hasattr(_motivation_c_module, "simulate_conditioned_zone_counts"):
        raise ImportError("The installed motivation extension is incompatible")
    _motivation_c = cast(_MotivationCExtension, _motivation_c_module)
except ImportError:  # The read-only frontend image does not build the nightly-only extension.
    _motivation_c = None

ObjectiveZone = Literal["title", "second", "third", "midtable", "relegation"]
SimulatedOutcome = Literal["home", "draw", "away"]
ZONES: tuple[ObjectiveZone, ...] = ("title", "second", "third", "midtable", "relegation")


@dataclass(frozen=True, slots=True)
class LeagueRules:
    teams: int
    matches_per_team: int
    european_cutoff: int
    relegation_places: int
    split_after: int | None = None

    @property
    def safe_cutoff(self) -> int:
        return self.teams - self.relegation_places


@dataclass(frozen=True, slots=True)
class PointAdjustment:
    season: str
    team: str
    points: int
    announced_date: date


@dataclass(frozen=True, slots=True)
class RankBounds:
    best: int
    worst: int

    @property
    def width(self) -> int:
        return self.worst - self.best


@dataclass(frozen=True, slots=True)
class TeamTableFeature:
    team: str
    points: int
    played: int
    remaining: int
    best_rank: int
    worst_rank: int
    exact_position_locked: bool
    objective_zone_locked: bool
    bounded_midtable: bool
    zone: ObjectiveZone | None


@dataclass(frozen=True, slots=True)
class FixtureProbability:
    match_id: str
    home: str
    away: str
    p_home: float
    p_draw: float
    p_away: float


def league_rules(season: str) -> LeagueRules:
    """Return rules for seasons covered by the operational xG evaluation."""
    if season == "2018/19":
        return LeagueRules(16, 37, 3, 2, split_after=30)
    if season == "2019/20":
        return LeagueRules(16, 37, 3, 3, split_after=30)
    if season == "2020/21":
        return LeagueRules(16, 30, 3, 1)
    if season >= "2021/22":
        return LeagueRules(18, 34, 3, 3)
    raise ValueError(f"Motivation rules are not audited for {season}")


def load_point_adjustments(path: Path) -> list[PointAdjustment]:
    with path.open(encoding="utf-8", newline="") as source:
        rows = csv.DictReader(source)
        required = {"season", "team_id", "points", "announced_date"}
        missing = required - set(rows.fieldnames or ())
        if missing:
            raise ValueError(f"Missing point-adjustment columns: {', '.join(sorted(missing))}")
        return [
            PointAdjustment(
                row["season"],
                row["team_id"],
                int(row["points"]),
                date.fromisoformat(row["announced_date"]),
            )
            for row in rows
        ]


def objective_zone(rank: int, rules: LeagueRules) -> ObjectiveZone:
    if rank == 1:
        return "title"
    if rank == 2 and rules.european_cutoff >= 2:
        return "second"
    if rank == 3 and rules.european_cutoff >= 3:
        return "third"
    if rank > rules.safe_cutoff:
        return "relegation"
    return "midtable"


def position_bounds(
    team: str,
    points: dict[str, int],
    remaining: dict[str, int],
    rank_groups: tuple[frozenset[str], ...] = (),
) -> RankBounds:
    for group_index, group in enumerate(rank_groups):
        if team not in group:
            continue
        offset = sum(len(candidate) for candidate in rank_groups[:group_index])
        upper = {candidate: points[candidate] + 3 * remaining[candidate] for candidate in group}
        best = offset + 1 + sum(points[candidate] > upper[team] for candidate in group if candidate != team)
        worst = offset + len(group) - sum(points[team] > upper[candidate] for candidate in group if candidate != team)
        return RankBounds(best, worst)
    upper = {candidate: value + 3 * remaining[candidate] for candidate, value in points.items()}
    best = 1 + sum(value > upper[team] for candidate, value in points.items() if candidate != team)
    worst = len(points) - sum(points[team] > value for candidate, value in upper.items() if candidate != team)
    return RankBounds(best, worst)


class TableTracker:
    """Maintain the table visible before a match-date batch is played."""

    def __init__(
        self,
        season: str,
        teams: set[str],
        rules: LeagueRules,
        adjustments: list[PointAdjustment] | None = None,
        split_groups: tuple[frozenset[str], ...] = (),
    ) -> None:
        if len(teams) != rules.teams:
            raise ValueError(f"{season} has {len(teams)} teams, expected {rules.teams}")
        self.season = season
        self.teams = tuple(sorted(teams, key=int))
        self.rules = rules
        self.adjustments = [row for row in adjustments or [] if row.season == season]
        self.played = Counter({team: 0 for team in self.teams})
        self.points = Counter({team: 0 for team in self.teams})
        self.goals_for = Counter({team: 0 for team in self.teams})
        self.goals_against = Counter({team: 0 for team in self.teams})
        self.wins = Counter({team: 0 for team in self.teams})
        self.rank_groups: tuple[frozenset[str], ...] = ()
        self.split_groups = split_groups
        if split_groups and (
            frozenset().union(*split_groups) != frozenset(self.teams)
            or sum(len(group) for group in split_groups) != len(self.teams)
        ):
            raise ValueError("Split groups must partition the season teams")

    def adjusted_points(self, as_of: date) -> dict[str, int]:
        output = dict(self.points)
        for adjustment in self.adjustments:
            if adjustment.announced_date <= as_of:
                output[adjustment.team] += adjustment.points
        return output

    def features(self, as_of: date, midtable_width: int = 4) -> dict[str, TeamTableFeature]:
        points = self.adjusted_points(as_of)
        remaining = {team: self.rules.matches_per_team - self.played[team] for team in self.teams}
        output = {}
        for team in self.teams:
            bounds = position_bounds(team, points, remaining, self.rank_groups)
            best_zone = objective_zone(bounds.best, self.rules)
            worst_zone = objective_zone(bounds.worst, self.rules)
            locked = best_zone == worst_zone
            output[team] = TeamTableFeature(
                team,
                points[team],
                self.played[team],
                remaining[team],
                bounds.best,
                bounds.worst,
                bounds.best == bounds.worst,
                locked,
                locked and best_zone == "midtable" and bounds.width <= midtable_width,
                best_zone if locked else None,
            )
        return output

    def observe(self, matches: list[Match]) -> None:
        if any(match.season != self.season for match in matches):
            raise ValueError("Table batch contains a different season")
        for match in matches:
            self.played[match.home] += 1
            self.played[match.away] += 1
            self.goals_for[match.home] += match.home_goals
            self.goals_for[match.away] += match.away_goals
            self.goals_against[match.home] += match.away_goals
            self.goals_against[match.away] += match.home_goals
            if match.home_goals > match.away_goals:
                self.points[match.home] += 3
                self.wins[match.home] += 1
            elif match.home_goals < match.away_goals:
                self.points[match.away] += 3
                self.wins[match.away] += 1
            else:
                self.points[match.home] += 1
                self.points[match.away] += 1
        if (
            self.rules.split_after is not None
            and not self.rank_groups
            and min(self.played.values()) >= self.rules.split_after
        ):
            if self.split_groups:
                self.rank_groups = self.split_groups
                return
            ordered = sorted(
                self.teams,
                key=lambda team: (
                    -self.points[team],
                    -(self.goals_for[team] - self.goals_against[team]),
                    -self.goals_for[team],
                    -self.wins[team],
                    int(team),
                ),
            )
            split = self.rules.teams // 2
            self.rank_groups = (frozenset(ordered[:split]), frozenset(ordered[split:]))


def simulate_zone_probabilities(
    teams: tuple[str, ...],
    starting_points: dict[str, int],
    fixtures: list[FixtureProbability],
    rules: LeagueRules,
    simulations: int,
    seed: int,
    rank_groups: tuple[frozenset[str], ...] = (),
    conditioned: tuple[str, SimulatedOutcome] | None = None,
) -> dict[str, np.ndarray]:
    """Simulate objective zones with random resolution of final point ties."""
    positions = {team: index for index, team in enumerate(teams)}
    points = np.broadcast_to(
        np.asarray([starting_points[team] for team in teams], dtype=np.int16),
        (simulations, len(teams)),
    ).copy()
    generator = np.random.default_rng(seed)
    conditioned_id, conditioned_outcome = conditioned or (None, None)
    for fixture in fixtures:
        if fixture.match_id == conditioned_id:
            home_win = np.full(simulations, conditioned_outcome == "home")
            draw = np.full(simulations, conditioned_outcome == "draw")
        else:
            sample = generator.random(simulations)
            home_win = sample < fixture.p_home
            draw = (sample >= fixture.p_home) & (sample < fixture.p_home + fixture.p_draw)
        points[:, positions[fixture.home]] += np.where(home_win, 3, np.where(draw, 1, 0)).astype(np.int16)
        points[:, positions[fixture.away]] += np.where(home_win, 0, np.where(draw, 1, 3)).astype(np.int16)

    groups = rank_groups or (frozenset(teams),)
    final_ranks = np.empty_like(points)
    offset = 0
    tie_noise = generator.random(points.shape)
    for group in groups:
        indices = np.asarray([positions[team] for team in teams if team in group])
        order = np.argsort(-(points[:, indices] + tie_noise[:, indices]), axis=1)
        ordered_indices = indices[order]
        ranks = np.arange(offset + 1, offset + len(indices) + 1, dtype=np.int16)
        np.put_along_axis(final_ranks, ordered_indices, np.broadcast_to(ranks, order.shape), axis=1)
        offset += len(indices)

    probabilities = {}
    for team, index in positions.items():
        counts = np.zeros(len(ZONES), dtype=float)
        for rank in range(1, rules.teams + 1):
            counts[ZONES.index(objective_zone(rank, rules))] += np.count_nonzero(final_ranks[:, index] == rank)
        probabilities[team] = counts / simulations
    return probabilities


def simulate_conditioned_zone_probabilities(
    teams: tuple[str, ...],
    starting_points: dict[str, int],
    fixtures: list[FixtureProbability],
    rules: LeagueRules,
    simulations: int,
    seed: int,
    conditioned_ids: list[str],
    rank_groups: tuple[frozenset[str], ...] = (),
    target_teams: set[str] | None = None,
) -> dict[tuple[str, SimulatedOutcome], dict[str, np.ndarray]]:
    """Evaluate forced outcomes, optionally ranking only teams consumed by the caller."""
    positions = {team: index for index, team in enumerate(teams)}
    initial_points = np.asarray([starting_points[team] for team in teams], dtype=np.int16)
    requested_teams = set(teams) if target_teams is None else target_teams
    unknown_teams = requested_teams - positions.keys()
    if unknown_teams:
        raise ValueError(f"Target teams are missing from the league: {sorted(unknown_teams)}")
    requested_team_list = [team for team in teams if team in requested_teams]
    groups = rank_groups or (frozenset(teams),)
    target_ids = set(conditioned_ids)
    target_fixture_indices = {
        fixture.match_id: fixture_index
        for fixture_index, fixture in enumerate(fixtures)
        if fixture.match_id in target_ids
    }
    if target_ids != target_fixture_indices.keys():
        missing = sorted(target_ids - target_fixture_indices.keys())
        raise ValueError(f"Conditioned fixtures are missing from the schedule: {missing}")
    conditions: list[tuple[str, SimulatedOutcome]] = []
    for match_id in conditioned_ids:
        conditions.extend(((match_id, "home"), (match_id, "away")))

    generator = np.random.default_rng(seed)
    if _motivation_c is not None:
        fixture_samples = generator.random((len(fixtures), simulations))
        tie_noise = generator.random((simulations, len(teams)))
        group_indices = [[positions[team] for team in teams if team in group] for group in groups]
        counts = _motivation_c.simulate_conditioned_zone_counts(
            initial_points,
            fixture_samples,
            tie_noise,
            np.asarray([positions[fixture.home] for fixture in fixtures], dtype=np.intp),
            np.asarray([positions[fixture.away] for fixture in fixtures], dtype=np.intp),
            np.asarray([fixture.p_home for fixture in fixtures], dtype=np.float64),
            np.asarray([fixture.p_draw for fixture in fixtures], dtype=np.float64),
            np.asarray([target_fixture_indices[match_id] for match_id in conditioned_ids], dtype=np.intp),
            np.asarray([index for indices in group_indices for index in indices], dtype=np.intp),
            np.cumsum([0, *(len(indices) for indices in group_indices)], dtype=np.intp),
            np.asarray(
                [ZONES.index(objective_zone(rank, rules)) for rank in range(1, rules.teams + 1)],
                dtype=np.intp,
            ),
            np.asarray([positions[team] for team in requested_team_list], dtype=np.intp),
        )
        return {
            condition: {
                team: counts[scenario, team_index] / simulations for team_index, team in enumerate(requested_team_list)
            }
            for scenario, condition in enumerate(conditions)
        }

    points = np.broadcast_to(initial_points, (simulations, len(teams))).copy()
    target_contributions: dict[str, tuple[int, int, np.ndarray, np.ndarray]] = {}
    for fixture in fixtures:
        sample = generator.random(simulations)
        home_win = sample < fixture.p_home
        draw = (sample >= fixture.p_home) & (sample < fixture.p_home + fixture.p_draw)
        home_points = np.where(home_win, 3, np.where(draw, 1, 0)).astype(np.int16)
        away_points = np.where(home_win, 0, np.where(draw, 1, 3)).astype(np.int16)
        home = positions[fixture.home]
        away = positions[fixture.away]
        points[:, home] += home_points
        points[:, away] += away_points
        if fixture.match_id in target_ids:
            target_contributions[fixture.match_id] = (home, away, home_points, away_points)
    if target_ids != target_contributions.keys():
        missing = sorted(target_ids - target_contributions.keys())
        raise ValueError(f"Conditioned fixtures are missing from the schedule: {missing}")

    scenario_points = np.broadcast_to(points, (len(conditions), *points.shape)).copy()
    for scenario, (match_id, outcome) in enumerate(conditions):
        home, away, home_points, away_points = target_contributions[match_id]
        scenario_points[scenario, :, home] -= home_points
        scenario_points[scenario, :, away] -= away_points
        if outcome == "home":
            scenario_points[scenario, :, home] += 3
        else:
            scenario_points[scenario, :, away] += 3

    target_ranks: dict[str, np.ndarray] = {}
    final_ranks = np.empty_like(scenario_points) if requested_teams == set(teams) else None
    tie_noise = generator.random(points.shape)
    offset = 0
    for group in groups:
        indices = np.asarray([positions[team] for team in teams if team in group])
        group_scores = scenario_points[:, :, indices] + tie_noise[None, :, indices]
        if final_ranks is not None:
            order = np.argsort(-group_scores, axis=2)
            ordered_indices = indices[order]
            ranks = np.arange(offset + 1, offset + len(indices) + 1, dtype=np.int16)
            np.put_along_axis(
                final_ranks,
                ordered_indices,
                np.broadcast_to(ranks, order.shape),
                axis=2,
            )
            offset += len(indices)
            continue
        for team in teams:
            if team not in requested_teams or team not in group:
                continue
            team_scores = scenario_points[:, :, positions[team]] + tie_noise[None, :, positions[team]]
            target_ranks[team] = offset + 1 + np.count_nonzero(group_scores > team_scores[:, :, None], axis=2)
        offset += len(indices)

    zone_by_rank = np.asarray([ZONES.index(objective_zone(rank, rules)) for rank in range(1, rules.teams + 1)])
    result = {}
    for scenario, condition in enumerate(conditions):
        result[condition] = {
            team: np.bincount(
                zone_by_rank[
                    (
                        final_ranks[scenario, :, positions[team]]
                        if final_ranks is not None
                        else target_ranks[team][scenario]
                    )
                    - 1
                ],
                minlength=len(ZONES),
            )
            / simulations
            for team in teams
            if team in requested_teams
        }
    return result


def result_leverage(win: np.ndarray, loss: np.ndarray) -> float:
    return float(0.5 * np.abs(win - loss).sum())


def simulation_seed(season: str, as_of: date, suffix: str = "") -> int:
    return zlib.crc32(f"{season}:{as_of.isoformat()}:{suffix}".encode())
