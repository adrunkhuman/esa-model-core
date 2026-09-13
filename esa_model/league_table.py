"""Ekstraklasa table state, ranking, and static season simulation."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from esa_model.baseline import Match

SIMULATIONS = 50_000


@dataclass(frozen=True, slots=True)
class SimulationFixture:
    home: int
    away: int
    p_home: float
    p_draw: float
    p_away: float
    score_matrix: np.ndarray | None = None
    home_rate: float | None = None
    away_rate: float | None = None


@dataclass(frozen=True, slots=True)
class TableState:
    points: np.ndarray
    goals_for: np.ndarray
    goals_against: np.ndarray
    wins: np.ndarray
    away_wins: np.ndarray
    head_to_head_points: np.ndarray
    head_to_head_goal_difference: np.ndarray


def empty_table(team_count: int) -> TableState:
    return TableState(
        np.zeros(team_count, dtype=np.int16),
        np.zeros(team_count, dtype=np.int16),
        np.zeros(team_count, dtype=np.int16),
        np.zeros(team_count, dtype=np.int16),
        np.zeros(team_count, dtype=np.int16),
        np.zeros((team_count, team_count), dtype=np.int16),
        np.zeros((team_count, team_count), dtype=np.int16),
    )


def copy_table(table: TableState) -> TableState:
    return TableState(
        table.points.copy(),
        table.goals_for.copy(),
        table.goals_against.copy(),
        table.wins.copy(),
        table.away_wins.copy(),
        table.head_to_head_points.copy(),
        table.head_to_head_goal_difference.copy(),
    )


def observe_table(table: TableState, positions: dict[str, int], matches: Sequence[Match]) -> None:
    for match in matches:
        home = positions[match.home]
        away = positions[match.away]
        home_win = match.home_goals > match.away_goals
        draw = match.home_goals == match.away_goals
        away_win = not home_win and not draw
        home_points = 3 if home_win else 1 if draw else 0
        away_points = 3 if away_win else 1 if draw else 0
        table.points[home] += home_points
        table.points[away] += away_points
        table.goals_for[home] += match.home_goals
        table.goals_for[away] += match.away_goals
        table.goals_against[home] += match.away_goals
        table.goals_against[away] += match.home_goals
        table.wins[home] += home_win
        table.wins[away] += away_win
        table.away_wins[away] += away_win
        table.head_to_head_points[home, away] += home_points
        table.head_to_head_points[away, home] += away_points
        difference = match.home_goals - match.away_goals
        table.head_to_head_goal_difference[home, away] += difference
        table.head_to_head_goal_difference[away, home] -= difference


def table_state_from_matches(
    team_ids: list[str], matches: list[Match], point_adjustments: dict[str, int]
) -> TableState:
    positions = {team: index for index, team in enumerate(team_ids)}
    size = len(team_ids)
    points = np.zeros(size, dtype=np.int16)
    goals_for = np.zeros(size, dtype=np.int16)
    goals_against = np.zeros(size, dtype=np.int16)
    wins = np.zeros(size, dtype=np.int16)
    away_wins = np.zeros(size, dtype=np.int16)
    head_to_head_points = np.zeros((size, size), dtype=np.int16)
    head_to_head_goal_difference = np.zeros((size, size), dtype=np.int16)
    for team, adjustment in point_adjustments.items():
        points[positions[team]] += adjustment
    for match in matches:
        home = positions[match.home]
        away = positions[match.away]
        home_points = 3 if match.home_goals > match.away_goals else 1 if match.home_goals == match.away_goals else 0
        away_points = 3 if match.away_goals > match.home_goals else 1 if match.home_goals == match.away_goals else 0
        points[home] += home_points
        points[away] += away_points
        goals_for[home] += match.home_goals
        goals_for[away] += match.away_goals
        goals_against[home] += match.away_goals
        goals_against[away] += match.home_goals
        wins[home] += match.home_goals > match.away_goals
        wins[away] += match.away_goals > match.home_goals
        away_wins[away] += match.away_goals > match.home_goals
        head_to_head_points[home, away] += home_points
        head_to_head_points[away, home] += away_points
        difference = match.home_goals - match.away_goals
        head_to_head_goal_difference[home, away] += difference
        head_to_head_goal_difference[away, home] -= difference
    return TableState(
        points,
        goals_for,
        goals_against,
        wins,
        away_wins,
        head_to_head_points,
        head_to_head_goal_difference,
    )


def _broadcast_stat(values: np.ndarray | None, batch: int, team_count: int) -> np.ndarray:
    initial = np.zeros(team_count, dtype=np.int16) if values is None else values.astype(np.int16)
    return np.broadcast_to(initial, (batch, team_count)).copy()


def _broadcast_pair_stat(values: np.ndarray | None, batch: int, team_count: int) -> np.ndarray:
    initial = np.zeros((team_count, team_count), dtype=np.int16) if values is None else values.astype(np.int16)
    return np.broadcast_to(initial, (batch, team_count, team_count)).copy()


def _fixture_values(
    fixture: SimulationFixture | tuple[int, int, float, float, float],
) -> tuple[int, int, float, float, float, np.ndarray | None]:
    if isinstance(fixture, SimulationFixture):
        return fixture.home, fixture.away, fixture.p_home, fixture.p_draw, fixture.p_away, fixture.score_matrix
    return *fixture, None


def rank_ekstraklasa(
    points: np.ndarray,
    goals_for: np.ndarray,
    goals_against: np.ndarray,
    wins: np.ndarray,
    away_wins: np.ndarray,
    head_to_head_points: np.ndarray,
    head_to_head_goal_difference: np.ndarray,
    random_values: np.ndarray,
) -> np.ndarray:
    """Rank one final table, omitting the unavailable discipline and fair-play criteria."""
    team_count = len(points)
    mini_points = np.zeros(team_count, dtype=np.int16)
    mini_goal_difference = np.zeros(team_count, dtype=np.int16)
    for point_total in np.unique(points):
        tied = np.flatnonzero(points == point_total)
        if len(tied) > 1:
            mini_points[tied] = head_to_head_points[np.ix_(tied, tied)].sum(axis=1)
            mini_goal_difference[tied] = head_to_head_goal_difference[np.ix_(tied, tied)].sum(axis=1)
    # Encodes the published tie-break order; cards and fair play are unavailable in the model data.
    return np.lexsort(
        (
            random_values,
            -away_wins,
            -wins,
            -goals_for,
            -(goals_for - goals_against),
            -mini_goal_difference,
            -mini_points,
            -points,
        )
    )


def rank_ekstraklasa_batch(
    points: np.ndarray,
    goals_for: np.ndarray,
    goals_against: np.ndarray,
    wins: np.ndarray,
    away_wins: np.ndarray,
    head_to_head_points: np.ndarray,
    head_to_head_goal_difference: np.ndarray,
    random_values: np.ndarray,
) -> np.ndarray:
    """Rank a batch of final tables with the same tie-breaks as ``rank_ekstraklasa``."""
    tied = points[:, :, None] == points[:, None, :]
    mini_points = np.sum(head_to_head_points * tied, axis=2)
    mini_goal_difference = np.sum(head_to_head_goal_difference * tied, axis=2)
    return np.lexsort(
        (
            random_values,
            -away_wins,
            -wins,
            -goals_for,
            -(goals_for - goals_against),
            -mini_goal_difference,
            -mini_points,
            -points,
        ),
        axis=1,
    )


def expected_points_from_fixtures(
    team_count: int, fixtures: Sequence[SimulationFixture | tuple[int, int, float, float, float]]
) -> np.ndarray:
    expected_points = np.zeros(team_count)
    for fixture in fixtures:
        home, away, p_home, p_draw, p_away, _ = _fixture_values(fixture)
        expected_points[home] += 3.0 * p_home + p_draw
        expected_points[away] += 3.0 * p_away + p_draw
    return expected_points


def simulate_seasons(
    team_count: int,
    fixtures: Sequence[SimulationFixture | tuple[int, int, float, float, float]],
    simulations: int = SIMULATIONS,
    seed: int = 20260716,
    starting_points: np.ndarray | None = None,
    starting_table: TableState | None = None,
    batch_size: int = 5_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if starting_points is not None and starting_table is not None:
        raise ValueError("Supply starting_points or starting_table, not both")
    initial = (
        starting_table.points
        if starting_table is not None
        else np.zeros(team_count)
        if starting_points is None
        else np.asarray(starting_points, dtype=float)
    )
    if initial.shape != (team_count,):
        raise ValueError("Starting points must contain one value per team")
    expected_points = initial + expected_points_from_fixtures(team_count, fixtures)

    generator = np.random.default_rng(seed)
    place_counts = np.zeros((team_count, team_count), dtype=np.int64)
    point_samples = np.empty((simulations, team_count), dtype=np.int16)
    score_cdfs: dict[int, np.ndarray] = {}
    remaining = simulations
    offset = 0
    while remaining:
        batch = min(batch_size, remaining)
        points = np.broadcast_to(initial.astype(np.int16), (batch, team_count)).copy()
        goals_for = _broadcast_stat(starting_table.goals_for if starting_table else None, batch, team_count)
        goals_against = _broadcast_stat(starting_table.goals_against if starting_table else None, batch, team_count)
        wins = _broadcast_stat(starting_table.wins if starting_table else None, batch, team_count)
        away_wins = _broadcast_stat(starting_table.away_wins if starting_table else None, batch, team_count)
        head_to_head_points = _broadcast_pair_stat(
            starting_table.head_to_head_points if starting_table else None, batch, team_count
        )
        head_to_head_goal_difference = _broadcast_pair_stat(
            starting_table.head_to_head_goal_difference if starting_table else None, batch, team_count
        )
        for fixture in fixtures:
            home, away, p_home, p_draw, _, score_matrix = _fixture_values(fixture)
            if score_matrix is None:
                draw = generator.random(batch)
                home_win = draw < p_home
                tied = (draw >= p_home) & (draw < p_home + p_draw)
                home_goals = home_win.astype(np.int16)
                away_goals = (~home_win & ~tied).astype(np.int16)
            else:
                score_cdf = score_cdfs.get(id(score_matrix))
                if score_cdf is None:
                    score_cdf = np.cumsum(score_matrix.ravel())
                    score_cdf[-1] = 1.0
                    score_cdfs[id(score_matrix)] = score_cdf
                scores = np.searchsorted(score_cdf, generator.random(batch))
                home_goals = (scores // score_matrix.shape[1]).astype(np.int16)
                away_goals = (scores % score_matrix.shape[1]).astype(np.int16)
                home_win = home_goals > away_goals
                tied = home_goals == away_goals
            home_points = np.where(home_win, 3, np.where(tied, 1, 0)).astype(np.int16)
            away_points = np.where(home_win, 0, np.where(tied, 1, 3)).astype(np.int16)
            points[:, home] += home_points
            points[:, away] += away_points
            goals_for[:, home] += home_goals
            goals_for[:, away] += away_goals
            goals_against[:, home] += away_goals
            goals_against[:, away] += home_goals
            wins[:, home] += home_win
            wins[:, away] += ~home_win & ~tied
            away_wins[:, away] += ~home_win & ~tied
            head_to_head_points[:, home, away] += home_points
            head_to_head_points[:, away, home] += away_points
            goal_difference = home_goals - away_goals
            head_to_head_goal_difference[:, home, away] += goal_difference
            head_to_head_goal_difference[:, away, home] -= goal_difference
        random_values = generator.random((batch, team_count))
        order = rank_ekstraklasa_batch(
            points,
            goals_for,
            goals_against,
            wins,
            away_wins,
            head_to_head_points,
            head_to_head_goal_difference,
            random_values,
        )
        for place in range(team_count):
            np.add.at(place_counts[:, place], order[:, place], 1)
        point_samples[offset : offset + batch] = points
        offset += batch
        remaining -= batch
    intervals = np.quantile(point_samples, (0.1, 0.9), axis=0).T
    return expected_points, place_counts / simulations * 100.0, intervals
