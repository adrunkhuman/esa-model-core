"""Chronological live-season simulation with date-by-date model updates."""

import copy
import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import date

import numpy as np

from esa_model.baseline import Match
from esa_model.derby_scoring_policy import (
    DERBY_SCORING_LOG_RATE_SHIFT,
    adjust_derby_scoring_rates,
    is_adopted_low_scoring_derby,
)
from esa_model.joint_covariance import JointCovarianceDCModel
from esa_model.league_table import TableState, rank_ekstraklasa
from esa_model.league_table import empty_table as _empty_table
from esa_model.league_table import observe_table as _observe_table
from esa_model.live_motivation import current_motivation_scores
from esa_model.motivation import PointAdjustment, TableTracker, league_rules
from esa_model.motivation_policy import MotivationPolicy, MotivationScores
from esa_model.posterior_simulation import conditional_simulation_fixture as _conditional_simulation_fixture
from esa_model.posterior_simulation import sample_fixture as _sample_fixture
from esa_model.relocated_home_policy import RelocatedHomeHfaPolicy, adopted_relocated_home_policy

CHRONOLOGICAL_DEV_SIMULATIONS = 10
MOTIVATION_PATH_SIMULATIONS = 500


@dataclass(frozen=True, slots=True)
class _ChronologicalProcessContext:
    model: JointCovarianceDCModel
    team_ids: list[str]
    fixtures: tuple[Match, ...]
    season: str
    policy: MotivationPolicy | None
    point_adjustments: tuple[PointAdjustment, ...]
    completed_matches: tuple[Match, ...]
    motivation_simulations: int
    future_xg_variance: tuple[float, float]
    relocation_policy: RelocatedHomeHfaPolicy


ChronologicalEventCallback = Callable[[JointCovarianceDCModel, date, np.random.Generator, np.ndarray], None]


_PROCESS_CONTEXT: _ChronologicalProcessContext | None = None


def _initialize_chronological_worker(context: _ChronologicalProcessContext) -> None:
    global _PROCESS_CONTEXT
    _PROCESS_CONTEXT = context


def _simulate_chronological_worker(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if _PROCESS_CONTEXT is None:
        raise RuntimeError("Chronological worker was not initialized")
    context = _PROCESS_CONTEXT
    return simulate_seasons_chronologically(
        context.model,
        context.team_ids,
        context.fixtures,
        context.season,
        context.policy,
        point_adjustments=context.point_adjustments,
        completed_matches=context.completed_matches,
        simulations=1,
        seed=seed,
        motivation_simulations=context.motivation_simulations,
        future_xg_variance=context.future_xg_variance,
        relocation_policy=context.relocation_policy,
    )


def simulate_seasons_chronologically(
    model: JointCovarianceDCModel,
    team_ids: list[str],
    fixtures: Sequence[Match],
    season: str,
    policy: MotivationPolicy | None = None,
    point_adjustments: Sequence[PointAdjustment] | None = None,
    completed_matches: Sequence[Match] = (),
    simulations: int = 1,
    seed: int = 20260716,
    motivation_simulations: int = MOTIVATION_PATH_SIMULATIONS,
    future_xg_variance: tuple[float, float] = (0.0, 0.0),
    workers: int = 1,
    relocation_policy: RelocatedHomeHfaPolicy | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate date-batched paths that update the model after each date.

    Workers preserve per-path seeds. ``relocation_policy=None`` loads the
    adopted operational policy; standalone callers should pass a policy.
    Returns expected points, finish-position percentages, and 10th/90th
    percentile point totals.
    """
    if simulations <= 0:
        raise ValueError("Simulations must be positive")
    if workers <= 0:
        raise ValueError("Workers must be positive")
    if not fixtures:
        raise ValueError("Chronological simulation requires dated fixtures")
    ordered_fixtures = sorted(fixtures, key=lambda match: (match.date, match.match_id))
    if any(match.season != season for match in ordered_fixtures):
        raise ValueError("Chronological fixtures must belong to one season")
    positions = {team: index for index, team in enumerate(team_ids)}
    if len(positions) != len(team_ids):
        raise ValueError("Team IDs must be unique")
    if any(match.home not in positions or match.away not in positions for match in ordered_fixtures):
        raise ValueError("Chronological fixture contains an unknown team")
    completed_matches = tuple(sorted(completed_matches, key=lambda match: (match.date, match.match_id)))
    if any(
        match.season != season or match.home not in positions or match.away not in positions
        for match in completed_matches
    ):
        raise ValueError("Chronological completed match belongs to a different season or team set")
    if {match.match_id for match in completed_matches} & {match.match_id for match in ordered_fixtures}:
        raise ValueError("Chronological completed and remaining matches overlap")
    adjustments = tuple(row for row in point_adjustments or () if row.season == season)
    if any(row.team not in positions for row in adjustments):
        raise ValueError("Point adjustment contains an unknown team")
    if not all(math.isfinite(value) and value >= 0.0 for value in future_xg_variance):
        raise ValueError("Future xG variances must be finite and nonnegative")
    selected_relocation_policy = relocation_policy if relocation_policy is not None else adopted_relocated_home_policy()
    if simulations > 1:
        path_seeds = [seed + path * 1_000_003 for path in range(simulations)]
        context = _ChronologicalProcessContext(
            model,
            team_ids,
            tuple(ordered_fixtures),
            season,
            policy,
            adjustments,
            completed_matches,
            motivation_simulations,
            future_xg_variance,
            selected_relocation_policy,
        )
        if workers > 1:
            with ProcessPoolExecutor(
                max_workers=min(workers, simulations),
                initializer=_initialize_chronological_worker,
                initargs=(context,),
            ) as executor:
                chunksize = max(1, simulations // (min(workers, simulations) * 4))
                path_results = list(executor.map(_simulate_chronological_worker, path_seeds, chunksize=chunksize))
        else:
            path_results = [
                simulate_seasons_chronologically(
                    model,
                    team_ids,
                    ordered_fixtures,
                    season,
                    policy,
                    point_adjustments=adjustments,
                    completed_matches=completed_matches,
                    simulations=1,
                    seed=path_seed,
                    motivation_simulations=motivation_simulations,
                    future_xg_variance=future_xg_variance,
                    relocation_policy=selected_relocation_policy,
                )
                for path_seed in path_seeds
            ]
        point_samples = np.asarray([result[0] for result in path_results])
        place_probabilities = sum((result[1] for result in path_results), np.zeros((len(team_ids), len(team_ids))))
        intervals = np.quantile(point_samples, (0.1, 0.9), axis=0).T
        return point_samples.mean(axis=0), place_probabilities / simulations, intervals

    table, order = _simulate_chronological_path(
        model,
        team_ids,
        ordered_fixtures,
        season,
        policy,
        adjustments,
        seed,
        motivation_simulations,
        future_xg_variance,
        completed_matches=completed_matches,
        relocation_policy=selected_relocation_policy,
    )
    points = table.points.astype(np.int16)
    place_counts = np.zeros((len(team_ids), len(team_ids)), dtype=np.int64)
    for place, team in enumerate(order):
        place_counts[team, place] += 1
    intervals = np.quantile(points[None, :], (0.1, 0.9), axis=0).T
    return points.astype(float), place_counts * 100.0, intervals


def _simulate_chronological_path(
    model: JointCovarianceDCModel,
    team_ids: list[str],
    fixtures: Sequence[Match],
    season: str,
    policy: MotivationPolicy | None,
    adjustments: Sequence[PointAdjustment],
    seed: int,
    motivation_simulations: int,
    future_xg_variance: tuple[float, float],
    *,
    completed_matches: Sequence[Match] = (),
    relocation_policy: RelocatedHomeHfaPolicy | None = None,
    extra_event_dates: Sequence[date] = (),
    event_callback: ChronologicalEventCallback | None = None,
    extra_event_generator: np.random.Generator | None = None,
) -> tuple[TableState, np.ndarray]:
    """Run one league path, optionally exposing date boundaries to side simulations."""
    positions = {team: index for index, team in enumerate(team_ids)}
    selected_relocation_policy = relocation_policy if relocation_policy is not None else adopted_relocated_home_policy()
    ordered_fixtures = sorted(fixtures, key=lambda match: (match.date, match.match_id))
    fixtures_by_date: dict[date, list[Match]] = defaultdict(list)
    for match in ordered_fixtures:
        fixtures_by_date[match.date].append(match)
    generator = np.random.default_rng(seed)
    active_teams = set(team_ids)
    path_model = _copy_chronological_model(model) if type(model) is JointCovarianceDCModel else copy.deepcopy(model)
    completed = list(completed_matches)
    remaining = list(ordered_fixtures)
    table = _empty_table(len(team_ids))
    _observe_table(table, positions, completed)
    rules = league_rules(season)
    motivation_tracker = (
        TableTracker(season, active_teams, rules, list(adjustments))
        if policy is not None and len(active_teams) == rules.teams
        else None
    )
    if motivation_tracker is not None:
        motivation_tracker.observe(completed)
    ordered_adjustments = sorted(adjustments, key=lambda row: row.announced_date)
    next_adjustment = 0
    beta = float(generator.choice(policy.grid, p=policy.beta_weights)) if policy is not None else 0.0
    gamma = float(generator.choice(policy.grid, p=policy.gamma_weights)) if policy is not None else 0.0
    callback_dates = set(extra_event_dates)
    for match_date in sorted(set(fixtures_by_date) | callback_dates):
        date_matches = fixtures_by_date.get(match_date, [])
        while (
            next_adjustment < len(ordered_adjustments)
            and ordered_adjustments[next_adjustment].announced_date <= match_date
        ):
            adjustment = ordered_adjustments[next_adjustment]
            table.points[positions[adjustment.team]] += adjustment.points
            next_adjustment += 1
        path_model.advance_to(match_date, season, active_teams)
        sampled_state = None
        if date_matches:
            sampled_state = path_model.sample_parameter_state(generator)
        elif event_callback is not None:
            # Cup-only dates must not consume the league path's random stream.
            sampled_state = path_model.sample_parameter_state(extra_event_generator or generator)
        if event_callback is not None and match_date in callback_dates:
            if sampled_state is None:
                raise AssertionError("Chronological event callback requires a sampled parameter state")
            event_callback(path_model, match_date, generator, sampled_state)
        if not date_matches:
            continue
        scores = (
            current_motivation_scores(
                path_model,
                season,
                match_date,
                active_teams,
                completed,
                remaining,
                policy,
                simulations=motivation_simulations,
                point_adjustments=list(adjustments),
                target_matches=date_matches,
                tracker=motivation_tracker,
            )
            if policy is not None
            else {}
        )
        if sampled_state is None:
            raise AssertionError("League date requires a sampled parameter state")
        update_offsets = {}
        for match in date_matches:
            predictive_rates = path_model.goal_rates(match)
            update_rates = path_model.goal_rates(match, strength_shrinkage=1.0)
            update_offsets[match.match_id] = (
                predictive_rates.home_log_rate_mean - update_rates.home_log_rate_mean,
                predictive_rates.away_log_rate_mean - update_rates.away_log_rate_mean,
            )
            if is_adopted_low_scoring_derby(match, getattr(path_model, "canonical_club_ids", None)):
                home_offset, away_offset = update_offsets[match.match_id]
                update_offsets[match.match_id] = (
                    home_offset + DERBY_SCORING_LOG_RATE_SHIFT,
                    away_offset + DERBY_SCORING_LOG_RATE_SHIFT,
                )
        # Build the complete date batch before observing any result from that date.
        predictions = [
            (
                match,
                _conditional_simulation_fixture(
                    adjust_derby_scoring_rates(
                        match,
                        selected_relocation_policy.adjust_rates(
                            match,
                            path_model.goal_rates_from_state(match, sampled_state),
                            path_model.home_advantage_from_state(sampled_state),
                        ),
                        getattr(path_model, "canonical_club_ids", None),
                    ),
                    path_model.globals.rho,
                    positions[match.home],
                    positions[match.away],
                    scores.get(match.match_id, MotivationScores()),
                    beta,
                    gamma,
                ),
            )
            for match in date_matches
        ]
        realized = []
        simulated_xg = {}
        simulated_xg_variances = {}
        for match, forecast in predictions:
            realized.append(_sample_fixture(match, forecast, generator))
            if forecast.score_matrix is None:
                raise ValueError("Chronological simulation requires score distributions")
            if forecast.home_rate is None or forecast.away_rate is None:
                raise ValueError("Chronological simulation requires conditional scoring rates")
            simulated_xg[match.match_id] = (
                _sample_xg(forecast.home_rate, future_xg_variance[0], generator),
                _sample_xg(forecast.away_rate, future_xg_variance[1], generator),
            )
            simulated_xg_variances[match.match_id] = future_xg_variance
        _observe_table(table, positions, realized)
        if motivation_tracker is not None:
            motivation_tracker.observe(realized)
        path_model.observe_with_xg(realized, simulated_xg, simulated_xg_variances, update_offsets)
        completed.extend(realized)
        played_ids = {match.match_id for match in date_matches}
        remaining = [match for match in remaining if match.match_id not in played_ids]
    order = rank_ekstraklasa(
        table.points,
        table.goals_for,
        table.goals_against,
        table.wins,
        table.away_wins,
        table.head_to_head_points,
        table.head_to_head_goal_difference,
        generator.random(len(team_ids)),
    )
    return table, order


class _ChronologicalSimulationModel(JointCovarianceDCModel):
    def _project_attack_gauge(self) -> None:
        self._pull_states()
        attack_positions = [self.index["attack", team] for team in sorted(self.active_teams)]
        weights = np.zeros(len(self.parameters))
        weights[attack_positions] = 1.0 / len(attack_positions)
        affected = np.zeros(len(self.parameters))
        for team in self.active_teams:
            affected[self.index["attack", team]] = 1.0
            affected[self.index["defense", team]] = 1.0
        covariance_weights = self.covariance @ weights
        weighted_covariance = weights @ self.covariance
        weighted_variance = float(weights @ covariance_weights)
        self.mean -= affected * float(weights @ self.mean)
        self.covariance -= np.outer(affected, weighted_covariance)
        self.covariance -= np.outer(covariance_weights, affected)
        self.covariance += weighted_variance * np.outer(affected, affected)
        self.covariance = (self.covariance + self.covariance.T) / 2.0


def _copy_chronological_model(model: JointCovarianceDCModel) -> JointCovarianceDCModel:
    clone = object.__new__(_ChronologicalSimulationModel)
    clone.__dict__ = model.__dict__.copy()
    clone.parameters = list(model.parameters)
    clone.index = dict(model.index)
    clone.mean = model.mean.copy()
    clone.covariance = model.covariance.copy()
    clone.states = {team: replace(state) for team, state in model.states.items()}
    clone.active_teams = set(model.active_teams)
    clone.adjusted_seasons = set(model.adjusted_seasons)
    clone.variance_scaled_seasons = set(model.variance_scaled_seasons)
    return clone


def _sample_xg(rate: float, variance: float, generator: np.random.Generator) -> float:
    if variance == 0.0:
        return rate
    return float(generator.gamma(rate**2 / variance, variance / rate))
