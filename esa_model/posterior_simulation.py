"""Season simulation from static posterior team-strength draws."""

import math
from collections.abc import Sequence

import numpy as np

from esa_model.baseline import MAX_GOALS, Match
from esa_model.derby_scoring_policy import (
    DERBY_SCORING_LOG_RATE_SHIFT,
    adjust_derby_scoring_rates,
    is_adopted_low_scoring_derby,
)
from esa_model.joint_covariance import JointCovarianceDCModel
from esa_model.league_table import (
    SIMULATIONS,
    SimulationFixture,
    TableState,
    _broadcast_pair_stat,
    _broadcast_stat,
    copy_table,
    empty_table,
    observe_table,
    rank_ekstraklasa,
    rank_ekstraklasa_batch,
)
from esa_model.motivation_policy import MotivationScores
from esa_model.relocated_home_policy import RelocatedHomeHfaPolicy, adopted_relocated_home_policy
from esa_model.stage1 import GoalRateForecast, dixon_coles_score_matrix


def conditional_simulation_fixture(
    rates: GoalRateForecast,
    rho: float,
    home: int,
    away: int,
    scores: MotivationScores,
    beta: float,
    gamma: float,
) -> SimulationFixture:
    settled_contrast = scores.away_settled - scores.home_settled
    urgency_contrast = scores.home_urgency - scores.away_urgency
    shift = beta * settled_contrast + gamma * urgency_contrast
    home_rate = math.exp(rates.home_log_rate_mean + shift)
    away_rate = math.exp(rates.away_log_rate_mean - shift)
    matrix = dixon_coles_score_matrix(home_rate, away_rate, rho)
    return SimulationFixture(
        home,
        away,
        float(np.tril(matrix, -1).sum()),
        float(np.trace(matrix)),
        float(np.triu(matrix, 1).sum()),
        matrix,
        home_rate,
        away_rate,
    )


def sample_fixture(match: Match, fixture: SimulationFixture, generator: np.random.Generator) -> Match:
    if fixture.score_matrix is None:
        raise ValueError("Chronological simulation requires score distributions")
    cdf = np.cumsum(fixture.score_matrix.ravel())
    cdf[-1] = 1.0
    score = int(np.searchsorted(cdf, generator.random()))
    return Match(
        match.match_id,
        match.date,
        match.season,
        match.home,
        match.away,
        score // fixture.score_matrix.shape[1],
        score % fixture.score_matrix.shape[1],
    )


def sample_conditional_scores(
    home_rates: np.ndarray,
    away_rates: np.ndarray,
    rho: float,
    generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample Dixon-Coles scores for a vector of conditional scoring rates."""
    if rho < 0.0:
        cap = -1.0 / rho * (1.0 - 1e-12)
    elif rho > 0.0:
        cap = math.sqrt(1.0 / rho) * (1.0 - 1e-12)
    else:
        cap = math.inf
    home_rates = np.minimum(home_rates, cap)
    away_rates = np.minimum(away_rates, cap)
    goals = np.arange(MAX_GOALS + 1)
    factorials = np.asarray([math.factorial(int(goal)) for goal in goals])
    home_pmf = np.exp(-home_rates[:, None]) * home_rates[:, None] ** goals / factorials
    away_pmf = np.exp(-away_rates[:, None]) * away_rates[:, None] ** goals / factorials
    matrices = home_pmf[:, :, None] * away_pmf[:, None, :]
    matrices[:, 0, 0] *= 1.0 - home_rates * away_rates * rho
    matrices[:, 0, 1] *= 1.0 + home_rates * rho
    matrices[:, 1, 0] *= 1.0 + away_rates * rho
    matrices[:, 1, 1] *= 1.0 - rho
    if np.any(matrices < -1e-12):
        raise ValueError("Dixon-Coles correction produced a negative outcome probability")
    cdf = np.cumsum(matrices.reshape(len(home_rates), -1), axis=1)
    cdf /= cdf[:, -1, None]
    sampled = np.sum(cdf < generator.random(len(home_rates))[:, None], axis=1)
    width = matrices.shape[2]
    return (sampled // width).astype(np.int16), (sampled % width).astype(np.int16)


def simulate_seasons_from_posterior(
    model: JointCovarianceDCModel,
    team_ids: list[str],
    fixtures: Sequence[Match],
    starting_table: TableState | None = None,
    simulations: int = SIMULATIONS,
    seed: int = 20260716,
    relocation_policy: RelocatedHomeHfaPolicy | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate exact tables with one fixed posterior state per season path.

    Unlike chronological simulation, results do not update team strength within
    the path. ``relocation_policy=None`` loads the adopted operational policy;
    standalone callers should pass a policy. Returns expected points,
    finish-position percentages, and 10th/90th percentile point totals.
    """
    if simulations <= 0:
        raise ValueError("Simulations must be positive")
    positions = {team: index for index, team in enumerate(team_ids)}
    if len(positions) != len(team_ids):
        raise ValueError("Team IDs must be unique")
    if any(match.home not in positions or match.away not in positions for match in fixtures):
        raise ValueError("Simulation fixture contains an unknown team")
    selected_relocation_policy = relocation_policy if relocation_policy is not None else adopted_relocated_home_policy()

    if type(model) is JointCovarianceDCModel:
        return _simulate_joint_covariance_seasons_from_posterior(
            model,
            team_ids,
            fixtures,
            positions,
            starting_table,
            simulations,
            seed,
            selected_relocation_policy,
        )

    generator = np.random.default_rng(seed)
    point_samples = np.empty((simulations, len(team_ids)), dtype=np.int16)
    place_counts = np.zeros((len(team_ids), len(team_ids)), dtype=np.int64)
    initial_table = starting_table or empty_table(len(team_ids))
    for simulation in range(simulations):
        table = copy_table(initial_table)
        sampled_state = model.sample_parameter_state(generator)
        for match in fixtures:
            rates = selected_relocation_policy.adjust_rates(
                match,
                model.goal_rates_from_state(match, sampled_state),
                model.home_advantage_from_state(sampled_state),
            )
            rates = adjust_derby_scoring_rates(match, rates, getattr(model, "canonical_club_ids", None))
            forecast = conditional_simulation_fixture(
                rates,
                model.globals.rho,
                positions[match.home],
                positions[match.away],
                MotivationScores(),
                0.0,
                0.0,
            )
            observe_table(table, positions, [sample_fixture(match, forecast, generator)])

        point_samples[simulation] = table.points
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
        for place, team in enumerate(order):
            place_counts[team, place] += 1

    intervals = np.quantile(point_samples, (0.1, 0.9), axis=0).T
    return point_samples.mean(axis=0), place_counts / simulations * 100.0, intervals


def _simulate_joint_covariance_seasons_from_posterior(
    model: JointCovarianceDCModel,
    team_ids: list[str],
    fixtures: Sequence[Match],
    positions: dict[str, int],
    starting_table: TableState | None,
    simulations: int,
    seed: int,
    relocation_policy: RelocatedHomeHfaPolicy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run static posterior paths in arrays, factoring the shared covariance once."""
    generator = np.random.default_rng(seed)
    model._pull_states()
    batch_size = 5_000
    covariance_factor = None if simulations <= batch_size else _covariance_factor(model.covariance)
    point_samples = np.empty((simulations, len(team_ids)), dtype=np.int16)
    place_counts = np.zeros((len(team_ids), len(team_ids)), dtype=np.int64)
    for offset in range(0, simulations, batch_size):
        batch = min(batch_size, simulations - offset)
        batch_states = (
            generator.multivariate_normal(
                model.mean,
                model.covariance,
                size=batch,
                check_valid="raise",
            )
            if covariance_factor is None
            else model.mean + generator.standard_normal((batch, len(model.mean))) @ covariance_factor.T
        )
        points, order = _simulate_joint_covariance_posterior_batch(
            model,
            team_ids,
            fixtures,
            positions,
            starting_table,
            batch_states,
            generator,
            relocation_policy,
        )
        point_samples[offset : offset + len(batch_states)] = points
        for place in range(len(team_ids)):
            np.add.at(place_counts[:, place], order[:, place], 1)
    intervals = np.quantile(point_samples, (0.1, 0.9), axis=0).T
    return point_samples.mean(axis=0), place_counts / simulations * 100.0, intervals


def _covariance_factor(covariance: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = max(float(np.max(np.abs(eigenvalues))), 1.0) * 1e-10
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError("Covariance is not positive-semidefinite")
    return eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))


def _simulate_joint_covariance_posterior_batch(
    model: JointCovarianceDCModel,
    team_ids: list[str],
    fixtures: Sequence[Match],
    positions: dict[str, int],
    starting_table: TableState | None,
    sampled_states: np.ndarray,
    generator: np.random.Generator,
    relocation_policy: RelocatedHomeHfaPolicy,
) -> tuple[np.ndarray, np.ndarray]:
    simulations = len(sampled_states)
    deviations = sampled_states - model.mean
    initial = starting_table or empty_table(len(team_ids))
    points = _broadcast_stat(initial.points, simulations, len(team_ids))
    goals_for = _broadcast_stat(initial.goals_for, simulations, len(team_ids))
    goals_against = _broadcast_stat(initial.goals_against, simulations, len(team_ids))
    wins = _broadcast_stat(initial.wins, simulations, len(team_ids))
    away_wins = _broadcast_stat(initial.away_wins, simulations, len(team_ids))
    head_to_head_points = _broadcast_pair_stat(initial.head_to_head_points, simulations, len(team_ids))
    head_to_head_goal_difference = _broadcast_pair_stat(
        initial.head_to_head_goal_difference,
        simulations,
        len(team_ids),
    )
    sampled_home_advantage = (
        sampled_states[:, model.index["home_advantage", ""]]
        if model.dynamic_home_advantage
        else np.full(simulations, model.home_advantage_mean)
    )

    for match in fixtures:
        marginal = model.goal_rates(match)
        home_log_rates = marginal.home_log_rate_mean + deviations @ model._design(match, True)
        away_log_rates = marginal.away_log_rate_mean + deviations @ model._design(match, False)
        home_log_rates += relocation_policy.adjustment(match, 1.0) * sampled_home_advantage
        if is_adopted_low_scoring_derby(match, model.canonical_club_ids):
            home_log_rates += DERBY_SCORING_LOG_RATE_SHIFT
            away_log_rates += DERBY_SCORING_LOG_RATE_SHIFT
        home_goals, away_goals = sample_conditional_scores(
            np.exp(home_log_rates),
            np.exp(away_log_rates),
            model.globals.rho,
            generator,
        )
        home = positions[match.home]
        away = positions[match.away]
        home_win = home_goals > away_goals
        tied = home_goals == away_goals
        away_win = ~home_win & ~tied
        home_points = np.where(home_win, 3, np.where(tied, 1, 0)).astype(np.int16)
        away_points_for_match = np.where(away_win, 3, np.where(tied, 1, 0)).astype(np.int16)
        points[:, home] += home_points
        points[:, away] += away_points_for_match
        goals_for[:, home] += home_goals
        goals_for[:, away] += away_goals
        goals_against[:, home] += away_goals
        goals_against[:, away] += home_goals
        wins[:, home] += home_win
        wins[:, away] += away_win
        away_wins[:, away] += away_win
        head_to_head_points[:, home, away] += home_points
        head_to_head_points[:, away, home] += away_points_for_match
        goal_difference = home_goals - away_goals
        head_to_head_goal_difference[:, home, away] += goal_difference
        head_to_head_goal_difference[:, away, home] -= goal_difference

    order = rank_ekstraklasa_batch(
        points,
        goals_for,
        goals_against,
        wins,
        away_wins,
        head_to_head_points,
        head_to_head_goal_difference,
        generator.random((simulations, len(team_ids))),
    )
    return points, order
