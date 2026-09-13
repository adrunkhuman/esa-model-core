"""Causal motivation scores for live match forecasts."""

from collections import defaultdict
from collections.abc import Sequence
from datetime import date

import numpy as np

from esa_model.baseline import Match
from esa_model.joint_covariance import JointCovarianceDCModel
from esa_model.live_forecast import operational_forecast
from esa_model.motivation import (
    ZONES,
    FixtureProbability,
    PointAdjustment,
    SimulatedOutcome,
    TableTracker,
    league_rules,
    result_leverage,
    simulate_conditioned_zone_probabilities,
    simulation_seed,
)
from esa_model.motivation_policy import MotivationPolicy, MotivationScores


def current_motivation_scores(
    model: JointCovarianceDCModel,
    season: str,
    as_of: date,
    teams: set[str],
    completed: list[Match],
    remaining: list[Match],
    policy: MotivationPolicy,
    simulations: int = 4_000,
    point_adjustments: list[PointAdjustment] | None = None,
    target_matches: Sequence[Match] | None = None,
    tracker: TableTracker | None = None,
    precomputed_fixture_probabilities: Sequence[FixtureProbability] | None = None,
) -> dict[str, MotivationScores]:
    """Calculate causal match-specific scores from the current table and remaining schedule."""
    if tracker is None:
        tracker = TableTracker(season, teams, league_rules(season), point_adjustments)
        completed_by_date: dict[date, list[Match]] = defaultdict(list)
        for match in completed:
            completed_by_date[match.date].append(match)
        for match_date in sorted(completed_by_date):
            tracker.observe(completed_by_date[match_date])
    targets = remaining if target_matches is None else target_matches
    remaining_counts = {team: tracker.rules.matches_per_team - tracker.played[team] for team in tracker.teams}
    if all(count > policy.urgency_remaining_gate for count in remaining_counts.values()):
        return {match.match_id: MotivationScores() for match in targets}

    if precomputed_fixture_probabilities is None:
        fixtures = []
        for match in remaining:
            forecast = operational_forecast(model, match)
            fixtures.append(
                FixtureProbability(
                    match.match_id,
                    match.home,
                    match.away,
                    forecast.p_home,
                    forecast.p_draw,
                    forecast.p_away,
                )
            )
    else:
        fixtures = list(precomputed_fixture_probabilities)
        if [fixture.match_id for fixture in fixtures] != [match.match_id for match in remaining]:
            raise ValueError("Precomputed motivation fixtures do not match the remaining schedule")
    points = tracker.adjusted_points(as_of)
    conditioned = simulate_conditioned_zone_probabilities(
        tracker.teams,
        points,
        fixtures,
        tracker.rules,
        simulations,
        simulation_seed(season, as_of, "all-targets"),
        [match.match_id for match in targets],
        tracker.rank_groups,
        {team for match in targets for team in (match.home, match.away)},
    )
    return _motivation_scores(targets, conditioned, remaining_counts, policy)


def _motivation_scores(
    targets: Sequence[Match],
    conditioned: dict[tuple[str, SimulatedOutcome], dict[str, np.ndarray]],
    remaining_counts: dict[str, int],
    policy: MotivationPolicy,
) -> dict[str, MotivationScores]:
    output = {}
    for match in targets:
        home_win = conditioned[match.match_id, "home"]
        away_win = conditioned[match.match_id, "away"]

        def team_scores(team: str, win: np.ndarray, loss: np.ndarray) -> tuple[float, float]:
            settled = 0.0
            if remaining_counts[team] <= policy.settled_remaining_gate:
                leverage = result_leverage(win, loss)
                settled = max(0.0, 1.0 - leverage / policy.settled_scale)
            urgency = 0.0
            if remaining_counts[team] <= policy.urgency_remaining_gate:
                objective_win = (
                    win[ZONES.index("title")],
                    sum(win[ZONES.index(zone)] for zone in ("title", "second", "third")),
                    1.0 - win[ZONES.index("relegation")],
                )
                objective_loss = (
                    loss[ZONES.index("title")],
                    sum(loss[ZONES.index(zone)] for zone in ("title", "second", "third")),
                    1.0 - loss[ZONES.index("relegation")],
                )
                urgency = min(
                    1.0,
                    max(
                        max(0.0, float(win_probability - loss_probability))
                        for win_probability, loss_probability in zip(objective_win, objective_loss, strict=True)
                    )
                    / policy.urgency_scale,
                )
            return settled, urgency

        home_settled, home_urgency = team_scores(match.home, home_win[match.home], away_win[match.home])
        away_settled, away_urgency = team_scores(match.away, away_win[match.away], home_win[match.away])
        output[match.match_id] = MotivationScores(
            home_settled,
            away_settled,
            home_urgency,
            away_urgency,
        )
    return output
