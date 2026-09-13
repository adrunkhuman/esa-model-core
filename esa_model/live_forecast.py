"""Operational match forecasts and probability attribution."""

import math

from esa_model.baseline import Match
from esa_model.derby_scoring_policy import adjust_derby_scoring_rates
from esa_model.evaluation import MatchForecast
from esa_model.joint_covariance import JointCovarianceDCModel
from esa_model.live_contracts import ProbabilityAdjustment, ProbabilityBreakdown
from esa_model.live_probability import live_uncertain_probabilities
from esa_model.motivation_policy import MotivationPolicy, MotivationScores
from esa_model.relocated_home_policy import relocated_home_goal_rates
from esa_model.stage1 import GoalRateForecast


def operational_goal_rates(model: JointCovarianceDCModel, match: Match) -> GoalRateForecast:
    rates = relocated_home_goal_rates(model, match)
    return adjust_derby_scoring_rates(match, rates, getattr(model, "canonical_club_ids", None))


def operational_forecast(model: JointCovarianceDCModel, match: Match) -> MatchForecast:
    rates = operational_goal_rates(model, match)
    probability = live_uncertain_probabilities(
        rates.home_log_rate_mean,
        rates.home_log_rate_variance,
        rates.away_log_rate_mean,
        rates.away_log_rate_variance,
        model.globals.rho,
        home_away_log_rate_covariance=rates.home_away_log_rate_covariance,
    )
    return MatchForecast(probability[3], probability[4], *probability[:3])


def _rate_probabilities(rates: GoalRateForecast, rho: float) -> tuple[float, float, float]:
    probability = live_uncertain_probabilities(
        rates.home_log_rate_mean,
        rates.home_log_rate_variance,
        rates.away_log_rate_mean,
        rates.away_log_rate_variance,
        rho,
        home_away_log_rate_covariance=rates.home_away_log_rate_covariance,
    )
    return probability[0], probability[1], probability[2]


def _probability_delta(
    treatment: tuple[float, float, float], parent: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        treatment[0] - parent[0],
        treatment[1] - parent[1],
        treatment[2] - parent[2],
    )


def match_probability_breakdown(
    model: JointCovarianceDCModel,
    match: Match,
    home_name: str,
    away_name: str,
    policy: MotivationPolicy | None = None,
    scores: MotivationScores | None = None,
) -> tuple[tuple[float, float, float], ProbabilityBreakdown]:
    """Attribute a forecast to a neutral certain-rate base and additive match effects."""
    regular_rates = model.goal_rates(match)
    no_home_advantage_rates = model.goal_rates_with_home_advantage_scale(match, 0.0)
    relocated_rates = relocated_home_goal_rates(model, match)
    operational_rates = adjust_derby_scoring_rates(match, relocated_rates, getattr(model, "canonical_club_ids", None))
    final_rates = policy.adjust_rates(operational_rates, scores or MotivationScores()) if policy else operational_rates

    active_teams = sorted(model.active_teams)
    league_difference = sum(model.states[team].attack_mean for team in active_teams) / len(active_teams) - sum(
        model.states[team].defense_mean for team in active_teams
    ) / len(active_teams)
    home_effect = model.strength_shrinkage * (
        model.states[match.home].attack_mean - model.states[match.away].defense_mean - league_difference
    )
    away_effect = model.strength_shrinkage * (
        model.states[match.away].attack_mean - model.states[match.home].defense_mean - league_difference
    )
    base_home_mean = no_home_advantage_rates.home_log_rate_mean - home_effect
    base_away_mean = no_home_advantage_rates.away_log_rate_mean - away_effect
    home_advantage_mean_effect = regular_rates.home_log_rate_mean - no_home_advantage_rates.home_log_rate_mean
    away_home_advantage_mean_effect = regular_rates.away_log_rate_mean - no_home_advantage_rates.away_log_rate_mean

    coalition_probabilities: dict[int, tuple[float, float, float]] = {}
    for coalition in range(16):
        includes_home_advantage = bool(coalition & 1)
        includes_uncertainty = bool(coalition & 2)
        uncertainty_rates = regular_rates if includes_home_advantage else no_home_advantage_rates
        rates = GoalRateForecast(
            base_home_mean
            + (home_advantage_mean_effect if includes_home_advantage else 0.0)
            + (home_effect if coalition & 4 else 0.0),
            uncertainty_rates.home_log_rate_variance if includes_uncertainty else 0.0,
            base_away_mean
            + (away_home_advantage_mean_effect if includes_home_advantage else 0.0)
            + (away_effect if coalition & 8 else 0.0),
            uncertainty_rates.away_log_rate_variance if includes_uncertainty else 0.0,
            uncertainty_rates.home_away_log_rate_covariance if includes_uncertainty else 0.0,
        )
        coalition_probabilities[coalition] = _rate_probabilities(rates, model.globals.rho)

    feature_adjustments: list[tuple[float, float, float]] = []
    for feature in range(4):
        values = [0.0, 0.0, 0.0]
        feature_mask = 1 << feature
        for coalition, parent in coalition_probabilities.items():
            if coalition & feature_mask:
                continue
            size = coalition.bit_count()
            weight = math.factorial(size) * math.factorial(3 - size) / math.factorial(4)
            treatment = coalition_probabilities[coalition | feature_mask]
            for outcome in range(3):
                values[outcome] += weight * (treatment[outcome] - parent[outcome])
        feature_adjustments.append((values[0], values[1], values[2]))

    base = coalition_probabilities[0]
    regular = _rate_probabilities(regular_rates, model.globals.rho)
    relocated = _rate_probabilities(relocated_rates, model.globals.rho)
    operational = _rate_probabilities(operational_rates, model.globals.rho)
    final = _rate_probabilities(final_rates, model.globals.rho)

    breakdown = ProbabilityBreakdown(
        base,
        (
            ProbabilityAdjustment("Home advantage", feature_adjustments[0]),
            ProbabilityAdjustment("Uncertainty", feature_adjustments[1]),
            ProbabilityAdjustment(f"{home_name} attack vs {away_name} defence", feature_adjustments[2]),
            ProbabilityAdjustment(f"{away_name} attack vs {home_name} defence", feature_adjustments[3]),
            ProbabilityAdjustment("Venue", _probability_delta(relocated, regular)),
            ProbabilityAdjustment("Derby", _probability_delta(operational, relocated)),
            ProbabilityAdjustment("Motivation", _probability_delta(final, operational)),
        ),
    )
    return final, breakdown
