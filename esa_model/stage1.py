"""Sequential uncertain-strength model with Dixon-Coles predictions."""

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from scipy.special import roots_hermitenorm
from scipy.stats import poisson

from esa_model.attendance import crowd_absence
from esa_model.baseline import (
    BURN_IN_SEASONS,
    HOLDOUT_SEASONS,
    MAX_GOALS,
    NAIVE,
    Match,
    Model,
    fit,
    load_matches,
    rps,
)
from esa_model.evaluation import MatchForecast, StagePrediction, run_walk_forward
from esa_model.team_identity import reconcile_state_team_ids

QUADRATURE_POINTS = 7
ANNUAL_DRIFT_VARIANCES = tuple(float(value) for value in np.geomspace(3.8e-4, 3.8e-1, 8))
SUMMER_VARIANCES = tuple(float(value) for value in np.geomspace(1e-4, 3e-1, 8))
RHO_VALUES = tuple(float(value) for value in np.linspace(-0.16, -0.04, 13)) + (0.0,)
STRENGTH_SHRINKAGE_VALUES = tuple(float(value) for value in np.linspace(0.5, 1.0, 11))
FAVORITE_SEGMENTS = ((0.0, 0.45), (0.45, 0.60), (0.60, 1.01))
FAVORITE_CALIBRATION_BINS = (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01)
DRAW_CALIBRATION_BINS = (0.0, 0.2, 0.25, 0.3, 0.35, 0.4, 1.01)


def output_stem(quick: bool, unshrunk: bool) -> str:
    variant = "stage1_unshrunk" if unshrunk else "stage1"
    return f"{variant}_quick" if quick else variant


@dataclass(slots=True)
class TeamState:
    attack_mean: float
    attack_variance: float
    defense_mean: float
    defense_variance: float


@dataclass(frozen=True, slots=True)
class FrozenGlobals:
    scoring_intercept: float
    home_advantage: float
    rho: float
    initial_attack_variance: float
    initial_defense_variance: float
    initial_home_advantage_variance: float = 1e-4
    crowd_effect: float = 0.0
    initial_scoring_variance: float = 1e-4
    initial_joint_scoring_variance: float = 0.0
    initial_joint_home_advantage_variance: float = 0.0
    initial_scoring_home_advantage_covariance: float = 0.0


@dataclass(frozen=True, slots=True)
class GoalRateForecast:
    home_log_rate_mean: float
    home_log_rate_variance: float
    away_log_rate_mean: float
    away_log_rate_variance: float
    home_away_log_rate_covariance: float = 0.0


@dataclass(frozen=True, slots=True)
class LeagueStateSnapshot:
    date: date
    season: str
    scoring_mean: float
    scoring_variance: float
    home_advantage_mean: float
    home_advantage_variance: float
    scoring_home_advantage_covariance: float


def freeze_globals(matches: list[Match], seasons: list[str]) -> tuple[FrozenGlobals, Model]:
    burn_in = [match for match in matches if match.season in set(seasons[:BURN_IN_SEASONS])]
    evaluation_date = max(match.date for match in burn_in) + timedelta(days=1)
    model = fit(burn_in, evaluation_date, xi=0.0)
    scoring_intercept = float(model.defense.mean())
    model_index = {team: index for index, team in enumerate(model.teams)}
    home_information = sum(
        math.exp(model.attack[model_index[match.home]] + model.defense[model_index[match.away]] + model.home_advantage)
        for match in burn_in
    )
    away_information = sum(
        math.exp(model.attack[model_index[match.away]] + model.defense[model_index[match.home]]) for match in burn_in
    )
    globals_ = FrozenGlobals(
        scoring_intercept=scoring_intercept,
        home_advantage=model.home_advantage,
        rho=model.rho,
        initial_attack_variance=max(float(np.var(model.attack)), 1e-4),
        initial_defense_variance=max(float(np.var(model.defense)), 1e-4),
        initial_home_advantage_variance=max(1.0 / home_information, 1e-4),
        initial_scoring_variance=max(1.0 / (home_information + away_information), 1e-4),
        initial_joint_scoring_variance=max(1.0 / away_information, 1e-4),
        initial_joint_home_advantage_variance=max(1.0 / home_information + 1.0 / away_information, 1e-4),
        initial_scoring_home_advantage_covariance=-1.0 / away_information,
    )
    return globals_, model


def update_pair(
    first_mean: float,
    first_variance: float,
    second_mean: float,
    second_variance: float,
    goals: float,
    predicted_goals: float,
    additional_variance: float = 0.0,
    observation_dispersion: float = 1.0,
) -> tuple[float, float, float, float]:
    """Update an attack and opposing defense from one observed goal count."""
    if observation_dispersion < 1.0:
        raise ValueError("Observation dispersion must be at least one")
    combined_variance = first_variance + second_variance + additional_variance
    scale = 1.0 / (observation_dispersion + predicted_goals * combined_variance)
    surprise = goals - predicted_goals
    first_mean += first_variance * scale * surprise
    second_mean -= second_variance * scale * surprise
    first_variance -= predicted_goals * first_variance**2 * scale
    second_variance -= predicted_goals * second_variance**2 * scale
    return first_mean, max(first_variance, 0.0), second_mean, max(second_variance, 0.0)


def _rate_cap(rho: float) -> float:
    if rho < 0:
        return -1.0 / rho * (1.0 - 1e-12)
    if rho > 0:
        return math.sqrt(1.0 / rho) * (1.0 - 1e-12)
    return math.inf


def uncertain_probabilities(
    home_log_rate_mean: float,
    home_log_rate_variance: float,
    away_log_rate_mean: float,
    away_log_rate_variance: float,
    rho: float,
    quadrature_points: int = QUADRATURE_POINTS,
    home_away_log_rate_covariance: float = 0.0,
) -> tuple[float, float, float, float, float]:
    """Integrate the DC score matrix over uncertain home and away scoring rates."""
    matrix, expected_home, expected_away = uncertain_score_matrix(
        home_log_rate_mean,
        home_log_rate_variance,
        away_log_rate_mean,
        away_log_rate_variance,
        rho,
        quadrature_points,
        home_away_log_rate_covariance,
    )
    return (
        float(np.tril(matrix, -1).sum()),
        float(np.trace(matrix)),
        float(np.triu(matrix, 1).sum()),
        expected_home,
        expected_away,
    )


def uncertain_score_matrix(
    home_log_rate_mean: float,
    home_log_rate_variance: float,
    away_log_rate_mean: float,
    away_log_rate_variance: float,
    rho: float,
    quadrature_points: int = QUADRATURE_POINTS,
    home_away_log_rate_covariance: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    """Integrate the normalized DC score matrix over uncertain scoring rates."""
    nodes, weights = roots_hermitenorm(quadrature_points)
    weights = weights / math.sqrt(2.0 * math.pi)
    cap = _rate_cap(rho)
    home_scale = math.sqrt(home_log_rate_variance)
    away_shared_scale = home_away_log_rate_covariance / home_scale if home_scale > 0.0 else 0.0
    away_independent_scale = math.sqrt(max(away_log_rate_variance - away_shared_scale**2, 0.0))
    home_rates = np.minimum(np.exp(home_log_rate_mean + home_scale * nodes), cap)
    away_rates = np.minimum(
        np.exp(away_log_rate_mean + away_shared_scale * nodes[:, None] + away_independent_scale * nodes[None, :]),
        cap,
    )
    goals = np.arange(MAX_GOALS + 1)
    home_pmf = poisson.pmf(goals[None, :], home_rates[:, None])
    away_pmf = poisson.pmf(goals[None, None, :], away_rates[:, :, None])
    home_grid = home_rates[:, None]
    away_grid = away_rates
    matrices = home_pmf[:, None, :, None] * away_pmf[:, :, None, :]
    matrices[:, :, 0, 0] *= 1.0 - home_grid * away_grid * rho
    matrices[:, :, 0, 1] *= 1.0 + home_grid * rho
    matrices[:, :, 1, 0] *= 1.0 + away_grid * rho
    matrices[:, :, 1, 1] *= 1.0 - rho
    if np.any(matrices < -1e-12):
        raise ValueError("Dixon-Coles correction produced a negative outcome probability")
    total = matrices.sum(axis=(2, 3))
    joint_weights = weights[:, None] * weights[None, :]
    matrix = np.einsum("ij,ijgh->gh", joint_weights, matrices / total[:, :, None, None])
    return (
        matrix,
        float(np.dot(weights, home_rates)),
        float(np.sum(joint_weights * away_rates)),
    )


def dixon_coles_score_matrix(home_rate: float, away_rate: float, rho: float) -> np.ndarray:
    """Return the normalized fixed-rate Dixon-Coles score distribution."""
    if not all(math.isfinite(value) and value > 0.0 for value in (home_rate, away_rate)):
        raise ValueError("Dixon-Coles scoring rates must be finite and positive")
    cap = _rate_cap(rho)
    home_rate = min(home_rate, cap)
    away_rate = min(away_rate, cap)
    goals = np.arange(MAX_GOALS + 1)
    matrix = poisson.pmf(goals, home_rate)[:, None] * poisson.pmf(goals, away_rate)[None, :]
    matrix[0, 0] *= 1.0 - home_rate * away_rate * rho
    matrix[0, 1] *= 1.0 + home_rate * rho
    matrix[1, 0] *= 1.0 + away_rate * rho
    matrix[1, 1] *= 1.0 - rho
    if np.any(matrix < -1e-12):
        raise ValueError("Dixon-Coles correction produced a negative outcome probability")
    return matrix / matrix.sum()


def initialize_team(globals_: FrozenGlobals) -> TeamState:
    return TeamState(0.0, globals_.initial_attack_variance, 0.0, globals_.initial_defense_variance)


def transition_season(
    states: dict[str, TeamState],
    previous_teams: set[str],
    current_teams: set[str],
    summer_variance: float,
) -> None:
    retained = previous_teams & current_teams
    promoted = current_teams - previous_teams
    for team in retained:
        states[team].attack_variance += summer_variance
        states[team].defense_variance += summer_variance
    if not promoted:
        return
    relegated_states = [states[team] for team in sorted(previous_teams - current_teams)]
    league_states = [states[team] for team in sorted(previous_teams)]
    prior_states = relegated_states or league_states
    attack_mean = float(np.mean([state.attack_mean for state in prior_states]))
    defense_mean = float(np.mean([state.defense_mean for state in prior_states]))
    attack_variance = max(float(np.var([state.attack_mean for state in league_states])), 1e-4)
    defense_variance = max(float(np.var([state.defense_mean for state in league_states])), 1e-4)
    for team in promoted:
        states[team] = TeamState(attack_mean, attack_variance, defense_mean, defense_variance)


def recenter(states: dict[str, TeamState], active_teams: set[str]) -> None:
    shift = float(np.mean([states[team].attack_mean for team in sorted(active_teams)]))
    for team in active_teams:
        states[team].attack_mean -= shift
        states[team].defense_mean -= shift


class SequentialDCModel:
    def __init__(
        self,
        annual_drift_variance: float,
        summer_variance: float,
        globals_: FrozenGlobals,
        strength_shrinkage: float = 1.0,
        canonical_club_ids: dict[tuple[str, str], str] | None = None,
        season_priors: dict[tuple[str, str], TeamState] | None = None,
        home_advantage_drift_variance: float = 0.0,
        dynamic_home_advantage: bool = False,
        crowd_effects_by_season: dict[str, float] | None = None,
        home_advantage_update_start_date: date | None = None,
        scoring_level_drift_variance: float = 0.0,
        dynamic_scoring_level: bool = False,
        xg_observations: dict[str, tuple[float, float]] | None = None,
        xg_weight: float = 0.0,
        xg_disagreement_variance: float = 0.0,
        xg_observation_variances: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        if not 0.0 < strength_shrinkage <= 1.0:
            raise ValueError("Strength shrinkage must be in (0, 1]")
        if not 0.0 <= xg_weight <= 1.0:
            raise ValueError("xG weight must be between zero and one")
        if xg_disagreement_variance < 0.0:
            raise ValueError("xG disagreement variance must be nonnegative")
        self.annual_drift_variance = annual_drift_variance
        self.summer_variance = summer_variance
        self.globals = globals_
        self.strength_shrinkage = strength_shrinkage
        self.canonical_club_ids = canonical_club_ids
        self.season_priors = season_priors
        self.home_advantage_drift_variance = home_advantage_drift_variance
        self.dynamic_home_advantage = dynamic_home_advantage
        self.crowd_effects_by_season = crowd_effects_by_season
        self.home_advantage_update_start_date = home_advantage_update_start_date
        self.scoring_level_drift_variance = scoring_level_drift_variance
        self.dynamic_scoring_level = dynamic_scoring_level
        self.xg_observations = xg_observations or {}
        self.xg_weight = xg_weight
        self.xg_disagreement_variance = xg_disagreement_variance
        self.xg_observation_variances = xg_observation_variances or {}
        self.scoring_level_mean = globals_.scoring_intercept
        self.scoring_level_variance = (
            globals_.initial_joint_scoring_variance
            if dynamic_scoring_level and dynamic_home_advantage and globals_.initial_joint_scoring_variance > 0.0
            else globals_.initial_scoring_variance
        )
        self.home_advantage_mean = globals_.home_advantage
        self.home_advantage_variance = (
            globals_.initial_joint_home_advantage_variance
            if dynamic_scoring_level and dynamic_home_advantage and globals_.initial_joint_home_advantage_variance > 0.0
            else globals_.initial_home_advantage_variance
        )
        self.scoring_home_advantage_covariance = (
            globals_.initial_scoring_home_advantage_covariance
            if dynamic_scoring_level and dynamic_home_advantage
            else 0.0
        )
        self.scoring_gauge_initialized = False
        self.league_state_history: list[LeagueStateSnapshot] = []
        self.states: dict[str, TeamState] = {}
        self.active_teams: set[str] = set()
        self.previous_date: date | None = None
        self.previous_season: str | None = None

    def advance_to(
        self,
        match_date: date,
        season: str,
        active_teams: set[str],
    ) -> None:
        season_changed = season != self.previous_season and self.previous_season is not None
        previous_teams = self.active_teams
        if self.previous_date is None:
            self.states.update({team: initialize_team(self.globals) for team in active_teams})
        else:
            if season_changed and self.canonical_club_ids is not None:
                previous_teams = reconcile_state_team_ids(
                    self.states,
                    self.previous_season,
                    previous_teams,
                    season,
                    active_teams,
                    self.canonical_club_ids,
                )
            elapsed_days = (match_date - self.previous_date).days
            elapsed_variance = self.annual_drift_variance * elapsed_days / 365.0
            for team in active_teams & self.states.keys():
                self.states[team].attack_variance += elapsed_variance
                self.states[team].defense_variance += elapsed_variance
            if (self.dynamic_scoring_level or self.dynamic_home_advantage) and (
                self.home_advantage_update_start_date is None or match_date >= self.home_advantage_update_start_date
            ):
                if self.dynamic_scoring_level:
                    self.scoring_level_variance += self.scoring_level_drift_variance * elapsed_days / 365.0
                if self.dynamic_home_advantage:
                    self.home_advantage_variance += self.home_advantage_drift_variance * elapsed_days / 365.0
        if season_changed:
            priors = {
                team: replace(self.season_priors[season, team])
                for team in active_teams
                if self.season_priors is not None and (season, team) in self.season_priors
            }
            transition_season(
                self.states,
                previous_teams,
                active_teams,
                self.summer_variance,
            )
            self.states.update(priors)
        for team in active_teams - self.states.keys():
            self.states[team] = initialize_team(self.globals)
        self.active_teams = active_teams
        self.previous_date = match_date
        self.previous_season = season
        scoring_updates_active = self.dynamic_scoring_level and (
            self.home_advantage_update_start_date is None or match_date >= self.home_advantage_update_start_date
        )
        if scoring_updates_active and (season_changed or not self.scoring_gauge_initialized):
            self._recenter_relative_strengths(transfer_defense_level=not self.scoring_gauge_initialized)
            self.scoring_gauge_initialized = True

    def goal_rates(
        self,
        match: Match,
        states: dict[str, TeamState] | None = None,
        strength_shrinkage: float | None = None,
        home_advantage_mean: float | None = None,
        home_advantage_variance: float | None = None,
    ) -> GoalRateForecast:
        current = states if states is not None else self.states
        home = current[match.home]
        away = current[match.away]
        shrinkage = self.strength_shrinkage if strength_shrinkage is None else strength_shrinkage
        hfa_mean = (
            self.home_advantage_mean
            if self.dynamic_home_advantage and home_advantage_mean is None
            else self.globals.home_advantage
            if home_advantage_mean is None
            else home_advantage_mean
        )
        hfa_variance = (
            self.home_advantage_variance
            if self.dynamic_home_advantage and home_advantage_variance is None
            else 0.0
            if home_advantage_variance is None
            else home_advantage_variance
        )
        scoring_mean = self.scoring_level_mean if self.dynamic_scoring_level else self.globals.scoring_intercept
        scoring_variance = self.scoring_level_variance if self.dynamic_scoring_level else 0.0
        global_covariance = (
            self.scoring_home_advantage_covariance
            if self.dynamic_scoring_level and self.dynamic_home_advantage
            else 0.0
        )
        crowd_effect = (
            self.crowd_effects_by_season.get(match.season, self.globals.crowd_effect)
            if self.crowd_effects_by_season is not None
            else self.globals.crowd_effect
        )
        home_strength_difference = home.attack_mean - away.defense_mean
        away_strength_difference = away.attack_mean - home.defense_mean
        if shrinkage < 1.0:
            active_states = [self.states[team] for team in sorted(self.active_teams)]
            league_strength_difference = float(
                np.mean([state.attack_mean for state in active_states])
                - np.mean([state.defense_mean for state in active_states])
            )
            home_strength_difference = league_strength_difference + shrinkage * (
                home_strength_difference - league_strength_difference
            )
            away_strength_difference = league_strength_difference + shrinkage * (
                away_strength_difference - league_strength_difference
            )
        return GoalRateForecast(
            home_log_rate_mean=(
                scoring_mean + home_strength_difference + hfa_mean + crowd_effect * crowd_absence(match.date)
            ),
            home_log_rate_variance=(
                home.attack_variance + away.defense_variance + scoring_variance + hfa_variance + 2.0 * global_covariance
            ),
            away_log_rate_mean=scoring_mean + away_strength_difference,
            away_log_rate_variance=away.attack_variance + home.defense_variance + scoring_variance,
            home_away_log_rate_covariance=scoring_variance + global_covariance,
        )

    def predict(self, match: Match) -> MatchForecast:
        rates = self.goal_rates(match)
        probability = uncertain_probabilities(
            rates.home_log_rate_mean,
            rates.home_log_rate_variance,
            rates.away_log_rate_mean,
            rates.away_log_rate_variance,
            self.globals.rho,
            home_away_log_rate_covariance=rates.home_away_log_rate_covariance,
        )
        return MatchForecast(probability[3], probability[4], *probability[:3])

    def observe(self, matches: list[Match]) -> None:
        participating = [team for match in matches for team in (match.home, match.away)]
        if len(participating) != len(set(participating)):
            raise ValueError(f"A team has multiple matches on {matches[0].date}")
        before = {team: replace(self.states[team]) for team in participating}
        after = {team: replace(self.states[team]) for team in participating}
        rates = [self.goal_rates(match, before, strength_shrinkage=1.0) for match in matches]
        home_rates = [math.exp(rate.home_log_rate_mean) for rate in rates]
        away_rates = [math.exp(rate.away_log_rate_mean) for rate in rates]
        for match in matches:
            self._apply_match_update(match, before, after)
        self.states.update(after)
        if (self.dynamic_scoring_level or self.dynamic_home_advantage) and (
            self.home_advantage_update_start_date is None or matches[0].date >= self.home_advantage_update_start_date
        ):
            self._update_league_state(matches, before, home_rates, away_rates)
        scoring_updates_active = self.dynamic_scoring_level and (
            self.home_advantage_update_start_date is None or matches[0].date >= self.home_advantage_update_start_date
        )
        if scoring_updates_active:
            self._recenter_relative_strengths(transfer_defense_level=True)
        else:
            recenter(self.states, self.active_teams)
        if self.dynamic_scoring_level or self.dynamic_home_advantage:
            self.league_state_history.append(
                LeagueStateSnapshot(
                    matches[0].date,
                    matches[0].season,
                    self.scoring_level_mean,
                    self.scoring_level_variance if self.dynamic_scoring_level else 0.0,
                    self.home_advantage_mean,
                    self.home_advantage_variance if self.dynamic_home_advantage else 0.0,
                    self.scoring_home_advantage_covariance,
                )
            )

    def _recenter_relative_strengths(self, transfer_defense_level: bool) -> None:
        recenter(self.states, self.active_teams)
        if self.dynamic_scoring_level:
            defense_shift = float(np.mean([self.states[team].defense_mean for team in sorted(self.active_teams)]))
            for team in self.active_teams:
                self.states[team].defense_mean -= defense_shift
            if transfer_defense_level:
                self.scoring_level_mean -= defense_shift

    def _update_league_state(
        self,
        matches: list[Match],
        before: dict[str, TeamState],
        home_rates: list[float],
        away_rates: list[float],
    ) -> None:
        covariance = np.array(
            [
                [
                    self.scoring_level_variance if self.dynamic_scoring_level else 0.0,
                    self.scoring_home_advantage_covariance
                    if self.dynamic_scoring_level and self.dynamic_home_advantage
                    else 0.0,
                ],
                [
                    self.scoring_home_advantage_covariance
                    if self.dynamic_scoring_level and self.dynamic_home_advantage
                    else 0.0,
                    self.home_advantage_variance if self.dynamic_home_advantage else 0.0,
                ],
            ]
        )
        information = np.zeros((2, 2))
        score = np.zeros(2)
        home_design = np.array([float(self.dynamic_scoring_level), float(self.dynamic_home_advantage)])
        away_design = np.array([float(self.dynamic_scoring_level), 0.0])
        for match, home_rate, away_rate in zip(matches, home_rates, away_rates, strict=True):
            home_observation, away_observation, has_xg = self._match_observations(match)
            home_xg_variance, away_xg_variance = self._xg_variances(match)
            observations = (
                (
                    home_design,
                    home_observation,
                    home_rate,
                    before[match.home].attack_variance + before[match.away].defense_variance,
                ),
                (
                    away_design,
                    away_observation,
                    away_rate,
                    before[match.away].attack_variance + before[match.home].defense_variance,
                ),
            )
            for (design, goals, rate, team_variance), xg_variance in zip(
                observations, (home_xg_variance, away_xg_variance), strict=True
            ):
                adjustment = self._observation_dispersion(rate, has_xg, xg_variance) + rate * team_variance
                information += rate / adjustment * np.outer(design, design)
                score += (goals - rate) / adjustment * design
        posterior = np.linalg.solve(np.eye(2) + covariance @ information, covariance)
        posterior = (posterior + posterior.T) / 2.0
        means = np.array([self.scoring_level_mean, self.home_advantage_mean]) + posterior @ score
        if self.dynamic_scoring_level:
            self.scoring_level_mean = float(means[0])
            self.scoring_level_variance = max(float(posterior[0, 0]), 0.0)
        if self.dynamic_home_advantage:
            self.home_advantage_mean = float(means[1])
            self.home_advantage_variance = max(float(posterior[1, 1]), 0.0)
        self.scoring_home_advantage_covariance = (
            float(posterior[0, 1]) if self.dynamic_scoring_level and self.dynamic_home_advantage else 0.0
        )

    def _apply_match_update(
        self,
        match: Match,
        before: dict[str, TeamState],
        after: dict[str, TeamState],
    ) -> None:
        home_before = before[match.home]
        away_before = before[match.away]
        home_after = after[match.home]
        away_after = after[match.away]
        rates = self.goal_rates(match, before, strength_shrinkage=1.0)
        home_observation, away_observation, has_xg = self._match_observations(match)
        home_xg_variance, away_xg_variance = self._xg_variances(match)
        home_rate = math.exp(rates.home_log_rate_mean)
        away_rate = math.exp(rates.away_log_rate_mean)
        (
            home_after.attack_mean,
            home_after.attack_variance,
            away_after.defense_mean,
            away_after.defense_variance,
        ) = update_pair(
            home_before.attack_mean,
            home_before.attack_variance,
            away_before.defense_mean,
            away_before.defense_variance,
            home_observation,
            home_rate,
            (
                (self.scoring_level_variance if self.dynamic_scoring_level else 0.0)
                + (self.home_advantage_variance if self.dynamic_home_advantage else 0.0)
                + 2.0 * self.scoring_home_advantage_covariance
            ),
            self._observation_dispersion(home_rate, has_xg, home_xg_variance),
        )
        (
            away_after.attack_mean,
            away_after.attack_variance,
            home_after.defense_mean,
            home_after.defense_variance,
        ) = update_pair(
            away_before.attack_mean,
            away_before.attack_variance,
            home_before.defense_mean,
            home_before.defense_variance,
            away_observation,
            away_rate,
            self.scoring_level_variance if self.dynamic_scoring_level else 0.0,
            self._observation_dispersion(away_rate, has_xg, away_xg_variance),
        )

    def _match_observations(self, match: Match) -> tuple[float, float, bool]:
        xg = self.xg_observations.get(match.match_id)
        if xg is None or self.xg_weight == 0.0:
            return float(match.home_goals), float(match.away_goals), False
        return (
            self.xg_weight * xg[0] + (1.0 - self.xg_weight) * match.home_goals,
            self.xg_weight * xg[1] + (1.0 - self.xg_weight) * match.away_goals,
            True,
        )

    def _xg_variances(self, match: Match) -> tuple[float, float]:
        return self.xg_observation_variances.get(
            match.match_id,
            (self.xg_disagreement_variance, self.xg_disagreement_variance),
        )

    def _observation_dispersion(self, predicted_goals: float, has_xg: bool, xg_variance: float | None = None) -> float:
        if not has_xg:
            return 1.0
        variance = self.xg_disagreement_variance if xg_variance is None else xg_variance
        return 1.0 + self.xg_weight**2 * variance / predicted_goals


def walk_forward(
    matches: list[Match],
    annual_drift_variance: float,
    summer_variance: float,
    evaluation_seasons: set[str],
    globals_: FrozenGlobals,
    strength_shrinkage: float = 1.0,
    canonical_club_ids: dict[tuple[str, str], str] | None = None,
    season_priors: dict[tuple[str, str], TeamState] | None = None,
    home_advantage_drift_variance: float = 0.0,
    dynamic_home_advantage: bool = False,
    crowd_effects_by_season: dict[str, float] | None = None,
    scoring_level_drift_variance: float = 0.0,
    dynamic_scoring_level: bool = False,
    league_state_history_output: list[LeagueStateSnapshot] | None = None,
    xg_observations: dict[str, tuple[float, float]] | None = None,
    xg_weight: float = 0.0,
    xg_disagreement_variance: float = 0.0,
    xg_observation_variances: dict[str, tuple[float, float]] | None = None,
) -> tuple[list[StagePrediction], dict[str, TeamState]]:
    home_advantage_update_start_date = league_state_update_start_date(
        matches, dynamic_home_advantage or dynamic_scoring_level
    )
    model = SequentialDCModel(
        annual_drift_variance,
        summer_variance,
        globals_,
        strength_shrinkage,
        canonical_club_ids,
        season_priors,
        home_advantage_drift_variance,
        dynamic_home_advantage,
        crowd_effects_by_season,
        home_advantage_update_start_date,
        scoring_level_drift_variance,
        dynamic_scoring_level,
        xg_observations,
        xg_weight,
        xg_disagreement_variance,
        xg_observation_variances,
    )
    predictions = run_walk_forward(matches, model, evaluation_seasons)
    if league_state_history_output is not None:
        league_state_history_output.extend(
            snapshot for snapshot in model.league_state_history if snapshot.season in evaluation_seasons
        )
    return predictions, model.states


def league_state_update_start_date(matches: list[Match], dynamic: bool) -> date | None:
    seasons = sorted({match.season for match in matches})
    if not dynamic or len(seasons) <= BURN_IN_SEASONS:
        return None
    return min(match.date for match in matches if match.season == seasons[BURN_IN_SEASONS])


def mean_rps(predictions: list[StagePrediction]) -> dict[str, float | int]:
    return {
        "matches": len(predictions),
        "model_rps": float(np.mean([prediction.rps for prediction in predictions])),
        "naive_rps": float(np.mean([rps(NAIVE, prediction.outcome) for prediction in predictions])),
    }


def calibration_table(
    predictions: list[StagePrediction], boundaries: tuple[float, ...], kind: str
) -> list[dict[str, float | int | str]]:
    rows = []
    for lower, upper in zip(boundaries, boundaries[1:], strict=False):
        if kind == "favorite":
            selected = [
                prediction for prediction in predictions if lower <= max(prediction.p_home, prediction.p_away) < upper
            ]
            predicted = [max(prediction.p_home, prediction.p_away) for prediction in selected]
            observed = [
                float(
                    (prediction.p_home >= prediction.p_away and prediction.outcome == 0)
                    or (prediction.p_away > prediction.p_home and prediction.outcome == 2)
                )
                for prediction in selected
            ]
        else:
            selected = [prediction for prediction in predictions if lower <= prediction.p_draw < upper]
            predicted = [prediction.p_draw for prediction in selected]
            observed = [float(prediction.outcome == 1) for prediction in selected]
        rows.append(
            {
                "bin": f"[{lower:.2f}, {min(upper, 1.0):.2f})",
                "matches": len(selected),
                "mean_predicted": float(np.mean(predicted)) if predicted else 0.0,
                "observed_rate": float(np.mean(observed)) if observed else 0.0,
            }
        )
    return rows


def segmented_rps(
    predictions: list[StagePrediction], reference: list[StagePrediction]
) -> list[dict[str, float | int | str]]:
    favorite_probability = {prediction.match_id: max(prediction.p_home, prediction.p_away) for prediction in reference}
    rows = []
    for lower, upper in FAVORITE_SEGMENTS:
        selected = [
            prediction for prediction in predictions if lower <= favorite_probability[prediction.match_id] < upper
        ]
        rows.append({"segment": f"[{lower:.2f}, {min(upper, 1.0):.2f})", **mean_rps(selected)})
    return rows


def write_predictions(path: Path, predictions: list[StagePrediction]) -> None:
    rows = []
    for prediction in predictions:
        row = asdict(prediction)
        row["date"] = prediction.date.isoformat()
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_baseline_predictions(path: Path, seasons: set[str]) -> list[StagePrediction]:
    with path.open(encoding="utf-8", newline="") as source:
        rows = [row for row in csv.DictReader(source) if row["season"] in seasons]
    return [
        StagePrediction(
            row["match_id"],
            date.fromisoformat(row["date"]),
            row["season"],
            row["home_team_id"],
            row["away_team_id"],
            int(row["home_goals"]),
            int(row["away_goals"]),
            float(row["home_xg"]),
            float(row["away_xg"]),
            float(row["p_home"]),
            float(row["p_draw"]),
            float(row["p_away"]),
            int(row["outcome"]),
            float(row["rps"]),
        )
        for row in rows
    ]


def evaluate_candidate(
    arguments: tuple[Path, float, float, float, set[str], FrozenGlobals],
) -> dict[str, float | int]:
    path, annual_drift_variance, summer_variance, strength_shrinkage, tuning_seasons, globals_ = arguments
    predictions, _ = walk_forward(
        load_matches(path),
        annual_drift_variance,
        summer_variance,
        tuning_seasons,
        globals_,
        strength_shrinkage,
    )
    return {
        "annual_drift_variance": annual_drift_variance,
        "summer_variance": summer_variance,
        "rho": globals_.rho,
        "strength_shrinkage": strength_shrinkage,
        **mean_rps(predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run one central grid candidate for development")
    parser.add_argument(
        "--unshrunk",
        action="store_true",
        help="Run the s=1 comparison variant and write stage1_unshrunk outputs",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    matches = load_matches(Path("data/matches.csv"))
    seasons = sorted({match.season for match in matches})
    evaluation_seasons = seasons[BURN_IN_SEASONS:]
    holdout_seasons = set(evaluation_seasons[-HOLDOUT_SEASONS:])
    tuning_seasons = set(evaluation_seasons) - holdout_seasons
    globals_, frozen_model = freeze_globals(matches, seasons)
    drift_grid = (ANNUAL_DRIFT_VARIANCES[3],) if args.quick else ANNUAL_DRIFT_VARIANCES
    summer_grid = (SUMMER_VARIANCES[3],) if args.quick else SUMMER_VARIANCES
    state_tasks = [
        (Path("data/matches.csv"), annual_drift_variance, summer_variance, 1.0, tuning_seasons, globals_)
        for annual_drift_variance in drift_grid
        for summer_variance in summer_grid
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        state_search = list(executor.map(evaluate_candidate, state_tasks))
    selected_state = min(state_search, key=lambda row: row["model_rps"])
    anchor_drift_variance = float(selected_state["annual_drift_variance"])
    selected_summer_variance = float(selected_state["summer_variance"])
    rho_grid = (globals_.rho,) if args.quick else tuple(sorted({*RHO_VALUES, globals_.rho}))
    rho_tasks = [
        (
            Path("data/matches.csv"),
            annual_drift_variance,
            selected_summer_variance,
            1.0,
            tuning_seasons,
            replace(globals_, rho=rho),
        )
        for annual_drift_variance in drift_grid
        for rho in rho_grid
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rho_search = list(executor.map(evaluate_candidate, rho_tasks))
    selected_rho_candidate = min(rho_search, key=lambda row: row["model_rps"])
    selected_rho = float(selected_rho_candidate["rho"])
    selected_globals = replace(globals_, rho=selected_rho)
    shrinkage_grid = (1.0,) if args.quick or args.unshrunk else STRENGTH_SHRINKAGE_VALUES
    shrinkage_tasks = [
        (
            Path("data/matches.csv"),
            annual_drift_variance,
            selected_summer_variance,
            strength_shrinkage,
            tuning_seasons,
            selected_globals,
        )
        for annual_drift_variance in drift_grid
        for strength_shrinkage in shrinkage_grid
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        shrinkage_search = list(executor.map(evaluate_candidate, shrinkage_tasks))
    selected = min(
        shrinkage_search,
        key=lambda row: (row["model_rps"], -float(row["strength_shrinkage"])),
    )
    selected_drift_variance = float(selected["annual_drift_variance"])
    selected_strength_shrinkage = float(selected["strength_shrinkage"])
    predictions, _ = walk_forward(
        matches,
        selected_drift_variance,
        selected_summer_variance,
        set(evaluation_seasons),
        selected_globals,
        selected_strength_shrinkage,
    )
    same_state_unshrunk_predictions, _ = walk_forward(
        matches,
        selected_drift_variance,
        selected_summer_variance,
        set(evaluation_seasons),
        selected_globals,
        1.0,
    )
    stage1_anchor_predictions, _ = walk_forward(
        matches,
        anchor_drift_variance,
        selected_summer_variance,
        set(evaluation_seasons),
        globals_,
        1.0,
    )
    tuning = [prediction for prediction in predictions if prediction.season in tuning_seasons]
    holdout = [prediction for prediction in predictions if prediction.season in holdout_seasons]
    same_state_unshrunk_tuning = [
        prediction for prediction in same_state_unshrunk_predictions if prediction.season in tuning_seasons
    ]
    same_state_unshrunk_holdout = [
        prediction for prediction in same_state_unshrunk_predictions if prediction.season in holdout_seasons
    ]
    stage1_anchor_tuning = [
        prediction for prediction in stage1_anchor_predictions if prediction.season in tuning_seasons
    ]
    stage1_anchor_holdout = [
        prediction for prediction in stage1_anchor_predictions if prediction.season in holdout_seasons
    ]
    baseline_holdout = load_baseline_predictions(Path("artifacts/baseline_predictions.csv"), holdout_seasons)
    result = {
        "model": {
            "variant": "unshrunk_comparison" if args.unshrunk else "canonical_shrinkage",
            "update": "joint diagonal Fisher/Laplace approximation",
            "drift": "elapsed calendar time",
            "quadrature_points": QUADRATURE_POINTS,
            "frozen_on_seasons": seasons[:BURN_IN_SEASONS],
            **asdict(selected_globals),
            "burn_in_rho": globals_.rho,
            "frozen_fit_converged": frozen_model.converged,
            "selected_annual_drift_variance": selected_drift_variance,
            "selected_summer_variance": selected_summer_variance,
            "selected_strength_shrinkage": selected_strength_shrinkage,
        },
        "tuning": mean_rps(tuning),
        "holdout": mean_rps(holdout),
        "holdout_favorite_segments": segmented_rps(holdout, baseline_holdout),
        "holdout_favorite_calibration": calibration_table(holdout, FAVORITE_CALIBRATION_BINS, "favorite"),
        "holdout_draw_calibration": calibration_table(holdout, DRAW_CALIBRATION_BINS, "draw"),
        "same_state_unshrunk_tuning": mean_rps(same_state_unshrunk_tuning),
        "same_state_unshrunk_holdout": mean_rps(same_state_unshrunk_holdout),
        "same_state_unshrunk_holdout_favorite_calibration": calibration_table(
            same_state_unshrunk_holdout, FAVORITE_CALIBRATION_BINS, "favorite"
        ),
        "same_state_unshrunk_holdout_draw_calibration": calibration_table(
            same_state_unshrunk_holdout, DRAW_CALIBRATION_BINS, "draw"
        ),
        "stage1_anchor_tuning": mean_rps(stage1_anchor_tuning),
        "stage1_anchor_holdout": mean_rps(stage1_anchor_holdout),
        "stage1_anchor_holdout_favorite_calibration": calibration_table(
            stage1_anchor_holdout, FAVORITE_CALIBRATION_BINS, "favorite"
        ),
        "stage1_anchor_holdout_draw_calibration": calibration_table(
            stage1_anchor_holdout, DRAW_CALIBRATION_BINS, "draw"
        ),
        "baseline_holdout": mean_rps(baseline_holdout),
        "baseline_holdout_favorite_segments": segmented_rps(baseline_holdout, baseline_holdout),
        "baseline_holdout_favorite_calibration": calibration_table(
            baseline_holdout, FAVORITE_CALIBRATION_BINS, "favorite"
        ),
        "baseline_holdout_draw_calibration": calibration_table(baseline_holdout, DRAW_CALIBRATION_BINS, "draw"),
        "state_grid_search": state_search,
        "rho_grid_search": rho_search,
        "shrinkage_grid_search": shrinkage_search,
    }
    stem = output_stem(args.quick, args.unshrunk)
    write_predictions(Path("artifacts") / f"{stem}_predictions.csv", predictions)
    (Path("artifacts") / f"{stem}_results.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
