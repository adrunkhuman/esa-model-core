"""Joint-covariance variant of the sequential Dixon-Coles model."""

import math
from dataclasses import replace
from datetime import date
from typing import Literal

import numpy as np

from esa_model.attendance import crowd_absence
from esa_model.baseline import Match
from esa_model.evaluation import MatchForecast, StagePrediction, run_walk_forward
from esa_model.stage1 import (
    FrozenGlobals,
    GoalRateForecast,
    TeamState,
    league_state_update_start_date,
    uncertain_probabilities,
)

Parameter = tuple[str, str]
HFA_KEY = ("", "home_advantage")
SeasonPriorCovariances = dict[str, dict[tuple[Parameter, Parameter], float]]
PriorTransitionMode = Literal["reset", "budgeted", "full", "diagonal"]
SeasonPriorPersistence = dict[str, tuple[float, float]]
SeasonTeamPriorPersistence = dict[str, dict[str, tuple[float, float]]]


class JointCovarianceDCModel:
    """Sequential Gaussian model with a joint team and HFA covariance matrix."""

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
        xg_observations: dict[str, tuple[float, float]] | None = None,
        xg_weight: float = 0.0,
        xg_disagreement_variance: float = 0.0,
        xg_observation_variances: dict[str, tuple[float, float]] | None = None,
        season_prior_covariances: SeasonPriorCovariances | None = None,
        season_mean_adjustments: dict[str, dict[str, tuple[float, float]]] | None = None,
        season_variance_scales: dict[str, dict[str, float]] | None = None,
        prior_transition_mode: PriorTransitionMode = "reset",
        season_prior_persistence: SeasonPriorPersistence | None = None,
        season_team_prior_persistence: SeasonTeamPriorPersistence | None = None,
    ) -> None:
        if not 0.0 < strength_shrinkage <= 1.0:
            raise ValueError("Strength shrinkage must be in (0, 1]")
        if not 0.0 <= xg_weight <= 1.0:
            raise ValueError("xG weight must be between zero and one")
        if xg_disagreement_variance < 0.0:
            raise ValueError("xG disagreement variance must be nonnegative")
        if prior_transition_mode not in ("reset", "budgeted", "full", "diagonal"):
            raise ValueError(f"Unknown prior transition mode: {prior_transition_mode}")
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
        self.xg_observations = xg_observations or {}
        self.xg_weight = xg_weight
        self.xg_disagreement_variance = xg_disagreement_variance
        self.xg_observation_variances = xg_observation_variances or {}
        self.season_prior_covariances = season_prior_covariances or {}
        self.season_mean_adjustments = season_mean_adjustments or {}
        self.season_variance_scales = season_variance_scales or {}
        self.prior_transition_mode = prior_transition_mode
        self.season_prior_persistence = season_prior_persistence or {}
        self.season_team_prior_persistence = season_team_prior_persistence or {}
        self.adjusted_seasons: set[str] = set()
        self.variance_scaled_seasons: set[str] = set()
        self.parameters: list[Parameter] = []
        self.index: dict[Parameter, int] = {}
        self.mean = np.empty(0)
        self.covariance = np.empty((0, 0))
        self.states: dict[str, TeamState] = {}
        self.active_teams: set[str] = set()
        self.previous_date: date | None = None
        self.previous_season: str | None = None

    @property
    def home_advantage_mean(self) -> float:
        if not self.dynamic_home_advantage:
            return self.globals.home_advantage
        return float(self.mean[self.index["home_advantage", ""]])

    @property
    def home_advantage_variance(self) -> float:
        if not self.dynamic_home_advantage:
            return 0.0
        index = self.index["home_advantage", ""]
        return float(self.covariance[index, index])

    @property
    def state_keys(self) -> list[tuple[str, str]]:
        return [HFA_KEY if kind == "home_advantage" else (team, kind) for kind, team in self.parameters]

    def _mean_vector(self) -> np.ndarray:
        self._pull_states()
        return self.mean.copy()

    def _parameter_list(self, teams: set[str]) -> list[Parameter]:
        parameters = [(kind, team) for kind in ("attack", "defense") for team in sorted(teams)]
        if self.dynamic_home_advantage:
            parameters.append(("home_advantage", ""))
        return parameters

    def _initialize(self, teams: set[str]) -> None:
        self.parameters = self._parameter_list(teams)
        self.index = {parameter: index for index, parameter in enumerate(self.parameters)}
        self.mean = np.zeros(len(self.parameters))
        variances = [
            self.globals.initial_attack_variance
            if kind == "attack"
            else self.globals.initial_defense_variance
            if kind == "defense"
            else self.globals.initial_home_advantage_variance
            for kind, _ in self.parameters
        ]
        self.covariance = np.diag(variances)
        if self.dynamic_home_advantage:
            self.mean[self.index["home_advantage", ""]] = self.globals.home_advantage
        self._sync_states()

    def _canonical_renames(self, season: str, teams: set[str]) -> dict[str, str]:
        if self.canonical_club_ids is None or self.previous_season is None:
            return {}
        previous_by_club = {self.canonical_club_ids[self.previous_season, team]: team for team in self.active_teams}
        current_by_club = {self.canonical_club_ids[season, team]: team for team in teams}
        return {
            previous_by_club[club]: current_by_club[club]
            for club in previous_by_club.keys() & current_by_club.keys()
            if previous_by_club[club] != current_by_club[club]
        }

    def _transition_season(self, season: str, teams: set[str]) -> None:
        renames = self._canonical_renames(season, teams)
        renamed_parameters = [(kind, renames.get(team, team)) for kind, team in self.parameters]
        old_index = {parameter: index for index, parameter in enumerate(renamed_parameters)}
        previous_teams = {renames.get(team, team) for team in self.active_teams}
        retained = previous_teams & teams
        relegated = previous_teams - teams
        source_teams = sorted(relegated or previous_teams)
        league_teams = sorted(previous_teams)
        attack_mean = float(np.mean([self.mean[old_index["attack", team]] for team in source_teams]))
        defense_mean = float(np.mean([self.mean[old_index["defense", team]] for team in source_teams]))
        attack_variance = max(float(np.var([self.mean[old_index["attack", team]] for team in league_teams])), 1e-4)
        defense_variance = max(float(np.var([self.mean[old_index["defense", team]] for team in league_teams])), 1e-4)

        new_parameters = self._parameter_list(teams)
        new_index = {parameter: index for index, parameter in enumerate(new_parameters)}
        copied = [
            parameter for parameter in new_parameters if parameter == ("home_advantage", "") or parameter[1] in retained
        ]
        mean = np.zeros(len(new_parameters))
        covariance = np.zeros((len(new_parameters), len(new_parameters)))
        for parameter in copied:
            mean[new_index[parameter]] = self.mean[old_index[parameter]]
        old_positions = [old_index[parameter] for parameter in copied]
        new_positions = [new_index[parameter] for parameter in copied]
        covariance[np.ix_(new_positions, new_positions)] = self.covariance[np.ix_(old_positions, old_positions)]
        for team in retained:
            covariance[new_index["attack", team], new_index["attack", team]] += self.summer_variance
            covariance[new_index["defense", team], new_index["defense", team]] += self.summer_variance
        for team in teams - retained:
            mean[new_index["attack", team]] = attack_mean
            mean[new_index["defense", team]] = defense_mean
            covariance[new_index["attack", team], new_index["attack", team]] = attack_variance
            covariance[new_index["defense", team], new_index["defense", team]] = defense_variance

        self.parameters = new_parameters
        self.index = new_index
        self.mean = mean
        self.covariance = covariance
        if self.season_priors is not None:
            propagated = {
                team: self.season_priors[season, team]
                for team in retained
                if (season, team) in self.season_priors and self.prior_transition_mode != "reset"
            }
            for team in teams - propagated.keys():
                prior = self.season_priors.get((season, team))
                if prior is not None:
                    self._reset_team_prior(team, prior)
            if propagated:
                self._propagate_team_priors(season, propagated)
        self._apply_season_prior_covariance(season)

    def _reset_team_prior(self, team: str, prior: TeamState) -> None:
        positions = [self.index["attack", team], self.index["defense", team]]
        self.mean[positions] = (prior.attack_mean, prior.defense_mean)
        self.covariance[positions, :] = 0.0
        self.covariance[:, positions] = 0.0
        self.covariance[positions, positions] = (prior.attack_variance, prior.defense_variance)

    def _propagate_team_priors(self, season: str, priors: dict[str, TeamState]) -> None:
        multipliers = np.ones(len(self.parameters))
        for team, prior in priors.items():
            try:
                team_persistence = self.season_team_prior_persistence.get(season, {})
                attack_persistence, defense_persistence = (
                    team_persistence[team] if team in team_persistence else self.season_prior_persistence[season]
                )
            except KeyError as error:
                raise ValueError(f"Missing prior persistence for propagated season {season}") from error
            if not all(math.isfinite(value) for value in (attack_persistence, defense_persistence)):
                raise ValueError(f"Prior persistence must be finite for {team} in {season}")
            attack = self.index["attack", team]
            defense = self.index["defense", team]
            multipliers[attack] = attack_persistence
            multipliers[defense] = defense_persistence
            self.mean[[attack, defense]] = (prior.attack_mean, prior.defense_mean)
        self.covariance *= np.outer(multipliers, multipliers)
        for team, prior in priors.items():
            self.covariance[self.index["attack", team], self.index["attack", team]] += prior.attack_variance
            self.covariance[self.index["defense", team], self.index["defense", team]] += prior.defense_variance

        if self.prior_transition_mode == "budgeted":
            role_scales = {}
            for role in ("attack", "defense"):
                positions = [self.index[role, team] for team in priors]
                target = float(
                    np.mean(
                        [
                            prior.attack_variance if role == "attack" else prior.defense_variance
                            for prior in priors.values()
                        ]
                    )
                )
                current = float(np.mean(self.covariance[positions, positions]))
                role_scales[role] = math.sqrt(target / current)
            for team in priors:
                multipliers[self.index["attack", team]] = role_scales["attack"]
                multipliers[self.index["defense", team]] = role_scales["defense"]
            self.covariance *= np.outer(multipliers, multipliers)
        elif self.prior_transition_mode == "diagonal":
            positions = [self.index[role, team] for role in ("attack", "defense") for team in priors]
            diagonal = np.diag(self.covariance).copy()
            self.covariance[positions, :] = 0.0
            self.covariance[:, positions] = 0.0
            self.covariance[positions, positions] = diagonal[positions]

        self.covariance = (self.covariance + self.covariance.T) / 2.0
        if float(np.linalg.eigvalsh(self.covariance).min()) < -1e-10:
            raise ValueError("Propagated season prior covariance must preserve positive semidefiniteness")

    def _apply_season_prior_covariance(self, season: str) -> None:
        entries = self.season_prior_covariances.get(season, {})
        for (left, right), value in entries.items():
            if left == right:
                raise ValueError("Season prior covariance entries must be off-diagonal")
            if left not in self.index or right not in self.index:
                raise ValueError(f"Season prior covariance references inactive parameters: {left}, {right}")
            if not math.isfinite(value):
                raise ValueError("Season prior covariance entries must be finite")
            left_index = self.index[left]
            right_index = self.index[right]
            self.covariance[left_index, right_index] = value
            self.covariance[right_index, left_index] = value
        if entries:
            self.covariance = (self.covariance + self.covariance.T) / 2.0
            if float(np.linalg.eigvalsh(self.covariance).min()) < -1e-10:
                raise ValueError("Season prior covariance must preserve positive semidefiniteness")

    def apply_team_variance_scales(self, scales: dict[str, float]) -> None:
        """Multiply active-team attack and defense diagonal variances exactly."""
        unknown = scales.keys() - self.active_teams
        if unknown:
            raise ValueError(f"Season variance scales reference inactive teams: {sorted(unknown)}")
        try:
            invalid_scale = any(not math.isfinite(scale) or not 1.0 <= scale <= 3.0 for scale in scales.values())
        except TypeError as error:
            raise ValueError("Season variance scales must be finite and within [1, 3]") from error
        if invalid_scale:
            raise ValueError("Season variance scales must be finite and within [1, 3]")
        means = self.mean.copy()
        off_diagonal = self.covariance.copy()
        diagonal = np.diag_indices_from(off_diagonal)
        off_diagonal[diagonal] = 0.0
        for team, scale in scales.items():
            attack = self.index["attack", team]
            defense = self.index["defense", team]
            self.covariance[attack, attack] *= scale
            self.covariance[defense, defense] *= scale
        if not np.array_equal(self.mean, means) or not np.array_equal(
            self.covariance - np.diag(np.diag(self.covariance)), off_diagonal
        ):
            raise AssertionError("Season variance scaling must preserve means and off-diagonal covariance")
        self.covariance = (self.covariance + self.covariance.T) / 2.0
        if float(np.linalg.eigvalsh(self.covariance).min()) < -1e-10:
            raise ValueError("Season variance scales must preserve positive semidefiniteness")
        self._sync_states()

    def _apply_season_variance_scales(self, season: str) -> None:
        if season in self.variance_scaled_seasons:
            return
        scales = self.season_variance_scales.get(season, {})
        if not scales:
            return
        self.apply_team_variance_scales(scales)
        self.variance_scaled_seasons.add(season)

    def advance_to(self, match_date: date, season: str, active_teams: set[str]) -> None:
        season_changed = self.previous_season is not None and season != self.previous_season
        if self.previous_date is None:
            self._initialize(active_teams)
        else:
            elapsed_days = (match_date - self.previous_date).days
            elapsed_variance = self.annual_drift_variance * elapsed_days / 365.0
            for team in self.active_teams:
                self.covariance[self.index["attack", team], self.index["attack", team]] += elapsed_variance
                self.covariance[self.index["defense", team], self.index["defense", team]] += elapsed_variance
            if self.dynamic_home_advantage and (
                self.home_advantage_update_start_date is None or match_date >= self.home_advantage_update_start_date
            ):
                hfa = self.index["home_advantage", ""]
                self.covariance[hfa, hfa] += self.home_advantage_drift_variance * elapsed_days / 365.0
            if season_changed:
                self._transition_season(season, active_teams)
        self.active_teams = set(active_teams)
        entering_season = self.previous_season is None or season_changed
        if entering_season:
            # This is after transition priors/covariances and before mean adjustments or predictions.
            self._apply_season_variance_scales(season)
        if entering_season and season in self.season_mean_adjustments and season not in self.adjusted_seasons:
            self._sync_states()
            self.adjust_team_means(self.season_mean_adjustments[season])
            self.adjusted_seasons.add(season)
        self.previous_date = match_date
        self.previous_season = season
        self._sync_states()

    def _design(
        self,
        match: Match,
        home: bool,
        include_hfa: bool = True,
        home_advantage_scale: float = 1.0,
    ) -> np.ndarray:
        design = np.zeros(len(self.parameters))
        attack_team = match.home if home else match.away
        defense_team = match.away if home else match.home
        design[self.index["attack", attack_team]] = 1.0
        design[self.index["defense", defense_team]] = -1.0
        if home and include_hfa and self.dynamic_home_advantage:
            design[self.index["home_advantage", ""]] = home_advantage_scale
        return design

    def goal_rates(self, match: Match, strength_shrinkage: float | None = None) -> GoalRateForecast:
        return self._goal_rates(match, strength_shrinkage, 1.0)

    def goal_rates_with_home_advantage_scale(
        self,
        match: Match,
        home_advantage_scale: float,
        strength_shrinkage: float | None = None,
    ) -> GoalRateForecast:
        if not math.isfinite(home_advantage_scale) or not 0.0 <= home_advantage_scale <= 1.0:
            raise ValueError("Home-advantage scale must be finite and within [0, 1]")
        return self._goal_rates(match, strength_shrinkage, home_advantage_scale)

    def _goal_rates(
        self,
        match: Match,
        strength_shrinkage: float | None,
        home_advantage_scale: float,
    ) -> GoalRateForecast:
        self._pull_states()
        shrinkage = self.strength_shrinkage if strength_shrinkage is None else strength_shrinkage
        home_design = self._design(match, True, home_advantage_scale=home_advantage_scale)
        away_design = self._design(match, False)
        home_difference = float(home_design @ self.mean)
        if self.dynamic_home_advantage:
            home_difference -= home_advantage_scale * self.home_advantage_mean
        away_difference = float(away_design @ self.mean)
        if shrinkage < 1.0:
            league_difference = float(
                np.mean([self.mean[self.index["attack", team]] for team in sorted(self.active_teams)])
                - np.mean([self.mean[self.index["defense", team]] for team in sorted(self.active_teams)])
            )
            home_difference = league_difference + shrinkage * (home_difference - league_difference)
            away_difference = league_difference + shrinkage * (away_difference - league_difference)
        crowd_effect = (
            self.crowd_effects_by_season.get(match.season, self.globals.crowd_effect)
            if self.crowd_effects_by_season is not None
            else self.globals.crowd_effect
        )
        return GoalRateForecast(
            self.globals.scoring_intercept
            + home_difference
            + home_advantage_scale * self.home_advantage_mean
            + crowd_effect * crowd_absence(match.date),
            max(float(home_design @ self.covariance @ home_design), 0.0),
            self.globals.scoring_intercept + away_difference,
            max(float(away_design @ self.covariance @ away_design), 0.0),
            float(home_design @ self.covariance @ away_design),
        )

    def sample_parameter_state(self, generator: np.random.Generator) -> np.ndarray:
        """Draw one coherent latent parameter state from the current posterior."""
        self._pull_states()
        return generator.multivariate_normal(self.mean, self.covariance, check_valid="raise")

    def home_advantage_from_state(self, state: np.ndarray) -> float:
        """Return the fixed or sampled home advantage represented by a parameter state."""
        if state.shape != self.mean.shape or not np.all(np.isfinite(state)):
            raise ValueError("Sampled parameter state has the wrong shape or non-finite values")
        return (
            float(state[self.index["home_advantage", ""]]) if self.dynamic_home_advantage else self.home_advantage_mean
        )

    def goal_rates_from_state(self, match: Match, state: np.ndarray) -> GoalRateForecast:
        """Project a fixed latent parameter state into conditional match scoring rates."""
        if state.shape != self.mean.shape or not np.all(np.isfinite(state)):
            raise ValueError("Sampled parameter state has the wrong shape or non-finite values")
        self._pull_states()
        marginal = self.goal_rates(match)
        home_design = self._design(match, True)
        away_design = self._design(match, False)
        deviation = state - self.mean
        return GoalRateForecast(
            marginal.home_log_rate_mean + float(home_design @ deviation),
            0.0,
            marginal.away_log_rate_mean + float(away_design @ deviation),
            0.0,
            0.0,
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
        self.observe_with_log_rate_offsets(matches, {})

    def observe_with_xg(
        self,
        matches: list[Match],
        xg_observations: dict[str, tuple[float, float]],
        xg_variances: dict[str, tuple[float, float]],
        log_rate_offsets: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Observe a simulated date batch with transient calibrated xG values."""
        match_ids = {match.match_id for match in matches}
        if xg_observations.keys() != match_ids or xg_variances.keys() != match_ids:
            raise ValueError("Simulated xG observations and variances must exactly cover the date batch")
        values = [value for pair in (*xg_observations.values(), *xg_variances.values()) for value in pair]
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("Simulated xG observations and variances must be finite and nonnegative")
        self.observe_with_log_rate_offsets(
            matches,
            log_rate_offsets or {},
            xg_observation_overrides=xg_observations,
            xg_variance_overrides=xg_variances,
        )

    def observe_with_log_rate_offsets(
        self,
        matches: list[Match],
        offsets: dict[str, tuple[float, float]],
        *,
        xg_observation_overrides: dict[str, tuple[float, float]] | None = None,
        xg_variance_overrides: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Observe a date batch after applying known temporary log-rate effects."""
        unknown = offsets.keys() - {match.match_id for match in matches}
        if unknown:
            raise ValueError(f"Observation offsets do not belong to this batch: {sorted(unknown)}")
        participating = [team for match in matches for team in (match.home, match.away)]
        if len(participating) != len(set(participating)):
            raise ValueError(f"A team has multiple matches on {matches[0].date}")
        hfa_updates_active = self.dynamic_home_advantage and (
            self.home_advantage_update_start_date is None or matches[0].date >= self.home_advantage_update_start_date
        )
        designs = []
        observations = []
        rates = []
        dispersions = []
        for match in matches:
            forecast = self.goal_rates(match, strength_shrinkage=1.0)
            home_offset, away_offset = offsets.get(match.match_id, (0.0, 0.0))
            if not all(math.isfinite(value) for value in (home_offset, away_offset)):
                raise ValueError("Observation log-rate offsets must be finite")
            home_observation, away_observation, has_xg = self._match_observations(match, xg_observation_overrides)
            home_xg_variance, away_xg_variance = self._xg_variances(match, xg_variance_overrides)
            for home, observation, log_rate, xg_variance in (
                (True, home_observation, forecast.home_log_rate_mean + home_offset, home_xg_variance),
                (False, away_observation, forecast.away_log_rate_mean + away_offset, away_xg_variance),
            ):
                rate = math.exp(log_rate)
                designs.append(self._design(match, home, include_hfa=hfa_updates_active))
                observations.append(observation)
                rates.append(rate)
                dispersions.append(
                    self._observation_dispersion(
                        rate,
                        has_xg,
                        xg_variance,
                        independent_xg=xg_observation_overrides is not None,
                    )
                )
        design = np.asarray(designs)
        rate = np.asarray(rates)
        innovation_covariance = np.diag(np.asarray(dispersions) / rate) + design @ self.covariance @ design.T
        covariance_design = self.covariance @ design.T
        gain_innovation = np.linalg.solve(
            innovation_covariance,
            (np.asarray(observations) - rate) / rate,
        )
        self.mean += covariance_design @ gain_innovation
        self.covariance -= covariance_design @ np.linalg.solve(innovation_covariance, design @ self.covariance)
        self.covariance = (self.covariance + self.covariance.T) / 2.0
        self._sync_states()
        self._project_attack_gauge()
        self._sync_states()

    def adjust_team_means(self, adjustments: dict[str, tuple[float, float]]) -> None:
        """Apply external attack/defense shifts without changing state uncertainty."""
        unknown = adjustments.keys() - self.active_teams
        if unknown:
            raise ValueError(f"Cannot adjust inactive teams: {sorted(unknown)}")
        self._pull_states()
        for team, (attack, defense) in adjustments.items():
            self.mean[self.index["attack", team]] += attack
            self.mean[self.index["defense", team]] += defense
        shift = float(np.mean([self.mean[self.index["attack", team]] for team in self.active_teams]))
        for team in self.active_teams:
            self.mean[self.index["attack", team]] -= shift
            self.mean[self.index["defense", team]] -= shift
        self._sync_states()

    def add_team_variance(self, adjustments: dict[str, tuple[float, float]]) -> None:
        """Add event uncertainty for active teams while preserving the attack gauge."""
        unknown = adjustments.keys() - self.active_teams
        if unknown:
            raise ValueError(f"Cannot adjust inactive teams: {sorted(unknown)}")
        if any(not math.isfinite(value) or value < 0.0 for values in adjustments.values() for value in values):
            raise ValueError("Variance adjustments must be finite and nonnegative")
        self._pull_states()
        for team, (attack, defense) in adjustments.items():
            self.covariance[self.index["attack", team], self.index["attack", team]] += attack
            self.covariance[self.index["defense", team], self.index["defense", team]] += defense
        self._sync_states()
        self._project_attack_gauge()
        self._sync_states()

    def _project_attack_gauge(self) -> None:
        self._pull_states()
        attack_positions = [self.index["attack", team] for team in sorted(self.active_teams)]
        shift_weights = np.zeros(len(self.parameters))
        shift_weights[attack_positions] = 1.0 / len(attack_positions)
        affected = np.zeros(len(self.parameters))
        for team in self.active_teams:
            affected[self.index["attack", team]] = 1.0
            affected[self.index["defense", team]] = 1.0
        projection = np.eye(len(self.parameters)) - np.outer(affected, shift_weights)
        self.mean = projection @ self.mean
        self.covariance = projection @ self.covariance @ projection.T
        self.covariance = (self.covariance + self.covariance.T) / 2.0

    def _sync_states(self) -> None:
        self.states = {
            team: TeamState(
                float(self.mean[self.index["attack", team]]),
                max(float(self.covariance[self.index["attack", team], self.index["attack", team]]), 0.0),
                float(self.mean[self.index["defense", team]]),
                max(float(self.covariance[self.index["defense", team], self.index["defense", team]]), 0.0),
            )
            for team in self.active_teams
        }

    def _pull_states(self) -> None:
        for team, state in self.states.items():
            attack = self.index["attack", team]
            defense = self.index["defense", team]
            self.mean[attack] = state.attack_mean
            self.mean[defense] = state.defense_mean
            self.covariance[attack, attack] = state.attack_variance
            self.covariance[defense, defense] = state.defense_variance

    def _match_observations(
        self,
        match: Match,
        overrides: dict[str, tuple[float, float]] | None = None,
    ) -> tuple[float, float, bool]:
        xg = (overrides or self.xg_observations).get(match.match_id)
        if xg is None or self.xg_weight == 0.0:
            return float(match.home_goals), float(match.away_goals), False
        return (
            self.xg_weight * xg[0] + (1.0 - self.xg_weight) * match.home_goals,
            self.xg_weight * xg[1] + (1.0 - self.xg_weight) * match.away_goals,
            True,
        )

    def _xg_variances(
        self,
        match: Match,
        overrides: dict[str, tuple[float, float]] | None = None,
    ) -> tuple[float, float]:
        return (overrides or self.xg_observation_variances).get(
            match.match_id,
            (self.xg_disagreement_variance, self.xg_disagreement_variance),
        )

    def _observation_dispersion(
        self,
        predicted_goals: float,
        has_xg: bool,
        xg_variance: float,
        *,
        independent_xg: bool = False,
    ) -> float:
        if not has_xg:
            return 1.0
        if independent_xg:
            return (1.0 - self.xg_weight) ** 2 + self.xg_weight**2 * xg_variance / predicted_goals
        return 1.0 + self.xg_weight**2 * xg_variance / predicted_goals


def walk_forward_joint_covariance(
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
    xg_observations: dict[str, tuple[float, float]] | None = None,
    xg_weight: float = 0.0,
    xg_disagreement_variance: float = 0.0,
    xg_observation_variances: dict[str, tuple[float, float]] | None = None,
    season_prior_covariances: SeasonPriorCovariances | None = None,
    season_mean_adjustments: dict[str, dict[str, tuple[float, float]]] | None = None,
    season_variance_scales: dict[str, dict[str, float]] | None = None,
    prior_transition_mode: PriorTransitionMode = "reset",
    season_prior_persistence: SeasonPriorPersistence | None = None,
    season_team_prior_persistence: SeasonTeamPriorPersistence | None = None,
) -> tuple[list[StagePrediction], dict[str, TeamState]]:
    model = JointCovarianceDCModel(
        annual_drift_variance,
        summer_variance,
        globals_,
        strength_shrinkage,
        canonical_club_ids,
        season_priors,
        home_advantage_drift_variance,
        dynamic_home_advantage,
        crowd_effects_by_season,
        league_state_update_start_date(matches, dynamic_home_advantage),
        xg_observations,
        xg_weight,
        xg_disagreement_variance,
        xg_observation_variances,
        season_prior_covariances,
        season_mean_adjustments,
        season_variance_scales,
        prior_transition_mode,
        season_prior_persistence,
        season_team_prior_persistence,
    )
    predictions = run_walk_forward(matches, model, evaluation_seasons)
    return predictions, {team: replace(state) for team, state in model.states.items()}


JointSequentialDCModel = JointCovarianceDCModel
