"""Causal I liga states and a drifting bridge into Ekstraklasa priors."""

import csv
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np

from esa_model.baseline import Match
from esa_model.stage1 import FrozenGlobals, SequentialDCModel, TeamState
from esa_model.stage2 import CovariateTarget, build_hybrid_rolling_priors, winter_cutoffs

TEAM_DRIFT_VARIANCE = 0.019682003781078598
SUMMER_VARIANCE = 0.0030916820425413915
HFA_DRIFT_VARIANCE = 0.0031622776601683794
CUP_OBSERVATION_DISPERSION = 2.0
TRANSITION_VARIANCE = 0.03
PROMOTED_ATTACK_VARIANCE_FLOOR = 0.01323176549009603
PROMOTED_DEFENSE_VARIANCE_FLOOR = 0.017292811381282303
CUP_SCORING, CUP_HFA, ATTACK_OFFSET, DEFENSE_OFFSET, CUP_BIAS, PROMOTION_BIAS, RELEGATION_BIAS = range(7)
BRIDGE_PROCESS_VARIANCES = np.asarray((0.0025, 0.0025, 0.01, 0.01, 0.0, 0.0, 0.0))
SOURCE_BIAS_VARIANCE = 0.01
TOP_TWO_TIERS = frozenset(("ekstraklasa", "i_liga"))


@dataclass(frozen=True, slots=True)
class AuxiliaryMatch:
    match_id: str
    date: date
    season: str
    competition: str
    home: str
    home_tier: str
    away: str
    away_tier: str
    home_goals: int
    away_goals: int


@dataclass(frozen=True, slots=True)
class LeagueHistory:
    snapshots: dict[tuple[str, str], TeamState]
    final_states: dict[tuple[str, str], TeamState]
    final_dates: dict[str, date]
    winter_states: dict[tuple[str, str], TeamState]
    winter_dates: dict[str, date]
    teams_by_season: dict[str, set[str]]


@dataclass(slots=True)
class BridgeState:
    mean: np.ndarray
    covariance: np.ndarray
    season: str
    cup_matches: int = 0
    cross_tier_matches: int = 0
    promotion_transitions: int = 0
    relegation_transitions: int = 0

    def copy(self) -> BridgeState:
        return BridgeState(
            self.mean.copy(),
            self.covariance.copy(),
            self.season,
            self.cup_matches,
            self.cross_tier_matches,
            self.promotion_transitions,
            self.relegation_transitions,
        )


@dataclass(frozen=True, slots=True)
class TransitionObservation:
    date: date
    season: str
    kind: str
    attack_value: float
    attack_variance: float
    defense_value: float
    defense_variance: float


@dataclass(frozen=True, slots=True)
class CupObservation:
    design: np.ndarray
    base_mean: float
    base_variance: float
    goals: int


def load_global_club_ids(path: Path) -> dict[str, str]:
    candidates: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            candidates[row["local_team_id"]].add(row["canonical_club_id"])
    ambiguous = {club_id: values for club_id, values in candidates.items() if len(values) != 1}
    if ambiguous:
        raise ValueError(f"Local 90minut IDs map to multiple canonical clubs: {ambiguous}")
    return {club_id: next(iter(values)) for club_id, values in candidates.items()}


def canonical_club_id(raw_id: str, global_ids: dict[str, str]) -> str:
    return global_ids.get(raw_id, f"90:{raw_id}")


def load_auxiliary_matches(path: Path, mapping_path: Path) -> list[AuxiliaryMatch]:
    global_ids = load_global_club_ids(mapping_path)
    output: list[AuxiliaryMatch] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            match_id = f"aux:{row['match_id']}"
            if match_id in seen:
                raise ValueError(f"Duplicate auxiliary match ID {match_id}")
            seen.add(match_id)
            output.append(
                AuxiliaryMatch(
                    match_id,
                    date.fromisoformat(row["date"]),
                    row["season"],
                    row["competition"],
                    canonical_club_id(row["home_club_id"], global_ids),
                    row["home_tier"],
                    canonical_club_id(row["away_club_id"], global_ids),
                    row["away_tier"],
                    int(row["home_goals"]),
                    int(row["away_goals"]),
                )
            )
    return sorted(output, key=lambda match: (match.date, match.match_id))


def top_two_cup_matches(matches: list[AuxiliaryMatch]) -> list[AuxiliaryMatch]:
    """Keep the historical bridge isolated from side-simulator lower tiers."""
    return [
        match
        for match in matches
        if match.competition == "polish_cup" and match.home_tier in TOP_TWO_TIERS and match.away_tier in TOP_TWO_TIERS
    ]


def canonical_ek_matches(matches: list[Match], canonical_ids: dict[tuple[str, str], str]) -> list[Match]:
    return [
        Match(
            f"ek:{match.match_id}",
            match.date,
            match.season,
            canonical_ids[match.season, match.home],
            canonical_ids[match.season, match.away],
            match.home_goals,
            match.away_goals,
        )
        for match in matches
    ]


def as_league_matches(matches: list[AuxiliaryMatch]) -> list[Match]:
    return [
        Match(
            match.match_id,
            match.date,
            match.season,
            match.home,
            match.away,
            match.home_goals,
            match.away_goals,
        )
        for match in matches
        if match.competition == "i_liga"
    ]


def extract_league_history(
    matches: list[Match],
    globals_: FrozenGlobals,
    query_matches: list[AuxiliaryMatch],
) -> LeagueHistory:
    teams_by_season: dict[str, set[str]] = defaultdict(set)
    matches_by_date: dict[date, list[Match]] = defaultdict(list)
    queries_by_date: dict[date, list[AuxiliaryMatch]] = defaultdict(list)
    last_date: dict[str, date] = {}
    for match in matches:
        teams_by_season[match.season].update((match.home, match.away))
        matches_by_date[match.date].append(match)
        last_date[match.season] = max(last_date.get(match.season, match.date), match.date)
    for match in query_matches:
        queries_by_date[match.date].append(match)
    cutoffs = winter_cutoffs(matches)
    model = SequentialDCModel(
        TEAM_DRIFT_VARIANCE,
        SUMMER_VARIANCE,
        globals_,
        home_advantage_drift_variance=HFA_DRIFT_VARIANCE,
        dynamic_home_advantage=True,
    )
    snapshots: dict[tuple[str, str], TeamState] = {}
    final_states: dict[tuple[str, str], TeamState] = {}
    winter_states: dict[tuple[str, str], TeamState] = {}
    dates = sorted(matches_by_date.keys() | queries_by_date.keys())
    for match_date in dates:
        date_matches = matches_by_date.get(match_date, [])
        date_queries = queries_by_date.get(match_date, [])
        season = date_matches[0].season if date_matches else date_queries[0].season
        if season not in teams_by_season:
            continue
        active = teams_by_season[season]
        model.advance_to(match_date, season, active)
        for query in date_queries:
            for _tier, club in ((query.home_tier, query.home), (query.away_tier, query.away)):
                if club in model.states:
                    snapshots[query.match_id, club] = replace(model.states[club])
        if date_matches:
            if match_date == cutoffs[season]:
                for club in active:
                    winter_states[season, club] = replace(model.states[club])
            model.observe(date_matches)
            if match_date == last_date[season]:
                for club in active:
                    final_states[season, club] = replace(model.states[club])
    return LeagueHistory(snapshots, final_states, last_date, winter_states, cutoffs, dict(teams_by_season))


def offseason_variance(source_date: date, target_date: date) -> float:
    if target_date < source_date:
        raise ValueError(f"Target date {target_date} precedes source date {source_date}")
    elapsed_years = (target_date - source_date).days / 365.0
    return SUMMER_VARIANCE + TEAM_DRIFT_VARIANCE * elapsed_years


def transition_observations(ek: LeagueHistory, liga: LeagueHistory) -> list[TransitionObservation]:
    observations: list[TransitionObservation] = []
    seasons = sorted(ek.teams_by_season)
    for index in range(1, len(seasons)):
        season = seasons[index]
        previous = seasons[index - 1]
        promoted = ek.teams_by_season[season] - ek.teams_by_season[previous]
        for club in promoted:
            source = liga.final_states.get((previous, club))
            target = ek.winter_states.get((season, club))
            if source is None or target is None:
                continue
            inactive_variance = offseason_variance(liga.final_dates[previous], ek.winter_dates[season])
            observations.append(
                TransitionObservation(
                    ek.winter_dates[season],
                    season,
                    "promotion",
                    target.attack_mean - source.attack_mean,
                    target.attack_variance + source.attack_variance + inactive_variance + TRANSITION_VARIANCE,
                    target.defense_mean - source.defense_mean,
                    target.defense_variance + source.defense_variance + inactive_variance + TRANSITION_VARIANCE,
                )
            )
        relegated = ek.teams_by_season[previous] - ek.teams_by_season[season]
        for club in relegated:
            source = ek.final_states.get((previous, club))
            target = liga.winter_states.get((season, club))
            if source is None or target is None:
                continue
            inactive_variance = offseason_variance(ek.final_dates[previous], liga.winter_dates[season])
            observations.append(
                TransitionObservation(
                    liga.winter_dates[season],
                    season,
                    "relegation",
                    source.attack_mean - target.attack_mean,
                    source.attack_variance + target.attack_variance + inactive_variance + TRANSITION_VARIANCE,
                    source.defense_mean - target.defense_mean,
                    source.defense_variance + target.defense_variance + inactive_variance + TRANSITION_VARIANCE,
                )
            )
    return sorted(observations, key=lambda row: (row.date, row.kind))


def initial_bridge(globals_: FrozenGlobals, first_season: str) -> BridgeState:
    mean = np.asarray((globals_.scoring_intercept, 0.0, -0.15, -0.15, 0.0, 0.0, 0.0))
    covariance = np.diag((0.1, 0.1, 0.15, 0.15, SOURCE_BIAS_VARIANCE, SOURCE_BIAS_VARIANCE, SOURCE_BIAS_VARIANCE))
    return BridgeState(mean, covariance, first_season)


def advance_bridge(
    state: BridgeState,
    season: str,
    process_variances: np.ndarray = BRIDGE_PROCESS_VARIANCES,
) -> None:
    elapsed = max(int(season[:4]) - int(state.season[:4]), 0)
    if elapsed:
        state.covariance += np.diag(process_variances * elapsed)
        state.season = season


def poisson_batch_update(
    state: BridgeState,
    observations: list[CupObservation],
) -> None:
    if not observations:
        return
    size = len(state.mean)
    information = np.zeros((size, size))
    score = np.zeros(size)
    for observation in observations:
        rate = math.exp(
            observation.base_mean + float(observation.design @ state.mean) + observation.base_variance / 2.0
        )
        adjustment = CUP_OBSERVATION_DISPERSION + rate * math.expm1(observation.base_variance)
        information += rate / adjustment * np.outer(observation.design, observation.design)
        score += (observation.goals - rate) / adjustment * observation.design
    precision = np.linalg.inv(state.covariance) + information
    posterior = np.linalg.inv(precision)
    state.mean += posterior @ score
    state.covariance = (posterior + posterior.T) / 2.0


def gaussian_linear_update(state: BridgeState, design: np.ndarray, value: float, variance: float) -> None:
    innovation_variance = variance + float(design @ state.covariance @ design)
    gain = state.covariance @ design / innovation_variance
    state.mean += gain * (value - float(design @ state.mean))
    state.covariance -= np.outer(gain, design @ state.covariance)
    state.covariance = (state.covariance + state.covariance.T) / 2.0


def gaussian_component_update(state: BridgeState, index: int, value: float, variance: float) -> None:
    design = np.zeros(len(state.mean))
    design[index] = 1.0
    gaussian_linear_update(state, design, value, variance)


def cup_observations(
    match: AuxiliaryMatch,
    ek: LeagueHistory,
    liga: LeagueHistory,
) -> list[CupObservation]:
    histories = {"ekstraklasa": ek, "i_liga": liga}
    home = histories[match.home_tier].snapshots.get((match.match_id, match.home))
    away = histories[match.away_tier].snapshots.get((match.match_id, match.away))
    if home is None or away is None:
        return []
    home_lower = float(match.home_tier == "i_liga")
    away_lower = float(match.away_tier == "i_liga")
    home_design = np.asarray((1.0, 1.0, home_lower, -away_lower, home_lower - away_lower, 0.0, 0.0))
    away_design = np.asarray((1.0, 0.0, away_lower, -home_lower, away_lower - home_lower, 0.0, 0.0))
    return [
        CupObservation(
            home_design,
            home.attack_mean - away.defense_mean,
            home.attack_variance + away.defense_variance,
            match.home_goals,
        ),
        CupObservation(
            away_design,
            away.attack_mean - home.defense_mean,
            away.attack_variance + home.defense_variance,
            match.away_goals,
        ),
    ]


def build_bridge_snapshots(
    cup_matches: list[AuxiliaryMatch],
    transitions: list[TransitionObservation],
    ek: LeagueHistory,
    liga: LeagueHistory,
    season_openings: dict[str, date],
    globals_: FrozenGlobals,
    process_variances: np.ndarray = BRIDGE_PROCESS_VARIANCES,
) -> dict[str, BridgeState]:
    first_season = min({match.season for match in cup_matches} | set(season_openings))
    state = initial_bridge(globals_, first_season)
    cups_by_date: dict[date, list[AuxiliaryMatch]] = defaultdict(list)
    transitions_by_date: dict[date, list[TransitionObservation]] = defaultdict(list)
    openings_by_date: dict[date, list[str]] = defaultdict(list)
    for match in cup_matches:
        cups_by_date[match.date].append(match)
    for row in transitions:
        transitions_by_date[row.date].append(row)
    for season, opening in season_openings.items():
        openings_by_date[opening].append(season)
    snapshots: dict[str, BridgeState] = {}
    for event_date in sorted(cups_by_date.keys() | transitions_by_date.keys() | openings_by_date.keys()):
        event_seasons = openings_by_date.get(event_date) or [
            (cups_by_date.get(event_date) or transitions_by_date[event_date])[0].season
        ]
        advance_bridge(state, event_seasons[0], process_variances)
        for season in openings_by_date.get(event_date, []):
            snapshots[season] = state.copy()
        observations: list[CupObservation] = []
        for match in cups_by_date.get(event_date, []):
            match_observations = cup_observations(match, ek, liga)
            if not match_observations:
                continue
            observations.extend(match_observations)
            state.cup_matches += 1
            state.cross_tier_matches += int(match.home_tier != match.away_tier)
        poisson_batch_update(state, observations)
        for row in transitions_by_date.get(event_date, []):
            bias_index = PROMOTION_BIAS if row.kind == "promotion" else RELEGATION_BIAS
            attack_design = np.zeros(len(state.mean))
            attack_design[ATTACK_OFFSET] = 1.0
            attack_design[bias_index] = 1.0
            defense_design = np.zeros(len(state.mean))
            defense_design[DEFENSE_OFFSET] = 1.0
            defense_design[bias_index] = 1.0
            gaussian_linear_update(state, attack_design, row.attack_value, row.attack_variance)
            gaussian_linear_update(state, defense_design, row.defense_value, row.defense_variance)
            if row.kind == "promotion":
                state.promotion_transitions += 1
            else:
                state.relegation_transitions += 1
    return snapshots


def build_stage3_priors(
    covariates: list[CovariateTarget],
    liga: LeagueHistory,
    bridge: dict[str, BridgeState],
    season_openings: dict[str, date],
) -> tuple[dict[tuple[str, str], TeamState], dict[str, int]]:
    priors, _ = build_hybrid_rolling_priors(covariates)
    added: dict[str, int] = defaultdict(int)
    seasons = sorted({row.season for row in covariates})
    for row in covariates:
        if not row.promoted or row.season not in bridge:
            continue
        index = seasons.index(row.season)
        if index == 0:
            continue
        previous = seasons[index - 1]
        source = liga.final_states.get((previous, row.canonical_club_id))
        if source is None:
            continue
        state = bridge[row.season]
        inactive_variance = offseason_variance(liga.final_dates[previous], season_openings[row.season])
        attack_design = np.zeros(len(state.mean))
        attack_design[ATTACK_OFFSET] = 1.0
        attack_design[PROMOTION_BIAS] = 1.0
        defense_design = np.zeros(len(state.mean))
        defense_design[DEFENSE_OFFSET] = 1.0
        defense_design[PROMOTION_BIAS] = 1.0
        priors[row.season, row.team_id] = TeamState(
            source.attack_mean + float(attack_design @ state.mean),
            max(
                source.attack_variance + inactive_variance + float(attack_design @ state.covariance @ attack_design),
                PROMOTED_ATTACK_VARIANCE_FLOOR,
            ),
            source.defense_mean + float(defense_design @ state.mean),
            max(
                source.defense_variance + inactive_variance + float(defense_design @ state.covariance @ defense_design),
                PROMOTED_DEFENSE_VARIANCE_FLOOR,
            ),
        )
        added[row.season] += 1
    return priors, dict(added)


def build_rank_priors(
    parent_priors: dict[tuple[str, str], TeamState],
    mapped_priors: dict[tuple[str, str], TeamState],
    parent_hedges: dict[tuple[str, str], TeamState],
    covariates: list[CovariateTarget],
    center_on_parent: bool,
) -> dict[tuple[str, str], TeamState]:
    priors = {key: replace(value) for key, value in parent_priors.items()}
    promoted_by_season: dict[str, list[CovariateTarget]] = defaultdict(list)
    for row in covariates:
        if row.promoted and (row.season, row.team_id) in mapped_priors and (row.season, row.team_id) in parent_hedges:
            promoted_by_season[row.season].append(row)
    for season, rows in promoted_by_season.items():
        mapped = [mapped_priors[season, row.team_id] for row in rows]
        attack_center = float(np.mean([state.attack_mean for state in mapped]))
        defense_center = float(np.mean([state.defense_mean for state in mapped]))
        for row, state in zip(rows, mapped, strict=True):
            hedge = parent_hedges[season, row.team_id]
            priors[season, row.team_id] = TeamState(
                hedge.attack_mean + state.attack_mean - attack_center if center_on_parent else state.attack_mean,
                hedge.attack_variance,
                hedge.defense_mean + state.defense_mean - defense_center if center_on_parent else state.defense_mean,
                hedge.defense_variance,
            )
    return priors
