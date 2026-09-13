"""Covariate-informed season-start priors for the sequential strength model."""

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np

from esa_model.baseline import Match, load_matches
from esa_model.stage1 import FrozenGlobals, SequentialDCModel, TeamState, freeze_globals, league_state_update_start_date
from esa_model.team_identity import load_canonical_club_ids

COVARIATE_START_SEASON = "2008/09"
RIDGE_ALPHA = 1.0
VARIANCE_FLOOR = 1e-4
FEATURE_NAMES = (
    "log_relative_squad_value",
    "promoted_flag",
    "promoted_liga1_points_rate_zscore",
    "last_season_mu",
)


@dataclass(frozen=True, slots=True)
class StateTarget:
    season: str
    cutoff_date: date
    team_id: str
    canonical_club_id: str
    matches_observed: int
    attack_mean: float
    attack_variance: float
    defense_mean: float
    defense_variance: float


@dataclass(frozen=True, slots=True)
class CovariateTarget:
    season: str
    team_id: str
    canonical_club_id: str
    promoted: bool
    log_relative_squad_value: float
    promoted_liga1_points_rate_zscore: float
    last_attack_mean: float
    last_defense_mean: float
    target_attack_mean: float
    target_attack_variance: float
    target_defense_mean: float
    target_defense_variance: float
    target_cutoff_date: date
    matches_observed: int

    def features(self, target: str) -> tuple[float, float, float, float]:
        last_mean = self.last_attack_mean if target == "attack" else self.last_defense_mean
        return (
            self.log_relative_squad_value,
            float(self.promoted),
            self.promoted_liga1_points_rate_zscore,
            last_mean,
        )

    def response(self, target: str) -> float:
        return self.target_attack_mean if target == "attack" else self.target_defense_mean


@dataclass(frozen=True, slots=True)
class RidgeFit:
    target: str
    intercept: float
    coefficients: tuple[float, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]

    def predict(self, row: CovariateTarget) -> float:
        values = row.features(self.target)
        standardized = [
            (value - mean) / scale
            for value, mean, scale in zip(values, self.feature_means, self.feature_scales, strict=True)
        ]
        return self.intercept + sum(
            coefficient * value for coefficient, value in zip(self.coefficients, standardized, strict=True)
        )


@dataclass(frozen=True, slots=True)
class PriorFit:
    attack: RidgeFit
    defense: RidgeFit
    promoted_attack_variance: float
    returning_attack_variance: float
    promoted_defense_variance: float
    returning_defense_variance: float


def winter_cutoffs(matches: list[Match]) -> dict[str, date]:
    dates: dict[str, list[date]] = defaultdict(list)
    for match in matches:
        if match.date.year == int(match.season[:4]) + 1:
            dates[match.season].append(match.date)
    missing = sorted({match.season for match in matches} - dates.keys())
    if missing:
        raise ValueError(f"Seasons have no post-winter matches: {missing}")
    return {season: min(season_dates) for season, season_dates in dates.items()}


def extract_state_history(
    matches: list[Match],
    annual_drift_variance: float,
    summer_variance: float,
    globals_: FrozenGlobals,
    canonical_ids: dict[tuple[str, str], str],
    home_advantage_drift_variance: float = 0.0,
    dynamic_home_advantage: bool = False,
    xg_observations: dict[str, tuple[float, float]] | None = None,
    xg_weight: float = 0.0,
    xg_observation_variances: dict[str, tuple[float, float]] | None = None,
) -> tuple[list[StateTarget], dict[tuple[str, str], TeamState]]:
    teams_by_season: dict[str, set[str]] = defaultdict(set)
    matches_by_date: dict[date, list[Match]] = defaultdict(list)
    last_date_by_season: dict[str, date] = {}
    for match in matches:
        teams_by_season[match.season].update((match.home, match.away))
        matches_by_date[match.date].append(match)
        last_date_by_season[match.season] = max(last_date_by_season.get(match.season, match.date), match.date)
    cutoffs = winter_cutoffs(matches)
    model = SequentialDCModel(
        annual_drift_variance,
        summer_variance,
        globals_,
        canonical_club_ids=canonical_ids,
        home_advantage_drift_variance=home_advantage_drift_variance,
        dynamic_home_advantage=dynamic_home_advantage,
        home_advantage_update_start_date=league_state_update_start_date(matches, dynamic_home_advantage),
        xg_observations=xg_observations,
        xg_weight=xg_weight,
        xg_observation_variances=xg_observation_variances,
    )
    appearances: dict[tuple[str, str], int] = defaultdict(int)
    targets: list[StateTarget] = []
    final_states: dict[tuple[str, str], TeamState] = {}
    for match_date, date_matches in sorted(matches_by_date.items()):
        season = date_matches[0].season
        model.advance_to(match_date, season, teams_by_season[season])
        if match_date == cutoffs[season]:
            for team in sorted(teams_by_season[season], key=int):
                state = model.states[team]
                targets.append(
                    StateTarget(
                        season,
                        match_date,
                        team,
                        canonical_ids[season, team],
                        appearances[season, team],
                        state.attack_mean,
                        state.attack_variance,
                        state.defense_mean,
                        state.defense_variance,
                    )
                )
        model.observe(date_matches)
        for match in date_matches:
            appearances[season, match.home] += 1
            appearances[season, match.away] += 1
        if match_date == last_date_by_season[season]:
            for team in teams_by_season[season]:
                final_states[season, canonical_ids[season, team]] = replace(model.states[team])
    return targets, final_states


def load_squad_features(path: Path) -> dict[tuple[str, str], float]:
    by_season: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if row["period"] == "season_start_prior":
                by_season[row["season"]].append(row)
    features: dict[tuple[str, str], float] = {}
    for season, rows in by_season.items():
        values = [float(row["squad_value_eur"]) for row in rows]
        median = float(np.median(values))
        for row in rows:
            features[season, row["local_team_id"]] = math.log(float(row["squad_value_eur"]) / median)
    return features


def load_promotion_features(path: Path) -> dict[tuple[str, str], tuple[bool, float]]:
    with path.open(encoding="utf-8", newline="") as source:
        return {
            (row["season"], row["team_id"]): (
                row["promoted_flag"] == "1",
                float(row["liga1_points_rate_zscore"]) if row["promoted_flag"] == "1" else 0.0,
            )
            for row in csv.DictReader(source)
        }


def assemble_covariate_targets(
    targets: list[StateTarget],
    previous_states: dict[tuple[str, str], TeamState],
    squad_features: dict[tuple[str, str], float],
    promotion_features: dict[tuple[str, str], tuple[bool, float]],
) -> list[CovariateTarget]:
    rows: list[CovariateTarget] = []
    for target in targets:
        if target.season < COVARIATE_START_SEASON:
            continue
        promoted, points_rate_zscore = promotion_features[target.season, target.team_id]
        previous = previous_states.get((target.season, target.canonical_club_id))
        if promoted:
            last_attack = last_defense = 0.0
        elif previous is not None:
            last_attack = previous.attack_mean
            last_defense = previous.defense_mean
        else:
            raise ValueError(
                f"Returning team {target.team_id} in {target.season} has no canonical previous-season state"
            )
        rows.append(
            CovariateTarget(
                target.season,
                target.team_id,
                target.canonical_club_id,
                promoted,
                squad_features[target.season, target.team_id],
                points_rate_zscore if promoted else 0.0,
                last_attack,
                last_defense,
                target.attack_mean,
                target.attack_variance,
                target.defense_mean,
                target.defense_variance,
                target.cutoff_date,
                target.matches_observed,
            )
        )
    return rows


def fit_ridge(rows: list[CovariateTarget], target: str, alpha: float = RIDGE_ALPHA) -> RidgeFit:
    if not rows:
        raise ValueError("Cannot fit a Stage 2 prior without training rows")
    features = np.asarray([row.features(target) for row in rows], dtype=float)
    response = np.asarray([row.response(target) for row in rows], dtype=float)
    means = features.mean(axis=0)
    scales = features.std(axis=0)
    scales[scales == 0.0] = 1.0
    design = np.column_stack((np.ones(len(rows)), (features - means) / scales))
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ response)
    return RidgeFit(
        target,
        float(coefficients[0]),
        tuple(float(value) for value in coefficients[1:]),
        tuple(float(value) for value in means),
        tuple(float(value) for value in scales),
    )


def expanding_residuals(
    rows: list[CovariateTarget], training_seasons: list[str], target: str
) -> list[tuple[bool, float]]:
    residuals: list[tuple[bool, float]] = []
    for index in range(1, len(training_seasons)):
        prior_seasons = set(training_seasons[:index])
        train = [row for row in rows if row.season in prior_seasons]
        if len(train) < 16:
            continue
        fit = fit_ridge(train, target)
        for row in rows:
            if row.season == training_seasons[index]:
                residuals.append((row.promoted, row.response(target) - fit.predict(row)))
    return residuals


def residual_variance(
    residuals: list[tuple[bool, float]],
    promoted: bool,
    training: list[CovariateTarget],
    target: str,
) -> float:
    selected = [residual for group, residual in residuals if group == promoted]
    if len(selected) < 2:
        selected = [residual for _, residual in residuals]
    if len(selected) < 2:
        fallback = [row for row in training if row.promoted == promoted] or training
        if len(fallback) < 2:
            fallback = training
        responses = [row.response(target) for row in fallback]
        posterior = [
            row.target_attack_variance if target == "attack" else row.target_defense_variance for row in fallback
        ]
        return max(float(np.var(responses, ddof=1)) + float(np.mean(posterior)), VARIANCE_FLOOR)
    return max(float(np.var(selected, ddof=1)), VARIANCE_FLOOR)


def fit_prior(rows: list[CovariateTarget], training_seasons: list[str]) -> PriorFit:
    training = [row for row in rows if row.season in set(training_seasons)]
    attack = fit_ridge(training, "attack")
    defense = fit_ridge(training, "defense")
    attack_residuals = expanding_residuals(rows, training_seasons, "attack")
    defense_residuals = expanding_residuals(rows, training_seasons, "defense")
    return PriorFit(
        attack,
        defense,
        residual_variance(attack_residuals, True, training, "attack"),
        residual_variance(attack_residuals, False, training, "attack"),
        residual_variance(defense_residuals, True, training, "defense"),
        residual_variance(defense_residuals, False, training, "defense"),
    )


def build_rolling_priors(rows: list[CovariateTarget]) -> tuple[dict[tuple[str, str], TeamState], dict[str, PriorFit]]:
    seasons = sorted({row.season for row in rows})
    priors: dict[tuple[str, str], TeamState] = {}
    fits: dict[str, PriorFit] = {}
    for index in range(1, len(seasons)):
        season = seasons[index]
        fit = fit_prior(rows, seasons[:index])
        fits[season] = fit
        for row in rows:
            if row.season != season:
                continue
            priors[season, row.team_id] = TeamState(
                fit.attack.predict(row),
                fit.promoted_attack_variance if row.promoted else fit.returning_attack_variance,
                fit.defense.predict(row),
                fit.promoted_defense_variance if row.promoted else fit.returning_defense_variance,
            )
    return priors, fits


def fit_returning_prior(rows: list[CovariateTarget], training_seasons: list[str]) -> PriorFit:
    full_fit = fit_prior(rows, training_seasons)
    training = [row for row in rows if row.season in set(training_seasons) and not row.promoted]
    attack = fit_ridge(training, "attack")
    defense = fit_ridge(training, "defense")
    attack_residuals = expanding_residuals([row for row in rows if not row.promoted], training_seasons, "attack")
    defense_residuals = expanding_residuals([row for row in rows if not row.promoted], training_seasons, "defense")
    return PriorFit(
        attack,
        defense,
        full_fit.promoted_attack_variance,
        residual_variance(attack_residuals, False, training, "attack"),
        full_fit.promoted_defense_variance,
        residual_variance(defense_residuals, False, training, "defense"),
    )


def build_hybrid_rolling_priors(
    rows: list[CovariateTarget],
) -> tuple[dict[tuple[str, str], TeamState], dict[str, PriorFit]]:
    seasons = sorted({row.season for row in rows})
    priors: dict[tuple[str, str], TeamState] = {}
    fits: dict[str, PriorFit] = {}
    for index in range(1, len(seasons)):
        season = seasons[index]
        fit = fit_returning_prior(rows, seasons[:index])
        fits[season] = fit
        for row in rows:
            if row.season != season or row.promoted:
                continue
            priors[season, row.team_id] = TeamState(
                fit.attack.predict(row),
                fit.returning_attack_variance,
                fit.defense.predict(row),
                fit.returning_defense_variance,
            )
    return priors, fits


def coefficient_rows(fits: dict[str, PriorFit]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for season, prior in sorted(fits.items()):
        for fit in (prior.attack, prior.defense):
            row: dict[str, object] = {"evaluation_season": season, "target": fit.target, "intercept": fit.intercept}
            row.update(dict(zip(FEATURE_NAMES, fit.coefficients, strict=True)))
            row["promoted_residual_variance"] = (
                prior.promoted_attack_variance if fit.target == "attack" else prior.promoted_defense_variance
            )
            row["returning_residual_variance"] = (
                prior.returning_attack_variance if fit.target == "attack" else prior.returning_defense_variance
            )
            output.append(row)
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_stage2_data(
    matches_path: Path = Path("data/matches.csv"),
    mapping_path: Path = Path("data/team_mapping.csv"),
    squad_path: Path = Path("data/squad_values.csv"),
    promotion_path: Path = Path("data/promoted_teams.csv"),
) -> tuple[list[Match], dict[tuple[str, str], str], list[CovariateTarget], FrozenGlobals]:
    matches = load_matches(matches_path)
    seasons = sorted({match.season for match in matches})
    globals_, _ = freeze_globals(matches, seasons)
    canonical_ids = load_canonical_club_ids(mapping_path)
    with Path("diagnostics/rolling_pseudo_holdout.json").open(encoding="utf-8") as source:
        stage1_folds = json.load(source)["folds"]
    parameters_by_season = {
        str(row["test_season"]): (
            float(row["selected_annual_drift_variance"]),
            float(row["selected_summer_variance"]),
        )
        for row in stage1_folds
    }
    covariate_seasons = [season for season in seasons if season >= COVARIATE_START_SEASON]
    earliest_parameters = parameters_by_season[min(parameters_by_season)]
    history_cache: dict[tuple[float, float], tuple[list[StateTarget], dict[tuple[str, str], TeamState]]] = {}
    targets = []
    previous_states: dict[tuple[str, str], TeamState] = {}
    for season in covariate_seasons:
        parameters = parameters_by_season.get(season, earliest_parameters)
        if parameters not in history_cache:
            history_cache[parameters] = extract_state_history(matches, *parameters, globals_, canonical_ids)
        history_targets, final_states = history_cache[parameters]
        season_targets = [target for target in history_targets if target.season == season]
        targets.extend(season_targets)
        previous_season = seasons[seasons.index(season) - 1]
        for target in season_targets:
            previous = final_states.get((previous_season, target.canonical_club_id))
            if previous is not None:
                previous_states[season, target.canonical_club_id] = previous
    rows = assemble_covariate_targets(
        targets,
        previous_states,
        load_squad_features(squad_path),
        load_promotion_features(promotion_path),
    )
    return matches, canonical_ids, rows, globals_


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=Path("diagnostics"))
    args = parser.parse_args()
    args.output_directory.mkdir(exist_ok=True)
    _, _, rows, _ = load_stage2_data()
    _, fits = build_hybrid_rolling_priors(rows)
    target_rows = []
    for row in rows:
        record = asdict(row)
        record["target_cutoff_date"] = row.target_cutoff_date.isoformat()
        target_rows.append(record)
    write_csv(args.output_directory / "stage2_covariate_targets.csv", target_rows)
    write_csv(args.output_directory / "stage2_coefficients.csv", coefficient_rows(fits))
    print(json.dumps({"team_seasons": len(rows), "seasons": len({row.season for row in rows})}, indent=2))


if __name__ == "__main__":
    main()
