"""Adopted promoted-team squad-value mean adjustments."""

from collections import defaultdict

from esa_model.stage2 import CovariateTarget, RidgeFit, fit_returning_prior


def raw_squad_coefficient(fit: RidgeFit) -> float:
    return float(fit.coefficients[0] / fit.feature_scales[0])


def historical_promoted_squad_adjustments(
    covariates: list[CovariateTarget], target_seasons: list[str] | None = None
) -> dict[tuple[str, str], tuple[float, float]]:
    all_seasons = sorted({row.season for row in covariates})
    targets = set(target_seasons or all_seasons[1:])
    adjustments: dict[tuple[str, str], tuple[float, float]] = {}
    for season in sorted(targets):
        fit = fit_returning_prior(covariates, [candidate for candidate in all_seasons if candidate < season])
        attack = raw_squad_coefficient(fit.attack)
        defense = raw_squad_coefficient(fit.defense)
        for row in covariates:
            if row.season == season and row.promoted:
                adjustments[season, row.team_id] = (
                    attack * row.log_relative_squad_value,
                    defense * row.log_relative_squad_value,
                )
    return adjustments


def prospective_promoted_squad_adjustments(
    covariates: list[CovariateTarget], season: str, promoted_log_relative_values: dict[str, float]
) -> dict[tuple[str, str], tuple[float, float]]:
    fit = fit_returning_prior(covariates, sorted({row.season for row in covariates}))
    attack = raw_squad_coefficient(fit.attack)
    defense = raw_squad_coefficient(fit.defense)
    return {(season, team): (attack * value, defense * value) for team, value in promoted_log_relative_values.items()}


def by_season(
    adjustments: dict[tuple[str, str], tuple[float, float]],
) -> dict[str, dict[str, tuple[float, float]]]:
    grouped: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for (season, team), adjustment in adjustments.items():
        grouped[season][team] = adjustment
    return dict(grouped)
