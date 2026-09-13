# ruff: noqa: I001 -- imports resolve as third-party only after export.

from datetime import date

import numpy as np
import pytest

from esa_model.baseline import Match, fit, probabilities, rps
from esa_model.chronological_simulation import simulate_seasons_chronologically
from esa_model.cup_projection import CupProjectionContext, CupSeason, CupTeam, project_cup
from esa_model.joint_covariance import JointCovarianceDCModel
from esa_model.posterior_simulation import simulate_seasons_from_posterior
from esa_model.relocated_home_policy import RelocatedHomeHfaPolicy, RelocatedHomePeriod
from esa_model.stage1 import FrozenGlobals, dixon_coles_score_matrix
from esa_model.stage3 import BridgeState


def synthetic_model() -> JointCovarianceDCModel:
    model = JointCovarianceDCModel(0.01, 0.02, FrozenGlobals(0.05, 0.15, -0.06, 0.1, 0.1))
    model.advance_to(date(2030, 7, 31), "2030/31", {"a", "b", "c", "d"})
    return model


def synthetic_fixtures() -> list[Match]:
    return [
        Match("s1", date(2030, 8, 1), "2030/31", "a", "b", 0, 0),
        Match("s2", date(2030, 8, 1), "2030/31", "c", "d", 0, 0),
        Match("s3", date(2030, 8, 8), "2030/31", "b", "c", 0, 0),
        Match("s4", date(2030, 8, 8), "2030/31", "d", "a", 0, 0),
    ]


def test_score_matrix_is_a_distribution() -> None:
    matrix = dixon_coles_score_matrix(1.4, 1.1, -0.06)
    assert np.all(matrix >= 0.0)
    assert np.isclose(float(matrix.sum()), 1.0, rtol=0.0, atol=1e-15)


def test_baseline_fits_caller_supplied_synthetic_matches() -> None:
    matches = [
        Match(f"m{index}", date(2030, 1, index), "2029/30", home, away, home_goals, away_goals)
        for index, (home, away, home_goals, away_goals) in enumerate(
            (
                ("a", "b", 2, 0),
                ("b", "c", 1, 1),
                ("c", "d", 0, 1),
                ("d", "a", 1, 2),
                ("b", "a", 0, 1),
                ("c", "b", 2, 1),
                ("d", "c", 1, 0),
                ("a", "d", 2, 2),
            ),
            start=1,
        )
    ]
    model = fit(matches, date(2030, 2, 1), xi=0.0)
    chance = probabilities(float(np.exp(model.home_advantage)), 1.0, model.rho)

    assert model.converged
    assert model.teams == ("a", "b", "c", "d")
    assert sum(chance) == pytest.approx(1.0)
    assert 0.0 <= rps(chance, 0) <= 1.0


def test_cup_projection_accepts_a_caller_constructed_context() -> None:
    model = synthetic_model()
    season = CupSeason(
        "2030/31",
        "active",
        {"a": CupTeam("a", "A", "ekstraklasa", 1), "b": CupTeam("b", "B", "ekstraklasa", 1)},
        {"final": ("a", "b")},
        {},
    )
    cutoff = date(2030, 8, 1)
    context = CupProjectionContext(
        {season.season: season},
        {},
        {},
        {(season.season, cutoff): BridgeState(np.zeros(7), np.eye(7) * 0.01, season.season)},
    )

    projection = project_cup(context, season.season, cutoff, "pre_match", model, simulations=32)

    assert {entry.team_id for entry in projection.entries} == {"a", "b"}
    assert sum(entry.probability for entry in projection.entries) == pytest.approx(1.0)


def test_static_posterior_simulation_needs_no_private_files() -> None:
    result = simulate_seasons_from_posterior(
        synthetic_model(),
        ["a", "b", "c", "d"],
        synthetic_fixtures(),
        simulations=16,
        seed=123,
        relocation_policy=RelocatedHomeHfaPolicy(()),
    )
    expected, places, intervals = result
    assert expected.shape == (4,)
    np.testing.assert_allclose(places.sum(axis=1), 100.0)
    assert intervals.shape == (4, 2)


def test_chronological_workers_preserve_fixed_seed_results_with_custom_policy() -> None:
    teams = ["a", "b", "c", "d"]
    fixtures = synthetic_fixtures()
    relocation_policy = RelocatedHomeHfaPolicy(
        (
            RelocatedHomePeriod(
                "synthetic-relocation",
                "a",
                date(2030, 8, 1),
                date(2030, 8, 1),
                "strong",
                60.0,
                False,
                frozenset(),
            ),
        )
    )
    assert relocation_policy.adjustment(fixtures[0], 1.0) == -0.75
    serial = simulate_seasons_chronologically(
        synthetic_model(),
        teams,
        fixtures,
        "2030/31",
        simulations=4,
        seed=456,
        workers=1,
        relocation_policy=relocation_policy,
    )
    parallel = simulate_seasons_chronologically(
        synthetic_model(),
        teams,
        fixtures,
        "2030/31",
        simulations=4,
        seed=456,
        workers=2,
        relocation_policy=relocation_policy,
    )
    for serial_value, parallel_value in zip(serial, parallel, strict=True):
        np.testing.assert_array_equal(serial_value, parallel_value)
