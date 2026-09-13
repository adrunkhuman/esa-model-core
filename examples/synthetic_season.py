"""Run a fixed-seed season simulation with entirely synthetic inputs."""

# ruff: noqa: I001 -- imports resolve as third-party only after export.

from datetime import date, timedelta
from typing import cast

import numpy as np

from esa_model.baseline import Match
from esa_model.joint_covariance import JointCovarianceDCModel
from esa_model.posterior_simulation import simulate_seasons_from_posterior
from esa_model.relocated_home_policy import RelocatedHomeHfaPolicy
from esa_model.stage1 import FrozenGlobals, GoalRateForecast


class SyntheticPosteriorModel:
    """Minimal model protocol for the static-posterior simulator."""

    globals = FrozenGlobals(0.05, 0.15, -0.06, 0.1, 0.1)
    home_advantage_mean = 0.15

    def sample_parameter_state(self, generator: np.random.Generator) -> np.ndarray:
        return generator.normal(0.0, 0.12, size=4)

    def home_advantage_from_state(self, state: np.ndarray) -> float:
        return self.home_advantage_mean

    def goal_rates_from_state(self, match: Match, state: np.ndarray) -> GoalRateForecast:
        teams = {"north": 0, "east": 1, "south": 2, "west": 3}
        home_strength = state[teams[match.home]]
        away_strength = state[teams[match.away]]
        return GoalRateForecast(
            self.globals.scoring_intercept + self.globals.home_advantage + home_strength - away_strength,
            0.0,
            self.globals.scoring_intercept + away_strength - home_strength,
            0.0,
        )


def fixtures() -> list[Match]:
    teams = ("north", "east", "south", "west")
    rows = []
    kickoff = date(2030, 8, 1)
    for round_index, shift in enumerate((1, 2, 3, 3, 2, 1)):
        for home_index in range(2):
            home = teams[(home_index + round_index) % 4]
            away = teams[(home_index + round_index + shift) % 4]
            if home == away:
                continue
            rows.append(
                Match(
                    f"synthetic-{round_index}-{home_index}",
                    kickoff + timedelta(days=7 * round_index),
                    "2030/31",
                    home,
                    away,
                    0,
                    0,
                )
            )
    return rows


def main() -> None:
    teams = ["north", "east", "south", "west"]
    expected, places, intervals = simulate_seasons_from_posterior(
        cast(JointCovarianceDCModel, SyntheticPosteriorModel()),
        teams,
        fixtures(),
        simulations=1_000,
        seed=42,
        relocation_policy=RelocatedHomeHfaPolicy(()),
    )
    for index, team in enumerate(teams):
        print(
            f"{team:5} xPts={expected[index]:5.2f} "
            f"80%=[{intervals[index, 0]:.0f}, {intervals[index, 1]:.0f}] "
            f"P(1st)={places[index, 0]:5.1f}%"
        )


if __name__ == "__main__":
    main()
