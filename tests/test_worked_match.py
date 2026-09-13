import runpy
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).parents[1] / "examples" / "worked_match.py"


def test_worked_match_documented_numbers() -> None:
    values = runpy.run_path(str(EXAMPLE))["worked_values"]()

    assert values.home_log_rate == pytest.approx(0.473144, abs=0.5e-6)
    assert values.away_log_rate == pytest.approx(0.223144, abs=0.5e-6)
    assert values.home_rate == pytest.approx(1.605032, abs=0.5e-6)
    assert values.away_rate == pytest.approx(1.250000, abs=0.5e-6)

    cells = values.low_score_cells
    assert cells["0-0"] == pytest.approx((0.057554, 1.160503, 0.066792), abs=0.5e-6)
    assert cells["0-1"] == pytest.approx((0.071942, 0.871597, 0.062705), abs=0.5e-6)
    assert cells["1-0"] == pytest.approx((0.092376, 0.900000, 0.083138), abs=0.5e-6)
    assert cells["1-1"] == pytest.approx((0.115470, 1.080000, 0.124708), abs=0.5e-6)

    assert values.fixed_outcomes == pytest.approx((0.447374, 0.264083, 0.288543), abs=0.5e-6)
    assert values.uncertain_outcomes == pytest.approx(
        (0.449587, 0.260903, 0.289510, 1.637456, 1.264142),
        abs=0.5e-6,
    )
    assert values.uncertain_outcomes[3:] == pytest.approx(
        values.analytic_uncertain_expected_goals,
        abs=1e-12,
    )
