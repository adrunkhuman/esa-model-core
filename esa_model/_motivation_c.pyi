import numpy as np
import numpy.typing as npt

def simulate_conditioned_zone_counts(
    starting_points: npt.NDArray[np.int16],
    fixture_samples: npt.NDArray[np.float64],
    tie_noise: npt.NDArray[np.float64],
    home_teams: npt.NDArray[np.intp],
    away_teams: npt.NDArray[np.intp],
    home_probabilities: npt.NDArray[np.float64],
    draw_probabilities: npt.NDArray[np.float64],
    conditioned_fixtures: npt.NDArray[np.intp],
    group_members: npt.NDArray[np.intp],
    group_offsets: npt.NDArray[np.intp],
    zones_by_rank: npt.NDArray[np.intp],
    requested_teams: npt.NDArray[np.intp],
) -> npt.NDArray[np.int64]: ...
