"""Fast scoring-rate integration for repeated live forecasts."""

import math
from functools import cache

import numpy as np
from scipy.special import roots_hermitenorm

from esa_model.baseline import MAX_GOALS
from esa_model.stage1 import QUADRATURE_POINTS

_GOALS = np.arange(MAX_GOALS + 1)
_FACTORIALS = np.asarray([math.factorial(goal) for goal in _GOALS])


@cache
def _quadrature_rule(points: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = roots_hermitenorm(points)
    return nodes, weights / math.sqrt(2.0 * math.pi)


def _poisson_pmf(rates: np.ndarray) -> np.ndarray:
    expanded = rates[..., None]
    return np.exp(-expanded) * expanded**_GOALS / _FACTORIALS


def live_uncertain_score_matrix(
    home_log_rate_mean: float,
    home_log_rate_variance: float,
    away_log_rate_mean: float,
    away_log_rate_variance: float,
    rho: float,
    quadrature_points: int = QUADRATURE_POINTS,
    home_away_log_rate_covariance: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    """Integrate the live DC score matrix without repeated SciPy setup."""
    nodes, weights = _quadrature_rule(quadrature_points)
    if rho < 0.0:
        cap = -1.0 / rho * (1.0 - 1e-12)
    elif rho > 0.0:
        cap = math.sqrt(1.0 / rho) * (1.0 - 1e-12)
    else:
        cap = math.inf
    home_scale = math.sqrt(home_log_rate_variance)
    away_shared_scale = home_away_log_rate_covariance / home_scale if home_scale > 0.0 else 0.0
    away_independent_scale = math.sqrt(max(away_log_rate_variance - away_shared_scale**2, 0.0))
    home_rates = np.minimum(np.exp(home_log_rate_mean + home_scale * nodes), cap)
    away_rates = np.minimum(
        np.exp(away_log_rate_mean + away_shared_scale * nodes[:, None] + away_independent_scale * nodes[None, :]),
        cap,
    )
    home_pmf = _poisson_pmf(home_rates)
    away_pmf = _poisson_pmf(away_rates)
    home_grid = home_rates[:, None]
    matrices = home_pmf[:, None, :, None] * away_pmf[:, :, None, :]
    matrices[:, :, 0, 0] *= 1.0 - home_grid * away_rates * rho
    matrices[:, :, 0, 1] *= 1.0 + home_grid * rho
    matrices[:, :, 1, 0] *= 1.0 + away_rates * rho
    matrices[:, :, 1, 1] *= 1.0 - rho
    if np.any(matrices < -1e-12):
        raise ValueError("Dixon-Coles correction produced a negative outcome probability")
    total = matrices.sum(axis=(2, 3))
    joint_weights = weights[:, None] * weights[None, :]
    matrix = np.einsum("ij,ijgh->gh", joint_weights, matrices / total[:, :, None, None])
    return matrix, float(np.dot(weights, home_rates)), float(np.sum(joint_weights * away_rates))


def live_uncertain_probabilities(
    home_log_rate_mean: float,
    home_log_rate_variance: float,
    away_log_rate_mean: float,
    away_log_rate_variance: float,
    rho: float,
    quadrature_points: int = QUADRATURE_POINTS,
    home_away_log_rate_covariance: float = 0.0,
) -> tuple[float, float, float, float, float]:
    matrix, expected_home, expected_away = live_uncertain_score_matrix(
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


def live_uncertain_probabilities_batch(
    home_log_rate_mean: np.ndarray,
    home_log_rate_variance: np.ndarray,
    away_log_rate_mean: np.ndarray,
    away_log_rate_variance: np.ndarray,
    rho: float,
    quadrature_points: int = QUADRATURE_POINTS,
    home_away_log_rate_covariance: np.ndarray | float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate live DC probabilities for parallel posterior lanes."""
    home_mean = np.asarray(home_log_rate_mean, dtype=float)
    home_variance = np.asarray(home_log_rate_variance, dtype=float)
    away_mean = np.asarray(away_log_rate_mean, dtype=float)
    away_variance = np.asarray(away_log_rate_variance, dtype=float)
    if home_mean.ndim != 1 or not (home_mean.shape == home_variance.shape == away_mean.shape == away_variance.shape):
        raise ValueError("Batched live-rate moments must be same-length vectors")
    covariance = np.broadcast_to(np.asarray(home_away_log_rate_covariance, dtype=float), home_mean.shape)
    if np.any(home_variance < 0.0) or np.any(away_variance < 0.0):
        raise ValueError("Batched live-rate variances must be nonnegative")

    nodes, weights = _quadrature_rule(quadrature_points)
    if rho < 0.0:
        cap = -1.0 / rho * (1.0 - 1e-12)
    elif rho > 0.0:
        cap = math.sqrt(1.0 / rho) * (1.0 - 1e-12)
    else:
        cap = math.inf
    home_scale = np.sqrt(home_variance)
    away_shared_scale = np.divide(covariance, home_scale, out=np.zeros_like(covariance), where=home_scale > 0.0)
    away_independent_scale = np.sqrt(np.maximum(away_variance - away_shared_scale**2, 0.0))
    home_rates = np.minimum(np.exp(home_mean[:, None] + home_scale[:, None] * nodes), cap)
    away_rates = np.minimum(
        np.exp(
            away_mean[:, None, None]
            + away_shared_scale[:, None, None] * nodes[None, :, None]
            + away_independent_scale[:, None, None] * nodes[None, None, :]
        ),
        cap,
    )
    home_pmf = _poisson_pmf(home_rates)
    away_pmf = _poisson_pmf(away_rates)
    home_grid = home_rates[:, :, None]
    matrices = home_pmf[:, :, None, :, None] * away_pmf[:, :, :, None, :]
    matrices[:, :, :, 0, 0] *= 1.0 - home_grid * away_rates * rho
    matrices[:, :, :, 0, 1] *= 1.0 + home_grid * rho
    matrices[:, :, :, 1, 0] *= 1.0 + away_rates * rho
    matrices[:, :, :, 1, 1] *= 1.0 - rho
    if np.any(matrices < -1e-12):
        raise ValueError("Dixon-Coles correction produced a negative outcome probability")
    total = matrices.sum(axis=(3, 4))
    joint_weights = weights[:, None] * weights[None, :]
    integrated = np.einsum("ij,bijgh->bgh", joint_weights, matrices / total[:, :, :, None, None])
    return (
        np.tril(integrated, -1).sum(axis=(1, 2)),
        np.trace(integrated, axis1=1, axis2=2),
        np.triu(integrated, 1).sum(axis=(1, 2)),
        home_rates @ weights,
        np.sum(joint_weights[None, :, :] * away_rates, axis=(1, 2)),
    )
