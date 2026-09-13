"""Read-only Polish Cup side-model probabilities and tournament simulation."""

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from esa_model.baseline import MAX_GOALS
from esa_model.stage1 import uncertain_probabilities
from esa_model.stage3 import BridgeState


@dataclass(frozen=True, slots=True)
class CupPolicy:
    """Adopted lower-tier Cup extrapolation policy."""

    extra_step_scale: float = 0.75
    extra_tier_sd: float = 0.3
    early_esa_penalty: float = 0.0


ADOPTED_CUP_POLICY = CupPolicy()


@dataclass(frozen=True, slots=True)
class CupTeamRating:
    """A stable team identity and its supplied attack/defense state."""

    team_id: str
    tier_rank: int
    attack_mean: float
    attack_variance: float
    defense_mean: float
    defense_variance: float


@dataclass(frozen=True, slots=True)
class CupRound:
    """Round metadata; only explicitly marked non-final rounds use the early policy."""

    name: str
    is_early: bool = False
    is_final: bool = False


@dataclass(frozen=True, slots=True)
class CupMatchProbabilities:
    """Regulation-time probabilities and the log-rate moments that produced them."""

    home_win: float
    draw: float
    away_win: float
    home_log_rate_mean: float
    home_log_rate_variance: float
    away_log_rate_mean: float
    away_log_rate_variance: float
    home_away_log_rate_covariance: float

    @property
    def regulation_probabilities(self) -> tuple[float, float, float]:
        return self.home_win, self.draw, self.away_win


@dataclass(frozen=True, slots=True)
class CupAdvancementProbabilities:
    """One-leg advancement probabilities.

    Extra time uses scaled 90-minute log-rate moments, but is not conditioned on
    the exact score that caused the 90-minute draw.
    """

    home: float
    away: float


@dataclass(frozen=True, slots=True)
class TournamentRules:
    """Known opening draw, staged entries, and round metadata for a knockout tournament."""

    first_round_pairings: tuple[tuple[str, str], ...]
    rounds: tuple[CupRound, ...]
    lower_tier_hosts: bool = True
    shootout_probability: float = 0.5
    later_entry_batches: tuple[CupEntryBatch, ...] = ()
    first_round_pairings_fixed: bool = False


@dataclass(frozen=True, slots=True)
class CupEntryBatch:
    """Teams joining immediately before a later round, indexed from zero."""

    round_index: int
    team_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CupTournamentResult:
    """Monte Carlo tournament wins, probabilities, and sampling error."""

    probabilities: dict[str, float]
    wins: dict[str, int]
    simulations: int
    standard_errors: dict[str, float]
    confidence_intervals_95: dict[str, tuple[float, float]]


def tier_coefficient(rank: int, extra_step_scale: float = ADOPTED_CUP_POLICY.extra_step_scale) -> float:
    """Map canonical rank 1..5 to the calibrated rank-2 bridge-offset scale."""
    if not 1 <= rank <= 5:
        raise ValueError(f"Unsupported canonical tier rank {rank}")
    return 0.0 if rank == 1 else 1.0 if rank == 2 else 1.0 + (rank - 2) * extra_step_scale


def _extra_tier_variance(rank: int, extra_tier_sd: float) -> float:
    return (max(rank - 2, 0) * extra_tier_sd) ** 2


def _validate_rating(rating: CupTeamRating) -> None:
    tier_coefficient(rating.tier_rank)
    values = (
        rating.attack_mean,
        rating.attack_variance,
        rating.defense_mean,
        rating.defense_variance,
    )
    if not rating.team_id or not all(math.isfinite(value) for value in values):
        raise ValueError("Cup ratings require a nonempty ID and finite state values")
    if rating.attack_variance < 0.0 or rating.defense_variance < 0.0:
        raise ValueError("Cup rating variances must be nonnegative")


def _bridge_moments(bridge: BridgeState) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(bridge.mean, dtype=float)
    covariance = np.asarray(bridge.covariance, dtype=float)
    if mean.shape != (7,) or covariance.shape != (7, 7):
        raise ValueError("Cup bridge must have seven means and a 7 by 7 covariance")
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError("Cup bridge moments must be finite")
    return mean, covariance


def _bridge_design(home_rank: int, away_rank: int, policy: CupPolicy, home: bool, neutral: bool) -> np.ndarray:
    scoring_rank, defending_rank = (home_rank, away_rank) if home else (away_rank, home_rank)
    lower_difference = float(scoring_rank > 1) - float(defending_rank > 1)
    return np.asarray(
        (
            1.0,
            float(home and not neutral),
            tier_coefficient(scoring_rank, policy.extra_step_scale),
            -tier_coefficient(defending_rank, policy.extra_step_scale),
            lower_difference,
            0.0,
            0.0,
        )
    )


def cup_match_probabilities(
    home: CupTeamRating,
    away: CupTeamRating,
    bridge: BridgeState,
    rho: float,
    round_metadata: CupRound,
    *,
    neutral: bool = False,
    policy: CupPolicy = ADOPTED_CUP_POLICY,
) -> CupMatchProbabilities:
    """Return regulation-time probabilities without changing ratings or the bridge."""
    _validate_rating(home)
    _validate_rating(away)
    mean, covariance = _bridge_moments(bridge)
    home_design = _bridge_design(home.tier_rank, away.tier_rank, policy, True, neutral)
    away_design = _bridge_design(home.tier_rank, away.tier_rank, policy, False, neutral)
    extra_variance = _extra_tier_variance(home.tier_rank, policy.extra_tier_sd) + _extra_tier_variance(
        away.tier_rank, policy.extra_tier_sd
    )
    home_mean = home.attack_mean - away.defense_mean + float(home_design @ mean)
    away_mean = away.attack_mean - home.defense_mean + float(away_design @ mean)
    if round_metadata.is_early and not round_metadata.is_final:
        if home.tier_rank == 1 and away.tier_rank >= 3:
            home_mean -= policy.early_esa_penalty
            away_mean += policy.early_esa_penalty
        elif away.tier_rank == 1 and home.tier_rank >= 3:
            home_mean += policy.early_esa_penalty
            away_mean -= policy.early_esa_penalty
    home_variance = (
        home.attack_variance + away.defense_variance + extra_variance + float(home_design @ covariance @ home_design)
    )
    away_variance = (
        away.attack_variance + home.defense_variance + extra_variance + float(away_design @ covariance @ away_design)
    )
    home_away_covariance = float(home_design @ covariance @ away_design)
    if home_variance < 0.0 or away_variance < 0.0:
        raise ValueError("Cup log-rate variances must be nonnegative")
    probabilities = _outcome_probabilities(
        home_mean,
        home_variance,
        away_mean,
        away_variance,
        rho,
        home_away_log_rate_covariance=home_away_covariance,
    )[:3]
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-10):
        raise ValueError("Cup outcome probabilities do not normalize")
    return CupMatchProbabilities(
        *probabilities,
        home_mean,
        home_variance,
        away_mean,
        away_variance,
        home_away_covariance,
    )


def advancement_probabilities(
    match: CupMatchProbabilities,
    rho: float,
    *,
    shootout_probability: float = 0.5,
) -> CupAdvancementProbabilities:
    """Approximate one-leg advancement, with ET unconditioned on the 90-minute score."""
    if not 0.0 <= shootout_probability <= 1.0:
        raise ValueError("Shootout probability must be between zero and one")
    extra_time = _outcome_probabilities(
        match.home_log_rate_mean + math.log(1.0 / 3.0),
        match.home_log_rate_variance,
        match.away_log_rate_mean + math.log(1.0 / 3.0),
        match.away_log_rate_variance,
        rho,
        home_away_log_rate_covariance=match.home_away_log_rate_covariance,
    )[:3]
    home = match.home_win + match.draw * (extra_time[0] + extra_time[1] * shootout_probability)
    away = match.away_win + match.draw * (extra_time[2] + extra_time[1] * (1.0 - shootout_probability))
    if not math.isclose(home + away, 1.0, abs_tol=1e-10):
        raise ValueError("Cup advancement probabilities do not normalize")
    return CupAdvancementProbabilities(home, away)


def _outcome_probabilities(
    home_log_rate_mean: float,
    home_log_rate_variance: float,
    away_log_rate_mean: float,
    away_log_rate_variance: float,
    rho: float,
    home_away_log_rate_covariance: float,
) -> tuple[float, float, float]:
    if home_log_rate_variance == away_log_rate_variance == home_away_log_rate_covariance == 0.0:
        return _fixed_outcome_probabilities(math.exp(home_log_rate_mean), math.exp(away_log_rate_mean), rho)
    return uncertain_probabilities(
        home_log_rate_mean,
        home_log_rate_variance,
        away_log_rate_mean,
        away_log_rate_variance,
        rho,
        home_away_log_rate_covariance=home_away_log_rate_covariance,
    )[:3]


def _fixed_outcome_probabilities(home_rate: float, away_rate: float, rho: float) -> tuple[float, float, float]:
    """Return normalized DC outcomes without materializing the score matrix."""
    if not all(math.isfinite(value) and value > 0.0 for value in (home_rate, away_rate)):
        raise ValueError("Dixon-Coles scoring rates must be finite and positive")
    if rho < 0.0:
        cap = -1.0 / rho * (1.0 - 1e-12)
    elif rho > 0.0:
        cap = math.sqrt(1.0 / rho) * (1.0 - 1e-12)
    else:
        cap = math.inf
    home_rate = min(home_rate, cap)
    away_rate = min(away_rate, cap)
    home_pmf = np.empty(MAX_GOALS + 1)
    away_pmf = np.empty(MAX_GOALS + 1)
    home_pmf[0] = math.exp(-home_rate)
    away_pmf[0] = math.exp(-away_rate)
    for goals in range(1, MAX_GOALS + 1):
        home_pmf[goals] = home_pmf[goals - 1] * home_rate / goals
        away_pmf[goals] = away_pmf[goals - 1] * away_rate / goals

    home = float(np.dot(home_pmf[1:], np.cumsum(away_pmf)[:-1]))
    draw = float(np.dot(home_pmf, away_pmf))
    away = float(np.dot(away_pmf[1:], np.cumsum(home_pmf)[:-1]))
    delta_00 = -home_pmf[0] * away_pmf[0] * home_rate * away_rate * rho
    delta_01 = home_pmf[0] * away_pmf[1] * home_rate * rho
    delta_10 = home_pmf[1] * away_pmf[0] * away_rate * rho
    delta_11 = -home_pmf[1] * away_pmf[1] * rho
    corrected_cells = (
        home_pmf[0] * away_pmf[0] + delta_00,
        home_pmf[0] * away_pmf[1] + delta_01,
        home_pmf[1] * away_pmf[0] + delta_10,
        home_pmf[1] * away_pmf[1] + delta_11,
    )
    if min(corrected_cells) < -1e-12:
        raise ValueError("Dixon-Coles correction produced a negative outcome probability")
    home += delta_10
    draw += delta_00 + delta_11
    away += delta_01
    total = home + draw + away
    return home / total, draw / total, away / total


def _hosted_pairing(
    first: CupTeamRating, second: CupTeamRating, lower_tier_hosts: bool, generator: np.random.Generator
) -> tuple[CupTeamRating, CupTeamRating]:
    if lower_tier_hosts and first.tier_rank != second.tier_rank:
        return (first, second) if first.tier_rank > second.tier_rank else (second, first)
    return (first, second) if generator.integers(2) == 0 else (second, first)


def _validate_tournament(
    ratings: Sequence[CupTeamRating], rules: TournamentRules, simulations: int
) -> dict[str, CupTeamRating]:
    entrant_count = len(ratings)
    if entrant_count < 2:
        raise ValueError("Tournament entrants must contain at least two teams")
    if simulations <= 0:
        raise ValueError("Tournament simulations must be positive")
    by_id = {rating.team_id: rating for rating in ratings}
    if len(by_id) != entrant_count:
        raise ValueError("Tournament team IDs must be unique")
    if not rules.rounds:
        raise ValueError("Tournament rules need at least one elimination stage")
    if not rules.rounds[-1].is_final or any(round_metadata.is_final for round_metadata in rules.rounds[:-1]):
        raise ValueError("Only the final tournament round may be marked final")
    if any(len(pairing) != 2 for pairing in rules.first_round_pairings):
        raise ValueError("Each known first-round pairing must contain two teams")
    paired_ids = [team_id for pairing in rules.first_round_pairings for team_id in pairing]
    entry_batches_by_round: dict[int, CupEntryBatch] = {}
    for batch in rules.later_entry_batches:
        if not 1 <= batch.round_index < len(rules.rounds):
            raise ValueError("Entry batches must join before a later tournament round")
        if batch.round_index in entry_batches_by_round:
            raise ValueError("Only one entry batch may join before each tournament round")
        entry_batches_by_round[batch.round_index] = batch
    entry_ids = [team_id for batch in rules.later_entry_batches for team_id in batch.team_ids]
    assigned_ids = paired_ids + entry_ids
    if len(set(assigned_ids)) != entrant_count or set(assigned_ids) != set(by_id):
        raise ValueError("Opening pairings and entry batches must cover every team exactly once")
    participants = len(paired_ids)
    for round_index, round_metadata in enumerate(rules.rounds):
        if round_index:
            participants += len(entry_batches_by_round.get(round_index, CupEntryBatch(round_index, ())).team_ids)
        if participants < 2 or participants % 2:
            raise ValueError(
                f"Tournament stage {round_metadata.name!r} must have an even participant count of at least two"
            )
        participants //= 2
    if participants != 1:
        raise ValueError("Tournament rounds must reduce the staged field to one winner")
    if not 0.0 <= rules.shootout_probability <= 1.0:
        raise ValueError("Shootout probability must be between zero and one")
    return by_id


def sample_tournament_path_state(
    ratings: Sequence[CupTeamRating],
    bridge: BridgeState,
    policy: CupPolicy,
    generator: np.random.Generator,
) -> tuple[dict[str, CupTeamRating], BridgeState, CupPolicy]:
    """Sample one persistent latent state shared by every match in a tournament path."""
    mean, covariance = _bridge_moments(bridge)
    sampled_bridge = BridgeState(
        generator.multivariate_normal(mean, covariance, check_valid="raise"),
        np.zeros_like(covariance),
        bridge.season,
    )
    sampled_ratings: dict[str, CupTeamRating] = {}
    for rating in ratings:
        extra_variance = _extra_tier_variance(rating.tier_rank, policy.extra_tier_sd)
        sampled_ratings[rating.team_id] = CupTeamRating(
            rating.team_id,
            rating.tier_rank,
            float(generator.normal(rating.attack_mean, math.sqrt(rating.attack_variance + extra_variance))),
            0.0,
            float(generator.normal(rating.defense_mean, math.sqrt(rating.defense_variance + extra_variance))),
            0.0,
        )
    return sampled_ratings, sampled_bridge, replace(policy, extra_tier_sd=0.0)


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    probability = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = (probability + z_squared / (2.0 * trials)) / denominator
    half_width = (
        z / denominator * math.sqrt(probability * (1.0 - probability) / trials + z_squared / (4.0 * trials * trials))
    )
    low = 0.0 if successes == 0 else max(0.0, center - half_width)
    high = 1.0 if successes == trials else min(1.0, center + half_width)
    return low, high


def tournament_final_win_probabilities(
    ratings: Sequence[CupTeamRating],
    bridge: BridgeState,
    rho: float,
    rules: TournamentRules,
    generator: np.random.Generator,
    *,
    simulations: int = 10_000,
    policy: CupPolicy = ADOPTED_CUP_POLICY,
) -> CupTournamentResult:
    """Estimate final-win probabilities with one shared latent state per path."""
    if not isinstance(generator, np.random.Generator):
        raise TypeError("Tournament simulation requires a numpy Generator")
    by_id = _validate_tournament(ratings, rules, simulations)
    wins = dict.fromkeys(by_id, 0)
    entry_batches_by_round = {batch.round_index: batch.team_ids for batch in rules.later_entry_batches}
    for _ in range(simulations):
        path_ratings, path_bridge, conditional_policy = sample_tournament_path_state(ratings, bridge, policy, generator)
        survivors: list[CupTeamRating] = []
        for round_index, round_metadata in enumerate(rules.rounds):
            if round_index == 0:
                opening_ids = [team_id for pairing in rules.first_round_pairings for team_id in pairing]
                if not rules.first_round_pairings_fixed:
                    generator.shuffle(opening_ids)
                pairs = list(zip(opening_ids[::2], opening_ids[1::2], strict=True))
                pairs = [(path_ratings[first], path_ratings[second]) for first, second in pairs]
            else:
                survivors.extend(path_ratings[team_id] for team_id in entry_batches_by_round.get(round_index, ()))
                generator.shuffle(survivors)
                pairs = list(zip(survivors[::2], survivors[1::2], strict=True))
            survivors = []
            for first, second in pairs:
                if round_index == 0 and rules.first_round_pairings_fixed:
                    home, away = first, second
                else:
                    home, away = _hosted_pairing(
                        first,
                        second,
                        rules.lower_tier_hosts and not round_metadata.is_final,
                        generator,
                    )
                match = cup_match_probabilities(
                    home,
                    away,
                    path_bridge,
                    rho,
                    round_metadata,
                    neutral=round_metadata.is_final,
                    policy=conditional_policy,
                )
                advancement = advancement_probabilities(match, rho, shootout_probability=rules.shootout_probability)
                survivors.append(home if generator.random() < advancement.home else away)
        wins[survivors[0].team_id] += 1
    probabilities = {team_id: count / simulations for team_id, count in wins.items()}
    standard_errors = {
        team_id: math.sqrt(probability * (1.0 - probability) / simulations)
        for team_id, probability in probabilities.items()
    }
    intervals = {team_id: _wilson_interval(count, simulations) for team_id, count in wins.items()}
    return CupTournamentResult(probabilities, wins, simulations, standard_errors, intervals)
