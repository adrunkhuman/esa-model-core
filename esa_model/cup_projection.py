"""Prepared, read-only Polish Cup projections for frontend snapshots."""

import hashlib
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np

from esa_model.cup_side import (
    CupEntryBatch,
    CupRound,
    CupTeamRating,
    TournamentRules,
    tournament_final_win_probabilities,
)
from esa_model.live_contracts import CupProbabilityEntry, CupProjection
from esa_model.stage1 import TeamState
from esa_model.stage3 import BridgeState

CUP_HISTORY_SIMULATIONS = 1_000
ROUND_KEYS = (
    "preliminary",
    "round_one",
    "round_of_32",
    "round_of_16",
    "quarterfinal",
    "semifinal",
    "final",
)
EARLY_ROUNDS = frozenset(("preliminary", "round_one", "round_of_32"))


@dataclass(frozen=True, slots=True)
class CupTeam:
    team_id: str
    name: str
    tier: str | None
    tier_rank: int | None


@dataclass(frozen=True, slots=True)
class CupSeason:
    season: str
    status: str
    teams: dict[str, CupTeam]
    entry_batches: dict[str, tuple[str, ...]]
    rounds: dict[str, dict[str, Any]]
    entry_boundaries: dict[str, date] = field(default_factory=dict)
    resolved_entries: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CupProjectionContext:
    """Read-only source data and causal side-model states shared by snapshots."""

    seasons: dict[str, CupSeason]
    canonical_ids: dict[str, str]
    shadow_states: dict[tuple[str, date], dict[str, TeamState]]
    bridge_states: dict[tuple[str, date], BridgeState]


def projection_seed(season: str, as_of: date, phase: str) -> int:
    """Return a process-independent deterministic seed for a published Cup snapshot."""
    digest = hashlib.blake2b(f"{season}|{as_of.isoformat()}|{phase}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _completed(fixture: dict[str, Any], as_of: date) -> bool:
    return (
        fixture.get("winner_id") is not None
        and fixture.get("date") is not None
        and date.fromisoformat(fixture["date"]) <= as_of
    )


def _round_metadata(key: str) -> CupRound:
    return CupRound(key.replace("_", " "), is_early=key in EARLY_ROUNDS, is_final=key == "final")


def _entries_for(season: CupSeason, key: str, as_of: date) -> tuple[str, ...]:
    """Return the round's entrants, using resolved identities after the draw boundary."""
    batch = season.entry_batches.get(key, ())
    boundary = season.entry_boundaries.get(key)
    if boundary is not None and boundary <= as_of:
        batch = season.resolved_entries[key]
    return batch


def build_tournament_rules(season: CupSeason, as_of: date) -> tuple[TournamentRules | None, str | None]:
    """Freeze known results and turn the remaining staged field into Cup-side rules."""
    rounds = [key for key in ROUND_KEYS if key in season.rounds or key in season.entry_batches]
    if "final" not in rounds:
        rounds.extend(key for key in ROUND_KEYS[ROUND_KEYS.index(rounds[-1]) + 1 :])
    fixtures_by_round = {key: list(season.rounds.get(key, {}).get("fixtures", [])) for key in rounds}
    final = fixtures_by_round.get("final", [])
    if final and _completed(final[0], as_of):
        return None, str(final[0]["winner_id"])
    start = next(
        (
            index
            for index, key in enumerate(rounds)
            if not fixtures_by_round[key] or any(not _completed(fixture, as_of) for fixture in fixtures_by_round[key])
        ),
        None,
    )
    if start is None:
        raise ValueError(f"Cup {season.season} has no remaining tournament stage")
    key = rounds[start]
    fixtures = fixtures_by_round[key]
    unplayed = [fixture for fixture in fixtures if not _completed(fixture, as_of)]
    round_data = season.rounds.get(key, {})
    known_at = round_data.get("pairings_known_at")
    fixed = bool(
        unplayed
        and round_data.get("pairings_known", False)
        and isinstance(known_at, str)
        and date.fromisoformat(known_at) <= as_of
    )
    completed_winners = tuple(str(fixture["winner_id"]) for fixture in fixtures if _completed(fixture, as_of))
    if fixed:
        pairs = tuple((str(row["home"]["team_id"]), str(row["away"]["team_id"])) for row in unplayed)
    else:
        prior_winners = tuple(str(row["winner_id"]) for row in fixtures_by_round[rounds[start - 1]]) if start else ()
        entrants = prior_winners + _entries_for(season, key, as_of)
        if len(entrants) % 2:
            raise ValueError(f"Cup {season.season} has an odd unpaired field in {key}")
        pairs = tuple(zip(entrants[::2], entrants[1::2], strict=True))
    later: list[CupEntryBatch] = []
    for index, later_key in enumerate(rounds[start + 1 :], 1):
        entrants = list(_entries_for(season, later_key, as_of))
        if index == 1:
            entrants.extend(completed_winners)
        if entrants:
            later.append(CupEntryBatch(index, tuple(entrants)))
    return (
        TournamentRules(
            pairs,
            tuple(_round_metadata(round_key) for round_key in rounds[start:]),
            later_entry_batches=tuple(later),
            first_round_pairings_fixed=fixed,
        ),
        None,
    )


def _state_for_esa(model: Any, raw_id: str, canonical_ids: dict[str, str], season: str) -> TeamState | None:
    if raw_id in model.states:
        return model.states[raw_id]
    canonical = canonical_ids.get(raw_id)
    if canonical is None:
        return None
    matches = [
        team for team in model.states if getattr(model, "canonical_club_ids", {}).get((season, team)) == canonical
    ]
    # Most operational snapshots retain raw IDs. The fallback intentionally only
    # accepts a unique configured canonical identity, never a name match.
    if len(matches) == 1:
        return model.states[matches[0]]
    return None


def _rating(
    team: CupTeam,
    model: Any,
    context: CupProjectionContext,
    season: str,
    as_of: date,
) -> CupTeamRating:
    rank = team.tier_rank or 5
    if rank == 1:
        state = _state_for_esa(model, team.team_id, context.canonical_ids, season)
    elif rank == 2:
        state = context.shadow_states[season, as_of].get(context.canonical_ids.get(team.team_id, f"90:{team.team_id}"))
    else:
        state = None
    if state is None:
        return CupTeamRating(team.team_id, rank, 0.0, 0.0, 0.0, 0.0)
    return CupTeamRating(
        team.team_id,
        rank,
        state.attack_mean,
        state.attack_variance,
        state.defense_mean,
        state.defense_variance,
    )


def project_cup(
    context: CupProjectionContext,
    season: str,
    as_of: date,
    phase: str,
    model: Any,
    *,
    simulations: int = CUP_HISTORY_SIMULATIONS,
    display: dict[str, tuple[str, str, str | None]] | None = None,
) -> CupProjection:
    """Project a snapshot without updating the supplied operational model."""
    if simulations <= 0:
        raise ValueError("Cup simulations must be positive")
    cup = context.seasons[season]
    rules, winner = build_tournament_rules(cup, as_of)
    if winner is None:
        assert rules is not None
        active_ids = {team_id for pair in rules.first_round_pairings for team_id in pair}
        active_ids.update(team_id for batch in rules.later_entry_batches for team_id in batch.team_ids)
        ratings = [_rating(cup.teams[team_id], model, context, season, as_of) for team_id in sorted(active_ids)]
        result = tournament_final_win_probabilities(
            ratings,
            context.bridge_states[season, as_of],
            model.globals.rho,
            rules,
            np.random.default_rng(projection_seed(season, as_of, phase)),
            simulations=simulations,
        )
        probabilities = result.probabilities
    else:
        active_ids = {winner}
        probabilities = {team_id: float(team_id == winner) for team_id in cup.teams}
    entries = []
    others = 0.0
    for team_id, team in cup.teams.items():
        if team_id not in active_ids:
            continue
        probability = probabilities.get(team_id, 0.0)
        if team.tier_rank in (1, 2):
            name, short_name, logo = (display or {}).get(team_id, (team.name, team.name, None))
            entries.append(
                CupProbabilityEntry(
                    team_id, name, short_name, "ekstraklasa" if team.tier_rank == 1 else "i_liga", probability, logo
                )
            )
        else:
            others += probability
    entries.sort(key=lambda entry: (-entry.probability, entry.name))
    return CupProjection(tuple(entries), others, simulations, as_of.isoformat())


def active_cup_team_ids(cup: CupSeason, as_of: date) -> set[str]:
    """Return teams still eligible to win the Cup at a snapshot cutoff."""
    rules, winner = build_tournament_rules(cup, as_of)
    if winner is not None:
        return {winner}
    if rules is None:
        raise ValueError("Unfinished Cup has no tournament rules")
    active = {team_id for pairing in rules.first_round_pairings for team_id in pairing}
    active.update(team_id for batch in rules.later_entry_batches for team_id in batch.team_ids)
    return active
