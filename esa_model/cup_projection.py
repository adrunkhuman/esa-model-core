"""Prepared, read-only Polish Cup projections for frontend snapshots."""

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from pathlib import Path
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
from esa_model.stage1 import SequentialDCModel, TeamState, freeze_globals
from esa_model.stage2 import load_stage2_data
from esa_model.stage3 import (
    BRIDGE_PROCESS_VARIANCES,
    HFA_DRIFT_VARIANCE,
    SUMMER_VARIANCE,
    TEAM_DRIFT_VARIANCE,
    AuxiliaryMatch,
    BridgeState,
    advance_bridge,
    as_league_matches,
    canonical_ek_matches,
    cup_observations,
    extract_league_history,
    initial_bridge,
    load_auxiliary_matches,
    poisson_batch_update,
    top_two_cup_matches,
)

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


def _season_key(value: str) -> int:
    return int(value[:4])


def _canonical_ids(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            raw = row["local_team_id"]
            canonical = row["canonical_club_id"]
            if raw in result and result[raw] != canonical:
                raise ValueError(f"Ambiguous Cup team mapping for {raw}")
            result[raw] = canonical
    return result


def load_cup_seasons(path: Path = Path("data/polish_cup_frontend.json")) -> dict[str, CupSeason]:
    """Load the checked Cup payload and reject malformed staged tournaments."""
    with path.open(encoding="utf-8") as source:
        payload = json.load(source)
    if payload.get("schema_version") != 2 or payload.get("competition") != "Polish Cup":
        raise ValueError("Unsupported Polish Cup frontend data")
    output: dict[str, CupSeason] = {}
    for row in payload.get("seasons", []):
        teams = {
            str(team["team_id"]): CupTeam(
                str(team["team_id"]), str(team["name"]), team.get("tier"), team.get("tier_rank")
            )
            for team in row["teams"]
        }
        batches: dict[str, tuple[str, ...]] = {}
        boundaries: dict[str, date] = {}
        resolved: dict[str, tuple[str, ...]] = {}
        for batch in row["entry_batches"]:
            key = str(batch["round"])
            if key in batches:
                raise ValueError(f"Duplicate Cup entry batch in {row['season']}")
            team_ids = tuple(map(str, batch["team_ids"]))
            known_team_ids = batch.get("known_team_ids")
            known_from = batch.get("known_from")
            if (known_team_ids is None) != (known_from is None):
                raise ValueError(f"Cup {row['season']} batch {key} needs both known_from and known_team_ids")
            if known_team_ids is not None:
                resolved[key] = tuple(map(str, known_team_ids))
                if len(resolved[key]) != len(team_ids):
                    raise ValueError(f"Cup {row['season']} batch {key} resolved field size mismatch")
                boundaries[key] = date.fromisoformat(str(known_from))
            batches[key] = team_ids
        round_rows = row["rounds"]
        rounds = {str(round_["key"]): round_ for round_ in round_rows}
        if row["season"] in output or len(teams) != len(row["teams"]):
            raise ValueError(f"Duplicate Cup season or entrant in {row['season']}")
        if len(rounds) != len(round_rows):
            raise ValueError(f"Duplicate Cup round in {row['season']}")
        covered = {team_id for batch in batches.values() for team_id in batch}
        covered.update(team_id for batch in resolved.values() for team_id in batch)
        if covered != set(teams):
            raise ValueError(f"Cup entry batches do not cover {row['season']}")
        for round_ in rounds.values():
            for fixture in round_.get("fixtures", []):
                sides = {str(fixture["home"]["team_id"]), str(fixture["away"]["team_id"])}
                if not sides <= set(teams):
                    raise ValueError(f"Unknown Cup fixture entrant in {row['season']}")
                winner = fixture.get("winner_id")
                if winner is not None and str(winner) not in sides:
                    raise ValueError(f"Invalid Cup winner in {row['season']}")
        _validate_staging(str(row["season"]), str(row["status"]), teams, batches, resolved)
        _validate_pairing_availability(str(row["season"]), round_rows)
        output[str(row["season"])] = CupSeason(
            str(row["season"]), str(row["status"]), teams, batches, rounds, boundaries, resolved
        )
    if not output:
        raise ValueError("Polish Cup frontend data has no seasons")
    return output


def _validate_staging(
    season: str,
    status: str,
    teams: dict[str, CupTeam],
    batches: dict[str, tuple[str, ...]],
    resolved: dict[str, tuple[str, ...]] | None = None,
) -> None:
    if status != "complete" and tuple(len(batches.get(key, ())) for key in ROUND_KEYS[:3]) != (22, 43, 5):
        raise ValueError(f"Unexpected current Cup entry batches for {season}")
    participants = len(batches.get("preliminary", ()))
    if participants < 2 or participants % 2:
        raise ValueError(f"Cup {season} preliminary field is invalid")
    for key in ROUND_KEYS:
        if key != "preliminary":
            participants += len(batches.get(key, ()))
        if participants < 2 or participants % 2:
            raise ValueError(f"Cup {season} cannot pair {key}")
        participants //= 2
    staged = {team_id for batch in batches.values() for team_id in batch}
    for batch in (resolved or {}).values():
        staged.update(batch)
    if participants != 1 or len(staged) != len(teams):
        raise ValueError(f"Cup {season} staging does not produce one winner")


def _validate_pairing_availability(season: str, rounds: list[dict[str, Any]]) -> None:
    """Require explicit, chronologically ordered draw-availability boundaries."""
    known_dates: list[date] = []
    for round_ in rounds:
        known = round_.get("pairings_known")
        known_at = round_.get("pairings_known_at")
        if not isinstance(known, bool):
            raise ValueError(f"Cup {season} round availability must declare pairings_known")
        if known:
            if not isinstance(known_at, str):
                raise ValueError(f"Known Cup pairings need pairings_known_at in {season}")
            try:
                known_dates.append(date.fromisoformat(known_at))
            except ValueError as error:
                raise ValueError(f"Invalid Cup pairings_known_at in {season}: {known_at!r}") from error
        elif known_at is not None:
            raise ValueError(f"Unknown Cup pairings must have null pairings_known_at in {season}")
    if known_dates != sorted(known_dates):
        raise ValueError(f"Cup pairing availability is out of round order in {season}")


def _cup_teams_by_season(seasons: dict[str, CupSeason]) -> dict[str, set[str]]:
    return {
        season: {team_id for team_id, team in data.teams.items() if team.tier_rank == 2}
        for season, data in seasons.items()
    }


def _prepare_shadow_states(
    seasons: dict[str, CupSeason],
    canonical_ids: dict[str, str],
    cutoffs: dict[str, set[date]],
    event_cutoffs: dict[str, set[date]],
    tier_path: Path,
    mapping_path: Path,
) -> dict[tuple[str, date], dict[str, TeamState]]:
    """Replay I liga once, recording states after every requested frontend date."""
    auxiliary = load_auxiliary_matches(tier_path, mapping_path)
    matches = as_league_matches(auxiliary)
    globals_, _ = freeze_globals(matches, sorted({match.season for match in matches}))
    matches_by_date: dict[date, list] = defaultdict(list)
    teams_by_season: dict[str, set[str]] = defaultdict(set)
    for match in matches:
        matches_by_date[match.date].append(match)
        teams_by_season[match.season].update((match.home, match.away))
    wanted = _cup_teams_by_season(seasons)
    model = SequentialDCModel(
        TEAM_DRIFT_VARIANCE,
        SUMMER_VARIANCE,
        globals_,
        home_advantage_drift_variance=HFA_DRIFT_VARIANCE,
        dynamic_home_advantage=True,
    )
    output: dict[tuple[str, date], dict[str, TeamState]] = {}
    events = sorted(
        set(matches_by_date)
        | {cutoff for dates in cutoffs.values() for cutoff in dates}
        | {cutoff for dates in event_cutoffs.values() for cutoff in dates}
    )
    for event_date in events:
        for match in matches_by_date.get(event_date, []):
            model.advance_to(event_date, match.season, teams_by_season[match.season])
        for season, dates in event_cutoffs.items():
            if event_date not in dates:
                continue
            active = {canonical_ids.get(team_id, f"90:{team_id}") for team_id in wanted.get(season, set())}
            model.advance_to(event_date, season, active)
            output[season, event_date] = {
                team_id: replace(model.states[team_id]) for team_id in active if team_id in model.states
            }
        if event_date in matches_by_date:
            model.observe(matches_by_date[event_date])
        for season, dates in cutoffs.items():
            if event_date not in dates:
                continue
            active = {canonical_ids.get(team_id, f"90:{team_id}") for team_id in wanted.get(season, set())}
            # A next-day query includes all same-day league results in post-match snapshots.
            model.advance_to(event_date + timedelta(days=1), season, active)
            output[season, event_date] = {
                team_id: replace(model.states[team_id]) for team_id in active if team_id in model.states
            }
    return output


def _prepare_bridge_states(
    cutoffs: dict[str, set[date]], event_cutoffs: dict[str, set[date]], tier_path: Path, mapping_path: Path
) -> dict[tuple[str, date], BridgeState]:
    """Replay only rank-1/rank-2 Cup results; lower-tier results never enter it."""
    ek_matches, ek_ids, _, ek_globals = load_stage2_data(mapping_path=mapping_path)
    auxiliary = load_auxiliary_matches(tier_path, mapping_path)
    cups = top_two_cup_matches(auxiliary)
    liga_matches = as_league_matches(auxiliary)
    liga_globals, _ = freeze_globals(liga_matches, sorted({match.season for match in liga_matches}))
    ek_history = extract_league_history(canonical_ek_matches(ek_matches, ek_ids), ek_globals, cups)
    liga_history = extract_league_history(liga_matches, liga_globals, cups)
    state = initial_bridge(ek_globals, min(match.season for match in cups))
    cups_by_date: dict[date, list[AuxiliaryMatch]] = defaultdict(list)
    for match in cups:
        cups_by_date[match.date].append(match)
    output: dict[tuple[str, date], BridgeState] = {}
    events = sorted(
        set(cups_by_date)
        | {cutoff for dates in cutoffs.values() for cutoff in dates}
        | {cutoff for dates in event_cutoffs.values() for cutoff in dates}
    )
    for event_date in events:
        for season, dates in event_cutoffs.items():
            if event_date in dates:
                advance_bridge(state, season, BRIDGE_PROCESS_VARIANCES)
                output[season, event_date] = state.copy()
        day_cups = sorted(cups_by_date.get(event_date, []), key=lambda match: match.match_id)
        if day_cups:
            advance_bridge(state, day_cups[0].season, BRIDGE_PROCESS_VARIANCES)
            observations = []
            for match in day_cups:
                observations.extend(cup_observations(match, ek_history, liga_history))
            poisson_batch_update(state, observations)
        for season, dates in cutoffs.items():
            if event_date in dates:
                advance_bridge(state, season, BRIDGE_PROCESS_VARIANCES)
                output[season, event_date] = state.copy()
    return output


def prepare_cup_projection_context(
    cutoffs: dict[str, set[date]],
    *,
    cup_path: Path = Path("data/polish_cup_frontend.json"),
    tier_path: Path = Path("data/tier_matches.csv"),
    mapping_path: Path = Path("data/team_mapping.csv"),
    event_cutoffs: dict[str, set[date]] | None = None,
    include_fixture_dates: bool = False,
) -> CupProjectionContext:
    """Prepare reusable causal state queries for all requested snapshot cutoffs."""
    seasons = load_cup_seasons(cup_path)
    prepared_cutoffs = {season: set(dates) for season, dates in cutoffs.items()}
    prepared_event_cutoffs = {season: set(dates) for season, dates in (event_cutoffs or {}).items()}
    missing = (set(prepared_cutoffs) | set(prepared_event_cutoffs)) - set(seasons)
    if missing:
        raise ValueError(f"Cup data is missing requested seasons: {sorted(missing)}")
    if include_fixture_dates:
        for season in set(prepared_cutoffs) | set(prepared_event_cutoffs):
            for round_ in seasons[season].rounds.values():
                prepared_event_cutoffs.setdefault(season, set()).update(
                    date.fromisoformat(str(fixture["date"]))
                    for fixture in round_.get("fixtures", [])
                    if fixture.get("date") is not None
                )
    overlap = {
        season: sorted(prepared_cutoffs.get(season, set()) & dates)
        for season, dates in prepared_event_cutoffs.items()
        if prepared_cutoffs.get(season, set()) & dates
    }
    if overlap:
        raise ValueError(f"Cup dates cannot be both snapshot and event cutoffs: {overlap}")
    canonical_ids = _canonical_ids(mapping_path)
    return CupProjectionContext(
        seasons,
        canonical_ids,
        _prepare_shadow_states(
            seasons, canonical_ids, prepared_cutoffs, prepared_event_cutoffs, tier_path, mapping_path
        ),
        _prepare_bridge_states(prepared_cutoffs, prepared_event_cutoffs, tier_path, mapping_path),
    )


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
