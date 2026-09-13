"""Canonical club identity across season-specific local team IDs."""

import csv
from pathlib import Path


def load_canonical_club_ids(path: Path) -> dict[tuple[str, str], str]:
    canonical: dict[tuple[str, str], str] = {}
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            club_id = row["canonical_club_id"]
            for year in range(int(row["valid_from_season"][:4]), int(row["valid_to_season"][:4]) + 1):
                season = f"{year}/{str(year + 1)[-2:]}"
                key = season, row["local_team_id"]
                if key in canonical:
                    raise ValueError(f"Duplicate canonical identity for season {season}, team {row['local_team_id']}")
                canonical[key] = club_id
    return canonical


def reconcile_state_team_ids[StateValue](
    states: dict[str, StateValue],
    previous_season: str,
    previous_teams: set[str],
    current_season: str,
    current_teams: set[str],
    canonical_ids: dict[tuple[str, str], str],
) -> set[str]:
    previous_by_club = {canonical_ids[previous_season, team]: team for team in previous_teams}
    current_by_club = {canonical_ids[current_season, team]: team for team in current_teams}
    reconciled = set(previous_teams)
    for club_id in previous_by_club.keys() & current_by_club.keys():
        old_id = previous_by_club[club_id]
        new_id = current_by_club[club_id]
        if old_id == new_id:
            continue
        states[new_id] = states.pop(old_id)
        reconciled.remove(old_id)
        reconciled.add(new_id)
    return reconciled
