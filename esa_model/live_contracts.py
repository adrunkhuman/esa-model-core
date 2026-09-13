"""Data contracts shared by live forecasts, storage, and presentation."""

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class SeasonTeam:
    team_id: str
    canonical_club_id: str
    code: str
    name: str
    short_name: str
    promoted: bool
    logo_url: str


@dataclass(frozen=True, slots=True)
class RatingChange:
    label: str
    change: float


@dataclass(frozen=True, slots=True)
class TeamRating:
    rank: int
    team_id: str
    name: str
    short_name: str
    promoted: bool
    logo_url: str
    weight: float
    attack: float
    defense: float
    actual_points: int
    expected_points: float
    points_interval_low: float
    points_interval_high: float
    offseason_change: float
    place_probabilities: list[float]
    rating_change_breakdown: tuple[RatingChange, ...] = ()
    europe_probability: float | None = None
    europe_league_probability: float | None = None
    europe_cup_win_probability: float | None = None
    europe_reallocated_probability: float | None = None
    europe_and_relegation_probability: float | None = None
    champions_league_probability: float | None = None
    europa_league_probability: float | None = None
    conference_league_probability: float | None = None
    grouped_probability_changes: tuple[float, float, float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class ProbabilityAdjustment:
    label: str
    probabilities: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class ProbabilityBreakdown:
    base: tuple[float, float, float]
    adjustments: tuple[ProbabilityAdjustment, ...]


@dataclass(frozen=True, slots=True)
class PanelMatch:
    date: str
    time: str
    home: str
    away: str
    home_logo_url: str
    away_logo_url: str
    home_goals: int | None = None
    away_goals: int | None = None
    p_home: float | None = None
    p_draw: float | None = None
    p_away: float | None = None
    probability_breakdown: ProbabilityBreakdown | None = None
    result_source: str | None = None
    result_status: str | None = None


@dataclass(frozen=True, slots=True)
class CupProbabilityEntry:
    """A named Polish Cup entrant probability, stored as a fraction in [0, 1]."""

    team_id: str
    name: str
    short_name: str
    tier: Literal["ekstraklasa", "i_liga"]
    probability: float
    logo_url: str | None = None

    def __post_init__(self) -> None:
        if not self.team_id or not self.name or not self.short_name:
            raise ValueError("Cup entries require a team ID, name, and short name")
        if self.tier not in ("ekstraklasa", "i_liga"):
            raise ValueError("Cup entry tier must be ekstraklasa or i_liga")
        if not math.isfinite(self.probability) or not 0.0 <= self.probability <= 1.0:
            raise ValueError("Cup entry probability must be a finite fraction between zero and one")


@dataclass(frozen=True, slots=True)
class CupProjection:
    """Polish Cup final-win probabilities; entries plus Others must total one."""

    entries: tuple[CupProbabilityEntry, ...]
    others_probability: float
    simulations: int
    as_of: str
    label: str = "Puchar Polski · win probability"

    def __post_init__(self) -> None:
        if len({entry.team_id for entry in self.entries}) != len(self.entries):
            raise ValueError("Cup projection contains duplicate team IDs")
        if any(entry.team_id == "others" for entry in self.entries):
            raise ValueError("Cup projection reserves the others team ID")
        if not math.isfinite(self.others_probability) or not 0.0 <= self.others_probability <= 1.0:
            raise ValueError("Cup Others probability must be a finite fraction between zero and one")
        if self.simulations <= 0 or not self.as_of or not self.label:
            raise ValueError("Cup projection requires simulations, as-of date, and label")
        probability_total = sum(entry.probability for entry in self.entries) + self.others_probability
        if not math.isclose(probability_total, 1.0, abs_tol=1e-9):
            raise ValueError("Cup probabilities including Others must normalize to one")


@dataclass(frozen=True, slots=True)
class LiveSnapshot:
    season: str
    as_of: str
    squad_as_of: str
    model: str
    simulations: int
    teams: list[TeamRating]
    parameters: dict[str, float]
    results_label: str = ""
    results: list[PanelMatch] = field(default_factory=list)
    fixtures_label: str = ""
    fixtures: list[PanelMatch] = field(default_factory=list)
    cup_projection: CupProjection | None = None
    data_status: Literal["opening", "provisional", "canonical"] = "opening"
    observed_through: str = ""
    input_hash: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)
