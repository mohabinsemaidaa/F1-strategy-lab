"""Pit-stop strategy simulator.

Uses the trained lap-time model to answer: for a given race, team and
conditions, which sequence of stints minimises total race time?

A strategy is a list of (compound, n_laps) stints, e.g.
    [("MEDIUM", 25), ("HARD", 32)]
Total time = sum of predicted lap times + a fixed pit-loss for every stop.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product
import pandas as pd
from .features import FEATURES

DEFAULT_PIT_LOSS = 22.0  # seconds lost per stop (entry + stop + exit), typical

@dataclass
class RaceContext:
    """Everything about the race that isn't the tyre strategy itself."""

    event: str
    team: str
    driver: str
    total_laps: int
    track_temp: float = 35.0
    air_temp: float = 25.0
    pit_loss: float = DEFAULT_PIT_LOSS

@dataclass
class StrategyResult:
    stints: list[tuple[str, int]]
    total_time: float
    n_stops: int
    lap_times: list[float] = field(default_factory=list)

    @property
    def label(self) -> str:
        return " → ".join(f"{c[0]}{n}" for c, n in self.stints)

def stint_frame(ctx: RaceContext, compound: str, start_lap: int, n_laps: int) -> pd.DataFrame:
    """Build the feature rows for one stint (tyre life resets to 1)."""
    laps = range(start_lap, start_lap + n_laps)
    rows = {
        "LapNumber": list(laps),
        "TyreLife": list(range(1, n_laps + 1)),
        "Stint": [1] * n_laps,  # stint index has little signal; keep constant
        "TrackTemp": [ctx.track_temp] * n_laps,
        "AirTemp": [ctx.air_temp] * n_laps,
        "Compound": [compound] * n_laps,
        "Event": [ctx.event] * n_laps,
        "Team": [ctx.team] * n_laps,
        "Driver": [ctx.driver] * n_laps,
    }
    return pd.DataFrame(rows)[FEATURES]

def simulate(model, ctx: RaceContext, stints: list[tuple[str, int]]) -> StrategyResult:
    """Predict total race time for one strategy."""
    if sum(n for _, n in stints) != ctx.total_laps:
        raise ValueError("Stint lengths must sum to total race laps.")

    lap_times: list[float] = []
    lap = 1
    for compound, n_laps in stints:
        frame = stint_frame(ctx, compound, start_lap=lap, n_laps=n_laps)
        lap_times.extend(model.predict(frame).tolist())
        lap += n_laps

    n_stops = len(stints) - 1
    total = sum(lap_times) + n_stops * ctx.pit_loss
    return StrategyResult(stints=stints, total_time=total, n_stops=n_stops, lap_times=lap_times)

def candidate_strategies(
    total_laps: int,
    compounds: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD"),
    step: int = 5,
    min_stint: int = 8,
) -> list[list[tuple[str, int]]]:
    """Enumerate sensible 1-stop and 2-stop strategies.
    F1 rules require using at least two different compounds, which this
    respects. Pit windows move in `step`-lap increments to keep the search
    small enough to run live in the dashboard.
    """
    candidates: list[list[tuple[str, int]]] = []

    # 1-stop: two stints, two different compounds.
    for c1, c2 in product(compounds, compounds):
        if c1 == c2:
            continue
        for pit in range(min_stint, total_laps - min_stint + 1, step):
            candidates.append([(c1, pit), (c2, total_laps - pit)])

    # 2-stop: three stints, at least two distinct compounds.
    for c1, c2, c3 in product(compounds, repeat=3):
        if len({c1, c2, c3}) < 2:
            continue
        for p1 in range(min_stint, total_laps - 2 * min_stint + 1, step):
            for p2 in range(p1 + min_stint, total_laps - min_stint + 1, step):
                candidates.append([(c1, p1), (c2, p2 - p1), (c3, total_laps - p2)])

    return candidates

def rank_strategies(model, ctx: RaceContext, top_n: int = 10, **kwargs) -> list[StrategyResult]:
    """Simulate all candidates and return the fastest `top_n`."""
    results = [
        simulate(model, ctx, stints)
        for stints in candidate_strategies(ctx.total_laps, **kwargs)
    ]
    results.sort(key=lambda r: r.total_time)
    return results[:top_n]