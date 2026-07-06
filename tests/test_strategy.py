"""Tests for the strategy simulator.

Uses a fake model with known physics (base pace + linear degradation +
fuel effect) so we can assert exact behaviour without training anything.
"""

import numpy as np
import pytest
from f1strategy.strategy import (
    RaceContext,
    candidate_strategies,
    rank_strategies,
    simulate,
)

DEG = {"SOFT": 0.10, "MEDIUM": 0.05, "HARD": 0.02}  # s per lap of tyre life
BASE = {"SOFT": 90.0, "MEDIUM": 90.6, "HARD": 91.2}
FUEL = 0.03  # s gained per lap as fuel burns off

class FakeModel:
    """Predicts base + degradation * tyre_life - fuel_effect * lap_number."""

    def predict(self, frame):
        out = []
        for _, row in frame.iterrows():
            t = (
                BASE[row["Compound"]]
                + DEG[row["Compound"]] * row["TyreLife"]
                - FUEL * row["LapNumber"]
            )
            out.append(t)
        return np.array(out)

CTX = RaceContext(
    event="Bahrain Grand Prix",
    team="McLaren",
    driver="NOR",
    total_laps=57,
    pit_loss=22.0,
)

def test_simulate_counts_pit_loss_per_stop():
    one_stop = simulate(FakeModel(), CTX, [("MEDIUM", 28), ("HARD", 29)])
    two_stop = simulate(FakeModel(), CTX, [("MEDIUM", 19), ("MEDIUM", 19), ("HARD", 19)])
    assert one_stop.n_stops == 1
    assert two_stop.n_stops == 2
    assert len(one_stop.lap_times) == CTX.total_laps

def test_simulate_rejects_wrong_lap_count():
    with pytest.raises(ValueError):
        simulate(FakeModel(), CTX, [("MEDIUM", 10), ("HARD", 10)])

def test_tyre_life_resets_after_pit_stop():
    result = simulate(FakeModel(), CTX, [("SOFT", 20), ("SOFT", 37)])
    # Lap 21 is on fresh softs: much faster than worn lap 20.
    assert result.lap_times[20] < result.lap_times[19]

def test_candidates_respect_two_compound_rule():
    for strat in candidate_strategies(57):
        compounds = {c for c, _ in strat}
        assert len(compounds) >= 2

def test_rank_strategies_returns_sorted_results():
    ranked = rank_strategies(FakeModel(), CTX, top_n=5)
    times = [r.total_time for r in ranked]
    assert times == sorted(times)
    assert len(ranked) == 5

def test_high_pit_loss_favours_fewer_stops():
    """With an enormous pit loss the best strategy must be a 1-stop."""
    slow_pits = RaceContext(**{**CTX.__dict__, "pit_loss": 200.0})
    best = rank_strategies(FakeModel(), slow_pits, top_n=1)[0]
    assert best.n_stops == 1