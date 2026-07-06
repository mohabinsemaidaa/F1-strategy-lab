"""Tests for lap cleaning and dataset building (synthetic data, no API)."""

import pandas as pd
import pytest
from f1strategy.features import FEATURES, TARGET, build_dataset, clean_laps

def make_raw_laps() -> pd.DataFrame:
    """Six laps: 4 good, 1 pit-in lap, 1 safety-car lap."""
    base = {
        "Driver": ["VER"] * 6,
        "Team": ["Red Bull Racing"] * 6,
        "LapNumber": [1, 2, 3, 4, 5, 6],
        "LapTime": [pd.Timedelta(seconds=t) for t in (92.0, 91.5, 91.8, 92.1, 96.0, 110.0)],
        "Stint": [1] * 6,
        "Compound": ["MEDIUM"] * 6,
        "TyreLife": [1, 2, 3, 4, 5, 6],
        "FreshTyre": [True] + [False] * 5,
        "TrackStatus": ["1", "1", "1", "1", "1", "4"],  # lap 6 = safety car
        "IsAccurate": [True] * 6,
        "PitInTime": pd.Series(
            [pd.NaT] * 4 + [pd.Timedelta(seconds=1)] + [pd.NaT], # lap 5 pits
            dtype="timedelta64[ns]",
        ),
        "PitOutTime": pd.Series([pd.NaT] * 6, dtype="timedelta64[ns]"),
        "Event": ["Bahrain Grand Prix"] * 6,
        "TrackTemp": [35.0] * 6,
        "AirTemp": [25.0] * 6,
    }
    return pd.DataFrame(base)

def test_clean_laps_removes_pit_and_safety_car_laps():
    clean = clean_laps(make_raw_laps())
    assert len(clean) == 4
    assert clean["LapNumber"].tolist() == [1, 2, 3, 4]

def test_clean_laps_converts_target_to_seconds():
    clean = clean_laps(make_raw_laps())
    assert TARGET in clean.columns
    assert clean[TARGET].iloc[0] == pytest.approx(92.0)

def test_clean_laps_applies_107_percent_rule():
    raw = make_raw_laps()
    # Makes lap 4 a huge outlier but keep it green-flag / non-pit.
    raw.loc[3, "LapTime"] = pd.Timedelta(seconds=140)
    clean = clean_laps(raw)
    assert 4 not in clean["LapNumber"].tolist()

def test_build_dataset_shapes_and_columns():
    X, y = build_dataset(make_raw_laps())
    assert list(X.columns) == FEATURES
    assert len(X) == len(y) == 4
    assert X.isna().sum().sum() == 0