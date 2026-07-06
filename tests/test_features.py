"""Tests for lap cleaning and dataset building (synthetic data, no API).

Timedelta columns are built via `td_seconds` (numpy, explicit ns unit) rather
than pd.Timedelta/pd.to_timedelta: those constructors trigger NumPy's
"generic unit" DeprecationWarning on some numpy/pandas combinations, and we
want this suite to pass with -W error::DeprecationWarning.
"""

import numpy as np
import pandas as pd
import pytest
from f1strategy.features import FEATURES, TARGET, build_dataset, clean_laps

def td_seconds(values) -> pd.Series:
    """Seconds (floats, NaN allowed) -> timedelta64[ns] Series, warning-free."""
    return pd.Series((np.asarray(values, dtype="float64") * 1e9).astype("timedelta64[ns]"))

def make_raw_laps() -> pd.DataFrame:
    """Six laps: 4 good, 1 pit-in lap, 1 safety-car lap."""
    nan = np.nan
    base = {
        "Driver": ["VER"] * 6,
        "Team": ["Red Bull Racing"] * 6,
        "LapNumber": [1, 2, 3, 4, 5, 6],
        "LapTime": td_seconds([92.0, 91.5, 91.8, 92.1, 96.0, 110.0]),
        "Stint": [1] * 6,
        "Compound": ["MEDIUM"] * 6,
        "TyreLife": [1, 2, 3, 4, 5, 6],
        "FreshTyre": [True] + [False] * 5,
        "TrackStatus": ["1", "1", "1", "1", "1", "4"],  # lap 6 = safety car
        "PitInTime": td_seconds([nan, nan, nan, nan, 1.0, nan]),  # lap 5 pits
        "PitOutTime": td_seconds([nan] * 6),
        "IsAccurate": [True] * 6,
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
    # Make lap 4 a huge outlier but keep it green-flag / non-pit.
    raw.loc[3, "LapTime"] = np.timedelta64(140_000_000_000, "ns")  # 140 s
    clean = clean_laps(raw)
    assert 4 not in clean["LapNumber"].tolist()

def test_build_dataset_shapes_and_columns():
    X, y = build_dataset(make_raw_laps())
    assert list(X.columns) == FEATURES
    assert len(X) == len(y) == 4
    assert X.isna().sum().sum() == 0