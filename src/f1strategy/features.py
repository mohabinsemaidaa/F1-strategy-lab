"""Turn raw FastF1 laps into a clean modelling dataset.

The target is lap time in seconds. The features capture the three main
physical drivers of race pace:

- fuel load        -> LapNumber (cars start heavy and get ~0.03s/lap faster)
- tyre wear        -> TyreLife + Compound (the degradation signal we care about)
- track conditions -> TrackTemp / AirTemp
plus categorical context: Event (circuit), Team (car performance), Driver.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

NUMERIC_FEATURES = ["LapNumber", "TyreLife", "Stint", "TrackTemp", "AirTemp"]
CATEGORICAL_FEATURES = ["Compound", "Event", "Team", "Driver"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "LapTimeSeconds"

def clean_laps(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the representative racing laps.

    Of course we have to remove:
    - in-laps / out-laps (pit entry & exit distort lap time)
    - laps under safety car, VSC, yellow flags (TrackStatus != "1")
    - laps FastF1 flags as inaccurate, and laps with no time at all
    - obvious outliers: slower than 107% of that driver's median race pace
    """
    out = df.copy()

    out = out[out["LapTime"].notna()]
    out = out[out["PitInTime"].isna() & out["PitOutTime"].isna()]
    out = out[out["TrackStatus"].astype(str) == "1"]
    if "IsAccurate" in out.columns:
        out = out[out["IsAccurate"].astype(bool)]

    out[TARGET] = out["LapTime"].dt.total_seconds()

    # 107% rule per driver per event: drop laps far off that driver's pace.
    median = out.groupby(["Event", "Driver"])[TARGET].transform("median")
    out = out[out[TARGET] <= 1.07 * median]

    # A tiny number of laps come through with unknown compound.
    out = out[out["Compound"].notna() & (out["Compound"] != "UNKNOWN")]

    return out.reset_index(drop=True)

def build_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) ready for the model pipeline."""
    clean = clean_laps(df)

    # Fills the occasional missing weather with the event median.
    for col in ("TrackTemp", "AirTemp"):
        if col not in clean.columns:
            clean[col] = np.nan
        clean[col] = clean.groupby("Event")[col].transform(
            lambda s: s.fillna(s.median())
        )
        clean[col] = clean[col].fillna(clean[col].median())

    X = clean[FEATURES].copy()
    y = clean[TARGET].copy()
    return X, y

def dataset_metadata(df: pd.DataFrame) -> dict:
    """Small summary saved alongside the model so the dashboard can offer
    sensible dropdowns and defaults without re-reading the raw data."""
    clean = clean_laps(df)
    meta = {
        "events": sorted(clean["Event"].unique().tolist()),
        "teams": sorted(clean["Team"].unique().tolist()),
        "drivers": sorted(clean["Driver"].unique().tolist()),
        "compounds": sorted(clean["Compound"].unique().tolist()),
        "median_track_temp": float(clean["TrackTemp"].median()),
        "median_air_temp": float(clean["AirTemp"].median()),
        "total_laps_by_event": (
            df.groupby("Event")["TotalLaps"].max().astype(int).to_dict()
            if "TotalLaps" in df.columns
            else {}
        ),
    }
    return meta