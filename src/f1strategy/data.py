"""Load and cache Formula 1 race data via the FastF1 API.

FastF1 downloads official timing data. Everything is cached locally so
repeat runs are fast and we don't hammer the API.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

CACHE_DIR = Path("data/fastf1_cache")
# Columns we keep from the raw FastF1 laps table.
RAW_COLUMNS = [
    "Driver",
    "Team",
    "LapNumber",
    "LapTime",
    "Stint",
    "Compound",
    "TyreLife",
    "FreshTyre",
    "TrackStatus",
    "IsAccurate",
    "PitInTime",
    "PitOutTime",
]

def enable_cache(cache_dir: Path = CACHE_DIR) -> None:
    """Turn on FastF1's on-disk cache (shit creates the folder if it needs to)."""
    import fastf1

    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

def load_race_laps(year: int, gp: str | int) -> pd.DataFrame:
    """Load all of the race laps for one Grand Prix, with per-lap weather attached.

    Parameters
    ----------
    year : season, e.g. 2024
    gp   : event name ("Bahrain") or round number (1)
    """
    import fastf1

    session = fastf1.get_session(year, gp, "R")
    session.load(laps=True, telemetry=False, weather=True, messages=False)
    laps = session.laps
    df = laps[RAW_COLUMNS].copy().reset_index(drop=True)
    # Per-lap weather, aligned by FastF1 itself.
    weather = laps.get_weather_data().reset_index(drop=True)
    df["AirTemp"] = weather["AirTemp"]
    df["TrackTemp"] = weather["TrackTemp"]

    df["Event"] = session.event["EventName"]
    df["Year"] = year
    df["TotalLaps"] = session.total_laps
    return df


def load_season_laps(year: int, rounds: list[int]) -> pd.DataFrame:
    """Load and concatenate several races from one season."""
    frames = []
    for rnd in rounds:
        print(f"Loading {year} round {rnd} ...")
        try:
            frames.append(load_race_laps(year, rnd))
        except Exception as exc:  # a session can be missing/broken upstream
            print(f"  skipped round {rnd}: {exc}")
    if not frames:
        raise RuntimeError("No races could be loaded.")
    return pd.concat(frames, ignore_index=True)