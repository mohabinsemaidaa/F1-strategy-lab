"""Download race data via FastF1 and save the raw lap dataset.
Usage:
    python scripts/build_dataset.py --year 2025 --rounds 1 2 3 4 5 6 7 8
"""

import argparse
from pathlib import Path
from f1strategy.data import enable_cache, load_season_laps

OUT = Path("data/raw_laps.parquet")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument(
        "--rounds", type=int, nargs="+", default=list(range(1, 11)),
        help="Round numbers to download (default: 1-10)",
    )
    args = parser.parse_args()

    enable_cache()
    df = load_season_laps(args.year, args.rounds)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT)
    print(f"\nSaved {len(df):,} laps from {df['Event'].nunique()} races -> {OUT}")

if __name__ == "__main__":
    main()