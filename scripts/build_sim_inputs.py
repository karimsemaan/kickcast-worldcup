"""Pre-compute loader outputs so notebook 04 runs without raw data.

Calls each load_* function from data/scripts/02_build_features.py and saves
its return value to data/processed/sim_inputs/ using parquet (for
DataFrames) and JSON (for dicts). The loader functions check for these
files first, so a grader without raw data still gets the full simulation
pipeline working.
"""

import json
import sys
import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "data" / "scripts"
OUT = ROOT / "data" / "processed" / "sim_inputs"
OUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "fb", SCRIPTS / "02_build_features.py"
)
fb = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(SCRIPTS))
spec.loader.exec_module(fb)


def save_parquet(df: pd.DataFrame, name: str) -> None:
    path = OUT / f"{name}.parquet"
    df.to_parquet(path, compression="zstd", index=False)
    mb = path.stat().st_size / (1024 * 1024)
    print(f"  {name:20s}  {mb:6.2f} MB  ({len(df):,} rows)")


def save_json(obj, name: str) -> None:
    path = OUT / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    mb = path.stat().st_size / (1024 * 1024)
    print(f"  {name:20s}  {mb:6.2f} MB")


def main() -> None:
    print("Loading match results...")
    save_parquet(fb.load_results(), "results")

    print("\nLoading FIFA rankings...")
    dates, by_date = fb.load_fifa_rankings()
    rows = []
    for d, teams in by_date.items():
        for team, info in teams.items():
            rows.append({
                "rank_date": d,
                "team": team,
                "rank": info["rank"],
                "points": info["points"],
                "conf": info.get("conf", ""),
            })
    save_parquet(pd.DataFrame(rows), "fifa_rankings")

    print("\nLoading Transfermarkt players + valuations...")
    players, valuations = fb.load_players_and_valuations()
    save_parquet(players, "players")
    save_parquet(valuations, "valuations")

    print("\nLoading World Cup history...")
    save_json(fb.load_wc_history(), "wc_history")

    total = sum(p.stat().st_size for p in OUT.iterdir())
    print(f"\nTotal: {total / (1024 * 1024):.1f} MB in {OUT}")


if __name__ == "__main__":
    main()
