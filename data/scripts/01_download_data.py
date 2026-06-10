"""
01_download_data.py -- Download all raw datasets for the KickCast project.

Sources:
  1. Transfermarkt (R2 CDN)          -> data/raw/transfermarkt/
  2. Injury histories (GitHub)        -> data/raw/injuries/
  3. International results (Kaggle)   -> data/raw/international_results/
  4. FIFA rankings (Kaggle)           -> data/raw/fifa_rankings/
  5. World Cup history (Kaggle)       -> data/raw/world_cup/
  6. Elo ratings (eloratings.net)     -> data/raw/elo/

Does NOT download EA FC / FIFA video game ratings.
"""

import gzip
import io
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data" / "raw"

TRANSFERMARKT_DIR = RAW / "transfermarkt"
INJURIES_DIR = RAW / "injuries"
INTL_RESULTS_DIR = RAW / "international_results"
FIFA_RANKINGS_DIR = RAW / "fifa_rankings"
WORLD_CUP_DIR = RAW / "world_cup"
ELO_DIR = RAW / "elo"

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
})

# All 48 qualified teams: (display name, eloratings.net URL slug)
QUALIFIED_TEAMS = [
    ("Mexico", "Mexico"),
    ("South Korea", "South_Korea"),
    ("South Africa", "South_Africa"),
    ("Czechia", "Czechia"),
    ("Canada", "Canada"),
    ("Switzerland", "Switzerland"),
    ("Qatar", "Qatar"),
    ("Bosnia and Herzegovina", "Bosnia_and_Herzegovina"),
    ("Brazil", "Brazil"),
    ("Morocco", "Morocco"),
    ("Scotland", "Scotland"),
    ("Haiti", "Haiti"),
    ("United States", "United_States"),
    ("Australia", "Australia"),
    ("Paraguay", "Paraguay"),
    ("Turkiye", "Turkey"),
    ("Germany", "Germany"),
    ("Ivory Coast", "Ivory_Coast"),
    ("Ecuador", "Ecuador"),
    ("Curacao", "Curacao"),
    ("Japan", "Japan"),
    ("Netherlands", "Netherlands"),
    ("Tunisia", "Tunisia"),
    ("Sweden", "Sweden"),
    ("Belgium", "Belgium"),
    ("Egypt", "Egypt"),
    ("Iran", "Iran"),
    ("New Zealand", "New_Zealand"),
    ("Spain", "Spain"),
    ("Uruguay", "Uruguay"),
    ("Saudi Arabia", "Saudi_Arabia"),
    ("Cape Verde", "Cape_Verde"),
    ("France", "France"),
    ("Senegal", "Senegal"),
    ("Norway", "Norway"),
    ("Iraq", "Iraq"),
    ("Argentina", "Argentina"),
    ("Algeria", "Algeria"),
    ("Austria", "Austria"),
    ("Jordan", "Jordan"),
    ("Portugal", "Portugal"),
    ("Colombia", "Colombia"),
    ("Uzbekistan", "Uzbekistan"),
    ("DR Congo", "DR_Congo"),
    ("England", "England"),
    ("Croatia", "Croatia"),
    ("Ghana", "Ghana"),
    ("Panama", "Panama"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def download_file(url: str, dest: Path, desc: str | None = None, headers: dict | None = None) -> bool:
    """Download a file with progress bar and resume support."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: check existing partial download
    resume_pos = dest.stat().st_size if dest.exists() else 0
    req_headers = dict(headers or {})
    if resume_pos:
        req_headers["Range"] = f"bytes={resume_pos}-"

    try:
        resp = SESSION.get(url, stream=True, headers=req_headers, timeout=60)
        if resp.status_code == 416:  # Range not satisfiable -> already complete
            print(f"  [OK] {desc or dest.name} (already downloaded)")
            return True
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        mode = "ab" if resume_pos and resp.status_code == 206 else "wb"
        if mode == "wb":
            resume_pos = 0

        with open(dest, mode) as f, tqdm(
            total=total + resume_pos if total else None,
            initial=resume_pos,
            unit="B",
            unit_scale=True,
            desc=desc or dest.name,
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        return True

    except Exception as e:
        print(f"  [FAIL] Failed to download {desc or url}: {e}")
        return False


def decompress_gz(gz_path: Path, csv_path: Path):
    """Decompress a .csv.gz file to .csv."""
    with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()


def run_kaggle_download(dataset: str, dest_dir: Path) -> bool:
    """Download a Kaggle dataset. Returns True on success."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset, "-p", str(dest_dir), "--unzip"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            return True
        print(f"  [FAIL] Kaggle download failed: {result.stderr.strip()}")
        return False
    except FileNotFoundError:
        print("  [FAIL] kaggle CLI not found in PATH")
        return False
    except subprocess.TimeoutExpired:
        print("  [FAIL] Kaggle download timed out")
        return False
    except Exception as e:
        print(f"  [FAIL] Kaggle error: {e}")
        return False


def csv_summary(path: Path) -> str:
    """Return a short summary of a CSV file."""
    try:
        df = pd.read_csv(path, low_memory=False)
        info = f"{len(df):,} rows × {len(df.columns)} cols"
        # Try to find a date column for range
        for col in ("date", "Date", "rank_date", "datetime"):
            if col in df.columns:
                dates = pd.to_datetime(df[col], errors="coerce").dropna()
                if len(dates) > 0:
                    info += f"  [{dates.min().date()} -> {dates.max().date()}]"
                break
        return info
    except Exception as e:
        return f"(could not read: {e})"


# ===========================================================================
# 1. Transfermarkt -- R2 CDN
# ===========================================================================
def download_transfermarkt():
    print("\n" + "=" * 60)
    print("1. TRANSFERMARKT (R2 CDN)")
    print("=" * 60)

    base_url = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
    files = [
        "players.csv.gz",
        "player_valuations.csv.gz",
        "appearances.csv.gz",
        "games.csv.gz",
        "clubs.csv.gz",
    ]

    for fname in files:
        csv_name = fname.replace(".gz", "")
        csv_path = TRANSFERMARKT_DIR / csv_name
        if csv_path.exists() and csv_path.stat().st_size > 0:
            print(f"  [OK] {csv_name} already exists -- skipping")
            continue

        gz_path = TRANSFERMARKT_DIR / fname
        url = f"{base_url}/{fname}"
        ok = download_file(url, gz_path, desc=fname)
        if ok and gz_path.exists():
            print(f"  Decompressing {fname} ...")
            decompress_gz(gz_path, csv_path)
            print(f"  [OK] {csv_name}")


# ===========================================================================
# 2. Injury histories -- GitHub sparse clone
# ===========================================================================
def download_injuries():
    print("\n" + "=" * 60)
    print("2. INJURY HISTORIES (GitHub: salimt/football-datasets)")
    print("=" * 60)

    all_injuries_path = INJURIES_DIR / "all_injuries.csv"
    if all_injuries_path.exists() and all_injuries_path.stat().st_size > 0:
        print("  [OK] all_injuries.csv already exists -- skipping")
        return

    INJURIES_DIR.mkdir(parents=True, exist_ok=True)
    clone_dir = INJURIES_DIR / "_football-datasets"

    # Clean up any previous failed attempt
    if clone_dir.exists():
        shutil.rmtree(clone_dir, ignore_errors=True)

    print("  Sparse-cloning repository (injury data only) ...")
    r = subprocess.run(
        ["git", "clone", "--filter=blob:none", "--sparse", "--depth=1",
         "https://github.com/salimt/football-datasets.git", str(clone_dir)],
        capture_output=True, text=True, timeout=300,
    )
    if r.returncode != 0:
        print(f"  [FAIL] Clone failed: {r.stderr.strip()}")
        return

    # The repo has a single consolidated CSV at datalake/transfermarkt/player_injuries/
    r = subprocess.run(
        ["git", "sparse-checkout", "set", "datalake/transfermarkt/player_injuries"],
        capture_output=True, text=True, cwd=str(clone_dir), timeout=300,
    )
    if r.returncode != 0:
        print(f"  [FAIL] sparse-checkout set failed: {r.stderr.strip()}")
        return

    r = subprocess.run(
        ["git", "checkout"],
        capture_output=True, text=True, cwd=str(clone_dir), timeout=600,
    )

    src_csv = clone_dir / "datalake" / "transfermarkt" / "player_injuries" / "player_injuries.csv"
    if src_csv.exists():
        shutil.copy2(src_csv, all_injuries_path)
        row_count = sum(1 for _ in open(all_injuries_path)) - 1
        print(f"  [OK] all_injuries.csv -- {row_count:,} injury records")
    else:
        print(f"  [FAIL] Injury CSV not found at {src_csv}")

    # Clean up clone
    shutil.rmtree(clone_dir, ignore_errors=True)


# ===========================================================================
# 3. International results -- Kaggle
# ===========================================================================
def download_international_results():
    print("\n" + "=" * 60)
    print("3. INTERNATIONAL MATCH RESULTS (Kaggle: martj42)")
    print("=" * 60)

    results_path = INTL_RESULTS_DIR / "results.csv"
    if results_path.exists() and results_path.stat().st_size > 0:
        print("  [OK] results.csv already exists -- skipping")
        return

    # Try Kaggle first
    ok = run_kaggle_download(
        "martj42/international-football-results-from-1872-to-2017",
        INTL_RESULTS_DIR,
    )
    if ok and results_path.exists():
        print("  [OK] Downloaded via Kaggle")
        return

    # Fallback: download from GitHub (martj42/international_results)
    print("  Trying GitHub fallback ...")
    github_base = "https://raw.githubusercontent.com/martj42/international_results/master"
    for fname in ["results.csv", "goalscorers.csv", "shootouts.csv"]:
        url = f"{github_base}/{fname}"
        download_file(url, INTL_RESULTS_DIR / fname, desc=fname)


# ===========================================================================
# 4. FIFA World Rankings -- FIFA API (primary) / Kaggle / GitHub fallback
# ===========================================================================
def _scrape_fifa_rankings_api() -> pd.DataFrame | None:
    """Scrape FIFA rankings directly from the official FIFA API.

    Steps:
      1. Fetch inside.fifa.com ranking page to extract schedule IDs from __NEXT_DATA__
      2. For each schedule, call the public api.fifa.com endpoint for that snapshot
      3. Normalize into a DataFrame matching our standard columns
    """
    import json
    import re

    print("  Fetching schedule IDs from inside.fifa.com ...")
    try:
        resp = SESSION.get("https://inside.fifa.com/fifa-world-ranking/men", timeout=30)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        print(f"  [FAIL] Could not reach inside.fifa.com: {e}")
        return None

    # Extract __NEXT_DATA__ JSON
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
    if not m:
        print("  [FAIL] Could not find __NEXT_DATA__ in FIFA page")
        return None

    try:
        next_data = json.loads(m.group(1))
        date_groups = next_data["props"]["pageProps"]["pageData"]["ranking"]["dates"]
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  [FAIL] Could not parse schedule data: {e}")
        return None

    # Flatten all schedule entries
    schedules = []
    for group in date_groups:
        for entry in group.get("dates", []):
            sid = entry.get("id", "")
            date_str = entry.get("matchWindowEndDate") or entry.get("iso", "")[:10]
            if sid and date_str:
                schedules.append((sid, date_str))

    print(f"  Found {len(schedules)} ranking snapshots ({schedules[-1][1]} to {schedules[0][1]})")

    # Fetch each snapshot from the API
    api_base = "https://api.fifa.com/api/v3/fifarankings/rankings/rankingsbyschedule"
    all_rows = []
    failed = 0

    for sid, date_str in tqdm(schedules, desc="  FIFA API snapshots", unit="snap"):
        url = f"{api_base}?rankingScheduleId={sid}&language=en"
        try:
            resp = SESSION.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            failed += 1
            time.sleep(0.3)
            continue

        results = data.get("Results", [])
        for team in results:
            # Extract English team name
            name_list = team.get("TeamName", [])
            team_name = name_list[0].get("Description", "") if name_list else ""
            country_code = team.get("IdCountry", "")

            all_rows.append({
                "rank": team.get("Rank"),
                "country_full": team_name,
                "country_abrv": country_code,
                "total_points": team.get("TotalPoints"),
                "previous_points": team.get("PrevPoints"),
                "rank_change": team.get("RankingMovement"),
                "confederation": team.get("ConfederationName", ""),
                "rank_date": date_str,
            })

        time.sleep(0.2)  # Be polite

    if failed:
        print(f"  [WARN] {failed}/{len(schedules)} snapshots failed")

    if all_rows:
        df = pd.DataFrame(all_rows)
        df = df.sort_values(["rank_date", "rank"]).reset_index(drop=True)
        return df
    return None


def download_fifa_rankings():
    print("\n" + "=" * 60)
    print("4. FIFA WORLD RANKINGS")
    print("=" * 60)

    ranking_path = FIFA_RANKINGS_DIR / "fifa_ranking.csv"

    # Check if already present
    if ranking_path.exists() and ranking_path.stat().st_size > 0:
        df = pd.read_csv(ranking_path, low_memory=False)
        if "rank_date" in df.columns:
            max_date = df["rank_date"].max()
            # If data extends past 2021, it's good enough
            if max_date >= "2024-01-01":
                print(f"  [OK] fifa_ranking.csv already present ({len(df):,} rows, through {max_date}) -- skipping")
                return

    FIFA_RANKINGS_DIR.mkdir(parents=True, exist_ok=True)

    # Strategy 1: Kaggle (most complete, if available)
    ok = run_kaggle_download("cashncarry/fifaworldranking", FIFA_RANKINGS_DIR)
    if ok and ranking_path.exists():
        print("  [OK] Downloaded via Kaggle")
        return

    # Strategy 2: Scrape directly from FIFA's public API (authoritative, 1993-present)
    print("  Kaggle unavailable. Scraping from official FIFA API ...")
    df_fifa = _scrape_fifa_rankings_api()
    if df_fifa is not None and len(df_fifa) > 0:
        df_fifa.to_csv(ranking_path, index=False)
        date_min = df_fifa["rank_date"].min()
        date_max = df_fifa["rank_date"].max()
        print(f"  [OK] fifa_ranking.csv -- {len(df_fifa):,} rows [{date_min} -> {date_max}]")
        return

    # Strategy 3: GitHub fallback mirrors (partial, 1993-2018 + 2018-2021)
    print("  FIFA API failed. Trying GitHub fallback mirrors ...")
    frames = []

    url1 = "https://raw.githubusercontent.com/tadhgfitzgerald/fifa_ranking/master/fifa_ranking.csv"
    if download_file(url1, FIFA_RANKINGS_DIR / "_rankings_1993_2018.csv", desc="Rankings 1993-2018"):
        try:
            df1 = pd.read_csv(FIFA_RANKINGS_DIR / "_rankings_1993_2018.csv", low_memory=False)
            if "rank_date" in df1.columns:
                frames.append(df1[["rank", "country_full", "country_abrv", "total_points",
                                   "previous_points", "rank_change", "rank_date"]])
        except Exception as e:
            print(f"  [WARN] Could not parse 1993-2018 data: {e}")

    url2 = "https://raw.githubusercontent.com/irisroatis/fifaranking20182021/main/fifarankings2018-2021.csv"
    if download_file(url2, FIFA_RANKINGS_DIR / "_rankings_2018_2021.csv", desc="Rankings 2018-2021"):
        try:
            df2 = pd.read_csv(FIFA_RANKINGS_DIR / "_rankings_2018_2021.csv", low_memory=False)
            col_map = {}
            for c in df2.columns:
                cl = c.lower().strip()
                if cl in ("rk", "rank"):
                    col_map[c] = "rank"
                elif cl == "country":
                    col_map[c] = "country_full"
                elif cl == "points":
                    col_map[c] = "total_points"
                elif "date" in cl:
                    col_map[c] = "rank_date"
            df2 = df2.rename(columns=col_map)
            keep = [c for c in ["rank", "country_full", "total_points", "rank_date"] if c in df2.columns]
            if keep:
                frames.append(df2[keep])
        except Exception as e:
            print(f"  [WARN] Could not parse 2018-2021 data: {e}")

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(ranking_path, index=False)
        print(f"  [OK] fifa_ranking.csv -- {len(combined):,} rows (GitHub fallback, partial coverage)")
        for f in FIFA_RANKINGS_DIR.glob("_rankings_*.csv"):
            f.unlink()
    else:
        print("  [FAIL] Could not download FIFA rankings from any source.")


# ===========================================================================
# 5. World Cup history -- Kaggle
# ===========================================================================
def download_world_cup_history():
    print("\n" + "=" * 60)
    print("5. HISTORICAL WORLD CUP DATA (Kaggle: piterfm)")
    print("=" * 60)

    existing = list(WORLD_CUP_DIR.glob("*.csv"))
    if existing:
        print(f"  [OK] {len(existing)} CSV file(s) already present -- skipping")
        return

    ok = run_kaggle_download("piterfm/fifa-football-world-cup", WORLD_CUP_DIR)
    if ok:
        print("  [OK] Downloaded via Kaggle")
        return

    # Fallback: jfjelstul/worldcup on GitHub (1930-2022)
    print("  Kaggle unavailable. Trying GitHub fallback (jfjelstul/worldcup) ...")
    base = "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv"
    files = ["matches.csv", "group_standings.csv", "tournaments.csv", "goals.csv"]
    any_ok = False
    for fname in files:
        ok = download_file(f"{base}/{fname}", WORLD_CUP_DIR / fname, desc=fname)
        any_ok = any_ok or ok
    if any_ok:
        print("  [OK] World Cup history downloaded from GitHub")


# ===========================================================================
# 6. Elo ratings -- eloratings.net (live TSV scrape)
# ===========================================================================
def download_elo_ratings():
    print("\n" + "=" * 60)
    print("6. ELO RATINGS (eloratings.net -- live)")
    print("=" * 60)

    ELO_DIR.mkdir(parents=True, exist_ok=True)
    base = "https://www.eloratings.net"
    elo_headers = {"X-Requested-With": "XMLHttpRequest"}

    def fetch_tsv(endpoint: str, desc: str) -> str | None:
        url = f"{base}/{endpoint}"
        try:
            resp = SESSION.get(url, headers=elo_headers, timeout=30)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            print(f"  [FAIL] Failed to fetch {desc}: {e}")
            return None

    # 6a. Lookup tables
    print("  Fetching lookup tables ...")
    teams_tsv = fetch_tsv("en.teams.tsv", "en.teams.tsv")
    tournaments_tsv = fetch_tsv("en.tournaments.tsv", "en.tournaments.tsv")

    # Parse team code -> name mapping
    code_to_name = {}
    if teams_tsv:
        with open(ELO_DIR / "en.teams.tsv", "w", encoding="utf-8") as f:
            f.write(teams_tsv)
        for line in teams_tsv.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 2:
                code_to_name[parts[0].strip()] = parts[1].strip()
        print(f"  [OK] en.teams.tsv -- {len(code_to_name)} team codes")

    # Parse tournament code -> name mapping
    tournament_map = {}
    if tournaments_tsv:
        with open(ELO_DIR / "en.tournaments.tsv", "w", encoding="utf-8") as f:
            f.write(tournaments_tsv)
        for line in tournaments_tsv.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 2:
                tournament_map[parts[0].strip()] = parts[1].strip()
        print(f"  [OK] en.tournaments.tsv -- {len(tournament_map)} tournament codes")

    # 6b. World rankings
    print("  Fetching current world rankings ...")
    world_tsv = fetch_tsv("World.tsv", "World.tsv")

    if world_tsv:
        with open(ELO_DIR / "World.tsv", "w", encoding="utf-8") as f:
            f.write(world_tsv)

        # Parse into elo_current.csv
        # World.tsv fields (no header, tab-separated):
        #   0=rank, 1=rank2, 2=team_code, 3=rating, 4=highest_rank, 5=highest_rating,
        #   6=avg_rank, 7=avg_rating, ...22=total_matches, 26=wins, 27=losses,
        #   28=draws, 29=goals_for, 30=goals_against
        rows = []
        for line in world_tsv.strip().split("\n"):
            fields = line.split("\t")
            if len(fields) < 4:
                continue
            try:
                rank = int(fields[0])
                team_code = fields[2].strip()
                rating = int(fields[3])
                team_name = code_to_name.get(team_code, team_code)
                rows.append({
                    "rank": rank,
                    "team_code": team_code,
                    "team": team_name,
                    "rating": rating,
                })
            except (ValueError, IndexError):
                continue

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(ELO_DIR / "elo_current.csv", index=False)
            print(f"  [OK] elo_current.csv -- {len(df)} teams (top: {rows[0]['team']} @ {rows[0]['rating']})")

    # 6c. Individual team match histories
    print(f"  Fetching match histories for {len(QUALIFIED_TEAMS)} qualified teams ...")
    all_matches = []
    failed_teams = []

    for display_name, url_slug in tqdm(QUALIFIED_TEAMS, desc="  Team histories", unit="team"):
        tsv_text = fetch_tsv(f"{url_slug}.tsv", display_name)
        if tsv_text is None:
            failed_teams.append(display_name)
            time.sleep(0.5)
            continue

        # Save raw TSV
        with open(ELO_DIR / f"{url_slug}.tsv", "w", encoding="utf-8") as f:
            f.write(tsv_text)

        # Parse match-by-match history
        # Team TSV fields (no header, tab-separated):
        #   0=year, 1=month, 2=day, 3=home_code, 4=away_code,
        #   5=home_goals, 6=away_goals, 7=tournament_code, 8=(empty/location),
        #   9=elo_exchanged, 10=home_elo_after, 11=away_elo_after,
        #   12=home_rank_change, 13=away_rank_change, 14=home_rank, 15=away_rank
        for line in tsv_text.strip().split("\n"):
            fields = line.split("\t")
            if len(fields) < 12:
                continue
            try:
                year = int(fields[0])
                month = int(fields[1])
                day = int(fields[2])
                date = f"{year:04d}-{month:02d}-{day:02d}"
                home_code = fields[3].strip()
                away_code = fields[4].strip()
                home_goals = fields[5].strip()
                away_goals = fields[6].strip()
                tournament_code = fields[7].strip()
                elo_exchanged = fields[9].strip()
                home_elo_after = fields[10].strip()
                away_elo_after = fields[11].strip()

                # Ranks may not always be present
                home_rank = fields[14].strip() if len(fields) > 14 else ""
                away_rank = fields[15].strip() if len(fields) > 15 else ""

                home_name = code_to_name.get(home_code, home_code)
                away_name = code_to_name.get(away_code, away_code)
                tournament_name = tournament_map.get(tournament_code, tournament_code)

                all_matches.append({
                    "date": date,
                    "home_team": home_name,
                    "away_team": away_name,
                    "home_team_code": home_code,
                    "away_team_code": away_code,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "tournament": tournament_name,
                    "tournament_code": tournament_code,
                    "elo_exchanged": elo_exchanged,
                    "home_elo_after": home_elo_after,
                    "away_elo_after": away_elo_after,
                    "home_rank": home_rank,
                    "away_rank": away_rank,
                })
            except (ValueError, IndexError):
                continue

        time.sleep(0.5)  # Rate limit

    if all_matches:
        df = pd.DataFrame(all_matches)
        # Remove duplicates (same match appears in both teams' histories)
        df = df.drop_duplicates(subset=["date", "home_team_code", "away_team_code"])
        df = df.sort_values("date").reset_index(drop=True)
        df.to_csv(ELO_DIR / "elo_match_history.csv", index=False)
        date_min = df["date"].min()
        date_max = df["date"].max()
        print(f"  [OK] elo_match_history.csv -- {len(df):,} matches [{date_min} -> {date_max}]")

    if failed_teams:
        print(f"  [WARN] Failed to fetch: {', '.join(failed_teams)}")


# ===========================================================================
# Summary
# ===========================================================================
def print_summary():
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    sections = [
        ("Transfermarkt", TRANSFERMARKT_DIR),
        ("Injuries", INJURIES_DIR),
        ("International Results", INTL_RESULTS_DIR),
        ("FIFA Rankings", FIFA_RANKINGS_DIR),
        ("World Cup History", WORLD_CUP_DIR),
        ("Elo Ratings", ELO_DIR),
    ]

    for section_name, directory in sections:
        print(f"\n  {section_name} ({directory.relative_to(PROJECT_ROOT)}):")
        csv_files = sorted(directory.glob("*.csv"))
        if not csv_files:
            print("    (no CSV files)")
        for csv_file in csv_files:
            size_mb = csv_file.stat().st_size / (1024 * 1024)
            summary = csv_summary(csv_file)
            print(f"    {csv_file.name:40s} {size_mb:7.1f} MB -- {summary}")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    print("KickCast -- Data Download Script")
    print(f"Project root: {PROJECT_ROOT}")

    download_transfermarkt()
    download_injuries()
    download_international_results()
    download_fifa_rankings()
    download_world_cup_history()
    download_elo_ratings()
    print_summary()

    print("\nDone.")
