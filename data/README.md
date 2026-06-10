# Data

Raw and processed datasets are **not redistributed in this repository**. Several sources
are Kaggle-hosted or scraped under terms that do not clearly allow re-hosting, so the
repo ships the *pipeline*, not the data. Run the three scripts below to rebuild
everything locally (Kaggle API credentials required: place `kaggle.json` in `~/.kaggle/`).

```bash
python data/scripts/01_download_data.py   # downloads all raw sources -> data/raw/
python data/scripts/02_build_features.py  # builds data/processed/feature_matrix.csv (21,371 x 38)
python data/scripts/03_create_splits.py   # chronological train/val/test splits
```

## Sources

| Source | What it provides | Where it lands |
|---|---|---|
| [International results (Kaggle, martj42)](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) | ~50k international matches, 1872-present | `data/raw/international_results/` |
| [eloratings.net](https://www.eloratings.net/) | Live Elo ratings + match-by-match history | `data/raw/elo/` |
| [FIFA rankings (Kaggle, cashncarry)](https://www.kaggle.com/datasets/cashncarry/fifaworldranking) | Historical FIFA world rankings | `data/raw/fifa_rankings/` |
| [Transfermarkt player scores (Kaggle, davidcariboo)](https://www.kaggle.com/datasets/davidcariboo/player-scores) | Player market values, appearances, managers | `data/raw/transfermarkt/` |
| [Injury histories (salimt/football-datasets)](https://github.com/salimt/football-datasets) | Historical player injury data | `data/raw/injuries/` |
| [World Cup history (Kaggle, piterfm)](https://www.kaggle.com/datasets/piterfm/fifa-football-world-cup) | All World Cup matches 1930-2022 | `data/raw/world_cup/` |

Each source keeps its own license/terms; consult them before any use beyond reproducing
this project.

## What IS checked in

`data/raw/world_cup_2026/` only: the 2026 tournament structure we compiled ourselves
(groups, fixtures, knockout bracket, venues as small CSVs). The simulation needs it and
it is factual public information, not a licensed dataset.
