# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML project for predicting FIFA World Cup 2026 player call-ups (convocações). The input is the EA FC 26 player dataset (`data/raw/FC26_20250921.csv`) enriched with a binary label `is_convocated` derived from the official FIFA squad lists PDF.

## CRISP-DM Mapping

| Phase | Status | Artifacts |
|---|---|---|
| Business Understanding | ✅ | Crispdm.pdf — predict call-ups, demo at science fair |
| Data Understanding | ✅ | FC26 CSV (18 405 rows, 110 cols) + FIFA PDF (48 × 26) |
| Data Preparation | ✅ | scripts 1–3 → `dataset_final.csv` (76 cols) |
| Modeling | ✅ | script 4 (feature importance), 5 (research), 6 (feira) |
| Evaluation | ✅ | `model_comparison.csv`, `confusion_matrix.png`, `analise_descritiva.png` |
| Deployment | ✅ | FastAPI backend + HTML/JS frontend (FIFA card generator) |

## Virtual Environments

There are **three separate venvs**:

- **Root `.venv/`** — minimal (numpy, pandas); used for non-ML scripts.
- **`ml/.venv/`** — full ML environment; includes pdfplumber, jupyter, ipykernel, pandas, scikit-learn, matplotlib. Always use this one when running `ml/scripts/`.
- **`backend/.venv/`** — FastAPI API server; includes fastapi, uvicorn, pydantic, joblib, scikit-learn, pandas.

```bash
# Activate the ML environment
source ml/.venv/bin/activate

# Activate the backend environment
source backend/.venv/bin/activate
```

## Running Scripts

Scripts use relative paths anchored to `ml/scripts/`, so run them from that directory:

```bash
cd ml/scripts
python 1-convocados_wc2026.py       # PDF → data/interim/convocados_wc2026.csv
python 2-add_is_convocated.py       # joins raw FC26 CSV + convocados → data/interim/fifa_with_convocados.csv
python 3-prepare_dataset.py         # cleans + encodes → data/processed/dataset_final.csv
python 4-feature_importance.py      # RandomForest importance → data/processed/feature_importance.png
python 5-train_research_model.py    # trains 3 models, saves best → ml/models/research_model.joblib
python 6-train_feira_model.py       # trains light model → ml/models/feira_model.joblib + meta.json
python 7-analise_descritiva.py      # descriptive analysis + overfitting diagnosis → data/processed/analise_descritiva.png
python 8-fix_overfitting.py         # regularization + optimal threshold → overwrites models + ml/models/thresholds.json
```

## Running the Backend

```bash
cd backend
source .venv/bin/activate
uvicorn main:app --reload    # serves at http://localhost:8000
```

Endpoints:
- `POST /predict` — JSON with `position`, `pace`, `shooting`, `passing`, `dribbling`, `defending`, `physic`, optional `overall`; returns `{ convocated, probability, overall }`
- `GET /positions` — list of valid positions from the trained model
- `GET /health` — liveness check

**Requires `ml/models/feira_model.joblib` and `ml/models/feira_model_meta.json`** (run script 6 first).

## Running the Frontend

Open `frontend/index.html` directly in a browser (no build step needed). The backend must be running at `http://localhost:8000`.

## Deployment Architecture

```
frontend/index.html  →  POST /predict  →  backend/main.py  →  ml/models/feira_model.joblib
                     ←  { convocated, probability, overall }
```

## Data Pipeline Architecture

```
data/raw/FC26_20250921.csv              ← EA FC 26 player attributes (18 405 rows, 110 cols)
data/raw/SquadLists-English.pdf         ← Official FIFA WC 2026 squad lists (48 teams × 26 players)
         ↓ script 1
data/interim/convocados_wc2026.csv      ← 1 248 called-up players (team, position, dob, club)
         ↓ script 2 (joins with raw CSV)
data/interim/fifa_with_convocados.csv   ← EA FC dataset + is_convocated (914 convocados matched)
         ↓ script 3
data/processed/dataset_final.csv        ← 76 cols, ready for ML (one-hot encoded, imputed)
         ↓ script 4
data/processed/feature_importance.png   ← top-30 feature importance chart (RandomForest)
         ↓ script 5
data/processed/model_comparison.csv     ← F1/Precision/Recall for RF, GB, LR
data/processed/confusion_matrix.png     ← confusion matrix of best research model
ml/models/research_model.joblib         ← best of 3 models by F1 (GradientBoosting)
         ↓ script 6
ml/models/feira_model.joblib            ← light model (19 features: 6 attrs + overall + 12 positions)
ml/models/feira_model_meta.json         ← feature column order + valid positions list
         ↓ script 7
data/processed/analise_descritiva.png   ← 8-panel figure: target dist, boxplots, correlation, learning curves, ROC, PR, prob dist
         ↓ script 8 (overwrites models)
ml/models/research_model.joblib         ← regularized model (gap 0.28 → 0.06)
ml/models/feira_model.joblib            ← regularized model (AUC-PR 0.304 → 0.341)
ml/models/thresholds.json              ← optimal thresholds: research=0.827, feira=0.852
data/processed/overfitting_fix.png     ← before/after comparison chart
```

## Name Matching Logic (script 2)

The `is_convocated` label is assigned via two-step fuzzy matching:
1. Exact normalized name match — uppercase, accent-stripped, **non-ASCII chars removed** (the FC26 CSV appends Arabic/Cyrillic script to some names like `Riyad Mahrezرياض محرز`), against `long_name`/`short_name` vs. `player_name` and `first_names + last_names` from the PDF.
2. Fallback: same `dob` + token overlap (≥2 tokens, or ≥1 if the call-up name has a single word) — catches nicknames and alternate spellings.

The CSV must be loaded with `read_padded_csv()` (defined in script 2), which applies `skipinitialspace=True` and strips column name/value padding — plain `pd.read_csv()` produces a KeyError on `long_name` due to the column padding.

## Dataset Preparation Decisions (script 3)

- **Dropped — data leakage**: `nation_team_id`, `nation_position`, `nation_jersey_number` (directly indicate national team membership).
- **Dropped — >50% nulls**: `work_rate` (100%), `goalkeeping_speed` (89%), `club_loaned_from` (93%), `player_tags` (95%), `player_traits` (59%).
- **Dropped — identifiers**: `player_id`, `player_url`, `player_face_url`, `short_name`, `long_name`, `dob`, `fifa_version`, `fifa_update`, `fifa_update_date`.
- **Dropped — position slot columns**: `ls`, `st`, `rs`, …, `gk` (28 cols).
- **Imputed with 0**: `pace`, `shooting`, `passing`, `dribbling`, `defending`, `physic` (NaN = GKs with no outfield stats), `release_clause_eur`.
- **One-hot encoded**: `preferred_foot`, `primary_position` (first position listed in `player_positions`).
- **Label-encoded**: remaining string columns (`league_name`, `club_name`, etc.).

## Modeling Decisions

**Research Model (script 5 — Ramificação 1):**
- Compares RandomForest, GradientBoosting, LogisticRegression on all 75 features.
- Best: GradientBoosting (F1=0.46, Precision=0.60, Recall=0.38).
- Class imbalance (~19:1) handled with `class_weight="balanced"` or `sample_weight`.

**Feira Model (script 6 → regularized by script 8 — Ramificação 2):**
- GradientBoosting on 19 features only: `pace`, `shooting`, `passing`, `dribbling`, `defending`, `physic`, `overall` + 12 position one-hot cols.
- Optimized for latency and logical coherence (4/4 sanity checks pass).
- `overall` can be provided explicitly or estimated from position-specific weights (approximating FIFA's internal formula per position).
- Regularization (script 8): `learning_rate=0.05`, `subsample=0.8`, `min_samples_leaf=15`. Optimal threshold=0.852 stored in `thresholds.json` and loaded automatically by the backend.

**Overfitting mitigation (script 8):**
- Research model gap (train−test F1): 0.28 → 0.06 after regularization.
- Feira model AUC-PR: 0.304 → 0.341. Backend uses `thresholds.json` for optimal threshold.

## Numbered Script Convention

Scripts in `ml/scripts/` are prefixed with a number indicating pipeline order. Next steps should continue with `7-...`, etc.
