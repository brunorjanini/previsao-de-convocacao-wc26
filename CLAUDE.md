# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Academic AI/ML project (USP discipline) that predicts which football players will be called up for the FIFA World Cup 2026. The model uses EA FC 26 player statistics as features and the official FIFA WC 2026 squad lists as ground truth labels.

The project follows the **CRISP-DM** methodology and is structured around two model branches:
- **Modelo de Pesquisa** — uses all features for deep statistical analysis (F1-Score, Confusion Matrix, Recall).
- **Modelo da Feira** — uses only the 6 main card stats + position for fast, interactive predictions at a live presentation.

**Current status:** Data science pipeline (Entrega 1) is complete. All 3 algorithms trained and evaluated for both branches. Next phase is the backend/frontend for Entrega 2.

## Environment Setup

The Python virtual environment lives at `ml/.venv` (Python 3.13).

```bash
# Activate the venv
source ml/.venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# Additional ML deps in venv: scikit-learn, xgboost, imbalanced-learn, matplotlib, pdfplumber, unidecode
```

> **Note:** `requirements.txt` currently only pins `numpy` and `pandas`. The remaining ML dependencies are installed in the venv but not pinned. TODO: freeze the full venv into `requirements.txt`.

## Running the Pipeline

Scripts must be run from within the `ml/scripts/` directory (relative paths are hardcoded to `../../data/`).

```bash
cd ml/scripts

# Step 1 — Extract WC2026 squad lists from PDF → data/interim/convocados_wc2026.csv
python 1-convocados_wc2026.py

# Step 2 — Match FIFA players against squad lists → data/interim/fifa_with_convocados.csv
python 2-add_is_convocated.py

# Step 3 — Clean & encode features → data/processed/dataset_final.csv
python 3-prepare_dataset.py

# Step 4 — Train each algorithm (both branches) → ml/models/*.joblib
python 4-1-decision_tree_model.py
python 4-2-xgboost_model.py
python 4-3-knn_model.py

# Step 5 — Consolidated comparison chart → data/processed/charts/comparison_all_models.png
python 5-compare_models.py
```

## Architecture

### CRISP-DM Data Preparation Structure

```
                    TRONCO COMUM (script 3)
                    ┌─────────────────────────┐
                    │ • Remove identifiers     │
                    │ • Remove leakage cols    │
                    │ • Impute nulls           │
                    │ • Encode categoricals    │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              ▼                                     ▼
   RAMIFICAÇÃO 1 — Pesquisa            RAMIFICAÇÃO 2 — Feira
   Todas as features (~70 cols)        Apenas 6 notas principais
                                       + posição primária + alvo
              │                                     │
              ▼                                     ▼
   research_model.joblib               feira_model.joblib
   (Modelo de Pesquisa)                (Modelo da Feira)
```

### Data Flow

```
data/raw/FC26_20250921.csv             ← EA FC 26 player stats (~18k players)
data/raw/SquadLists-English.pdf        ← FIFA official WC2026 squads (48 teams × 26 players)
        │
        ▼ script 1
data/interim/convocados_wc2026.csv     ← 1248 called-up players
        │
        ▼ script 2
data/interim/fifa_with_convocados.csv  ← FC26 stats + is_convocated column
        │
        ▼ script 3
data/processed/dataset_final.csv       ← ML-ready: cleaned, imputed, one-hot encoded
        │
        ▼ scripts 4-1 / 4-2 / 4-3
ml/models/research_{dt,xgb,knn}.joblib ← per-algorithm Pesquisa models
ml/models/feira_{dt,xgb,knn}.joblib    ← per-algorithm Feira models
ml/models/research_model.joblib        ← best Pesquisa model (XGBoost)
ml/models/feira_model.joblib           ← best Feira model (XGBoost)
data/processed/model_comparison.csv    ← metrics for all 6 trained models
        │
        ▼ script 5
data/processed/charts/comparison_all_models.png  ← consolidated comparison chart
data/processed/charts/{dt,xgb,knn}_*.png         ← per-algorithm charts
```

### Current Model State

All three algorithms trained and evaluated for both branches. Best model in each branch is XGBoost, saved as the canonical `*_model.joblib`.

| Branch   | Model         | F1     | Precision | Recall | F1-Macro |
|----------|---------------|--------|-----------|--------|----------|
| pesquisa | Decision Tree | 0.1872 | 0.1832    | 0.1913 | 0.5718   |
| pesquisa | **XGBoost**   | **0.3536** | **0.3765** | **0.3333** | **0.6609** |
| pesquisa | KNN           | 0.2763 | 0.1875    | 0.5246 | 0.6004   |
| feira    | Decision Tree | 0.1961 | 0.2439    | 0.1639 | 0.5806   |
| feira    | **XGBoost**   | **0.2870** | **0.2490** | **0.3388** | **0.6213** |
| feira    | KNN           | 0.2199 | 0.1518    | 0.3989 | 0.5713   |

### Key Design Decisions

**Name matching (script 2):** Two-stage fuzzy match — (1) exact normalized name match (strips accents, non-ASCII chars, punctuation), (2) DOB fallback with token overlap ≥ 2. This handles players with artistic names, transliterations (Arabic/Cyrillic), and name-order differences.

**Feature engineering (script 3):**
- Drops identifier columns, leakage columns (`nation_team_id`, `nation_position`, `nation_jersey_number`), and columns with >50% nulls.
- Drops `league_id`, `league_name`, `club_team_id`, `club_name` — redundant with `league_level` or high-cardinality identifiers with no genuine predictive value.
- GK outfield stats (`pace`, `shooting`, etc.) are imputed with 0 (semantically correct in FC26 context).
- `release_clause_eur` NaN → 0 (no clause).
- `player_positions` → `primary_position` (first listed position only).
- One-hot encodes `preferred_foot` and `primary_position`; label-encodes remaining categoricals.

**Class imbalance:** ~914 convocated vs ~17,491 not-convocated (~19:1 ratio).
- Decision Tree: `class_weight="balanced"`
- XGBoost: `scale_pos_weight = n_neg / n_pos` (≈19.14)
- KNN: `RandomOverSampler` on training set only; test set evaluated on original distribution

**Known CSV quirk (pandas 3.x + Python 3.13):** `pd.get_dummies()` saves boolean dtype columns as `'True'`/`'False'` strings in CSV. All scripts 4-x handle this with a post-load conversion loop using `str.strip().map({"True": 1, "False": 0})`.

**Feira model feature set:** `pace`, `shooting`, `passing`, `dribbling`, `defending`, `physic` + all `primary_position_*` one-hot columns. Scripts 4-x filter `dataset_final.csv` to these columns for the Feira branch.

**Saved models:** Each script 4-x saves its own per-algorithm `.joblib` files and updates `model_comparison.csv`. After each run, it reads the full CSV and copies the best F1 model to `research_model.joblib` / `feira_model.joblib`. Running all three scripts in order always leaves the overall best in the canonical files.

### Directories

- `ml/scripts/` — numbered pipeline scripts (run in order)
- `ml/models/` — serialized trained models (`.joblib`): 6 per-algorithm + 2 canonical best + `feira_model_meta.json`
- `data/raw/` — original source files (do not modify)
- `data/interim/` — intermediate CSVs produced by scripts 1–2
- `data/processed/` — final ML-ready dataset, `model_comparison.csv`, and `charts/` directory
- `backend/` — FastAPI app serving `feira_model.joblib` (`main.py` + `requirements.txt`)
- `frontend/` — React + Vite + Tailwind CSS app; FIFA card generator UI for live audience interaction

### Frontend

The frontend is a React 19 + TypeScript + Vite + Tailwind CSS v4 app at `frontend/`. It calls the backend API hardcoded at `http://localhost:8000` (see `frontend/src/api.ts`).

**Prerequisites:** Node.js with npm.

**Running in development:**
```bash
cd frontend
npm install        # first time only
npm run dev        # → http://localhost:5173
```

**Building for production:**
```bash
cd frontend
npm run build      # outputs to frontend/dist/
npm run preview    # preview the production build locally
```

**Key components:**
- `PlayerForm` — collects position + 6 stat sliders + player name + optional webcam photo
- `FootballField` — visualizes the selected position on a pitch diagram
- `PaniniCard` — renders the FIFA-style prediction card (convocated/not) with the player photo
- `SendCardModal` — sends the generated card image by email via the backend `/send-card` endpoint
- `WebcamCapture` — captures a photo from the user's webcam to use as the card photo

The app requires the **backend running at port 8000** before it can make predictions.

### Backend API

The backend is a FastAPI app at `backend/main.py`. It loads `feira_model.joblib` at startup.

**Required files at startup:**
- `ml/models/feira_model.joblib` — trained Feira model (XGBoost)
- `ml/models/feira_model_meta.json` — feature columns and valid positions metadata
- `ml/models/thresholds.json` — optional; overrides the default 0.5 classification threshold

**Endpoints:**
- `POST /predict` — receives player attributes, returns `{ convocated, probability, overall }`
- `GET /positions` — lists valid positions the model accepts
- `GET /health` — sanity check

**Running the API:**
```bash
# Install backend deps (separate from the ML venv)
pip install -r backend/requirements.txt

# Start the server (from repo root)
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

**Example request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"position":"ST","pace":90,"shooting":85,"passing":70,"dribbling":88,"defending":30,"physic":75}'
# → {"convocated":true,"probability":0.9673,"overall":83}
```

**`overall` field:** If omitted in the request, the API estimates it using position-specific attribute weights (same logic EA FC 26 uses). If provided, uses the value directly. Returned in the response for display purposes only — not used as a model feature.

### Deployment Plan

- **Entrega 1 ✅ Complete:** 3-algorithm pipeline (DT, XGBoost, KNN) trained for both branches; feature importance and comparison charts generated; best model (XGBoost) saved for each branch.
- **Entrega 2 — Backend ✅ Complete:** FastAPI serving `feira_model.joblib` with `/predict`, `/positions`, and `/health` endpoints.
- **Entrega 2 — Frontend ✅ Complete:** React + Vite app with FIFA card generator UI; webcam photo capture, stat sliders, football field position display, and email delivery of the generated card.

**To run the full stack locally:**
1. Start the backend: `uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000` (from repo root, with ML venv active)
2. Start the frontend: `cd frontend && npm install && npm run dev`
3. Open `http://localhost:5173` in the browser.
