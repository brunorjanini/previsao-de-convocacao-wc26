# -*- coding: utf-8 -*-
"""
API FastAPI — Previsão de Convocação WC 2026

Endpoints:
  POST /predict   — recebe atributos do jogador, retorna veredito de convocação
  GET  /positions — lista posições válidas para o modelo
"""

import json
import os
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv

import base64
import smtplib
from email.message import EmailMessage

import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR    = Path(__file__).parent
MODELS_DIR  = BASE_DIR.parent / "ml" / "models"
MODEL_PATH  = MODELS_DIR / "feira_model.joblib"
META_PATH   = MODELS_DIR / "feira_model_meta.json"
THRESH_PATH = MODELS_DIR / "thresholds.json"

load_dotenv()

EMAIL_USER     = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

_model     = None
_meta      = None
_threshold = 0.5          # default; substituído pelo ótimo se disponível

class PredictionOutput(BaseModel):
    convocated:  bool
    probability: float
    overall:     int

class SendCardRequest(BaseModel):
    email: str
    image: str
    player_name: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _meta, _threshold
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Modelo não encontrado: {MODEL_PATH}. "
            "Execute os scripts ml/scripts/4-2-xgboost_model.py (e os demais 4-x) primeiro."
        )
    _model = joblib.load(MODEL_PATH)
    with open(META_PATH, encoding="utf-8") as f:
        _meta = json.load(f)
    if THRESH_PATH.exists():
        with open(THRESH_PATH, encoding="utf-8") as f:
            _threshold = json.load(f).get("feira_model", 0.5)
    print(f"Modelo carregado: {MODEL_PATH}")
    print(f"Threshold ótimo:  {_threshold}")
    print(f"Posições válidas: {_meta['positions']}")
    yield


app = FastAPI(
    title="WC26 Convocação API",
    description="Prediz se um jogador seria convocado para a Copa do Mundo 2026.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pesos por posição para estimar o overall como o FIFA faz (simplificado)
_POSITION_WEIGHTS: dict[str, dict[str, float]] = {
    "ST":  {"pace": .25, "shooting": .45, "passing": .05, "dribbling": .15, "defending": .05, "physic": .05},
    "LW":  {"pace": .30, "shooting": .25, "passing": .10, "dribbling": .25, "defending": .05, "physic": .05},
    "RW":  {"pace": .30, "shooting": .25, "passing": .10, "dribbling": .25, "defending": .05, "physic": .05},
    "LM":  {"pace": .25, "shooting": .15, "passing": .20, "dribbling": .20, "defending": .10, "physic": .10},
    "RM":  {"pace": .25, "shooting": .15, "passing": .20, "dribbling": .20, "defending": .10, "physic": .10},
    "CAM": {"pace": .10, "shooting": .20, "passing": .30, "dribbling": .25, "defending": .05, "physic": .10},
    "CM":  {"pace": .10, "shooting": .15, "passing": .30, "dribbling": .20, "defending": .15, "physic": .10},
    "CDM": {"pace": .05, "shooting": .05, "passing": .25, "dribbling": .15, "defending": .35, "physic": .15},
    "LB":  {"pace": .20, "shooting": .05, "passing": .15, "dribbling": .12, "defending": .33, "physic": .15},
    "RB":  {"pace": .20, "shooting": .05, "passing": .15, "dribbling": .12, "defending": .33, "physic": .15},
    "CB":  {"pace": .05, "shooting": .05, "passing": .10, "dribbling": .05, "defending": .55, "physic": .20},
    "GK":  {"pace": .05, "shooting": .05, "passing": .15, "dribbling": .05, "defending": .05, "physic": .20},
}
_DEFAULT_WEIGHTS = {"pace": .17, "shooting": .17, "passing": .17, "dribbling": .17, "defending": .17, "physic": .17}


def _estimate_overall(attrs: dict, position: str) -> int:
    w = _POSITION_WEIGHTS.get(position.upper(), _DEFAULT_WEIGHTS)
    return min(99, int(sum(attrs.get(k, 0) * v for k, v in w.items())))


class PlayerInput(BaseModel):
    position:  str = Field(..., description="Posição primária (ex.: ST, CB, GK)")
    pace:      int = Field(..., ge=0, le=99)
    shooting:  int = Field(..., ge=0, le=99)
    passing:   int = Field(..., ge=0, le=99)
    dribbling: int = Field(..., ge=0, le=99)
    defending: int = Field(..., ge=0, le=99)
    physic:    int = Field(..., ge=0, le=99)
    overall:   int | None = Field(None, ge=0, le=99, description="Se fornecido, usa direto; senão calcula por posição")


class PredictionOutput(BaseModel):
    convocated:  bool
    probability: float
    overall:     int


def build_feature_vector(player: PlayerInput) -> tuple[np.ndarray, int]:
    feature_cols    = _meta["feature_cols"]
    position_prefix = _meta["position_prefix"]

    attrs = {
        "pace":      player.pace,
        "shooting":  player.shooting,
        "passing":   player.passing,
        "dribbling": player.dribbling,
        "defending": player.defending,
        "physic":    player.physic,
    }
    overall = player.overall if player.overall is not None else _estimate_overall(attrs, player.position)
    attrs["overall"] = overall

    row = {col: 0 for col in feature_cols}
    for k, v in attrs.items():
        if k in row:
            row[k] = v
    pos_col = position_prefix + player.position.upper()
    if pos_col in row:
        row[pos_col] = 1

    return pd.DataFrame([row], columns=feature_cols), overall


@app.post("/predict", response_model=PredictionOutput)
def predict(player: PlayerInput):
    valid_positions = _meta["positions"]
    if player.position.upper() not in valid_positions:
        raise HTTPException(
            status_code=422,
            detail=f"Posição inválida '{player.position}'. Válidas: {valid_positions}",
        )

    X, overall = build_feature_vector(player)
    prob       = float(_model.predict_proba(X)[0][1])
    convocated = bool(prob >= _threshold)

    return PredictionOutput(convocated=convocated, probability=round(prob, 4), overall=overall)


@app.get("/positions")
def positions():
    return {"positions": _meta["positions"]}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/send-card")
def send_card(req: SendCardRequest):
    if not EMAIL_USER or not EMAIL_PASSWORD:
        raise HTTPException(
            status_code=500,
            detail="EMAIL_USER ou EMAIL_PASSWORD não configurados no .env."
        )

    try:
        image_bytes = base64.b64decode(req.image.split(",")[1])

        msg = EmailMessage()
        msg["Subject"] = "Sua Carta FIFA World Cup 2026"
        msg["From"]    = EMAIL_USER
        msg["To"]      = req.email
        msg.set_content(
            f"Olá!\n\nSegue em anexo a carta FIFA gerada para {req.player_name}.\n\n"
            "Obrigado por utilizar nosso preditor de convocação da Copa do Mundo 2026."
        )
        msg.add_attachment(
            image_bytes,
            maintype="image",
            subtype="png",
            filename=f"{req.player_name}.png",
        )

        # Tenta SSL na porta 465 primeiro; se falhar, tenta STARTTLS na 587
        smtp_host = "smtp.gmail.com"
        try:
            with smtplib.SMTP_SSL(smtp_host, 465) as smtp:
                smtp.login(EMAIL_USER, EMAIL_PASSWORD)
                smtp.send_message(msg)
        except Exception:
            with smtplib.SMTP(smtp_host, 587) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.login(EMAIL_USER, EMAIL_PASSWORD)
                smtp.send_message(msg)

        return {"success": True, "message": f"Email enviado para {req.email}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao enviar email: {str(e)}")