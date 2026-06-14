# -*- coding: utf-8 -*-
"""
Treina o Modelo da Feira (Ramificação 2 do CRISP-DM).

Usa apenas os 6 atributos principais (pace, shooting, passing, dribbling,
defending, physic) + overall + posição primária (one-hot).

Saídas:
  ml/models/feira_model.joblib      — modelo serializado
  ml/models/feira_model_meta.json   — colunas de entrada e posições válidas
"""

import os
import json
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report

INPUT       = "../../data/processed/dataset_final.csv"
OUT_MODEL   = "../../ml/models/feira_model.joblib"
OUT_META    = "../../ml/models/feira_model_meta.json"
RANDOM_STATE = 42

BASE_ATTRS   = ["pace", "shooting", "passing", "dribbling", "defending", "physic", "overall"]
POSITION_PREFIX = "primary_position_"


def compute_overall(attrs: dict) -> int:
    """Calcula overall aproximado como média ponderada dos 6 atributos."""
    weights = dict(pace=1, shooting=1, passing=1, dribbling=1, defending=1, physic=1)
    total = sum(attrs.get(k, 0) * w for k, w in weights.items())
    return int(total / sum(weights.values()))


def build_input(attrs: dict, position: str, feature_cols: list) -> np.ndarray:
    """Monta vetor de features para predição a partir de atributos e posição."""
    row = {col: 0 for col in feature_cols}
    for k, v in attrs.items():
        if k in row:
            row[k] = v
    pos_col = POSITION_PREFIX + position.upper()
    if pos_col in row:
        row[pos_col] = 1
    return np.array([[row[c] for c in feature_cols]])


def sanity_check(model, feature_cols: list):
    """Testa coerência lógica do modelo com jogadores fictícios."""
    cases = [
        {
            "name": "Craque óbvio (ST)",
            "attrs": {"pace": 95, "shooting": 92, "passing": 80, "dribbling": 90, "defending": 40, "physic": 85, "overall": 92},
            "position": "ST",
            "expected": 1,
        },
        {
            "name": "Zagueiro elite (CB)",
            "attrs": {"pace": 75, "shooting": 35, "passing": 70, "dribbling": 60, "defending": 93, "physic": 88, "overall": 88},
            "position": "CB",
            "expected": 1,
        },
        {
            "name": "Atacante fraco (ST)",
            "attrs": {"pace": 50, "shooting": 40, "passing": 45, "dribbling": 48, "defending": 20, "physic": 55, "overall": 48},
            "position": "ST",
            "expected": 0,
        },
        {
            "name": "Jogador mediano (CM)",
            "attrs": {"pace": 65, "shooting": 55, "passing": 68, "dribbling": 64, "defending": 58, "physic": 62, "overall": 66},
            "position": "CM",
            "expected": 0,
        },
    ]

    print("\n=== Testes de sanidade ===")
    ok = 0
    for case in cases:
        x = build_input(case["attrs"], case["position"], feature_cols)
        pred = model.predict(x)[0]
        prob = model.predict_proba(x)[0][1]
        status = "OK" if pred == case["expected"] else "FALHOU"
        if pred == case["expected"]:
            ok += 1
        print(f"  [{status}] {case['name']}: pred={pred}, prob={prob:.2f}, esperado={case['expected']}")
    print(f"Sanidade: {ok}/{len(cases)} corretos")


def main():
    df = pd.read_csv(INPUT)

    pos_cols = [c for c in df.columns if c.startswith(POSITION_PREFIX)]
    feature_cols = BASE_ATTRS + pos_cols
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df["is_convocated"]

    print(f"Modelo da Feira — {len(feature_cols)} features: {feature_cols}")
    print(f"Convocados: {int(y.sum())} / {len(y)}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # GradientBoosting sem class_weight nativo → usa sample_weight
    classes, counts = np.unique(y_train, return_counts=True)
    weight_map = {c: len(y_train) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    sample_weights = y_train.map(weight_map).values

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)

    y_pred = model.predict(X_test)
    print("Classification report (test set):")
    print(classification_report(y_test, y_pred, target_names=["não convocado", "convocado"]))

    print(f"F1:        {f1_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")

    sanity_check(model, feature_cols)

    positions = [c.replace(POSITION_PREFIX, "") for c in pos_cols]
    meta = {
        "feature_cols":     feature_cols,
        "base_attrs":       BASE_ATTRS,
        "position_prefix":  POSITION_PREFIX,
        "positions":        positions,
    }

    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
    joblib.dump(model, OUT_MODEL)
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nModelo salvo em: {OUT_MODEL}")
    print(f"Meta salvo em:   {OUT_META}")
    print(f"Posições válidas: {positions}")


if __name__ == "__main__":
    main()
