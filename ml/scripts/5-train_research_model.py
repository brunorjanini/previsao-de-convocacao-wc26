# -*- coding: utf-8 -*-
"""
Treina e compara 3 modelos sobre dataset_final.csv (Ramificação 1 — Modelo de Pesquisa).

Saídas:
  data/processed/model_comparison.csv   — métricas por modelo
  data/processed/confusion_matrix.png   — matriz do melhor modelo (por F1)
  ml/models/research_model.joblib       — melhor modelo serializado
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

INPUT        = "../../data/processed/dataset_final.csv"
OUT_CSV      = "../../data/processed/model_comparison.csv"
OUT_CM_PNG   = "../../data/processed/confusion_matrix.png"
OUT_MODEL    = "../../ml/models/research_model.joblib"
RANDOM_STATE = 42


MODELS = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        random_state=RANDOM_STATE,
    ),
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}


def evaluate(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "model":     name,
        "f1":        round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1_macro":  round(f1_score(y_test, y_pred, average="macro"), 4),
    }, y_pred


def main():
    df = pd.read_csv(INPUT)
    print(f"Dataset: {df.shape[0]} linhas × {df.shape[1]} colunas")
    print(f"Convocados: {int(df['is_convocated'].sum())} / {len(df)}\n")

    X = df.drop(columns=["is_convocated"])
    y = df["is_convocated"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    results = []
    preds   = {}
    for name, model in MODELS.items():
        print(f"Treinando {name}...")
        row, y_pred = evaluate(name, model, X_train, X_test, y_train, y_test)
        results.append(row)
        preds[name] = (model, y_pred)
        print(classification_report(y_test, y_pred, target_names=["não convocado", "convocado"]))

    # Comparação
    df_cmp = pd.DataFrame(results).sort_values("f1", ascending=False)
    print("\n=== Comparação de modelos ===")
    print(df_cmp.to_string(index=False))

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df_cmp.to_csv(OUT_CSV, index=False)
    print(f"\nMétricas salvas em: {OUT_CSV}")

    # Melhor modelo
    best_name = df_cmp.iloc[0]["model"]
    best_model, best_pred = preds[best_name]
    print(f"\nMelhor modelo: {best_name} (F1 = {df_cmp.iloc[0]['f1']})")

    # Matriz de confusão
    cm = confusion_matrix(y_test, best_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["não convocado", "convocado"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Matriz de Confusão — {best_name}")
    plt.tight_layout()
    fig.savefig(OUT_CM_PNG, dpi=150)
    print(f"Matriz de confusão salva em: {OUT_CM_PNG}")

    # Serializa melhor modelo
    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
    joblib.dump(best_model, OUT_MODEL)
    print(f"Modelo salvo em: {OUT_MODEL}")


if __name__ == "__main__":
    main()
