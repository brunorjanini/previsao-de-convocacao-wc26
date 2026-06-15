# -*- coding: utf-8 -*-
"""
Script 4-3 — KNN (K-Nearest Neighbors)
Branch Pesquisa: todas as features do dataset_final.csv
Branch Feira: 6 notas principais + posição primária
Desbalanceamento tratado via RandomOverSampler no conjunto de treino.
KNN não possui feature_importances_ — gráfico de importância não é gerado.
"""

import shutil
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
)
from imblearn.over_sampling import RandomOverSampler

# ── Constantes ────────────────────────────────────────────────────────────────
RANDOM_STATE   = 42
TEST_SIZE      = 0.2
N_NEIGHBORS    = 7
ALGO_NAME      = "knn"
ALGO_LABEL     = "KNN"

DATASET_PATH   = Path("../../data/processed/dataset_final.csv")
CHARTS_DIR     = Path("../../data/processed/charts")
MODELS_DIR     = Path("../models")
COMPARISON_CSV = Path("../../data/processed/model_comparison.csv")

FEIRA_STATS  = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
POS_PREFIX   = "primary_position_"

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_branches():
    df = pd.read_csv(DATASET_PATH, skipinitialspace=True)
    # pandas 3.x + Python 3.13: get_dummies salva booleanos como 'True'/'False' no CSV
    for col in df.select_dtypes(include=["object", "str"]).columns:
        if df[col].str.strip().isin(["True", "False"]).all():
            df[col] = df[col].str.strip().map({"True": 1, "False": 0})
    print(f"Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
    print(f"Distribuição is_convocated:\n{df['is_convocated'].value_counts().to_string()}\n")

    X_all = df.drop(columns=["is_convocated"])
    y = df["is_convocated"]
    pos_cols = [c for c in X_all.columns if c.startswith(POS_PREFIX)]
    X_feira = X_all[FEIRA_STATS + pos_cols]
    return X_all, X_feira, y


def train_knn(X_train, y_train):
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_res, y_res = ros.fit_resample(X_train, y_train)
    # Escala é essencial para KNN (distância euclidiana)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=N_NEIGHBORS, metric="euclidean", n_jobs=-1)),
    ])
    pipeline.fit(X_res, y_res)
    return pipeline


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "f1_macro":  round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "cm":        confusion_matrix(y_test, y_pred),
    }


def update_comparison_csv(metrics_p, metrics_f):
    new_rows = pd.DataFrame([
        {"branch": "pesquisa", "model": ALGO_NAME,
         "accuracy": metrics_p["accuracy"], "precision": metrics_p["precision"],
         "recall": metrics_p["recall"], "f1": metrics_p["f1"], "f1_macro": metrics_p["f1_macro"]},
        {"branch": "feira", "model": ALGO_NAME,
         "accuracy": metrics_f["accuracy"], "precision": metrics_f["precision"],
         "recall": metrics_f["recall"], "f1": metrics_f["f1"], "f1_macro": metrics_f["f1_macro"]},
    ])

    if COMPARISON_CSV.exists():
        existing = pd.read_csv(COMPARISON_CSV)
        existing = existing[existing["model"] != ALGO_NAME]
        updated = pd.concat([existing, new_rows], ignore_index=True)
    else:
        updated = new_rows

    updated.to_csv(COMPARISON_CSV, index=False)
    print(f"model_comparison.csv → {len(updated)} linhas totais")
    return updated


def select_and_save_best(comparison_df):
    for branch, prefix in [("pesquisa", "research"), ("feira", "feira")]:
        sub = comparison_df[comparison_df["branch"] == branch]
        if sub.empty:
            continue
        best = sub.loc[sub["f1"].idxmax()]
        src = MODELS_DIR / f"{prefix}_{best['model']}.joblib"
        dst = MODELS_DIR / f"{prefix}_model.joblib"
        if src.exists():
            shutil.copy(src, dst)
            print(f"Melhor {branch}: {best['model']} (F1={best['f1']:.4f}) → {dst.name}")

# ── Gráficos ──────────────────────────────────────────────────────────────────
METRIC_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def plot_metrics(metrics_p, metrics_f):
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    metric_names = ["precision", "recall", "f1", "f1_macro"]
    x = np.arange(len(metric_names))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{ALGO_LABEL} — Métricas por Branch", fontsize=14, fontweight="bold")

    for ax, metrics, title in [
        (axes[0], metrics_p, "Pesquisa (todas as features)"),
        (axes[1], metrics_f, "Feira (6 notas + posição)"),
    ]:
        vals = [metrics[m] for m in metric_names]
        bars = ax.bar(x, vals, width=0.5, color=METRIC_COLORS)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(["Precision", "Recall", "F1", "F1-Macro"], rotation=15)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out = CHARTS_DIR / f"{ALGO_NAME}_metrics.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {out}")


def plot_confusion_matrices(metrics_p, metrics_f):
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"{ALGO_LABEL} — Matrizes de Confusão", fontsize=14, fontweight="bold")

    for ax, metrics, title in [
        (axes[0], metrics_p, f"Pesquisa  |  F1 = {metrics_p['f1']:.3f}"),
        (axes[1], metrics_f, f"Feira  |  F1 = {metrics_f['f1']:.3f}"),
    ]:
        disp = ConfusionMatrixDisplay(
            confusion_matrix=metrics["cm"],
            display_labels=["Não conv.", "Convocado"],
        )
        disp.plot(ax=ax, colorbar=False, cmap="Greens")
        ax.set_title(title, fontsize=11)

    plt.tight_layout()
    out = CHARTS_DIR / f"{ALGO_NAME}_confusion_matrices.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {out}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print(f"  {ALGO_LABEL}  (k={N_NEIGHBORS}, oversampling=RandomOverSampler)")
    print(f"{'='*60}\n")

    X_pesquisa, X_feira, y = load_branches()
    print(f"Features Pesquisa: {X_pesquisa.shape[1]}  |  Features Feira: {X_feira.shape[1]}\n")

    Xp_tr, Xp_te, yp_tr, yp_te = train_test_split(
        X_pesquisa, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    Xf_tr, Xf_te, yf_tr, yf_te = train_test_split(
        X_feira, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    print(f"Treino Pesquisa: {len(yp_tr)} amostras → oversampling para balancear")
    print("Treinando Branch Pesquisa...")
    model_p = train_knn(Xp_tr, yp_tr)

    print(f"Treino Feira: {len(yf_tr)} amostras → oversampling para balancear")
    print("Treinando Branch Feira...")
    model_f = train_knn(Xf_tr, yf_tr)

    metrics_p = evaluate(model_p, Xp_te, yp_te)
    metrics_f = evaluate(model_f, Xf_te, yf_te)

    print(f"\nPesquisa → F1={metrics_p['f1']:.4f} | P={metrics_p['precision']:.4f} | R={metrics_p['recall']:.4f} | F1-mac={metrics_p['f1_macro']:.4f}")
    print(f"Feira    → F1={metrics_f['f1']:.4f} | P={metrics_f['precision']:.4f} | R={metrics_f['recall']:.4f} | F1-mac={metrics_f['f1_macro']:.4f}\n")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_p, MODELS_DIR / f"research_{ALGO_NAME}.joblib")
    joblib.dump(model_f, MODELS_DIR / f"feira_{ALGO_NAME}.joblib")
    print(f"Modelos salvos: research_{ALGO_NAME}.joblib, feira_{ALGO_NAME}.joblib\n")

    plot_metrics(metrics_p, metrics_f)
    plot_confusion_matrices(metrics_p, metrics_f)
    print("(KNN não possui feature_importances_ — gráfico de importância não gerado)")

    comparison_df = update_comparison_csv(metrics_p, metrics_f)
    select_and_save_best(comparison_df)

    print(f"\n{'='*60}")
    print("Concluído — KNN")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
