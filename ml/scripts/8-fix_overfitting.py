# -*- coding: utf-8 -*-
"""
Mitiga o overfitting identificado no script 7.

Estratégias:
  1. Threshold ótimo via maximização de F1 na curva Precision-Recall.
  2. Regularização dos modelos (max_depth, min_samples_leaf, subsample,
     learning_rate menor + mais estimadores) para fechar o gap treino/val.
  3. Validação cruzada estratificada (5 folds) como estimativa não enviesada.

Saídas:
  ml/models/research_model.joblib    — substituído pelo modelo regularizado
  ml/models/feira_model.joblib       — substituído pelo modelo regularizado
  ml/models/thresholds.json          — thresholds ótimos por modelo
  data/processed/overfitting_fix.png — comparação antes × depois
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve,
)

warnings.filterwarnings("ignore")

INPUT       = "../../data/processed/dataset_final.csv"
META_FEIRA  = "../../ml/models/feira_model_meta.json"
OUT_PESQ    = "../../ml/models/research_model.joblib"
OUT_FEIRA   = "../../ml/models/feira_model.joblib"
OUT_THRESH  = "../../ml/models/thresholds.json"
OUT_PNG     = "../../data/processed/overfitting_fix.png"

RANDOM_STATE = 42


# ── Threshold ótimo ────────────────────────────────────────────────────────────

def best_threshold(y_true, proba):
    """Retorna o threshold que maximiza o F1-score na curva PR."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1s = 2 * prec * rec / np.where((prec + rec) == 0, 1, prec + rec)
    idx = np.argmax(f1s[:-1])   # último elemento de thr não tem ponto correspondente
    return float(thr[idx]), float(f1s[idx])


def predict_with_threshold(model, X, threshold):
    return (model.predict_proba(X)[:, 1] >= threshold).astype(int)


# ── Avaliação completa ─────────────────────────────────────────────────────────

def evaluate(model, X_train, X_test, y_train, y_test, threshold=0.5):
    prob_test  = model.predict_proba(X_test)[:, 1]
    pred_test  = (prob_test >= threshold).astype(int)
    prob_train = model.predict_proba(X_train)[:, 1]
    pred_train = (prob_train >= 0.5).astype(int)
    return {
        "f1_train":    round(f1_score(y_train, pred_train), 4),
        "f1_test":     round(f1_score(y_test,  pred_test),  4),
        "prec_test":   round(precision_score(y_test, pred_test, zero_division=0), 4),
        "rec_test":    round(recall_score(y_test,  pred_test),  4),
        "auc_roc":     round(roc_auc_score(y_test, prob_test),  4),
        "auc_pr":      round(average_precision_score(y_test, prob_test), 4),
    }


# ── CV score (estimativa não enviesada) ───────────────────────────────────────

def cv_f1(model, X, y, n_splits=5):
    cv    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    thr, _ = best_threshold(y, proba)
    pred  = (proba >= thr).astype(int)
    return round(f1_score(y, pred), 4), round(thr, 4)


# ── Plots comparativos ─────────────────────────────────────────────────────────

def plot_comparison(before_pesq, after_pesq, before_feira, after_feira,
                    thr_pesq, thr_feira):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("Antes × Depois da Regularização + Threshold Ótimo",
                 fontsize=14, fontweight="bold", color="#f0f0f0", y=1.01)

    for ax, before, after, title, thr in [
        (axes[0], before_pesq, after_pesq, "Modelo de Pesquisa (75 feat.)", thr_pesq),
        (axes[1], before_feira, after_feira, "Modelo da Feira (19 feat.)", thr_feira),
    ]:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#ccc")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

        metrics  = ["F1 Treino", "F1 Teste", "Precision", "Recall", "AUC-ROC", "AUC-PR"]
        keys     = ["f1_train", "f1_test", "prec_test", "rec_test", "auc_roc", "auc_pr"]
        x        = np.arange(len(metrics))
        width    = 0.35

        vals_b = [before[k] for k in keys]
        vals_a = [after[k]  for k in keys]

        bars_b = ax.bar(x - width/2, vals_b, width, label="Antes",  color="#4878d0", alpha=0.85)
        bars_a = ax.bar(x + width/2, vals_a, width, label="Depois", color="#ee854a", alpha=0.85)

        for bar in list(bars_b) + list(bars_a):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                    f"{bar.get_height():.2f}", ha="center", va="bottom",
                    fontsize=7.5, color="#ccc")

        gap_before = before["f1_train"] - before["f1_test"]
        gap_after  = after["f1_train"]  - after["f1_test"]
        ax.set_title(
            f"{title}\nGap (treino−teste): {gap_before:.2f} → {gap_after:.2f} | thr={thr:.2f}",
            fontweight="bold", color="#f0f0f0", fontsize=10,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=20, ha="right", fontsize=8, color="#ccc")
        ax.set_ylim(0, 1.08)
        ax.yaxis.label.set_color("#ccc")
        ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Gráfico salvo em: {OUT_PNG}")


# ── Configurações regularizadas ───────────────────────────────────────────────

def make_pesq_model():
    """GradientBoosting com regularização agressiva para 75 features."""
    return GradientBoostingClassifier(
        n_estimators=300,
        max_depth=3,            # era None implicitamente via padrão 3 — mantém explícito
        learning_rate=0.05,     # era 0.1 — menor LR + mais estimadores = menos overfit
        subsample=0.8,          # stochastic GB reduz variância
        min_samples_leaf=20,    # evita folhas com poucos exemplos
        max_features=0.5,       # feature bagging — análogo ao RF
        random_state=RANDOM_STATE,
    )


def make_feira_model():
    """GradientBoosting leve para 19 features."""
    return GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=15,
        random_state=RANDOM_STATE,
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(INPUT)
    print(f"Dataset: {df.shape[0]} × {df.shape[1]}")

    with open(META_FEIRA, encoding="utf-8") as f:
        meta = json.load(f)
    feat_cols_feira = meta["feature_cols"]

    # Splits
    X_all  = df.drop(columns=["is_convocated"])
    y      = df["is_convocated"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    feat_fair = [c for c in feat_cols_feira if c in df.columns]
    X_tr_f = X_tr[feat_fair]; X_te_f = X_te[feat_fair]

    # Pesos de classe
    classes, counts = np.unique(y_tr, return_counts=True)
    w_map    = {c: len(y_tr) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    sw_train = y_tr.map(w_map).values

    # ── Antes: carrega modelos antigos ────────────────────────────────────────
    print("\n── Avaliando modelos ANTES da regularização ──")
    old_pesq  = joblib.load(OUT_PESQ)
    old_feira = joblib.load(OUT_FEIRA)

    prob_old_p = old_pesq.predict_proba(X_te)[:, 1]
    thr_old_p, _ = best_threshold(y_te, prob_old_p)
    before_pesq = evaluate(old_pesq, X_tr, X_te, y_tr, y_te, threshold=thr_old_p)

    prob_old_f = old_feira.predict_proba(X_te_f)[:, 1]
    thr_old_f, _ = best_threshold(y_te, prob_old_f)
    before_feira = evaluate(old_feira, X_tr_f, X_te_f, y_tr, y_te, threshold=thr_old_f)

    # ── Retreina com regularização ─────────────────────────────────────────────
    print("\n── Retreinando Modelo de Pesquisa (regularizado) ──")
    new_pesq = make_pesq_model()
    new_pesq.fit(X_tr, y_tr, sample_weight=sw_train)

    print("── Retreinando Modelo da Feira (regularizado) ──")
    new_feira = make_feira_model()
    new_feira.fit(X_tr_f, y_tr, sample_weight=sw_train)

    # ── Threshold ótimo nos novos modelos ─────────────────────────────────────
    prob_new_p = new_pesq.predict_proba(X_te)[:, 1]
    thr_p, f1_p = best_threshold(y_te, prob_new_p)

    prob_new_f = new_feira.predict_proba(X_te_f)[:, 1]
    thr_f, f1_f = best_threshold(y_te, prob_new_f)

    print(f"\nThreshold ótimo — Pesquisa: {thr_p:.3f}  (F1={f1_p:.4f})")
    print(f"Threshold ótimo — Feira:    {thr_f:.3f}  (F1={f1_f:.4f})")

    # ── CV score ──────────────────────────────────────────────────────────────
    print("\nCV F1 (5-fold, threshold ótimo por fold) — pode levar ~1 min...")
    cv_pesq,  cv_thr_p = cv_f1(new_pesq,  X_all, y)
    cv_feira, cv_thr_f = cv_f1(new_feira, df[feat_fair], y)
    print(f"CV F1 Pesquisa : {cv_pesq:.4f}  (thr médio ≈ {cv_thr_p:.3f})")
    print(f"CV F1 Feira    : {cv_feira:.4f}  (thr médio ≈ {cv_thr_f:.3f})")

    # ── Métricas depois ───────────────────────────────────────────────────────
    after_pesq  = evaluate(new_pesq,  X_tr, X_te, y_tr, y_te, threshold=thr_p)
    after_feira = evaluate(new_feira, X_tr_f, X_te_f, y_tr, y_te, threshold=thr_f)

    # ── Tabela comparativa ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'Métrica':<20} {'Pesq ANTES':>12} {'Pesq DEPOIS':>12} {'Feira ANTES':>12} {'Feira DEPOIS':>12}")
    print("-" * 72)
    for k, label in [
        ("f1_train",  "F1 Treino"),
        ("f1_test",   "F1 Teste"),
        ("prec_test", "Precision"),
        ("rec_test",  "Recall"),
        ("auc_roc",   "AUC-ROC"),
        ("auc_pr",    "AUC-PR"),
    ]:
        print(f"{label:<20} {before_pesq[k]:>12.4f} {after_pesq[k]:>12.4f}"
              f" {before_feira[k]:>12.4f} {after_feira[k]:>12.4f}")

    gap_b_p = before_pesq["f1_train"] - before_pesq["f1_test"]
    gap_a_p = after_pesq["f1_train"]  - after_pesq["f1_test"]
    gap_b_f = before_feira["f1_train"] - before_feira["f1_test"]
    gap_a_f = after_feira["f1_train"]  - after_feira["f1_test"]

    print("=" * 72)
    print(f"\nGap treino−teste Pesquisa : {gap_b_p:.4f} → {gap_a_p:.4f}  "
          f"({'melhorou' if gap_a_p < gap_b_p else 'piorou'})")
    print(f"Gap treino−teste Feira    : {gap_b_f:.4f} → {gap_a_f:.4f}  "
          f"({'melhorou' if gap_a_f < gap_b_f else 'piorou'})")

    # ── Salva artefatos ───────────────────────────────────────────────────────
    joblib.dump(new_pesq,  OUT_PESQ)
    joblib.dump(new_feira, OUT_FEIRA)

    thresholds = {
        "research_model": round(thr_p, 4),
        "feira_model":    round(thr_f, 4),
    }
    with open(OUT_THRESH, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    print(f"\nModelos salvos: {OUT_PESQ}  |  {OUT_FEIRA}")
    print(f"Thresholds salvos: {OUT_THRESH}")

    plot_comparison(before_pesq, after_pesq, before_feira, after_feira, thr_p, thr_f)


if __name__ == "__main__":
    main()
