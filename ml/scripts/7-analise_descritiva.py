# -*- coding: utf-8 -*-
"""
Análise descritiva do dataset_final.csv + diagnóstico de overfitting.

Saída:
  data/processed/analise_descritiva.png  — figura multi-painel (8 subplots)
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
)

warnings.filterwarnings("ignore")

# ── Caminhos ───────────────────────────────────────────────────────────────────
INPUT        = "../../data/processed/dataset_final.csv"
MODEL_PESQ   = "../../ml/models/research_model.joblib"
MODEL_FEIRA  = "../../ml/models/feira_model.joblib"
META_FEIRA   = "../../ml/models/feira_model_meta.json"
OUTPUT_PNG   = "../../data/processed/analise_descritiva.png"

RANDOM_STATE = 42
MAIN_ATTRS   = ["pace", "shooting", "passing", "dribbling", "defending", "physic", "overall"]

sns.set_theme(style="darkgrid", palette="muted")


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(INPUT)
    print(f"Dataset: {df.shape[0]} linhas × {df.shape[1]} colunas")
    print(f"Convocados: {int(df['is_convocated'].sum())} / {len(df)}")
    return df


def split(df, feature_cols=None):
    X = df[feature_cols] if feature_cols else df.drop(columns=["is_convocated"])
    y = df["is_convocated"]
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


def diagnose(f1_train, f1_val, gap_thresh=0.15, low_thresh=0.35):
    gap = f1_train - f1_val
    if gap > gap_thresh and f1_val < 0.55:
        return "overfitting"
    if f1_train < low_thresh and f1_val < low_thresh:
        return "underfitting"
    return "ajuste razoável"


# ── 1. Distribuição da variável alvo ──────────────────────────────────────────

def plot_target_dist(ax, df):
    counts = df["is_convocated"].value_counts().sort_index()
    labels = ["Não convocado (0)", "Convocado (1)"]
    colors = ["#4878d0", "#ee854a"]
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
                f"{val:,}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_title("1. Distribuição da variável alvo", fontweight="bold")
    ax.set_ylabel("Contagem de jogadores")
    ratio = counts[0] / counts[1]
    ax.text(0.97, 0.95, f"Razão: {ratio:.0f}:1", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="#888")
    ax.set_ylim(0, counts.max() * 1.15)


# ── 2. Atributos principais por classe ────────────────────────────────────────

def plot_attrs_by_class(ax, df):
    attrs_present = [c for c in MAIN_ATTRS if c in df.columns]
    melted = df[attrs_present + ["is_convocated"]].melt(
        id_vars="is_convocated", var_name="Atributo", value_name="Valor"
    )
    melted["Classe"] = melted["is_convocated"].map({0: "Não conv.", 1: "Convocado"})
    sns.boxplot(data=melted, x="Atributo", y="Valor", hue="Classe",
                ax=ax, palette={"Não conv.": "#4878d0", "Convocado": "#ee854a"},
                linewidth=0.8, fliersize=2)
    ax.set_title("2. Atributos principais por classe", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Valor do atributo")
    ax.legend(title="", loc="lower right", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)


# ── 3. Correlação com is_convocated ───────────────────────────────────────────

def plot_correlation(ax, df):
    corr = df.corr(numeric_only=True)["is_convocated"].drop("is_convocated")
    top20 = corr.abs().nlargest(20).index
    corr_top = corr[top20].sort_values()
    colors = ["#ee854a" if v < 0 else "#4878d0" for v in corr_top]
    ax.barh(corr_top.index, corr_top.values, color=colors, edgecolor="none")
    ax.axvline(0, color="white", linewidth=0.8, linestyle="--")
    ax.set_title("3. Top-20 correlações com is_convocated", fontweight="bold")
    ax.set_xlabel("Correlação de Pearson")
    ax.tick_params(axis="y", labelsize=7)


# ── 4. Curvas de aprendizado ───────────────────────────────────────────────────

def plot_learning_curves(ax_pesq, ax_feira, df, model_pesq, model_feira, meta_feira):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_sizes_pct = np.linspace(0.10, 1.0, 6)

    # Modelo de pesquisa — todas as features
    X_pesq = df.drop(columns=["is_convocated"])
    y = df["is_convocated"]

    ts_p, tr_p, val_p = learning_curve(
        model_pesq, X_pesq, y,
        train_sizes=train_sizes_pct, cv=cv,
        scoring="f1", n_jobs=-1,
    )
    tr_mean_p  = tr_p.mean(axis=1)
    val_mean_p = val_p.mean(axis=1)
    diag_p = diagnose(tr_mean_p[-1], val_mean_p[-1])

    ax_pesq.plot(ts_p, tr_mean_p,  "o-", label="Treino",    color="#4878d0")
    ax_pesq.fill_between(ts_p, tr_p.min(1), tr_p.max(1), alpha=0.15, color="#4878d0")
    ax_pesq.plot(ts_p, val_mean_p, "s--", label="Validação", color="#ee854a")
    ax_pesq.fill_between(ts_p, val_p.min(1), val_p.max(1), alpha=0.15, color="#ee854a")
    ax_pesq.set_title("4a. Curva de aprendizado — Modelo de Pesquisa (GB, 75 feat.)", fontweight="bold")
    ax_pesq.set_xlabel("Amostras de treino")
    ax_pesq.set_ylabel("F1-score")
    ax_pesq.legend(fontsize=8)
    ax_pesq.text(0.02, 0.06, f"DIAGNÓSTICO: {diag_p.upper()}", transform=ax_pesq.transAxes,
                 fontsize=9, fontweight="bold",
                 color="#27ae60" if diag_p == "ajuste razoável" else "#c0392b")

    # Modelo da feira — 19 features
    feat_cols = meta_feira["feature_cols"]
    X_feira   = df[[c for c in feat_cols if c in df.columns]]

    ts_f, tr_f, val_f = learning_curve(
        model_feira, X_feira, y,
        train_sizes=train_sizes_pct, cv=cv,
        scoring="f1", n_jobs=-1,
    )
    tr_mean_f  = tr_f.mean(axis=1)
    val_mean_f = val_f.mean(axis=1)
    diag_f = diagnose(tr_mean_f[-1], val_mean_f[-1])

    ax_feira.plot(ts_f, tr_mean_f,  "o-", label="Treino",    color="#4878d0")
    ax_feira.fill_between(ts_f, tr_f.min(1), tr_f.max(1), alpha=0.15, color="#4878d0")
    ax_feira.plot(ts_f, val_mean_f, "s--", label="Validação", color="#ee854a")
    ax_feira.fill_between(ts_f, val_f.min(1), val_f.max(1), alpha=0.15, color="#ee854a")
    ax_feira.set_title("4b. Curva de aprendizado — Modelo da Feira (GB, 19 feat.)", fontweight="bold")
    ax_feira.set_xlabel("Amostras de treino")
    ax_feira.set_ylabel("F1-score")
    ax_feira.legend(fontsize=8)
    ax_feira.text(0.02, 0.06, f"DIAGNÓSTICO: {diag_f.upper()}", transform=ax_feira.transAxes,
                  fontsize=9, fontweight="bold",
                  color="#27ae60" if diag_f == "ajuste razoável" else "#c0392b")

    return diag_p, diag_f, tr_mean_p[-1], val_mean_p[-1], tr_mean_f[-1], val_mean_f[-1]


# ── 5. ROC e Precision-Recall ─────────────────────────────────────────────────

def plot_roc_pr(ax_roc, ax_pr, df, model_pesq, model_feira, meta_feira):
    y = df["is_convocated"]

    # Modelo de pesquisa
    X_pesq = df.drop(columns=["is_convocated"])
    _, X_test_p, _, y_test_p = split(df)
    prob_p = model_pesq.predict_proba(X_test_p)[:, 1]

    fpr_p, tpr_p, _ = roc_curve(y_test_p, prob_p)
    auc_p = roc_auc_score(y_test_p, prob_p)
    prec_p, rec_p, _ = precision_recall_curve(y_test_p, prob_p)
    ap_p = average_precision_score(y_test_p, prob_p)

    # Modelo da feira
    feat_cols = meta_feira["feature_cols"]
    X_feira   = df[[c for c in feat_cols if c in df.columns]]
    _, X_test_f, _, y_test_f = split(df, feature_cols=[c for c in feat_cols if c in df.columns])
    prob_f = model_feira.predict_proba(X_test_f)[:, 1]

    fpr_f, tpr_f, _ = roc_curve(y_test_f, prob_f)
    auc_f = roc_auc_score(y_test_f, prob_f)
    prec_f, rec_f, _ = precision_recall_curve(y_test_f, prob_f)
    ap_f = average_precision_score(y_test_f, prob_f)

    # ROC
    ax_roc.plot(fpr_p, tpr_p, label=f"Pesquisa  AUC={auc_p:.3f}", color="#4878d0", linewidth=2)
    ax_roc.plot(fpr_f, tpr_f, label=f"Feira     AUC={auc_f:.3f}", color="#ee854a", linewidth=2, linestyle="--")
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax_roc.set_title("5a. Curva ROC", fontweight="bold")
    ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
    ax_roc.legend(fontsize=8)

    # Precision-Recall
    ax_pr.plot(rec_p, prec_p, label=f"Pesquisa  AP={ap_p:.3f}", color="#4878d0", linewidth=2)
    ax_pr.plot(rec_f, prec_f, label=f"Feira     AP={ap_f:.3f}", color="#ee854a", linewidth=2, linestyle="--")
    baseline = y_test_p.mean()
    ax_pr.axhline(baseline, color="gray", linestyle=":", linewidth=0.9, label=f"Baseline ({baseline:.3f})")
    ax_pr.set_title("5b. Curva Precision-Recall", fontweight="bold")
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.legend(fontsize=8)

    return auc_p, ap_p, auc_f, ap_f, prob_p, y_test_p


# ── 6. Distribuição de probabilidades ─────────────────────────────────────────

def plot_prob_dist(ax, prob_pesq, y_test):
    ax.hist(prob_pesq[y_test == 0], bins=40, alpha=0.65, color="#4878d0",
            label="Não convocado (0)", density=True)
    ax.hist(prob_pesq[y_test == 1], bins=40, alpha=0.75, color="#ee854a",
            label="Convocado (1)", density=True)
    ax.axvline(0.5, color="white", linestyle="--", linewidth=1, label="Threshold=0.5")
    ax.set_title("6. Distribuição dos scores P(convocado) — Modelo de Pesquisa", fontweight="bold")
    ax.set_xlabel("Probabilidade predita")
    ax.set_ylabel("Densidade")
    ax.legend(fontsize=8)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    df          = load_data()
    model_pesq  = joblib.load(MODEL_PESQ)
    model_feira = joblib.load(MODEL_FEIRA)
    with open(META_FEIRA, encoding="utf-8") as f:
        meta_feira = json.load(f)

    fig = plt.figure(figsize=(20, 28))
    fig.patch.set_facecolor("#1a1a2e")

    gs = gridspec.GridSpec(
        4, 4,
        figure=fig,
        hspace=0.48, wspace=0.38,
        top=0.95, bottom=0.04, left=0.06, right=0.97,
    )

    ax_target  = fig.add_subplot(gs[0, 0])
    ax_attrs   = fig.add_subplot(gs[0, 1:3])
    ax_corr    = fig.add_subplot(gs[0, 3])
    ax_lc_pesq = fig.add_subplot(gs[1, :2])
    ax_lc_fair = fig.add_subplot(gs[1, 2:])
    ax_roc     = fig.add_subplot(gs[2, :2])
    ax_pr      = fig.add_subplot(gs[2, 2:])
    ax_prob    = fig.add_subplot(gs[3, :])

    for ax in fig.axes:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#ccc")
        ax.xaxis.label.set_color("#ccc")
        ax.yaxis.label.set_color("#ccc")
        ax.title.set_color("#f0f0f0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    fig.suptitle(
        "Análise Descritiva & Diagnóstico de Overfitting — WC 2026 Convocação",
        fontsize=16, fontweight="bold", color="#f0f0f0", y=0.975,
    )

    plot_target_dist(ax_target, df)
    plot_attrs_by_class(ax_attrs, df)
    plot_correlation(ax_corr, df)

    print("\nCalculando curvas de aprendizado (pode levar ~2 min)...")
    diag_p, diag_f, f1_tr_p, f1_val_p, f1_tr_f, f1_val_f = plot_learning_curves(
        ax_lc_pesq, ax_lc_fair, df, model_pesq, model_feira, meta_feira
    )

    auc_p, ap_p, auc_f, ap_f, prob_pesq, y_test = plot_roc_pr(
        ax_roc, ax_pr, df, model_pesq, model_feira, meta_feira
    )

    plot_prob_dist(ax_prob, prob_pesq, y_test)

    fig.savefig(OUTPUT_PNG, dpi=150, facecolor=fig.get_facecolor())
    print(f"\nFigura salva em: {OUTPUT_PNG}")

    # ── Tabela resumo ────────────────────────────────────────────────────────────
    feat_cols = meta_feira["feature_cols"]
    X_test_f_full = df[[c for c in feat_cols if c in df.columns]]
    _, X_test_f, _, y_test_f = train_test_split(
        X_test_f_full, df["is_convocated"],
        test_size=0.2, random_state=RANDOM_STATE, stratify=df["is_convocated"]
    )
    pred_p = model_pesq.predict(
        df.drop(columns=["is_convocated"]).iloc[len(df) - len(y_test):]
    )
    _, X_test_pf, _, y_test_pf = train_test_split(
        df.drop(columns=["is_convocated"]), df["is_convocated"],
        test_size=0.2, random_state=RANDOM_STATE, stratify=df["is_convocated"]
    )
    f1_test_p  = f1_score(y_test_pf, model_pesq.predict(X_test_pf))
    f1_test_f  = f1_score(y_test_f,  model_feira.predict(X_test_f))

    print("\n" + "=" * 60)
    print(f"{'Modelo':<22} {'F1 Treino':>10} {'F1 Val':>10} {'AUC-ROC':>10} {'AUC-PR':>10}")
    print("-" * 60)
    print(f"{'Pesquisa (75 feat.)':<22} {f1_tr_p:>10.4f} {f1_val_p:>10.4f} {auc_p:>10.4f} {ap_p:>10.4f}")
    print(f"{'Feira (19 feat.)':<22} {f1_tr_f:>10.4f} {f1_val_f:>10.4f} {auc_f:>10.4f} {ap_f:>10.4f}")
    print("=" * 60)
    print(f"\nDIAGNÓSTICO Modelo de Pesquisa : {diag_p.upper()}")
    print(f"DIAGNÓSTICO Modelo da Feira    : {diag_f.upper()}")


if __name__ == "__main__":
    main()
