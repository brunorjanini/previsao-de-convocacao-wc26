# -*- coding: utf-8 -*-
"""
Treina um RandomForestClassifier sobre dataset_final.csv e reporta
a importância de cada feature (Gini impurity).

Saída:
  data/processed/feature_importance.png  — gráfico top-30
  tabela completa impressa no stdout
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")   # backend sem display (ambiente headless)
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

INPUT  = "../../data/processed/dataset_final.csv"
OUTPUT_PNG = "../../data/processed/feature_importance.png"

TOP_N = 30
RANDOM_STATE = 42


def main():
    df = pd.read_csv(INPUT)
    print(f"Dataset: {df.shape[0]} linhas × {df.shape[1]} colunas")

    X = df.drop(columns=["is_convocated"])
    y = df["is_convocated"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",   # lida com desbalanceamento (914 vs 17491)
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # Avaliação rápida
    y_pred = rf.predict(X_test)
    print("\nClassification report (test set):")
    print(classification_report(y_test, y_pred, target_names=["não convocado", "convocado"]))

    # Feature importance
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)

    print(f"\nFeature importance completa ({len(importances)} features):")
    print(importances.to_string())

    # Gráfico top-30
    top = importances.head(TOP_N)
    fig, ax = plt.subplots(figsize=(10, 8))
    top[::-1].plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.set_xlabel("Importância (Gini)")
    ax.set_title(f"Top {TOP_N} features — RandomForest · previsão de convocação WC26")
    ax.set_xlim(0, top.iloc[0] * 1.15)
    for i, (val, name) in enumerate(zip(top[::-1], top[::-1].index)):
        ax.text(val + top.iloc[0] * 0.01, i, f"{val:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150)
    print(f"\nGráfico salvo em: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
