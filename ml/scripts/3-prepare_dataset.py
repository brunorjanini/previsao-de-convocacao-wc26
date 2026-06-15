# -*- coding: utf-8 -*-
"""
Prepara o dataset final para ML a partir de data/interim/fifa_with_convocados.csv.

- Remove colunas identificadoras, com >50% de nulos e de data leakage
- Remove colunas de posição por slot (ls, st, ..., gk)
- Imputa valores ausentes em colunas numéricas remanescentes
- Codifica variáveis categóricas remanescentes (one-hot)
- Salva em data/processed/dataset_final.csv
"""

import pandas as pd
import numpy as np

INPUT  = "../../data/interim/fifa_with_convocados.csv"
OUTPUT = "../../data/processed/dataset_final.csv"

# ------------------------------------------------------------------
# Colunas a remover explicitamente
# ------------------------------------------------------------------
DROP_IDENTIFIERS = [
    "player_id", "player_url", "player_face_url",
    "fifa_version", "fifa_update", "fifa_update_date",
    "short_name", "long_name",
    "dob", "club_joined_date",
    "league_id", "league_name",
      "club_team_id", "club_name","club_jersey_number",
      "nationality_id", "nationality_name",
      "club_contract_valid_until",
]

# data leakage: indicam participação em seleção nacional no momento do snapshot
DROP_LEAKAGE = [
    "nation_team_id", "nation_position", "nation_jersey_number",
]

# 100 % nulos ou semântica redundante / não preditiva
DROP_OTHER = [
    "work_rate",           # 100 % NaN
    "goalkeeping_speed",   # 88 % NaN
    "club_loaned_from",    # 93 % NaN
    "player_tags",         # 95 % NaN
    "player_traits",       # 59 % NaN
    "real_face",           # binário sem poder preditivo
    "body_type",           # muito ruidoso / inconsistente
]

# Posições por slot
SLOT_POSITIONS = [
    "ls", "st", "rs", "lw", "lf", "cf", "rf", "rw",
    "lam", "cam", "ram", "lm", "lcm", "cm", "rcm", "rm",
    "lwb", "ldm", "cdm", "rdm", "rwb",
    "lb", "lcb", "cb", "rcb", "rb", "gk",
]

ALL_DROP = set(DROP_IDENTIFIERS + DROP_LEAKAGE + DROP_OTHER + SLOT_POSITIONS)

# ------------------------------------------------------------------
def main():
    df = pd.read_csv(INPUT, low_memory=False)
    print(f"Input shape: {df.shape}")

    # 1) Remove colunas explícitas
    to_drop = [c for c in ALL_DROP if c in df.columns]
    df.drop(columns=to_drop, inplace=True)
    print(f"Após remoção de colunas explícitas: {df.shape}")

    # 2) Remove colunas com >50 % de nulos (restante)
    null_pct = df.isnull().mean()
    high_null = null_pct[null_pct > 0.5].index.tolist()
    if high_null:
        print(f"Removendo por >50% nulos: {high_null}")
        df.drop(columns=high_null, inplace=True)

    # 3) Imputa nulos em colunas numéricas
    # pace/shooting/passing/dribbling/defending/physic: NaN = goleiros (sem stats de linha)
    # → imputa com 0 (semanticamente correto no contexto do jogo)
    outfield_stats = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
    for col in outfield_stats:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # release_clause_eur: NaN = sem cláusula → 0
    if "release_clause_eur" in df.columns:
        df["release_clause_eur"] = df["release_clause_eur"].fillna(0)

    # Demais numéricas: imputa com mediana
    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # 4) Trata categóricas remanescentes
    # preferred_foot → one-hot
    # player_positions → top-1 posição (primária)
    # league_name → mantida como string (pode ser label-encoded depois se necessário)
    # nationality_name → mantida

    if "player_positions" in df.columns:
        df["primary_position"] = df["player_positions"].str.split(",").str[0].str.strip()
        df.drop(columns=["player_positions"], inplace=True)

    cat_onehot = ["preferred_foot", "primary_position"]
    existing_cat = [c for c in cat_onehot if c in df.columns]
    df = pd.get_dummies(df, columns=existing_cat, drop_first=False)

    # Converte restantes string/object para category codes (league_name, club_name, etc.)
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].astype("category").cat.codes

    # 5) Garante que is_convocated é a última coluna
    target = df.pop("is_convocated")
    df["is_convocated"] = target

    # 6) Salva
    df.to_csv(OUTPUT, index=False)

    print(f"\nDataset final salvo em: {OUTPUT}")
    print(f"Shape final: {df.shape}")
    print(f"\nDistribuição de is_convocated:")
    print(df["is_convocated"].value_counts().to_string())
    print(f"\nColunas mantidas ({len(df.columns)}):")
    print(", ".join(df.columns.tolist()))


if __name__ == "__main__":
    main()
