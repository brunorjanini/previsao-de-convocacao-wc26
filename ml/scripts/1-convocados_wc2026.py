# -*- coding: utf-8 -*-
"""
Extrai os jogadores convocados do PDF 'SquadLists-English.pdf' (FIFA World Cup 2026)
para um DataFrame e marca, em outro DataFrame, a variável is_convocated (1/0).

Requisitos: pip install pdfplumber pandas unidecode
"""

import re
import logging
import pdfplumber
import pandas as pd
import os

logging.getLogger("pdfminer").setLevel(logging.ERROR)

TEAM_RE = re.compile(r"^(.+?)\s*\(([A-Z]{3})\)$")


def extract_convocados(pdf_path: str) -> pd.DataFrame:
    """Lê o PDF da FIFA e retorna um DataFrame com 1 linha por jogador."""
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Nome e código da seleção (ex.: "Brazil (BRA)") nas primeiras linhas
            team_name, team_code = None, None
            for line in (page.extract_text() or "").split("\n")[:8]:
                m = TEAM_RE.match(line.strip())
                if m:
                    team_name, team_code = m.group(1), m.group(2)
                    break

            for table in page.extract_tables():
                if not table:
                    continue
                header = [(c or "").strip() for c in table[0]]
                if "PLAYER NAME" not in header:
                    continue  # ignora a tabela do treinador
                idx = {h: i for i, h in enumerate(header)}
                for r in table[1:]:
                    if not r:
                        continue
                    name = (r[idx["PLAYER NAME"]] or "").strip()
                    if not name:
                        continue
                    rows.append({
                        "team": team_name,
                        "team_code": team_code,
                        "number": (r[idx.get("#", 0)] or "").strip(),
                        "position": (r[idx["POS"]] or "").strip(),
                        "player_name": name,
                        "first_names": (r[idx["FIRST NAME(S)"]] or "").strip(),
                        "last_names": (r[idx["LAST NAME(S)"]] or "").strip(),
                        "shirt_name": (r[idx["NAME ON SHIRT"]] or "").strip(),
                        "dob": (r[idx["DOB"]] or "").strip(),
                        "club": (r[idx["CLUB"]] or "").strip(),
                        "height_cm": (r[idx["HEIGHT (CM)"]] or "").strip(),
                    })

    df = pd.DataFrame(rows)
    df["dob"] = pd.to_datetime(df["dob"], format="%d/%m/%Y", errors="coerce")
    df["height_cm"] = pd.to_numeric(df["height_cm"], errors="coerce")
    return df


def normalize_name(s: pd.Series) -> pd.Series:
    """Normaliza nomes para o cruzamento: maiúsculas, sem acentos,
    sem pontuação e com espaços únicos."""
    try:
        from unidecode import unidecode
        s = s.fillna("").map(unidecode)
    except ImportError:
        # fallback sem unidecode: remove acentos via NFKD
        import unicodedata
        s = s.fillna("").map(
            lambda x: "".join(c for c in unicodedata.normalize("NFKD", x)
                              if not unicodedata.combining(c))
        )
    return (s.str.upper()
             .str.replace(r"[^A-Z ]", " ", regex=True)
             .str.replace(r"\s+", " ", regex=True)
             .str.strip())


def flag_convocated(df_base: pd.DataFrame,
                    df_convocados: pd.DataFrame,
                    name_col: str = "player_name") -> pd.DataFrame:
    """Adiciona a coluna is_convocated (1/0) em df_base.

    df_base       : DataFrame maior, com uma coluna de nome de jogador.
    df_convocados : DataFrame gerado por extract_convocados().
    name_col      : nome da coluna de jogador em df_base.
    """
    convocados_norm = set(normalize_name(df_convocados["player_name"]))
    # Também aceita o formato "Nome Sobrenome" (PLAYER NAME vem como "SOBRENOME Nome")
    convocados_norm |= set(normalize_name(
        df_convocados["first_names"] + " " + df_convocados["last_names"]
    ))

    base_norm = normalize_name(df_base[name_col])
    df_base = df_base.copy()
    df_base["is_convocated"] = base_norm.isin(convocados_norm).astype(int)
    return df_base


if __name__ == "__main__":
    current_dir = os.getcwd()
    df_convocados = extract_convocados("../../data/raw/SquadLists-English.pdf")
    df_convocados.to_csv("../../data/interim/convocados_wc2026.csv", index=False)
    print(f"arquivo salvo em: ../../data/interim/convocados_wc2026.csv")
    print(f"total de convocados extraídos: {len(df_convocados)}")
   
