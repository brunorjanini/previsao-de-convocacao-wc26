# -*- coding: utf-8 -*-
"""
Adiciona a coluna `is_convocated` (1/0) ao CSV do dataset FIFA/EA FC,
cruzando com o CSV de convocados da Copa 2026.
 
Formato do CSV FIFA (1º argumento) — valores com padding de espaços e
campos entre aspas contendo vírgulas, ex.:
 
       252371, /player/252371/jude-bellingham/260004/ , ..., J. Bellingham ,
       Jude Victor William Bellingham , "CAM, CM", ..., 2003-06-29, ...
 
Formato do CSV de convocados (2º argumento):
 
    team,team_code,number,position,player_name,first_names,last_names,
    shirt_name,dob,club,height_cm
 
Estratégia de match (em ordem):
  1. Nome completo normalizado: long_name/short_name do dataset comparado a
     first_names + last_names e a player_name (nas duas ordens de palavras).
  2. Fallback por data de nascimento: mesmo `dob` E sobreposição de >= 2
     tokens entre os nomes (>= 1 se o nome do convocado tiver 1 só palavra).
     Captura apelidos e grafias diferentes (ex.: nomes artísticos).
 
Uso:
    python add_is_convocated.py fifa_players.csv convocados_wc2026.csv saida.csv
"""
 
import unicodedata
import pandas as pd
 
 
# ----------------------------- normalização ------------------------------- #

def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", str(text))
        if not unicodedata.combining(c)
    )


def normalize(text) -> str:
    """Maiúsculas, sem acento, sem pontuação, espaços únicos.

    Descarta chars não-ASCII após strip_accents para evitar que scripts
    alternativos (árabe, cirílico etc.) corrompam a tokenização — o FC26
    armazena nomes como 'Riyad Mahrezرياض محرز'.
    """
    if pd.isna(text):
        return ""
    t = strip_accents(str(text))
    t = "".join(c if c.isascii() else " " for c in t)   # remove não-ASCII
    t = t.upper()
    t = "".join(c if c.isalpha() or c.isspace() else " " for c in t)
    return " ".join(t.split())
 
 
def tokens(text) -> frozenset:
    return frozenset(normalize(text).split())
 
 
def parse_dob(series: pd.Series) -> pd.Series:
    """Aceita AAAA-MM-DD (ISO, formato do dataset) e DD/MM/AAAA (PDF FIFA)."""
    s = series.astype(str).str.strip()
    out = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    mask = out.isna()
    if mask.any():
        out[mask] = pd.to_datetime(s[mask], format="%d/%m/%Y", errors="coerce")
    return out
 
 
def read_padded_csv(path: str) -> pd.DataFrame:
    """Lê CSV com padding de espaços nos cabeçalhos e nos valores.
 
    - skipinitialspace remove os espaços APÓS a vírgula (antes do valor),
      o que também garante que campos entre aspas ("CAM, CM") sejam
      reconhecidos corretamente mesmo com espaço antes da aspa;
    - em seguida, remove os espaços à direita dos valores de texto e
      dos nomes de coluna.
    """
    df = pd.read_csv(path, skipinitialspace=True, low_memory=False)
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            df[col] = df[col].str.strip()
    return df
 
 
# ------------------------------- matching --------------------------------- #
 
def add_is_convocated(df_fifa: pd.DataFrame, df_conv: pd.DataFrame) -> pd.DataFrame:
    df_fifa = df_fifa.copy()
 
    # --- 1) conjunto de nomes completos normalizados dos convocados --------
    full = (df_conv["first_names"].fillna("") + " " + df_conv["last_names"].fillna(""))
    name_set = set(full.map(normalize))
    name_set |= set(df_conv["player_name"].map(normalize))
    # player_name vem como "SOBRENOME Nome" -> gera também "Nome SOBRENOME"
    name_set |= {
        " ".join(reversed(normalize(s).split())) for s in df_conv["player_name"]
    }
    name_set.discard("")
 
    # --- 2) índice por dob para o fallback por tokens ------------------------
    conv_dob = parse_dob(df_conv["dob"])
    conv_tokens = [
        tokens(a) | tokens(b) for a, b in zip(full, df_conv["player_name"])
    ]
    dob_index: dict = {}
    for dob, toks in zip(conv_dob, conv_tokens):
        if pd.notna(dob):
            dob_index.setdefault(dob, []).append(toks)
 
    # --- aplica nas linhas do dataset FIFA -----------------------------------
    fifa_long = df_fifa["long_name"].map(normalize)
    fifa_short = (df_fifa["short_name"].map(normalize)
                  if "short_name" in df_fifa.columns else fifa_long)
    fifa_dob = (parse_dob(df_fifa["dob"])
                if "dob" in df_fifa.columns
                else pd.Series(pd.NaT, index=df_fifa.index))
    fifa_tokens = df_fifa["long_name"].map(tokens)
 
    def is_convocated(i) -> int:
        # 1) match exato de nome (long_name pode ter sobrenomes extras,
        #    então também testa se o nome do convocado está contido nele)
        if fifa_long.iat[i] in name_set or fifa_short.iat[i] in name_set:
            return 1
        # 2) fallback: mesmo dob + sobreposição de tokens
        dob = fifa_dob.iat[i]
        if pd.notna(dob) and dob in dob_index:
            ftoks = fifa_tokens.iat[i]
            for ctoks in dob_index[dob]:
                needed = 1 if len(ctoks) <= 1 else 2
                if len(ftoks & ctoks) >= needed:
                    return 1
        return 0
 
    df_fifa["is_convocated"] = [is_convocated(i) for i in range(len(df_fifa))]
    return df_fifa
# --------------------------------- CLI ------------------------------------ #

def main():
    fifa_path = "../../data/raw/FC26_20250921.csv"
    conv_path = "../../data/interim/convocados_wc2026.csv"
    out_path = "../../data/interim/fifa_with_convocados.csv"

    df_fifa = read_padded_csv(fifa_path)
    df_conv = pd.read_csv(conv_path, encoding="utf-8")

    df_out = add_is_convocated(df_fifa, df_conv)
    df_out.to_csv(out_path, index=False)

    n = int(df_out["is_convocated"].sum())
    print(f"Arquivo salvo em: {out_path}")
    print(f"Jogadores marcados como convocados: {n} de {len(df_out)} linhas "
          f"({len(df_conv)} convocados na lista)")


if __name__ == "__main__":
    main()
