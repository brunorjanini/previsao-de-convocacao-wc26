# Previsão de Convocação — Copa do Mundo FIFA 2026

Trabalho da disciplina de Inteligência Artificial (USP): desenvolvimento de um modelo de Machine Learning para prever a convocação de jogadores para a Copa do Mundo FIFA 2026, seguindo a metodologia CRISP-DM.

## Objetivo

Dado o dataset de atributos de jogadores do EA FC 26, prever se um jogador foi convocado para a Copa do Mundo 2026 (`is_convocated = 1/0`), usando as listas oficiais da FIFA como fonte de verdade.

**Demo interativa**: o usuário insere sua foto e seus atributos de jogo → o sistema gera uma carta no estilo EA FC 26 e diz se ele seria convocado para a Copa.

## Pipeline de dados

```
data/raw/FC26_20250921.csv            ← atributos de 18.405 jogadores (EA FC 26)
data/raw/SquadLists-English.pdf       ← listas oficiais FIFA WC 2026 (48 seleções × 26 jogadores)
         ↓ script 1
data/interim/convocados_wc2026.csv    ← 1.248 convocados extraídos do PDF
         ↓ script 2
data/interim/fifa_with_convocados.csv ← dataset FC26 + label is_convocated (914 matches)
         ↓ script 3
data/processed/dataset_final.csv      ← 76 features, pronto para ML
         ↓ script 4
data/processed/feature_importance.png ← top-30 features (RandomForest)
         ↓ script 5
data/processed/model_comparison.csv   ← comparação RF / GradientBoosting / LR
data/processed/confusion_matrix.png   ← matriz de confusão do melhor modelo
ml/models/research_model.joblib       ← modelo de pesquisa (GradientBoosting)
         ↓ script 6
ml/models/feira_model.joblib          ← modelo da feira (19 features, resposta <1ms)
ml/models/feira_model_meta.json       ← metadados: colunas e posições válidas
```

## Scripts

Todos os scripts ficam em `ml/scripts/` e devem ser executados a partir desse diretório:

```bash
cd ml/scripts

# 1. Extrai convocados do PDF da FIFA
python 1-convocados_wc2026.py

# 2. Cruza com o dataset do FC26 e adiciona a label is_convocated
python 2-add_is_convocated.py

# 3. Limpa, imputa e codifica para ML
python 3-prepare_dataset.py

# 4. Treina RandomForest e gera gráfico de feature importance
python 4-feature_importance.py

# 5. Modelo de Pesquisa: compara RF, GradientBoosting, LogisticRegression
python 5-train_research_model.py

# 6. Modelo da Feira: treina modelo leve + testes de sanidade
python 6-train_feira_model.py

# 7. Análise descritiva + diagnóstico de overfitting
python 7-analise_descritiva.py

# 8. Regularização + threshold ótimo (mitiga overfitting identificado no script 7)
python 8-fix_overfitting.py
```

## Resultados de Modelagem

### Feature Importance (top-6)

| # | Feature | Importância |
|---|---------|-------------|
| 1 | overall | 0.068 |
| 2 | value_eur | 0.066 |
| 3 | nationality_id | 0.065 |
| 4 | wage_eur | 0.056 |
| 5 | release_clause_eur | 0.051 |
| 6 | movement_reactions | 0.044 |

### Comparação de Modelos (Modelo de Pesquisa)

| Modelo | F1 | Precision | Recall | F1 Macro |
|---|---|---|---|---|
| GradientBoosting | 0.46 | 0.60 | 0.38 | 0.72 |
| RandomForest | 0.38 | 0.51 | 0.31 | 0.68 |
| LogisticRegression | 0.31 | 0.19 | 0.76 | 0.61 |

Desbalanceamento 19:1 (914 convocados / 17.491 não convocados) — todos os modelos usam `class_weight="balanced"`.

## Análise Descritiva e Diagnóstico

Figura gerada: `data/processed/analise_descritiva.png` (script 7)

| Painel | Conteúdo |
|---|---|
| 1. Distribuição alvo | Barplot 0 vs 1 — documenta o desbalanceamento 19:1 (17.491 × 914) |
| 2. Atributos por classe | Boxplot dos 6 atributos + overall agrupado — convocados têm overall e value sistematicamente maiores |
| 3. Correlação | Top-20 features mais correlacionadas com `is_convocated` — lideradas por `overall`, `value_eur`, `nationality_id` |
| 4a. Curva aprendizado (Pesquisa) | F1 treino vs validação em 6 tamanhos de treino — gap crescente indica overfitting |
| 4b. Curva aprendizado (Feira) | Mesmo diagnóstico com 19 features — gap menor mas ainda presente |
| 5a. ROC | AUC-ROC: Pesquisa 0.931, Feira 0.833 — ambos discriminam bem por probabilidade |
| 5b. Precision-Recall | AP: Pesquisa 0.499, Feira 0.304 — Pesquisa recupera melhor os convocados |
| 6. Distribuição P(convocado) | Histograma separado por classe real — Pesquisa mostra separação parcial; convocados concentrados em alta probabilidade |

### Diagnóstico de Overfitting

| Modelo | F1 Treino | F1 Validação | AUC-ROC | AUC-PR | Diagnóstico |
|---|---|---|---|---|---|
| Pesquisa (75 feat.) | 0.82 | 0.44 | 0.931 | 0.499 | **Overfitting** (gap 0.38) |
| Feira (19 feat.) | 0.40 | 0.22 | 0.833 | 0.304 | **Overfitting** (gap 0.18) |

O overfitting é moderado pelo desbalanceamento extremo (19:1): o modelo aprende os convocados do treino com maior facilidade do que generaliza para os do teste. O AUC-ROC alto (0.93) indica que o **ranking** de probabilidades está correto — o modelo sabe ordenar quem tem mais chance de ser convocado — mas o threshold padrão (0.5) não é ótimo.

### Mitigação do overfitting (script 8)

Gráfico: `data/processed/overfitting_fix.png`

| Métrica | Pesquisa Antes | Pesquisa Depois | Feira Antes | Feira Depois |
|---|---|---|---|---|
| F1 Treino | 0.83 | 0.47 | 0.35 | 0.32 |
| F1 Teste | 0.55 | 0.53 | 0.38 | 0.40 |
| AUC-ROC | 0.931 | 0.931 | 0.833 | 0.845 |
| AUC-PR | 0.499 | 0.509 | 0.304 | 0.341 |
| Gap treino−teste | 0.28 | **0.06** | −0.03 | −0.07 |

**Estratégias aplicadas**: `learning_rate=0.05`, `subsample=0.8`, `min_samples_leaf=20`, `max_features=0.5` + threshold ótimo via curva Precision-Recall (Pesquisa: 0.827, Feira: 0.852). O backend usa automaticamente o threshold salvo em `ml/models/thresholds.json`.

## Ambiente

```bash
# Instalar dependências do pipeline ML
pip install -r ml/requirements.txt

# Ou ativar o venv já configurado
source ml/.venv/bin/activate
```

## Executando o Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
# API disponível em http://localhost:8000
# Docs interativos: http://localhost:8000/docs
```

## Executando o Frontend

Abra `frontend/index.html` diretamente no navegador (sem build step).  
O backend deve estar rodando em `http://localhost:8000`.

Funcionalidades:
1. **Foto**: upload local (FileReader — nada enviado ao servidor)
2. **Posição**: select preenchido via `GET /positions`
3. **Atributos**: sliders para OVR + PAC, CHU, PAS, DRI, DEF, FÍS (0–99)
4. **Carta FIFA**: renderizada em tempo real com visual dourado
5. **Veredito**: convocado ou não, com % de chance

## Estrutura do projeto

```
├── data/
│   ├── raw/          ← dados originais (não modificar)
│   ├── interim/      ← dados intermediários gerados pelos scripts
│   └── processed/    ← dataset final e artefatos de avaliação
├── ml/
│   ├── scripts/      ← pipeline numerado (1→ 2→ 3→ 4→ 5→ 6→ ...)
│   ├── models/       ← modelos serializados (.joblib)
│   └── requirements.txt
├── backend/          ← API FastAPI (POST /predict, GET /positions)
│   ├── main.py
│   └── requirements.txt
└── frontend/         ← interface HTML/CSS/JS (gerador de cartas FIFA)
    ├── index.html
    ├── style.css
    └── app.js
```
