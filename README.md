# Previsão de Convocação — Copa do Mundo FIFA 2026

Projeto acadêmico da disciplina de Inteligência Artificial (USP) que usa Machine Learning para prever quais jogadores serão convocados para a Copa do Mundo de 2026, com base nos atributos do jogo EA FC 26.

---

## Problema

Dado o perfil estatístico de um jogador (notas do EA FC 26), o modelo responde:

> **Esse jogador seria convocado para a Copa 2026?**

É uma tarefa de **classificação binária** com classe altamente desbalanceada (~914 convocados vs ~17.491 não convocados em ~18.400 jogadores).

---

## Metodologia — CRISP-DM

| Fase | O que foi feito |
|------|----------------|
| **Business Understanding** | Entender quais fatores matemáticos explicam as convocações e apresentar isso de forma interativa |
| **Data Understanding** | Cruzamento de atributos do EA FC 26 com listas oficiais da FIFA (48 seleções × 26 jogadores) |
| **Data Preparation** | Pipeline de limpeza em tronco comum → duas ramificações (Pesquisa / Feira) |
| **Modeling** | Árvore de Decisão, XGBoost e KNN treinados para cada ramificação |
| **Evaluation** | Pesquisa: F1-Score, Recall, Matriz de Confusão · Feira: coerência interativa e latência |
| **Deployment** | Entrega 1: relatório + gráficos ✅ · Entrega 2: API + gerador de cartas FIFA |

---

## Dois Modelos

O projeto treina dois modelos com propósitos distintos:

### Modelo de Pesquisa (`research_model.joblib`)
- Usa **todas as features** disponíveis após a limpeza (~70 colunas).
- Avaliado por rigor estatístico: F1-Score, Recall, Matriz de Confusão.
- Objetivo: explicar com precisão científica quais atributos mais influenciam a convocação.

### Modelo da Feira (`feira_model.joblib`)
- Usa apenas as **6 notas principais** da carta do jogador: `pace`, `shooting`, `passing`, `dribbling`, `defending`, `physic` + posição primária.
- Avaliado pela coerência interativa: deve responder em milissegundos e não cometer erros grotescos (ex.: convocar atacante com `shooting = 15`).
- Objetivo: experiência interativa ao vivo — a plateia insere os atributos e vê a predição instantaneamente.

---

## Resultados — Comparação de Modelos

| Branch   | Algoritmo     | Precision | Recall | F1     | F1-Macro |
|----------|---------------|-----------|--------|--------|----------|
| Pesquisa | Árvore de Decisão | 0.183 | 0.191 | 0.187 | 0.572 |
| Pesquisa | **XGBoost ★** | **0.377** | **0.333** | **0.354** | **0.661** |
| Pesquisa | KNN           | 0.188     | 0.525  | 0.276  | 0.600    |
| Feira    | Árvore de Decisão | 0.244 | 0.164 | 0.196 | 0.581 |
| Feira    | **XGBoost ★** | **0.249** | **0.339** | **0.287** | **0.621** |
| Feira    | KNN           | 0.152     | 0.399  | 0.220  | 0.571    |

★ Melhor modelo em cada branch — salvo como `research_model.joblib` e `feira_model.joblib`.

---

## Setup

### Pipeline de ML

```bash
# 1. Clone o repositório
git clone <repo-url>
cd previsao-de-convocacao-wc26

# 2. Ative o ambiente virtual (Python 3.13)
source ml/.venv/bin/activate

# 3. Instale dependências
pip install -r ml/requirements.txt
```

### Backend (API)

> Requer o ambiente virtual do ML ativo (`source ml/.venv/bin/activate`) ou as dependências do backend instaladas separadamente.

```bash
# Instale as dependências do backend
pip install -r backend/requirements.txt

# Inicie o servidor (a partir da raiz do repositório)
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Endpoints disponíveis:

| Método | Rota | Descrição |
|--------|------|-----------|
| `POST` | `/predict` | Recebe atributos do jogador e retorna `{ convocated, probability, overall }` |
| `GET` | `/positions` | Lista as posições válidas aceitas pelo modelo |
| `GET` | `/health` | Verifica se o modelo foi carregado |

**Exemplo:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"position":"ST","pace":90,"shooting":85,"passing":70,"dribbling":88,"defending":30,"physic":75}'
# → {"convocated":true,"probability":0.9673,"overall":83}
```

### Frontend (gerador de cartas FIFA)

> Requer Node.js e npm. O **backend deve estar rodando** na porta 8000 antes de abrir o frontend.

```bash
cd frontend

# Instale as dependências (apenas na primeira vez)
npm install

# Inicie o servidor de desenvolvimento
npm run dev
# → http://localhost:5173
```

Para gerar o build de produção:

```bash
cd frontend
npm run build      # gera frontend/dist/
npm run preview    # visualiza o build localmente
```

### Rodando o sistema completo (backend + frontend)

Abra dois terminais a partir da raiz do repositório:

**Terminal 1 — Backend:**
```bash
source ml/.venv/bin/activate
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm install   # apenas na primeira vez
npm run dev
```

Acesse `http://localhost:5173` no navegador.

---

## Pipeline

Os scripts devem ser executados **dentro de `ml/scripts/`** (os caminhos relativos apontam para `../../data/`).



```bash
# Passo 1 - Ative o ambiente virtual (Python 3.13)
source ml/.venv/bin/activate

cd ml/scripts

# Passo 2 — Extrai as listas de convocados do PDF oficial da FIFA
python 1-convocados_wc2026.py
# → data/interim/convocados_wc2026.csv  (1.248 jogadores convocados)

# Passo 3 — Cruza jogadores do EA FC 26 com os convocados (fuzzy match por nome + DOB)
python 2-add_is_convocated.py
# → data/interim/fifa_with_convocados.csv

# Passo 4 — Limpeza, imputação e encoding → dataset ML-ready
python 3-prepare_dataset.py
# → data/processed/dataset_final.csv

# Passo 5 — Treina os 3 algoritmos (ambas as branches)
python 4-1-decision_tree_model.py
python 4-2-xgboost_model.py
python 4-3-knn_model.py
# → ml/models/{research,feira}_{dt,xgb,knn}.joblib
# → ml/models/research_model.joblib  (melhor: XGBoost)
# → ml/models/feira_model.joblib     (melhor: XGBoost)
# → data/processed/model_comparison.csv

# Passo 6 — Gráfico consolidado comparando todos os modelos
python 5-compare_models.py
# → data/processed/charts/comparison_all_models.png
```

---

## Estrutura de Diretórios

```
previsao-de-convocacao-wc26/
├── data/
│   ├── raw/                            ← Fontes originais (não modificar)
│   │   ├── FC26_20250921.csv           ← ~18k jogadores do EA FC 26
│   │   └── SquadLists-English.pdf      ← Listas oficiais FIFA WC2026
│   ├── interim/                        ← CSVs intermediários (scripts 1–2)
│   └── processed/
│       ├── dataset_final.csv           ← Dataset ML-ready (~18k linhas, ~71 colunas)
│       ├── model_comparison.csv        ← Métricas de todos os 6 modelos treinados
│       └── charts/                     ← Gráficos gerados pelos scripts 4-x e 5
├── ml/
│   ├── scripts/                        ← Pipeline numerado (rodar em ordem)
│   │   ├── 1-convocados_wc2026.py
│   │   ├── 2-add_is_convocated.py
│   │   ├── 3-prepare_dataset.py
│   │   ├── 4-1-decision_tree_model.py
│   │   ├── 4-2-xgboost_model.py
│   │   ├── 4-3-knn_model.py
│   │   └── 5-compare_models.py
│   ├── models/                         ← Modelos treinados (.joblib)
│   │   ├── research_model.joblib       ← Melhor modelo Pesquisa (XGBoost)
│   │   ├── feira_model.joblib          ← Melhor modelo Feira (XGBoost)
│   │   ├── feira_model_meta.json       ← Metadados de features/posições (requerido pela API)
│   │   ├── research_{dt,xgb,knn}.joblib
│   │   └── feira_{dt,xgb,knn}.joblib
│   └── .venv/                          ← Ambiente virtual Python 3.13
├── backend/                            ← API FastAPI servindo o Modelo da Feira
│   ├── main.py                         ← Endpoints /predict /positions /health
│   └── requirements.txt
├── frontend/                           ← React + Vite + Tailwind CSS (gerador de cartas FIFA)
│   ├── src/
│   │   ├── api.ts                      ← Chamadas HTTP ao backend (http://localhost:8000)
│   │   ├── App.tsx                     ← Layout principal (campo | formulário | carta)
│   │   └── components/
│   │       ├── PlayerForm.tsx          ← Formulário de atributos + webcam
│   │       ├── PaniniCard.tsx          ← Carta estilo FIFA com resultado
│   │       ├── FootballField.tsx       ← Campo visual com posição destacada
│   │       ├── StatSlider.tsx          ← Slider de atributos (0–99)
│   │       ├── WebcamCapture.tsx       ← Captura de foto pela webcam
│   │       └── SendCardModal.tsx       ← Envio da carta por e-mail
│   ├── package.json
│   └── vite.config.ts
├── requirements.txt
└── CLAUDE.md                           ← Guia operacional interno (para Claude Code)
```

---

## Status Atual

| Componente | Status |
|---|---|
| Script 1 — extração de convocados do PDF | ✅ Completo |
| Script 2 — cruzamento fuzzy por nome + DOB | ✅ Completo |
| Script 3 — preparação do dataset | ✅ Completo |
| Script 4-1 — Árvore de Decisão (Pesquisa + Feira) | ✅ Completo |
| Script 4-2 — XGBoost (Pesquisa + Feira) | ✅ Completo |
| Script 4-3 — KNN (Pesquisa + Feira) | ✅ Completo |
| Script 5 — gráfico comparativo consolidado | ✅ Completo |
| Gráficos de métricas e importância de features | ✅ Completo (11 PNGs) |
| Modelos salvos (`.joblib`) | ✅ 8 arquivos (6 por-algoritmo + 2 canônicos) |
| Backend / API FastAPI | ✅ Completo (`/predict`, `/positions`, `/health`) |
| Frontend / gerador de cartas | ✅ Completo (React + Vite, `http://localhost:5173`) |

---

## Entregas

- **Entrega 1 ✅:** Pipeline de dados + 3 algoritmos treinados e avaliados para ambas as branches. Gráficos de métricas, matrizes de confusão, importância de features e comparação consolidada gerados em `data/processed/charts/`.
- **Entrega 2 — Backend ✅:** API FastAPI servindo o `feira_model.joblib` com endpoints `/predict`, `/positions` e `/health`. Calcula ou estima o `overall` por posição.
- **Entrega 2 — Frontend ✅:** Interface React com gerador de cartas estilo FIFA. A plateia insere atributos, tira uma foto pela webcam e recebe a predição em uma carta FIFA animada. Inclui envio da carta por e-mail.
