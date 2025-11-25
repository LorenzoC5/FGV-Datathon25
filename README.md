# Estratégia de Trading com Criptomoedas - Datathon

## Visão Geral

Este projeto implementa uma estratégia completa de trading algorítmico para criptomoedas utilizando técnicas de Machine Learning e Reinforcement Learning. A estratégia transforma um problema de regressão (predição de retornos contínuos) em classificação através de **binning supervisionado com Decision Tree**, e utiliza modelos MLP e RL para alocação de portfólio. A estratégia utiliza **dados semanais** e o método **Tree** para binning.

## Estrutura do Projeto

```
Datathon/
├── src/                    # Código fonte principal
│   ├── data/              # Coleta e preparação de dados
│   │   ├── __init__.py
│   │   ├── data_fetcher.py    # Busca dados da API Binance
│   │   └── indicators.py      # Cálculo de indicadores técnicos
│   ├── binning/           # Métodos de binning supervisionado
│   │   ├── __init__.py
│   │   └── decision_tree.py   # Binning com Decision Tree
│   ├── models/            # Modelos de ML
│   │   ├── __init__.py
│   │   ├── mlp_pipeline.py    # Pipeline MLP para classificação
│   │   └── rl_pipeline.py      # Pipeline RL para alocação de portfólio
│   └── utils/             # Utilitários
│       ├── __init__.py
│       └── visualization.py   # Funções de visualização
├── scripts/               # Scripts executáveis
│   └── run_pipeline.py   # Pipeline completo
├── data/                  # Dados
│   ├── raw/               # Dados brutos (cache .pkl)
│   └── processed/         # Dados processados (CSVs de indicadores)
└── results/               # Resultados
    ├── models/            # Modelos treinados (PPO .zip)
    ├── plots/             # Gráficos e visualizações
    └── reports/            # Relatórios e métricas (CSVs, HTMLs)
```

**Nota**: O projeto foi reorganizado para manter todo o pipeline em arquivos `.py`. Os arquivos antigos foram movidos para a estrutura modular acima. Veja `ESTRUTURA.md` para detalhes completos da reorganização.

## Estratégia

### 1. Binning Supervisionado (Decision Tree)

A estratégia transforma retornos contínuos em 5 classes discretas usando **Decision Tree**:

- Utiliza `DecisionTreeRegressor` com `max_leaf_nodes=5`
- Particiona os retornos em 5 bins baseado na estrutura da árvore
- Cada folha da árvore representa uma classe de retorno
- Método supervisionado que aprende os cortes ótimos dos dados de treino

### 2. Pipeline MLP (Semanal)

- **Frequência**: Dados semanais (1w)
- **Entrada**: Indicadores técnicos + bins de retorno (Tree)
- **Modelo**: MLPClassifier com arquitetura (32, 64)
- **Objetivo**: Classificar retornos futuros em 5 classes
- **Períodos**:
  - Treino: 2020-01-01 até 2023-12-31
  - Teste: 2024-01-01 até 2025-12-31

### 3. Pipeline RL (Reinforcement Learning - Semanal)

- **Frequência**: Dados semanais (1w)
- **Ambiente**: `CryptoPortfolioEnv` (Gymnasium)
- **Algoritmo**: PPO (Proximal Policy Optimization)
- **Entrada**: 
  - Sentimento (S[t]): vetor [positivo, neutro, negativo]
  - Probabilidades de bins (P[t]): matriz (N, 5) de probabilidades por cripto (Tree)
  - Retornos realizados (R[t+1])
  - Máscara de elegibilidade (M[t])
- **Ação**: Alocação de portfólio (pesos por criptomoeda)
- **Recompensa**: Sharpe ratio local ou mean-variance

## Criptomoedas Analisadas

- ADA (Cardano)
- BNB (Binance Coin)
- BTC (Bitcoin)
- ETH (Ethereum)
- SOL (Solana)
- TRX (Tron)
- USDC (USD Coin)
- XRP (Ripple)
- DOGE (Dogecoin)

## Instalação

```bash
# Instalar dependências
pip install -r requirements.txt

# Dependências adicionais para RL
pip install gymnasium stable-baselines3 plotly
```

## Uso

### Pipeline Completo

```bash
python scripts/run_pipeline.py
```

### Passos Individuais

1. **Gerar Indicadores Técnicos**
```bash
python src/data/indicators.py
```

2. **Treinar Modelos MLP**
```bash
python src/models/mlp_pipeline.py
```

3. **Treinar e Testar Modelos RL (Semanal)**
```bash
python src/models/rl_pipeline.py --interval 1w --total_steps 150000 --seed 42
```

### Calcular Bins (Tree - Semanal)

```python
from src.data.data_fetcher import prepare_binance_data_for_training, BINANCE_SYMBOLS
from src.binning.decision_tree import create_supervised_static_bins

# Buscar dados semanais
X_train_data, Y_train_data = prepare_binance_data_for_training(
    symbols=BINANCE_SYMBOLS,
    interval='1w',  # Semanal
    start_time='2020-01-01',
    end_time='2023-12-31'
)

# Calcular bins usando Decision Tree
bins_tree = create_supervised_static_bins(X_train_data, Y_train_data)
```

## Indicadores Técnicos

O pipeline calcula 15 indicadores técnicos:

1. `price_close` - Preço de fechamento
2. `ret_1` - Log-retorno 1 período
3. `ret_5` - Log-retorno acumulado 5 períodos
4. `ret_20` - Log-retorno acumulado 20 períodos
5. `sma_10_ratio` - Preço / SMA(10)
6. `sma_50_ratio` - Preço / SMA(50)
7. `ema_10_ratio` - Preço / EMA(10)
8. `ema_50_ratio` - Preço / EMA(50)
9. `vol_10` - Volatilidade (desvio padrão) 10 períodos
10. `vol_50` - Volatilidade 50 períodos
11. `rsi_14` - RSI clássico (14 períodos)
12. `bb_width_20` - Largura das Bandas de Bollinger (20)
13. `bb_pos_20` - Posição relativa nas bandas (20)
14. `volume_zscore_20` - Z-score de volume (20)
15. `pv_corr_20` - Correlação retorno x variação de volume (20)

Os indicadores são selecionados para baixa correlação entre si (threshold < 0.3).

## Resultados

Os resultados são salvos em:

- **MLP (Semanal - Tree)**: `results/plots/confusion_matrix_weekly_tree.png`
- **RL (Semanal)**: `results/models/ppo_model_1w_*.zip`
- **Backtests**: `results/backtest_results_1w_*.csv`
- **Dashboards**: `results/dashboard_1w_*.html`
- **Métricas**: `results/comparative_metrics_1w_*.csv` e `.tex`

## Métricas de Avaliação

### MLP
- Accuracy
- F1-Score (macro e micro)
- Matriz de Confusão

### RL
- Sharpe Ratio
- Retorno Acumulado
- Drawdown Máximo
- Volatilidade
- Comparação com benchmarks (Buy & Hold, Equal Weight)

## Períodos de Dados

- **Treino**: 2020-01-01 até 2023-12-31 (incluso)
- **Teste**: 2024-01-01 até 2025-12-31 (incluso)

## Cache de Dados

Os dados da Binance são automaticamente cacheados em arquivos `.pkl` para evitar requisições repetidas. Os arquivos de cache são salvos em `data/raw/` com o formato:

```
crypto_data_{symbols}_{interval}_{start}_{end}.pkl
```

## Notas Técnicas

- **Frequência**: Todos os modelos utilizam dados semanais (1w)
- **Binning**: Apenas método Tree (Decision Tree)
- Todos os retornos são calculados como log-retornos: `log(close[t] / close[t-1])`
- Os bins são criados apenas com base nos retornos, não em features técnicas
- O ambiente RL inclui custos de transação e penalidades por turnover
- O modelo RL utiliza lookback de 1 período por padrão

## Contribuição

Este projeto foi desenvolvido para o Datathon FGV. Para questões ou melhorias, por favor abra uma issue ou pull request.

## Estrutura Modular

O projeto foi organizado em módulos Python para facilitar manutenção e reutilização:

- **`src/data/`**: Funções de coleta e processamento de dados
- **`src/binning/`**: Algoritmo de binning supervisionado (Decision Tree)
- **`src/models/`**: Pipelines de treinamento (MLP e RL)
- **`src/utils/`**: Funções auxiliares e visualizações

Cada módulo possui um `__init__.py` que exporta as funções principais, permitindo imports limpos:

```python
from src.data import prepare_binance_data_for_training
from src.binning.decision_tree import create_supervised_static_bins
from src.models.mlp_pipeline import main as train_mlp
```

## Notas de Migração

Se você estava usando os arquivos antigos (`main.py`, `main2.py`), as funções foram movidas para:

- `main.py` → `src/data/data_fetcher.py` + `src/binning/`
- `main2.py` → `scripts/run_pipeline.py`
- `indicators.py` → `src/data/indicators.py`
- `mlp_pipeline.py` → `src/models/mlp_pipeline.py`
- `rl_pipeline.py` → `src/models/rl_pipeline.py`

Atualize seus imports conforme necessário.

## Licença

Este projeto é fornecido "como está" para fins educacionais e de pesquisa.

