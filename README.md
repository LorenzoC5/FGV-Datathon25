# Estratégia de Trading com Criptomoedas - Datathon FGV 25

## Visão Geral

Este projeto implementa uma estratégia completa de trading algorítmico para criptomoedas utilizando técnicas de Machine Learning e Reinforcement Learning. A estratégia transforma um problema de regressão (predição de retornos contínuos) em classificação através de **binning supervisionado com Decision Tree**, e utiliza modelos MLP e RL para alocação de portfólio.

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

## Informações Técnicas Detalhadas

### Metodologia

#### 1. Coleta e Preparação de Dados

- **Fonte de Dados**: API Binance (klines)
- **Formato de Dados**: OHLCV (Open, High, Low, Close, Volume)
- **Cache**: Dados são cacheados em arquivos `.pkl` para otimização
- **Período de Treino**: 2020-01-01 a 2023-12-31 (4 anos)
- **Período de Teste**: 2024-01-01 a 2025-12-31 (2 anos)
- **Frequência**: Semanal (1w)
- **Retornos**: Log-retornos calculados como `ln(close[t] / close[t-1])`

#### 2. Binning Supervisionado (Decision Tree)

**Algoritmo**: `DecisionTreeRegressor` (scikit-learn)

**Parâmetros**:
- `max_leaf_nodes=5`: Define 5 bins discretos
- `criterion='squared_error'`: Minimiza variância dentro dos bins
- `min_samples_split=2`: Mínimo de amostras para dividir um nó
- `min_samples_leaf=1`: Mínimo de amostras em uma folha

**Processo**:
1. Treina uma árvore de decisão usando retornos como variável alvo
2. Cada folha da árvore representa um bin
3. Os bins são ordenados do menor ao maior retorno
4. Bins são aplicados de forma estática em todo o conjunto de dados

**Vantagens**:
- Método supervisionado que aprende cortes ótimos dos dados
- Adapta-se à distribuição dos retornos
- Não requer normalização prévia

#### 3. Pipeline MLP (Multi-Layer Perceptron)

**Modelo**: `MLPClassifier` (scikit-learn)

**Arquitetura**:
- **Camada de Entrada**: Dimensão variável (depende do número de features)
- **Camada Oculta 1**: 32 neurônios
- **Camada Oculta 2**: 64 neurônios
- **Camada de Saída**: 5 neurônios (softmax para 5 classes)

**Hiperparâmetros**:
- `hidden_layer_sizes=(32, 64)`: Arquitetura da rede
- `activation='relu'`: Função de ativação ReLU
- `solver='adam'`: Otimizador Adam
- `alpha=0.0001`: Regularização L2
- `learning_rate='constant'`: Taxa de aprendizado constante
- `learning_rate_init=0.001`: Taxa de aprendizado inicial
- `max_iter=500`: Máximo de iterações
- `random_state=42`: Semente para reprodutibilidade

**Preprocessamento**:
- **StandardScaler**: Normalização de features numéricas (média=0, desvio=1)
- **OneHotEncoder**: Codificação de features categóricas (símbolos)
- **Validação**: Divisão treino/teste temporal (sem shuffle)

**Features de Entrada**:
- 15 indicadores técnicos (normalizados)
- Features categóricas (símbolos das criptomoedas)
- Total: ~20-25 features por amostra

#### 4. Pipeline RL (Reinforcement Learning)

**Framework**: Gymnasium + Stable-Baselines3

**Algoritmo**: PPO (Proximal Policy Optimization)

**Configuração do Ambiente** (`CryptoPortfolioEnv`):

**Estado (State Space)**:
- **Sentimento (S[t])**: Vetor [positivo, neutro, negativo] - 3 dimensões
- **Probabilidades de Bins (P[t])**: Matriz (N, 5) onde N = número de criptos - 45 dimensões (9 criptos × 5 bins)
- **Retornos Realizados (R[t+1])**: Vetor de retornos - 9 dimensões
- **Máscara de Elegibilidade (M[t])**: Vetor binário - 9 dimensões
- **Pesos Anteriores (W[t-1])**: Vetor de alocação anterior - 9 dimensões
- **EWMA Volatilidade**: Volatilidade exponencialmente ponderada - 9 dimensões
- **Retorno Líquido Anterior**: Retorno líquido do portfólio - 1 dimensão
- **Total**: ~89 dimensões

**Ação (Action Space)**:
- **Tipo**: Box (contínuo)
- **Dimensão**: N (número de criptomoedas) = 9
- **Range**: [0, 1] (logits)
- **Normalização**: Softmax para garantir soma = 1 (pesos de portfólio)

**Recompensa (Reward)**:
- **Tipo**: Sharpe Ratio Local ou Mean-Variance
- **Cálculo**: `(retorno_médio - taxa_livre_risco) / volatilidade`
- **Janela**: Rolling window de N períodos
- **Penalizações**:
  - Custos de transação: 0.1% por transação
  - Turnover: Penalidade por mudanças excessivas de posição

**Hiperparâmetros PPO**:
- `learning_rate=3e-4`: Taxa de aprendizado
- `n_steps=2048`: Passos por atualização
- `batch_size=64`: Tamanho do batch
- `n_epochs=10`: Épocas por atualização
- `gamma=0.99`: Fator de desconto
- `gae_lambda=0.95`: Parâmetro GAE (Generalized Advantage Estimation)
- `clip_range=0.2`: Clipping do PPO
- `ent_coef=0.01`: Coeficiente de entropia
- `vf_coef=0.5`: Coeficiente de valor
- `total_timesteps=150000`: Total de passos de treinamento

**Arquitetura da Rede Neural (PPO)**:
- **Policy Network**: MLP com 2 camadas ocultas (64, 64 neurônios)
- **Value Network**: MLP com 2 camadas ocultas (64, 64 neurônios)
- **Ativação**: Tanh
- **Função de Valor**: Critic network separado

#### 5. Indicadores Técnicos

**Cálculo de Indicadores**:

1. **Retornos**:
   - `ret_1`: `ln(close[t] / close[t-1])`
   - `ret_5`: `ln(close[t] / close[t-5])`
   - `ret_20`: `ln(close[t] / close[t-20])`

2. **Médias Móveis**:
   - `SMA(n)`: Média aritmética simples
   - `EMA(n)`: Média exponencialmente ponderada
   - Ratios: `price / SMA(n)` e `price / EMA(n)`

3. **Volatilidade**:
   - `vol_n`: Desvio padrão dos retornos em janela de n períodos

4. **RSI (Relative Strength Index)**:
   - `RSI_14`: Indicador de momentum (0-100)
   - Fórmula: `100 - (100 / (1 + RS))` onde `RS = média_gains / média_losses`

5. **Bandas de Bollinger**:
   - `bb_width_20`: `(upper_band - lower_band) / middle_band`
   - `bb_pos_20`: `(price - lower_band) / (upper_band - lower_band)`

6. **Volume**:
   - `volume_zscore_20`: Z-score do volume em janela de 20 períodos
   - `pv_corr_20`: Correlação entre retornos e variação de volume

**Seleção de Features**:
- Threshold de correlação: < 0.3
- Remove features altamente correlacionadas
- Mantém diversidade de informações

#### 6. Processamento de Sentimento

- **Fonte**: Arquivo CSV (`data/raw/daily_sentiment_full.csv`)
- **Agregação**: Dados diários agregados para frequência semanal
- **Formato**: [positivo, neutro, negativo] por criptomoeda
- **Normalização**: Soma = 1 (probabilidades)

#### 7. Benchmarks

**Estratégias de Comparação**:

1. **Buy & Hold Equal Weight**:
   - Alocação inicial igual (1/N)
   - Sem rebalanceamento
   - Sem custos de transação

2. **Markowitz Sharpe Optimization**:
   - Otimização de Sharpe Ratio
   - Rebalanceamento periódico
   - Considera matriz de covariância

3. **Equal Weight Rebalanced**:
   - Alocação igual (1/N)
   - Rebalanceamento periódico
   - Com custos de transação

### Requisitos do Sistema

**Hardware Mínimo**:
- CPU: 4 cores
- RAM: 8 GB
- Espaço em Disco: 5 GB (dados + modelos)

**Hardware Recomendado**:
- CPU: 8+ cores
- RAM: 16+ GB
- GPU: Opcional (não utilizada no código atual)
- Espaço em Disco: 10+ GB

**Software**:
- Python 3.8+
- pip ou conda
- Sistema operacional: Windows, Linux ou macOS

**Dependências Principais**:
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- requests >= 2.25.0
- matplotlib >= 3.5.0
- gymnasium >= 0.28.0
- stable-baselines3 >= 2.0.0
- plotly >= 5.0.0

### Tempo de Execução Aproximado

- **Geração de Indicadores**: 5-10 minutos
- **Treinamento MLP**: 2-5 minutos
- **Treinamento RL**: 30-60 minutos (150k steps)
- **Pipeline Completo**: 40-75 minutos

### Reprodutibilidade

- **Seeds Fixos**: `random_state=42` em todos os modelos
- **Ordenação Temporal**: Dados mantêm ordem cronológica
- **Sem Shuffle**: Divisão treino/teste temporal
- **Cache Determinístico**: Dados cacheados garantem consistência

### Limitações e Considerações

1. **Lookback Limitado**: RL utiliza apenas 1 período de histórico
2. **Custos de Transação**: Fixo em 0.1% (pode variar na prática)
3. **Slippage**: Não considerado no modelo
4. **Liquidez**: Assume liquidez infinita
5. **Dados de Sentimento**: Requer arquivo externo pré-processado
6. **Frequência Fixa**: Apenas estratégia semanal implementada

## Contribuição

Este projeto foi desenvolvido para o Datathon FGV. Para questões ou melhorias, por favor abra uma issue ou pull request.

## Estrutura Modular

O projeto foi organizado em módulos Python para facilitar manutenção e reutilização:

- **`src/data/`**: Funções de coleta e processamento de dados
- **`src/binning/`**: Algoritmo de binning supervisionado (Decision Tree)
- **`src/models/`**: Pipelines de treinamento (MLP e RL)
- **`src/utils/`**: Funções auxiliares e visualizações