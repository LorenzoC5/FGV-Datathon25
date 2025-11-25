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

### Equações Matemáticas

#### 1. Retornos Logarítmicos

O retorno logarítmico é calculado como:

$$
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right) = \ln(P_t) - \ln(P_{t-1})
$$

onde:
- $r_t$ é o retorno logarítmico no período $t$
- $P_t$ é o preço de fechamento no período $t$
- $P_{t-1}$ é o preço de fechamento no período anterior

**Retornos Acumulados**:
$$
r_{t-k:t} = \sum_{i=t-k+1}^{t} r_i = \ln\left(\frac{P_t}{P_{t-k}}\right)
$$

#### 2. Indicadores Técnicos

**Média Móvel Simples (SMA)**:
$$
\text{SMA}_n(t) = \frac{1}{n} \sum_{i=0}^{n-1} P_{t-i}
$$

**Média Móvel Exponencial (EMA)**:
$$
\text{EMA}_n(t) = \alpha \cdot P_t + (1-\alpha) \cdot \text{EMA}_n(t-1)
$$

onde $\alpha = \frac{2}{n+1}$ é o fator de suavização.

**Ratios de Preço**:
$$
\text{ratio}_{\text{SMA}}(t) = \frac{P_t}{\text{SMA}_n(t)}
$$

**Volatilidade (Desvio Padrão dos Retornos)**:
$$
\sigma_n(t) = \sqrt{\frac{1}{n-1} \sum_{i=0}^{n-1} (r_{t-i} - \bar{r}_n)^2}
$$

onde $\bar{r}_n = \frac{1}{n} \sum_{i=0}^{n-1} r_{t-i}$ é a média dos retornos.

**RSI (Relative Strength Index)**:
$$
\text{RS}_n(t) = \frac{\text{AG}_n(t)}{\text{AL}_n(t)}
$$

$$
\text{RSI}_n(t) = 100 - \frac{100}{1 + \text{RS}_n(t)}
$$

onde:
- $\text{AG}_n(t)$ é a média dos ganhos nos últimos $n$ períodos
- $\text{AL}_n(t)$ é a média das perdas nos últimos $n$ períodos

**Bandas de Bollinger**:
$$
\text{BB}_{\text{middle}}(t) = \text{SMA}_{20}(t)
$$

$$
\text{BB}_{\text{upper}}(t) = \text{SMA}_{20}(t) + 2 \cdot \sigma_{20}(t)
$$

$$
\text{BB}_{\text{lower}}(t) = \text{SMA}_{20}(t) - 2 \cdot \sigma_{20}(t)
$$

$$
\text{BB}_{\text{width}}(t) = \frac{\text{BB}_{\text{upper}}(t) - \text{BB}_{\text{lower}}(t)}{\text{BB}_{\text{middle}}(t)}
$$

$$
\text{BB}_{\text{pos}}(t) = \frac{P_t - \text{BB}_{\text{lower}}(t)}{\text{BB}_{\text{upper}}(t) - \text{BB}_{\text{lower}}(t)}
$$

**Z-score de Volume**:
$$
\text{volume\_zscore}_n(t) = \frac{V_t - \bar{V}_n}{\sigma_{V,n}}
$$

onde:
- $V_t$ é o volume no período $t$
- $\bar{V}_n$ é a média do volume nos últimos $n$ períodos
- $\sigma_{V,n}$ é o desvio padrão do volume nos últimos $n$ períodos

**Correlação Preço-Volume**:
$$
\text{pv\_corr}_n(t) = \frac{\sum_{i=0}^{n-1} (r_{t-i} - \bar{r}_n)(\Delta V_{t-i} - \overline{\Delta V}_n)}{\sqrt{\sum_{i=0}^{n-1} (r_{t-i} - \bar{r}_n)^2 \sum_{i=0}^{n-1} (\Delta V_{t-i} - \overline{\Delta V}_n)^2}}
$$

onde $\Delta V_t = V_t - V_{t-1}$ é a variação do volume.

#### 3. Binning Supervisionado (Decision Tree)

O Decision Tree busca minimizar a variância dentro dos bins:

$$
\text{Var}_{\text{within}} = \sum_{b=1}^{B} \sum_{i \in \text{bin}_b} (r_i - \bar{r}_b)^2
$$

onde:
- $B = 5$ é o número de bins
- $\bar{r}_b$ é a média dos retornos no bin $b$
- $\text{bin}_b$ contém os índices das amostras no bin $b$

O algoritmo escolhe os cortes que minimizam $\text{Var}_{\text{within}}$ sujeito a:
$$
|\text{bin}_b| \geq \text{min\_samples\_leaf} \quad \forall b
$$

#### 4. MLP (Multi-Layer Perceptron)

**Forward Pass**:

Camada oculta 1:
$$
\mathbf{h}_1 = \text{ReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)
$$

Camada oculta 2:
$$
\mathbf{h}_2 = \text{ReLU}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2)
$$

Camada de saída (logits):
$$
\mathbf{z} = \mathbf{W}_3 \mathbf{h}_2 + \mathbf{b}_3
$$

Função de ativação ReLU:
$$
\text{ReLU}(x) = \max(0, x)
$$

**Softmax** (normalização para probabilidades):
$$
P(y = k | \mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^{5} e^{z_j}}
$$

**Função de Perda (Cross-Entropy)**:
$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{5} y_{i,k} \log(P(y_i = k | \mathbf{x}_i))
$$

onde:
- $N$ é o número de amostras
- $y_{i,k}$ é 1 se a classe verdadeira da amostra $i$ é $k$, 0 caso contrário

**Regularização L2**:
$$
\mathcal{L}_{\text{total}} = \mathcal{L} + \alpha \sum_{l=1}^{3} ||\mathbf{W}_l||_2^2
$$

onde $\alpha = 0.0001$ é o coeficiente de regularização.

**Otimizador Adam**:
$$
\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1) \nabla_\theta \mathcal{L}_t
$$

$$
\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2) (\nabla_\theta \mathcal{L}_t)^2
$$

$$
\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}
$$

$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}
$$

onde:
- $\beta_1 = 0.9$, $\beta_2 = 0.999$ são os momentos de decaimento
- $\eta = 0.001$ é a taxa de aprendizado
- $\epsilon = 10^{-8}$ é um termo de estabilização

#### 5. Reinforcement Learning (PPO)

**Normalização de Ações (Softmax)**:
$$
w_i = \frac{e^{a_i}}{\sum_{j=1}^{N} e^{a_j}}
$$

onde:
- $a_i$ são os logits da ação (antes da normalização)
- $w_i$ são os pesos normalizados do portfólio
- $\sum_{i=1}^{N} w_i = 1$ (restrição de soma unitária)

**Retorno do Portfólio**:
$$
R_p(t) = \sum_{i=1}^{N} w_i(t) \cdot r_i(t+1)
$$

**Retorno Líquido (após custos de transação)**:
$$
R_{\text{net}}(t) = R_p(t) - \tau \cdot \sum_{i=1}^{N} |w_i(t) - w_i(t-1)|
$$

onde $\tau = 0.001$ (0.1%) é a taxa de custo de transação.

**Sharpe Ratio Local**:
$$
\text{SR}(t) = \frac{\bar{R}_{\text{net}} - r_f}{\sigma_{R_{\text{net}}}}
$$

onde:
- $\bar{R}_{\text{net}} = \frac{1}{W} \sum_{i=t-W+1}^{t} R_{\text{net}}(i)$ é a média dos retornos líquidos em uma janela de $W$ períodos
- $\sigma_{R_{\text{net}}} = \sqrt{\frac{1}{W-1} \sum_{i=t-W+1}^{t} (R_{\text{net}}(i) - \bar{R}_{\text{net}})^2}$ é o desvio padrão
- $r_f$ é a taxa livre de risco (assumida como 0)

**Recompensa**:
$$
r_t = \text{SR}(t) - \lambda_{\text{turnover}} \cdot \text{Turnover}(t)
$$

onde:
$$
\text{Turnover}(t) = \sum_{i=1}^{N} |w_i(t) - w_i(t-1)|
$$

e $\lambda_{\text{turnover}}$ é o coeficiente de penalização por turnover.

**EWMA (Exponentially Weighted Moving Average) da Volatilidade**:
$$
\sigma_{\text{EWMA}}(t) = \alpha \cdot |r(t)| + (1-\alpha) \cdot \sigma_{\text{EWMA}}(t-1)
$$

onde $\alpha$ é o fator de decaimento (tipicamente 0.1-0.3).

**Função de Valor (Value Function)**:
$$
V^\pi(s_t) = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \mid s_t \right]
$$

onde:
- $\gamma = 0.99$ é o fator de desconto
- $\pi$ é a política (policy)

**GAE (Generalized Advantage Estimation)**:
$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

$$
\hat{A}_t = \delta_t + (\gamma \lambda) \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + \cdots
$$

onde $\lambda = 0.95$ é o parâmetro GAE.

**Função Objetivo do PPO**:
$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

onde:
$$
r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}
$$

é a razão de probabilidades e $\epsilon = 0.2$ é o clipping range.

**Função Objetivo Total**:
$$
L(\theta) = L^{\text{CLIP}}(\theta) - c_1 L^{\text{VF}}(\theta) + c_2 S[\pi_\theta](s_t)
$$

onde:
- $L^{\text{VF}}(\theta) = (V_\theta(s_t) - \hat{V}_t)^2$ é a função de perda do value network
- $S[\pi_\theta](s_t) = -\sum_{a} \pi_\theta(a|s_t) \log \pi_\theta(a|s_t)$ é a entropia (para exploração)
- $c_1 = 0.5$ e $c_2 = 0.01$ são coeficientes

#### 6. Normalização de Features

**StandardScaler**:
$$
x_{\text{normalized}} = \frac{x - \mu}{\sigma}
$$

onde:
- $\mu = \frac{1}{N} \sum_{i=1}^{N} x_i$ é a média
- $\sigma = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (x_i - \mu)^2}$ é o desvio padrão

#### 7. Métricas de Avaliação

**Accuracy**:
$$
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
$$

**F1-Score (Macro)**:
$$
F_1 = \frac{1}{C} \sum_{k=1}^{C} F_{1,k}
$$

onde:
$$
F_{1,k} = \frac{2 \cdot \text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}
$$

**Sharpe Ratio (Global)**:
$$
\text{Sharpe} = \frac{\bar{R} - r_f}{\sigma_R} \cdot \sqrt{T}
$$

onde $T$ é o número de períodos (anualizado).

**Drawdown Máximo**:
$$
\text{DD}(t) = \frac{\text{Peak}(t) - V(t)}{\text{Peak}(t)}
$$

$$
\text{MDD} = \max_t \text{DD}(t)
$$

onde:
- $\text{Peak}(t) = \max_{s \leq t} V(s)$ é o valor máximo até o período $t$
- $V(t)$ é o valor do portfólio no período $t$

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