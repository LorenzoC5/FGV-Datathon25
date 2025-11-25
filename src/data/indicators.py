import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# Adicionar diretório raiz ao path para imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

from src.data.data_fetcher import fetch_binance_klines_multiple, BINANCE_SYMBOLS


# ============================================================
# Configuração básica
# ============================================================

# Período total
GLOBAL_START = "2020-01-01"
GLOBAL_END = "2025-12-31"

# Split treino / teste
TRAIN_END = "2023-12-31"  # inclusive (treino: 2020-2023)
TEST_START = "2024-01-01"  # teste: 2024-2025

# Diretório de saída
OUTPUT_DIR = "data/processed"


# ============================================================
# Cálculo de indicadores técnicos
# ============================================================

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula um conjunto de indicadores técnicos para um DataFrame OHLCV.

    Cada linha (timestamp) será representada por um vetor de indicadores
    calculados usando apenas informações até aquele momento (sem look-ahead).

    Conjunto alvo: 15 indicadores, incluindo preço e relações preço-volume.

    Indicadores calculados (antes da filtragem de ortogonalidade):
      1.  price_close          - preço de fechamento
      2.  ret_1                - log-retorno 1 período
      3.  ret_5                - log-retorno acumulado 5 períodos
      4.  ret_20               - log-retorno acumulado 20 períodos
      5.  sma_10_ratio         - close / SMA(10)
      6.  sma_50_ratio         - close / SMA(50)
      7.  ema_10_ratio         - close / EMA(10)
      8.  ema_50_ratio         - close / EMA(50)
      9.  vol_10               - desvio padrão dos retornos em 10 períodos
      10. vol_50               - desvio padrão dos retornos em 50 períodos
      11. rsi_14               - RSI clássico
      12. bb_width_20          - largura das bandas de Bollinger (20)
      13. bb_pos_20            - posição relativa nas bandas (20)
      14. volume_zscore_20     - z-score de volume (20)
      15. pv_corr_20           - correlação 20 períodos entre retorno e variação de volume
    """
    df = df.copy()

    features = pd.DataFrame(index=df.index)

    # Preço (obrigatório)
    features["price_close"] = df["close"]

    # Retornos logarítmicos
    ret_1 = np.log(df["close"] / df["close"].shift(1))
    features["ret_1"] = ret_1
    features["ret_5"] = np.log(df["close"] / df["close"].shift(5))
    features["ret_20"] = np.log(df["close"] / df["close"].shift(20))

    # Médias móveis simples
    sma_10 = df["close"].rolling(10).mean()
    sma_50 = df["close"].rolling(50).mean()
    features["sma_10_ratio"] = df["close"] / sma_10
    features["sma_50_ratio"] = df["close"] / sma_50

    # EMAs
    ema_10 = df["close"].ewm(span=10, adjust=False).mean()
    ema_50 = df["close"].ewm(span=50, adjust=False).mean()
    features["ema_10_ratio"] = df["close"] / ema_10
    features["ema_50_ratio"] = df["close"] / ema_50

    # Volatilidade dos retornos
    features["vol_10"] = ret_1.rolling(10).std()
    features["vol_50"] = ret_1.rolling(50).std()

    # RSI 14
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    features["rsi_14"] = 100 - (100 / (1 + rs))

    # Bandas de Bollinger 20
    sma_20 = df["close"].rolling(20).mean()
    std_20 = df["close"].rolling(20).std()
    upper_20 = sma_20 + 2 * std_20
    lower_20 = sma_20 - 2 * std_20
    features["bb_width_20"] = (upper_20 - lower_20) / sma_20
    features["bb_pos_20"] = (df["close"] - lower_20) / (upper_20 - lower_20)

    # Volume z-score (20)
    vol_ma_20 = df["volume"].rolling(20).mean()
    vol_std_20 = df["volume"].rolling(20).std()
    features["volume_zscore_20"] = (df["volume"] - vol_ma_20) / vol_std_20

    # Correlação retorno x variação de volume (20 períodos)
    vol_change = df["volume"].pct_change()
    features["pv_corr_20"] = ret_1.rolling(20).corr(vol_change)

    # Remover linhas iniciais com NaNs
    features = features.dropna()

    return features


# ============================================================
# Seleção de indicadores quase ortogonais
# ============================================================

def select_orthogonal_features(
    features: pd.DataFrame,
    mandatory: List[str],
    corr_threshold: float = 0.3,
) -> List[str]:
    """
    Seleciona subconjunto de indicadores com baixa correlação entre si.

    Estratégia:
      - Mantém todos os indicadores obrigatórios (mandatory), por exemplo, price_close.
      - Para os demais, usa um algoritmo guloso:
          * Ordena os candidatos por variância decrescente.
          * Adiciona um candidato se |correlation| com todos os já selecionados < corr_threshold.
    """
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()

    # Garantir que os obrigatórios estão presentes
    mandatory = [c for c in mandatory if c in numeric_cols]

    # Candidatos são todos os numéricos exceto os obrigatórios
    candidates = [c for c in numeric_cols if c not in mandatory]

    # Matriz de correlação no conjunto de treino
    corr = features[numeric_cols].corr()

    selected = list(mandatory)

    # Ordenar candidatos por variância (descrescente)
    var_order = features[candidates].var().sort_values(ascending=False).index.tolist()

    for col in var_order:
        ok = True
        for s in selected:
            if abs(corr.loc[col, s]) > corr_threshold:
                ok = False
                break
        if ok:
            selected.append(col)

    return selected


# ============================================================
# Construção de DataFrame de indicadores para múltiplas moedas
# ============================================================

def build_indicator_dataset(
    symbols: List[str],
    fetch_interval: str,
    start_time: str,
    end_time: str,
    train_end: str,
    target_freq: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constrói dataset de indicadores para todas as moedas e separa em treino e teste.

    Cada linha do DataFrame final representa:
        - timestamp
        - symbol
        - vetor de indicadores técnicos até aquele momento.

    Parâmetros:
        fetch_interval: intervalo utilizado para buscar dados na Binance
                        (ex: '1h' para estratégia diária, '1w' para semanal).
        target_freq: frequência alvo para os indicadores.
            - None  -> usa a própria frequência de fetch (ex: semanal direto em '1w').
            - '1D'  -> agrega indicadores intradiários para frequência diária
                      usando o último valor de cada dia.

    A seleção de indicadores quase ortogonais é feita com base apenas nos dados de treino.
    """
    print(f"\nBuscando dados para intervalo {fetch_interval}...")
    klines_data = fetch_binance_klines_multiple(
        symbols=symbols,
        interval=fetch_interval,
        start_time=start_time,
        end_time=end_time,
        limit=50000,
        delay=0.1,
    )

    all_features = []

    for symbol in symbols:
        if symbol not in klines_data:
            print(f"Aviso: sem dados para {symbol} no intervalo {fetch_interval}")
            continue

        df_ohlcv = klines_data[symbol]
        feats = compute_indicators(df_ohlcv)

        # Se for estratégia diária com dados intradiários, agregamos para '1D'
        if target_freq is not None:
            # Usar o último valor de cada dia (todos os indicadores são filtrados sem look-ahead)
            feats = feats.resample(target_freq).last()

        if feats.empty:
            print(f"Aviso: indicadores vazios para {symbol} no intervalo {fetch_interval}")
            continue

        feats = feats.copy()
        feats["symbol"] = symbol.replace("USDT", "").replace("USD", "")
        feats["timestamp"] = feats.index

        all_features.append(feats)

    if not all_features:
        raise ValueError(f"Nenhum indicador calculado para intervalo {fetch_interval}")

    full_df = pd.concat(all_features, axis=0).reset_index(drop=True)

    # Separar treino e teste por data
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"])
    train_mask = full_df["timestamp"] <= pd.to_datetime(train_end)

    train_df = full_df[train_mask].copy()
    test_df = full_df[~train_mask].copy()

    # Selecionar indicadores quase ortogonais usando apenas o treino
    feature_cols = [
        c
        for c in train_df.columns
        if c not in ["symbol", "timestamp"]
    ]

    mandatory = ["price_close"]  # preço deve estar entre os indicadores

    selected_features = select_orthogonal_features(
        train_df[feature_cols], mandatory=mandatory, corr_threshold=0.3
    )

    print(f"\nIntervalo de busca {fetch_interval}, frequência alvo {target_freq or 'raw'}:")
    print(f"- Total de indicadores calculados: {len(feature_cols)}")
    print(f"- Indicadores selecionados (quase ortogonais): {len(selected_features)}")
    print(f"- Indicadores: {selected_features}")

    # Manter apenas colunas selecionadas + symbol + timestamp
    cols_to_keep = ["timestamp", "symbol"] + selected_features
    train_df = train_df[cols_to_keep].sort_values(["symbol", "timestamp"])
    test_df = test_df[cols_to_keep].sort_values(["symbol", "timestamp"])

    return train_df, test_df


# ============================================================
# Pipeline principal
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Usar as mesmas moedas definidas em src.data.data_fetcher.BINANCE_SYMBOLS
    symbols = BINANCE_SYMBOLS
    print("Símbolos utilizados:", symbols)

    # Estratégia: apenas semanal
    configs = {
        # Estratégia semanal: usar dados já semanais (1w) diretamente
        "weekly": {
            "interval_label": "1w",
            "fetch_interval": "1w",
            "target_freq": None,
            "suffix": "weekly",
        },
    }

    for key, cfg in configs.items():
        interval_label = cfg["interval_label"]
        fetch_interval = cfg["fetch_interval"]
        target_freq = cfg["target_freq"]
        suffix = cfg["suffix"]

        print("\n" + "=" * 60)
        print(f"Construindo dataset de indicadores - {suffix.upper()} ({interval_label})")
        print("=" * 60)

        train_df, test_df = build_indicator_dataset(
            symbols=symbols,
            fetch_interval=fetch_interval,
            start_time=GLOBAL_START,
            end_time=GLOBAL_END,
            train_end=TRAIN_END,
            target_freq=target_freq,
        )

        # Salvar em arquivos separados
        train_path = os.path.join(OUTPUT_DIR, f"indicators_{suffix}_train_2020_2023.csv")
        test_path = os.path.join(OUTPUT_DIR, f"indicators_{suffix}_test_2024_2025.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"\nArquivos salvos para {suffix}:")
        print(f"- Treino: {train_path}")
        print(f"- Teste: {test_path}")

    print("\n" + "=" * 60)
    print("Pipeline de indicadores concluído com sucesso.")
    print(f"Arquivos disponíveis no diretório: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()


