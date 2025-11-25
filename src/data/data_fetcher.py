import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from scipy.optimize import minimize, differential_evolution
from typing import Optional, Union
from datetime import datetime, timedelta
import requests
import time
import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Union
import matplotlib
matplotlib.use('Agg')  # Usar backend não-interativo para não mostrar figuras
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, ConnectionPatch
from matplotlib.patches import Circle, Ellipse
import matplotlib.patheffects as path_effects

# ============================================================
# Configuração: Lista das 10 criptomoedas para análise
# ============================================================
# Lista das 10 criptomoedas especificadas
# Nota: USDT foi substituído por referência a dólar (USD)
CRYPTO_LIST = ['ADA', 'BNB', 'BTC', 'ETH', 'SOL', 'TRX', 'USDC', 'XRP', 'DOGE', 'USD']

# Converter para formato Binance (adicionar USD como par de negociação)
# Na Binance, os pares são principalmente contra USDT, mas usaremos USD como referência
# Nota: USDC é uma stablecoin e pode não ter par negociável, mas tentaremos buscar
BINANCE_SYMBOLS = []
for crypto in CRYPTO_LIST:
    if crypto == 'USD':
        # Para USD, não há par negociável direto, então pulamos
        continue
    elif crypto == 'USDC':
        # USDC pode ter par USDCUSDT
        BINANCE_SYMBOLS.append('USDCUSDT')
    else:
        # Usar USDT como par (que é o padrão na Binance, equivalente a dólar)
        BINANCE_SYMBOLS.append(f'{crypto}USDT')


def get_binance_symbols(cryptos: Optional[List[str]] = None) -> List[str]:
    """
    Retorna a lista de símbolos Binance para as criptomoedas especificadas.
    
    Args:
        cryptos (List[str], optional): Lista de criptomoedas. Se None, usa CRYPTO_LIST completa.
    
    Returns:
        List[str]: Lista de símbolos no formato Binance (ex: ['BTCUSDT', 'ETHUSDT', ...])
    """
    if cryptos is None:
        return BINANCE_SYMBOLS.copy()
    
    symbols = []
    for crypto in cryptos:
        crypto_upper = crypto.upper()
        if crypto_upper == 'USD':
            # Para USD, não há par negociável direto, então pulamos
            continue
        elif crypto_upper == 'USDC':
            # USDC pode ter par USDCUSDT
            symbols.append('USDCUSDT')
        else:
            # Usar USDT como par (equivalente a dólar)
            symbols.append(f'{crypto_upper}USDT')
    return symbols


def save_crypto_data(data: Dict[str, pd.DataFrame], filepath: str = 'crypto_data.pkl'):
    """
    Salva dados de criptomoedas em arquivo pickle.
    
    Args:
        data (Dict[str, pd.DataFrame]): Dicionário com símbolos como chaves e DataFrames como valores.
        filepath (str): Caminho do arquivo para salvar. Padrão: 'crypto_data.pkl'.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Dados salvos em {filepath}")


def load_crypto_data(filepath: str = 'crypto_data.pkl') -> Dict[str, pd.DataFrame]:
    """
    Carrega dados de criptomoedas de arquivo pickle.
    
    Args:
        filepath (str): Caminho do arquivo para carregar. Padrão: 'crypto_data.pkl'.
    
    Returns:
        Dict[str, pd.DataFrame]: Dicionário com símbolos como chaves e DataFrames como valores.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo {filepath} não encontrado")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"Dados carregados de {filepath}")
    return data


def get_cache_filename(symbols: List[str], interval: str, start_time: Optional[str], 
                       end_time: Optional[str]) -> str:
    """
    Gera nome de arquivo de cache baseado nos parâmetros.
    
    Args:
        symbols: Lista de símbolos
        interval: Intervalo de tempo
        start_time: Data de início
        end_time: Data de fim
    
    Returns:
        str: Nome do arquivo de cache
    """
    import hashlib
    
    # Criar hash dos símbolos para evitar nomes muito longos
    symbols_str = '_'.join(sorted(symbols))
    if len(symbols_str) > 50:
        symbols_hash = hashlib.md5(symbols_str.encode()).hexdigest()[:8]
        symbols_str = f"all_{symbols_hash}"
    
    # Simplificar datas
    start_str = str(start_time)[:10].replace('-', '') if start_time else 'None'
    end_str = str(end_time)[:10].replace('-', '') if end_time else 'None'
    
    filename = f"crypto_data_{symbols_str}_{interval}_{start_str}_{end_str}.pkl"
    # Remover caracteres inválidos para nome de arquivo
    filename = filename.replace('/', '-').replace(':', '-').replace(' ', '_')
    return filename


def fetch_binance_klines(
    symbol: str,
    interval: str = '1h',
    start_time: Optional[Union[str, datetime]] = None,
    end_time: Optional[Union[str, datetime]] = None,
    limit: int = 1000
) -> pd.DataFrame:
    """
    Busca dados de klines (candles) da API pública da Binance.
    Se o período solicitado exceder 1000 candles, faz múltiplas requisições automaticamente.
    
    Args:
        symbol (str): Par de negociação (ex: 'BTCUSDT', 'ETHUSDT'). Use formato sem separador.
        interval (str): Intervalo de tempo. Opções: '1m', '3m', '5m', '15m', '30m', '1h', '2h', 
                        '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'. Padrão: '1h' (hora em hora).
        start_time (str, datetime, optional): Data/hora de início. Se None, busca desde o início disponível.
        end_time (str, datetime, optional): Data/hora de fim. Se None, usa a data/hora atual.
        limit (int): Número máximo de candles desejado. Se > 1000, faz múltiplas requisições. Padrão: 1000.
    
    Returns:
        pd.DataFrame: DataFrame com colunas ['open', 'high', 'low', 'close', 'volume']
                     e índice DatetimeIndex.
    
    Example:
        >>> df = fetch_binance_klines('BTCUSDT', interval='1h', start_time='2023-01-01', limit=2000)
    """
    
    base_url = "https://api.binance.com/api/v3/klines"
    max_limit_per_request = 1000
    all_data = []
    
    # Converter start_time e end_time para datetime
    if start_time is not None:
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
    if end_time is not None:
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)
    else:
        end_time = datetime.now()
    
    current_start = start_time
    remaining_limit = limit
    
    # Fazer requisições múltiplas se necessário
    while remaining_limit > 0 and (current_start is None or current_start < end_time):
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': min(remaining_limit, max_limit_per_request)
        }
        
        if current_start is not None:
            params['startTime'] = int(current_start.timestamp() * 1000)
        
        if end_time is not None:
            params['endTime'] = int(end_time.timestamp() * 1000)
        
        # Fazer requisição à API
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Erro ao buscar dados da Binance para {symbol}: {e}")
        
        if not data:
            break
        
        all_data.extend(data)
        remaining_limit -= len(data)
        
        # Se recebemos menos dados do que solicitamos, chegamos ao fim
        if len(data) < max_limit_per_request:
            break
        
        # Atualizar start_time para a próxima requisição (último timestamp + 1 intervalo)
        if current_start is not None:
            last_timestamp = pd.to_datetime(data[-1][0], unit='ms')
            # Calcular próximo intervalo baseado no intervalo especificado
            interval_map = {
                '1m': timedelta(minutes=1), '3m': timedelta(minutes=3), '5m': timedelta(minutes=5),
                '15m': timedelta(minutes=15), '30m': timedelta(minutes=30), '1h': timedelta(hours=1),
                '2h': timedelta(hours=2), '4h': timedelta(hours=4), '6h': timedelta(hours=6),
                '8h': timedelta(hours=8), '12h': timedelta(hours=12), '1d': timedelta(days=1),
                '3d': timedelta(days=3), '1w': timedelta(weeks=1), '1M': timedelta(days=30)
            }
            interval_delta = interval_map.get(interval, timedelta(hours=1))
            current_start = last_timestamp + interval_delta
        else:
            break
        
        # Pequeno delay para evitar rate limiting
        time.sleep(0.1)
    
    if not all_data:
        raise ValueError(f"Nenhum dado retornado para {symbol} no período especificado")
    
    # Converter para DataFrame
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Converter tipos e processar timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    
    # Remover duplicatas (caso haja sobreposição)
    df = df.drop_duplicates(subset=['timestamp'])
    
    # Ordenar por timestamp
    df = df.sort_values('timestamp')
    
    # Usar timestamp como índice
    df.set_index('timestamp', inplace=True)
    
    # Retornar apenas as colunas principais
    return df[['open', 'high', 'low', 'close', 'volume']]


def fetch_binance_klines_multiple(
    symbols: List[str],
    interval: str = '1h',
    start_time: Optional[Union[str, datetime]] = None,
    end_time: Optional[Union[str, datetime]] = None,
    limit: int = 1000,
    delay: float = 0.1
) -> Dict[str, pd.DataFrame]:
    """
    Busca dados de klines para múltiplos símbolos da Binance.
    
    Args:
        symbols (List[str]): Lista de pares de negociação (ex: ['BTCUSDT', 'ETHUSDT']).
        interval (str): Intervalo de tempo. Padrão: '1h'.
        start_time (str, datetime, optional): Data/hora de início.
        end_time (str, datetime, optional): Data/hora de fim.
        limit (int): Número máximo de candles por requisição.
        delay (float): Delay entre requisições (em segundos) para evitar rate limiting. Padrão: 0.1s.
    
    Returns:
        Dict[str, pd.DataFrame]: Dicionário com símbolos como chaves e DataFrames como valores.
    """
    
    results = {}
    
    for symbol in symbols:
        try:
            print(f"Buscando dados para {symbol}...")
            df = fetch_binance_klines(symbol, interval, start_time, end_time, limit)
            results[symbol] = df
            time.sleep(delay)  # Delay para evitar rate limiting
        except Exception as e:
            print(f"Erro ao buscar {symbol}: {e}")
            continue
    
    return results


def calculate_features_from_klines(df: pd.DataFrame, periods: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
    """
    Calcula features técnicas a partir de dados de klines da Binance.
    
    NOTA: Esta função não é mais usada no pipeline principal, pois os bins são criados
    apenas com base nos retornos. Mantida apenas para referência ou uso futuro.
    
    Args:
        df (pd.DataFrame): DataFrame com colunas ['open', 'high', 'low', 'close', 'volume'].
        periods (List[int]): Períodos para médias móveis e outras features. Padrão: [5, 10, 20, 50].
    
    Returns:
        pd.DataFrame: DataFrame com features calculadas.
    """
    
    features_df = pd.DataFrame(index=df.index)
    
    # Retornos (log returns)
    features_df['return'] = np.log(df['close'] / df['close'].shift(1))
    features_df['return_5'] = np.log(df['close'] / df['close'].shift(5))
    features_df['return_10'] = np.log(df['close'] / df['close'].shift(10))
    
    # Médias móveis simples
    for period in periods:
        features_df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        features_df[f'price_sma_ratio_{period}'] = df['close'] / features_df[f'sma_{period}']
    
    # Médias móveis exponenciais
    for period in [12, 26]:
        features_df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
    # MACD
    if 12 in [12, 26] and 26 in [12, 26]:
        features_df['macd'] = features_df['ema_12'] - features_df['ema_26']
        features_df['macd_signal'] = features_df['macd'].ewm(span=9, adjust=False).mean()
        features_df['macd_diff'] = features_df['macd'] - features_df['macd_signal']
    
    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features_df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    for period in [20]:
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        features_df[f'bb_upper_{period}'] = sma + (std * 2)
        features_df[f'bb_lower_{period}'] = sma - (std * 2)
        features_df[f'bb_width_{period}'] = (features_df[f'bb_upper_{period}'] - features_df[f'bb_lower_{period}']) / sma
        features_df[f'bb_position_{period}'] = (df['close'] - features_df[f'bb_lower_{period}']) / (features_df[f'bb_upper_{period}'] - features_df[f'bb_lower_{period}'])
    
    # Volatilidade (rolling standard deviation dos retornos)
    for period in [5, 10, 20]:
        features_df[f'volatility_{period}'] = features_df['return'].rolling(window=period).std()
    
    # Volume features
    features_df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    features_df['volume_ratio'] = df['volume'] / features_df['volume_sma_20']
    
    # High-Low range
    features_df['hl_range'] = (df['high'] - df['low']) / df['close']
    features_df['hl_range_5'] = features_df['hl_range'].rolling(window=5).mean()
    
    # Remover colunas auxiliares que não são features
    cols_to_remove = ['ema_12', 'ema_26', 'macd', 'macd_signal', 'bb_upper_20', 'bb_lower_20']
    for col in cols_to_remove:
        if col in features_df.columns:
            features_df.drop(columns=[col], inplace=True)
    
    # Remover linhas com NaN (resultado de cálculos de médias móveis)
    features_df.dropna(inplace=True)
    
    return features_df


def prepare_binance_data_for_training(
    symbols: List[str],
    interval: str = '1h',
    start_time: Optional[Union[str, datetime]] = None,
    end_time: Optional[Union[str, datetime]] = None,
    limit: int = 1000,
    use_cache: bool = True,
    cache_filepath: Optional[str] = None,
    save_cache: bool = True
) -> tuple:
    """
    Busca dados da Binance, calcula APENAS os retornos e prepara no formato para treinamento.
    Suporta cache para evitar requisições repetidas.
    
    NOTA: Esta função calcula apenas os retornos (log returns) a partir dos preços de fechamento.
    Não calcula features técnicas, pois os bins são criados apenas com base nos retornos.
    
    Args:
        symbols (List[str]): Lista de símbolos (ex: ['BTCUSDT', 'ETHUSDT']).
        interval (str): Intervalo de tempo. Padrão: '1h'.
        start_time (str, datetime, optional): Data/hora de início.
        end_time (str, datetime, optional): Data/hora de fim.
        limit (int): Número máximo de candles por requisição.
        use_cache (bool): Se True, tenta carregar dados do cache antes de fazer requisições.
        cache_filepath (str, optional): Caminho do arquivo de cache. Se None, gera automaticamente.
        save_cache (bool): Se True, salva os dados em cache após buscar da API.
    
    Returns:
        tuple: (X_train_data, Y_train_data) onde:
               - X_train_data: dict com símbolos como chaves e DataFrames vazios (apenas para compatibilidade)
               - Y_train_data: dict com símbolos como chaves e Series de retornos (log returns)
    """
    
    # Tentar carregar do cache
    if use_cache:
        if cache_filepath is None:
            start_str = str(start_time) if start_time else None
            end_str = str(end_time) if end_time else None
            cache_filepath = get_cache_filename(symbols, interval, start_str, end_str)
        
        if os.path.exists(cache_filepath):
            try:
                print(f"Carregando dados do cache: {cache_filepath}")
                cached_data = load_crypto_data(cache_filepath)
                # Verificar se temos todos os símbolos necessários
                if all(symbol in cached_data for symbol in symbols):
                    print("Todos os dados encontrados no cache!")
                    klines_data = cached_data
                else:
                    print("Cache incompleto, buscando dados da Binance...")
                    klines_data = None
            except Exception as e:
                print(f"Erro ao carregar cache: {e}. Buscando dados da Binance...")
                klines_data = None
        else:
            print(f"Cache não encontrado: {cache_filepath}. Buscando dados da Binance...")
            klines_data = None
    else:
        klines_data = None
    
    # Buscar dados de klines se não encontrados no cache
    if klines_data is None:
        print("Buscando dados da Binance...")
        klines_data = fetch_binance_klines_multiple(symbols, interval, start_time, end_time, limit)
        
        if not klines_data:
            raise ValueError("Nenhum dado foi retornado da Binance")
        
        # Salvar no cache se solicitado
        if save_cache and use_cache:
            if cache_filepath is None:
                start_str = str(start_time) if start_time else None
                end_str = str(end_time) if end_time else None
                cache_filepath = get_cache_filename(symbols, interval, start_str, end_str)
            save_crypto_data(klines_data, cache_filepath)
    
    X_train_data = {}
    Y_train_data = {}
    
    # Processar cada símbolo
    for symbol in symbols:
        if symbol not in klines_data:
            print(f"Aviso: {symbol} não encontrado nos dados retornados")
            continue
        
        df = klines_data[symbol]
        
        # Calcular apenas os retornos (log returns)
        print(f"Calculando retornos para {symbol}...")
        returns = np.log(df['close'] / df['close'].shift(1))
        
        # Remover primeira linha (que tem NaN devido ao shift)
        returns = returns.iloc[1:]
        
        # Remover NaN restantes (se houver)
        returns = returns.dropna()
        
        if len(returns) == 0:
            print(f"Aviso: Nenhum retorno válido calculado para {symbol}")
            continue
        
        # Usar nome simplificado (ex: 'BTCUSDT' -> 'BTC', 'USDCUSDT' -> 'USDC')
        # Tratar USDC antes de remover USD para evitar que vire 'C'
        if symbol == 'USDCUSDT':
            crypto_name = 'USDC'
        else:
            crypto_name = symbol.replace('USDT', '').replace('USD', '')
        
        # Criar DataFrame vazio para X (apenas para manter compatibilidade com a função de bins)
        # Mas não será usado para criar os bins, apenas para alinhamento temporal
        X_dummy = pd.DataFrame(index=returns.index)
        
        X_train_data[crypto_name] = X_dummy
        Y_train_data[crypto_name] = returns
        
        print(f"{crypto_name}: {len(returns)} retornos calculados")
    
    return X_train_data, Y_train_data


# ============================================================
# MÉTODO 1: BINNING SUPERVISIONADO (DECISION TREE)
# ============================================================
def create_supervised_static_bins(
    X_train_data: dict, 
    Y_train_data: dict, 
    datastart: Optional[Union[str, pd.Timestamp]] = None, 
    dataend: Optional[Union[str, pd.Timestamp]] = None
) -> dict:
    """
    Transforma um problema de regressão em classificação usando particionamento supervisionado
    com DecisionTreeRegressor, focado APENAS nos retornos (Y_train_data).
    
    Args:
        X_train_data (dict): Dicionário de features (usado apenas para filtragem temporal).
        Y_train_data (dict): Dicionário de retornos contínuos (usado para criar os bins).
        datastart (str, datetime, optional): Data/hora de início do período de treino.
        dataend (str, datetime, optional): Data/hora de fim do período de treino.
    
    Returns:
        dict: Dicionário com bins por criptomoeda (ex: [-inf, c1, c2, c3, c4, +inf])
    """
    
    # Converter datastart e dataend para datetime se forem strings
    if datastart is not None and isinstance(datastart, str):
        datastart = pd.to_datetime(datastart)
    if dataend is not None and isinstance(dataend, str):
        dataend = pd.to_datetime(dataend)
    
    bin_intervals_dict = {}
    
    # Iterar por cada criptomoeda nos dicionários de entrada
    for crypto_name in X_train_data.keys():
        
        # Verificar se a criptomoeda existe em ambos os dicionários
        if crypto_name not in Y_train_data:
            print(f"[AVISO] {crypto_name}: Nao encontrado em Y_train_data, pulando.")
            continue
        
        # Extrair dados de treino para esta criptomoeda
        X_train = X_train_data[crypto_name].copy()
        y_train_cont = Y_train_data[crypto_name].copy()
        
        # ============================================================
        # FILTRAGEM TEMPORAL
        # ============================================================
        has_datetime_index_X = isinstance(X_train, pd.DataFrame) and isinstance(X_train.index, pd.DatetimeIndex)
        has_datetime_index_y = isinstance(y_train_cont, pd.Series) and isinstance(y_train_cont.index, pd.DatetimeIndex)
        
        if has_datetime_index_X and has_datetime_index_y:
            if not X_train.index.equals(y_train_cont.index):
                common_index = X_train.index.intersection(y_train_cont.index)
                if len(common_index) == 0:
                    print(f"[AVISO] {crypto_name}: Indices X e y nao se sobrepoem, pulando.")
                    continue
                X_train = X_train.loc[common_index]
                y_train_cont = y_train_cont.loc[common_index]
        
        if datastart is not None or dataend is not None:
            if has_datetime_index_X and has_datetime_index_y:
                if datastart is not None:
                    X_train = X_train[X_train.index >= datastart]
                    y_train_cont = y_train_cont[y_train_cont.index >= datastart]
                if dataend is not None:
                    X_train = X_train[X_train.index <= dataend]
                    y_train_cont = y_train_cont[y_train_cont.index <= dataend]
                
                if len(X_train) == 0 or len(y_train_cont) == 0:
                    print(f"[AVISO] {crypto_name}: Nenhum dado no periodo {datastart} a {dataend}, pulando.")
                    continue
            else:
                import warnings
                warnings.warn(f"Dados para '{crypto_name}' nao tem indices datetime. Filtragem temporal ignorada.")
        
        y_train_values = y_train_cont.values if isinstance(y_train_cont, pd.Series) else y_train_cont
        
        if len(y_train_values) < 5:
            print(f"[AVISO] {crypto_name}: Dados insuficientes ({len(y_train_values)} amostras), pulando.")
            continue
        
        # ============================================================
        # FASE 1: Particionamento Supervisionado (APENAS RETORNOS)
        # ============================================================
        # Usamos o próprio retorno como "feature" para a árvore
        X_dummy = y_train_values.reshape(-1, 1)  # Formato (n_samples, 1)
        
        tree = DecisionTreeRegressor(
            max_leaf_nodes=5,
            random_state=42
        )
        tree.fit(X_dummy, y_train_values)
        
        # ============================================================
        # FASE 2: Extração e Ordenação dos Bins (As "Folhas")
        # ============================================================
        leaf_ids = tree.apply(X_dummy)
        
        df_temp = pd.DataFrame({
            'leaf_id': leaf_ids,
            'return': y_train_values
        })
        
        leaf_stats = df_temp.groupby('leaf_id')['return'].agg(['min', 'max', 'mean']).reset_index()
        leaf_stats_sorted = leaf_stats.sort_values('mean').reset_index(drop=True)
        
        num_leaves = len(leaf_stats_sorted)
        if num_leaves != 5:
            print(f"[AVISO] {crypto_name}: Arvore gerou {num_leaves} folhas em vez de 5, pulando.")
            continue
        
        # ============================================================
        # FASE 3: Definição dos Limites Estáticos (Os "Intervalos")
        # ============================================================
        cut_points = []
        for i in range(num_leaves - 1):
            max_current_bin = leaf_stats_sorted.iloc[i]['max']
            min_next_bin = leaf_stats_sorted.iloc[i + 1]['min']
            cut_point = (max_current_bin + min_next_bin) / 2.0
            cut_points.append(cut_point)
        
        # ============================================================
        # FASE 4: Saída Final
        # ============================================================
        bin_intervals = [-np.inf] + cut_points + [np.inf]
        bin_intervals_dict[crypto_name] = np.array(bin_intervals)
        
        print(f"[OK] {crypto_name} (Tree): Bins criados")
    
    return bin_intervals_dict


def visualize_decision_tree_process(
    crypto_name: str,
    y_train_values: np.ndarray,
    tree: DecisionTreeRegressor,
    bin_intervals: np.ndarray,
    output_dir: str = 'decision_tree_graphs',
    save_plots: bool = True
):
    """
    Cria visualizações completas do processo de Decision Tree para binning.
    
    Args:
        crypto_name (str): Nome da criptomoeda
        y_train_values (np.ndarray): Valores de retorno usados para treinar a árvore
        tree (DecisionTreeRegressor): Árvore de decisão treinada
        bin_intervals (np.ndarray): Intervalos dos bins criados
        output_dir (str): Diretório para salvar os gráficos
        save_plots (bool): Se True, salva os gráficos em arquivos
    """
    # Criar diretório de saída se não existir
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
    
    # ============================================================
    # GRÁFICO 1: Distribuição de Retornos com Bin Boundaries
    # ============================================================
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Subplot 1: Histograma com bin boundaries
    ax1 = axes[0]
    n, bins_hist, patches = ax1.hist(y_train_values, bins=50, alpha=0.7, 
                                      color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Colorir as barras do histograma de acordo com os bins
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
    bin_colors = []
    
    # Determinar qual bin cada barra do histograma pertence
    for i, (bin_start, bin_end) in enumerate(zip(bins_hist[:-1], bins_hist[1:])):
        bin_center = (bin_start + bin_end) / 2
        # Encontrar qual bin interval contém este valor
        bin_idx = 0
        for j in range(len(bin_intervals) - 1):
            if bin_intervals[j] == -np.inf:
                if bin_center <= bin_intervals[j + 1]:
                    bin_idx = j
                    break
            elif bin_intervals[j + 1] == np.inf:
                if bin_center > bin_intervals[j]:
                    bin_idx = j
                    break
            else:
                if bin_intervals[j] < bin_center <= bin_intervals[j + 1]:
                    bin_idx = j
                    break
        patches[i].set_facecolor(colors[bin_idx % len(colors)])
        patches[i].set_alpha(0.6)
    
    # Desenhar linhas verticais para os cut points
    cut_points = bin_intervals[1:-1]  # Remover -inf e +inf
    for cut_point in cut_points:
        ax1.axvline(cut_point, color='red', linestyle='--', linewidth=2, 
                   label='Cut Point' if cut_point == cut_points[0] else '')
    
    # Adicionar anotações para os cut points
    for i, cut_point in enumerate(cut_points):
        ax1.text(cut_point, ax1.get_ylim()[1] * 0.95, f'c{i+1}\n{cut_point:.4f}',
                ha='center', va='top', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax1.set_xlabel('Retorno (Log Return)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequência', fontsize=12, fontweight='bold')
    ax1.set_title(f'Distribuição de Retornos com Bin Boundaries - {crypto_name}', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Adicionar estatísticas
    stats_text = f'Média: {np.mean(y_train_values):.6f}\n'
    stats_text += f'Desvio Padrão: {np.std(y_train_values):.6f}\n'
    stats_text += f'Min: {np.min(y_train_values):.6f}\n'
    stats_text += f'Max: {np.max(y_train_values):.6f}\n'
    stats_text += f'Número de amostras: {len(y_train_values)}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Subplot 2: Box plot por bin
    ax2 = axes[1]
    
    # Calcular estatísticas por bin
    leaf_ids = tree.apply(y_train_values.reshape(-1, 1))
    df_temp = pd.DataFrame({
        'leaf_id': leaf_ids,
        'return': y_train_values
    })
    leaf_stats = df_temp.groupby('leaf_id')['return'].agg(['min', 'max', 'mean', 'count']).reset_index()
    leaf_stats_sorted = leaf_stats.sort_values('mean').reset_index(drop=True)
    
    # Criar dados para box plot
    bin_data = []
    bin_labels = []
    for i, row in leaf_stats_sorted.iterrows():
        mask = (df_temp['leaf_id'] == row['leaf_id'])
        bin_data.append(df_temp[mask]['return'].values)
        bin_labels.append(f'Bin {i+1}\n(μ={row["mean"]:.4f})')
    
    bp = ax2.boxplot(bin_data, labels=bin_labels, patch_artist=True, 
                     showmeans=True, meanline=True)
    
    # Colorir os boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax2.set_ylabel('Retorno (Log Return)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Distribuição de Retornos por Bin - {crypto_name}', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Adicionar informações sobre cada bin
    info_text = 'Informações dos Bins:\n'
    for i, row in leaf_stats_sorted.iterrows():
        info_text += f'Bin {i+1}: n={int(row["count"])}, '
        info_text += f'μ={row["mean"]:.4f}, '
        info_text += f'[{row["min"]:.4f}, {row["max"]:.4f}]\n'
    
    ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, 
            fontsize=8, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_plots:
        filename = os.path.join(output_dir, f'decision_tree_bins_{crypto_name}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"[OK] Gráfico salvo: {filename}")
    
    plt.close()  # Fechar figura em vez de mostrar
    
    # ============================================================
    # GRÁFICO 2: Estrutura da Árvore de Decisão
    # ============================================================
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plotar a árvore
    plot_tree(tree, filled=True, feature_names=['Return'], 
              class_names=None, ax=ax, fontsize=10, 
              rounded=True, precision=4)
    
    ax.set_title(f'Estrutura da Árvore de Decisão - {crypto_name}', 
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_plots:
        filename = os.path.join(output_dir, f'decision_tree_structure_{crypto_name}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"[OK] Gráfico da árvore salvo: {filename}")
    
    plt.close()  # Fechar figura em vez de mostrar
    
    # ============================================================
    # GRÁFICO 3: Comparação Visual dos Bins
    # ============================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Criar um gráfico de barras horizontais mostrando os intervalos
    y_positions = np.arange(len(bin_intervals) - 1)
    bin_widths = []
    bin_centers = []
    bin_labels = []
    
    for i in range(len(bin_intervals) - 1):
        if bin_intervals[i] == -np.inf:
            left = np.min(y_train_values) - 0.1 * (np.max(y_train_values) - np.min(y_train_values))
        else:
            left = bin_intervals[i]
        
        if bin_intervals[i + 1] == np.inf:
            right = np.max(y_train_values) + 0.1 * (np.max(y_train_values) - np.min(y_train_values))
        else:
            right = bin_intervals[i + 1]
        
        width = right - left
        center = (left + right) / 2
        
        bin_widths.append(width)
        bin_centers.append(center)
        
        # Contar amostras neste bin
        if bin_intervals[i] == -np.inf:
            mask = (y_train_values <= bin_intervals[i + 1])
        elif bin_intervals[i + 1] == np.inf:
            mask = (y_train_values > bin_intervals[i])
        else:
            mask = (y_train_values > bin_intervals[i]) & (y_train_values <= bin_intervals[i + 1])
        
        count = np.sum(mask)
        mean_val = np.mean(y_train_values[mask]) if count > 0 else 0
        
        # Formatar labels dos intervalos
        left_str = f'{bin_intervals[i]:.4f}' if bin_intervals[i] != -np.inf else '-∞'
        right_str = f'{bin_intervals[i+1]:.4f}' if bin_intervals[i+1] != np.inf else '+∞'
        
        label = f'Bin {i+1}\n[{left_str}, {right_str}]\n'
        label += f'n={count}, μ={mean_val:.4f}'
        bin_labels.append(label)
    
    # Criar barras horizontais
    bars = ax.barh(y_positions, bin_widths, left=[c - w/2 for c, w in zip(bin_centers, bin_widths)],
                   color=colors[:len(bin_intervals)-1], alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Adicionar valores nas barras
    for i, (bar, center, width, label) in enumerate(zip(bars, bin_centers, bin_widths, bin_labels)):
        ax.text(center, i, label, ha='center', va='center', 
               fontsize=9, fontweight='bold', color='white')
    
    # Adicionar linha para a distribuição de retornos
    ax2_twin = ax.twiny()
    n, bins_hist, _ = ax2_twin.hist(y_train_values, bins=50, alpha=0.3, 
                                     color='gray', edgecolor='none')
    ax2_twin.set_xlabel('Densidade de Retornos', fontsize=10, color='gray')
    ax2_twin.tick_params(axis='x', labelcolor='gray')
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f'Bin {i+1}' for i in range(len(bin_intervals) - 1)])
    ax.set_xlabel('Retorno (Log Return)', fontsize=12, fontweight='bold')
    ax.set_title(f'Visualização dos Bins Criados pela Árvore de Decisão - {crypto_name}', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_plots:
        filename = os.path.join(output_dir, f'decision_tree_bins_comparison_{crypto_name}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"[OK] Gráfico de comparação salvo: {filename}")
    
    plt.close()  # Fechar figura em vez de mostrar
    
    # ============================================================
    # Imprimir texto da árvore
    # ============================================================
    print(f"\n{'='*60}")
    print(f"ÁRVORE DE DECISÃO - {crypto_name}")
    print(f"{'='*60}")
    tree_text = export_text(tree, feature_names=['Return'], decimals=4)
    print(tree_text)
    print(f"{'='*60}\n")


def create_article_decision_tree_figure(
    crypto_name: str,
    y_train_values: np.ndarray,
    tree: DecisionTreeRegressor,
    bin_intervals: np.ndarray,
    output_dir: str = 'decision_tree_graphs',
    filename: str = None,
    language: str = 'pt'
) -> str:
    """
    Cria uma figura profissional da árvore de decisão adequada para artigo acadêmico.
    Inclui diagrama explicativo do processo e visualização limpa da árvore.
    
    Args:
        crypto_name (str): Nome da criptomoeda
        y_train_values (np.ndarray): Valores de retorno usados para treinar a árvore
        tree (DecisionTreeRegressor): Árvore de decisão treinada
        bin_intervals (np.ndarray): Intervalos dos bins criados
        output_dir (str): Diretório para salvar os gráficos
        filename (str, optional): Nome do arquivo. Se None, gera automaticamente.
        language (str): Idioma ('pt' para português, 'en' para inglês)
    
    Returns:
        str: Caminho do arquivo salvo
    """
    # Textos em português e inglês
    texts = {
        'pt': {
            'title': 'Processo de Binning Supervisionado usando Árvore de Decisão',
            'subtitle': f'Criptomoeda: {crypto_name}',
            'step1': '1. Dados de Retornos',
            'step2': '2. Treinamento da Árvore',
            'step3': '3. Extração das Folhas',
            'step4': '4. Definição dos Bins',
            'returns': 'Retornos\nContínuos',
            'tree': 'Árvore de\nDecisão',
            'leaves': '5 Folhas\n(Classes)',
            'bins': '5 Bins\nFinais',
            'distribution': 'Distribuição dos Retornos',
            'tree_structure': 'Estrutura da Árvore de Decisão',
            'bins_result': 'Bins Criados',
            'cut_points': 'Pontos de Corte',
            'samples': 'Amostras',
            'mean': 'Média',
            'bin': 'Bin'
        },
        'en': {
            'title': 'Supervised Binning Process using Decision Tree',
            'subtitle': f'Cryptocurrency: {crypto_name}',
            'step1': '1. Return Data',
            'step2': '2. Tree Training',
            'step3': '3. Leaf Extraction',
            'step4': '4. Bin Definition',
            'returns': 'Continuous\nReturns',
            'tree': 'Decision\nTree',
            'leaves': '5 Leaves\n(Classes)',
            'bins': '5 Final\nBins',
            'distribution': 'Return Distribution',
            'tree_structure': 'Decision Tree Structure',
            'bins_result': 'Created Bins',
            'cut_points': 'Cut Points',
            'samples': 'Samples',
            'mean': 'Mean',
            'bin': 'Bin'
        }
    }
    
    t = texts[language]
    
    # Criar diretório de saída
    os.makedirs(output_dir, exist_ok=True)
    
    # Configurar estilo para artigo
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            plt.style.use('default')
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3, 
                          left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # ============================================================
    # PARTE SUPERIOR: Diagrama de Fluxo do Processo
    # ============================================================
    ax_flow = fig.add_subplot(gs[0, :])
    ax_flow.axis('off')
    ax_flow.set_xlim(0, 10)
    ax_flow.set_ylim(0, 3)
    
    # Título
    title_text = ax_flow.text(5, 2.5, t['title'], 
                              ha='center', va='center',
                              fontsize=18, fontweight='bold',
                              color='#35633a')
    subtitle_text = ax_flow.text(5, 2.1, t['subtitle'],
                                 ha='center', va='center',
                                 fontsize=14, style='italic',
                                 color='#35633a')
    
    # Textos do processo (sem caixas) - distribuídos igualmente
    text_color = '#35633a'
    
    # Distribuir 4 elementos igualmente no espaço de 0.5 a 9.5
    # Espaçamento: (9.5 - 0.5) / 3 = 3.0
    x_positions = [1.5, 4.5, 7.5, 9.5]  # Posições igualmente espaçadas
    
    # Passo 1: Dados
    ax_flow.text(x_positions[0], 0.9, t['step1'], ha='center', va='center', 
                fontsize=11, fontweight='bold', color=text_color)
    ax_flow.text(x_positions[0], 0.6, t['returns'], ha='center', va='center', 
                fontsize=10, color=text_color)
    
    # Passo 2: Árvore
    ax_flow.text(x_positions[1], 0.9, t['step2'], ha='center', va='center', 
                fontsize=11, fontweight='bold', color=text_color)
    ax_flow.text(x_positions[1], 0.6, t['tree'], ha='center', va='center', 
                fontsize=10, color=text_color)
    
    # Passo 3: Folhas
    ax_flow.text(x_positions[2], 0.9, t['step3'], ha='center', va='center', 
                fontsize=11, fontweight='bold', color=text_color)
    ax_flow.text(x_positions[2], 0.6, t['leaves'], ha='center', va='center', 
                fontsize=10, color=text_color)
    
    # Passo 4: Bins
    ax_flow.text(x_positions[3], 0.9, t['step4'], ha='center', va='center', 
                fontsize=11, fontweight='bold', color=text_color)
    ax_flow.text(x_positions[3], 0.6, t['bins'], ha='center', va='center', 
                fontsize=10, color=text_color)
    
    
    # ============================================================
    # PARTE INFERIOR ESQUERDA: Distribuição com Bins
    # ============================================================
    ax_dist = fig.add_subplot(gs[1, 0])
    
    # Histograma
    n, bins_hist, patches = ax_dist.hist(y_train_values, bins=50, 
                                         alpha=0.6, color='steelblue', 
                                         edgecolor='black', linewidth=0.5)
    
    # Colorir por bin
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
    for i, (bin_start, bin_end) in enumerate(zip(bins_hist[:-1], bins_hist[1:])):
        bin_center = (bin_start + bin_end) / 2
        bin_idx = 0
        for j in range(len(bin_intervals) - 1):
            if bin_intervals[j] == -np.inf:
                if bin_center <= bin_intervals[j + 1]:
                    bin_idx = j
                    break
            elif bin_intervals[j + 1] == np.inf:
                if bin_center > bin_intervals[j]:
                    bin_idx = j
                    break
            else:
                if bin_intervals[j] < bin_center <= bin_intervals[j + 1]:
                    bin_idx = j
                    break
        patches[i].set_facecolor(colors[bin_idx % len(colors)])
        patches[i].set_alpha(0.6)
    
    # Linhas dos cut points
    cut_points = bin_intervals[1:-1]
    for i, cut_point in enumerate(cut_points):
        ax_dist.axvline(cut_point, color='red', linestyle='--', 
                       linewidth=2, alpha=0.8)
        ax_dist.text(cut_point, ax_dist.get_ylim()[1] * 0.9, 
                    f'c{i+1}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', 
                             facecolor='yellow', alpha=0.7))
    
    ax_dist.set_xlabel('Retorno (Log Return)', fontsize=12, fontweight='bold')
    ax_dist.set_ylabel('Frequência', fontsize=12, fontweight='bold')
    ax_dist.set_title(t['distribution'], fontsize=13, fontweight='bold')
    ax_dist.grid(True, alpha=0.3)
    
    # ============================================================
    # PARTE INFERIOR DIREITA: Estrutura da Árvore
    # ============================================================
    ax_tree = fig.add_subplot(gs[1, 1])
    ax_tree.axis('off')
    
    # Plotar árvore com estilo limpo
    plot_tree(tree, filled=True, feature_names=['Return'], 
              ax=ax_tree, fontsize=9, rounded=True, precision=3,
              node_ids=True, proportion=True)
    
    ax_tree.set_title(t['tree_structure'], fontsize=13, fontweight='bold', pad=10)
    
    # ============================================================
    # PARTE INFERIOR: Tabela de Bins
    # ============================================================
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')
    
    # Calcular estatísticas por bin
    leaf_ids = tree.apply(y_train_values.reshape(-1, 1))
    df_temp = pd.DataFrame({
        'leaf_id': leaf_ids,
        'return': y_train_values
    })
    leaf_stats = df_temp.groupby('leaf_id')['return'].agg(['min', 'max', 'mean', 'count']).reset_index()
    leaf_stats_sorted = leaf_stats.sort_values('mean').reset_index(drop=True)
    
    # Criar tabela
    table_data = []
    headers = [t['bin'], 'Intervalo', t['samples'], t['mean'], 'Min', 'Max']
    
    for i in range(len(bin_intervals) - 1):
        # Determinar intervalo
        left = bin_intervals[i]
        right = bin_intervals[i + 1]
        left_str = f'{left:.4f}' if left != -np.inf else '-∞'
        right_str = f'{right:.4f}' if right != np.inf else '+∞'
        interval_str = f'[{left_str}, {right_str}]'
        
        # Estatísticas
        mask = (df_temp['leaf_id'] == leaf_stats_sorted.iloc[i]['leaf_id'])
        count = int(np.sum(mask))
        mean_val = leaf_stats_sorted.iloc[i]['mean']
        min_val = leaf_stats_sorted.iloc[i]['min']
        max_val = leaf_stats_sorted.iloc[i]['max']
        
        table_data.append([
            f"Bin {i+1}",
            interval_str,
            f"{count}",
            f"{mean_val:.6f}",
            f"{min_val:.6f}",
            f"{max_val:.6f}"
        ])
    
    # Desenhar tabela
    table = ax_table.table(cellText=table_data, colLabels=headers,
                          cellLoc='center', loc='center',
                          bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Estilizar cabeçalho
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Colorir linhas alternadas
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('white')
    
    ax_table.set_title(t['bins_result'], fontsize=13, fontweight='bold', pad=10)
    
    # Salvar figura
    if filename is None:
        filename = f'decision_tree_article_{crypto_name}.png'
    
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[OK] Figura para artigo salva: {filepath}")
    
    plt.close()
    
    return filepath


def create_supervised_static_bins_with_visualization(
    X_train_data: dict, 
    Y_train_data: dict, 
    datastart: Optional[Union[str, pd.Timestamp]] = None, 
    dataend: Optional[Union[str, pd.Timestamp]] = None,
    visualize: bool = False,
    output_dir: str = 'decision_tree_graphs'
) -> tuple:
    """
    Versão estendida de create_supervised_static_bins que também retorna informações
    para visualização e pode gerar gráficos automaticamente.
    
    Args:
        X_train_data (dict): Dicionário de features (usado apenas para filtragem temporal).
        Y_train_data (dict): Dicionário de retornos contínuos (usado para criar os bins).
        datastart (str, datetime, optional): Data/hora de início do período de treino.
        dataend (str, datetime, optional): Data/hora de fim do período de treino.
        visualize (bool): Se True, gera gráficos para cada criptomoeda.
        output_dir (str): Diretório para salvar os gráficos.
    
    Returns:
        tuple: (bin_intervals_dict, visualization_data) onde:
               - bin_intervals_dict: dict com bins por criptomoeda
               - visualization_data: dict com dados para visualização (tree, y_values, etc.)
    """
    
    # Converter datastart e dataend para datetime se forem strings
    if datastart is not None and isinstance(datastart, str):
        datastart = pd.to_datetime(datastart)
    if dataend is not None and isinstance(dataend, str):
        dataend = pd.to_datetime(dataend)
    
    bin_intervals_dict = {}
    visualization_data = {}
    
    # Iterar por cada criptomoeda nos dicionários de entrada
    for crypto_name in X_train_data.keys():
        
        # Verificar se a criptomoeda existe em ambos os dicionários
        if crypto_name not in Y_train_data:
            print(f"[AVISO] {crypto_name}: Nao encontrado em Y_train_data, pulando.")
            continue
        
        # Extrair dados de treino para esta criptomoeda
        X_train = X_train_data[crypto_name].copy()
        y_train_cont = Y_train_data[crypto_name].copy()
        
        # ============================================================
        # FILTRAGEM TEMPORAL
        # ============================================================
        has_datetime_index_X = isinstance(X_train, pd.DataFrame) and isinstance(X_train.index, pd.DatetimeIndex)
        has_datetime_index_y = isinstance(y_train_cont, pd.Series) and isinstance(y_train_cont.index, pd.DatetimeIndex)
        
        if has_datetime_index_X and has_datetime_index_y:
            if not X_train.index.equals(y_train_cont.index):
                common_index = X_train.index.intersection(y_train_cont.index)
                if len(common_index) == 0:
                    print(f"[AVISO] {crypto_name}: Indices X e y nao se sobrepoem, pulando.")
                    continue
                X_train = X_train.loc[common_index]
                y_train_cont = y_train_cont.loc[common_index]
        
        if datastart is not None or dataend is not None:
            if has_datetime_index_X and has_datetime_index_y:
                if datastart is not None:
                    X_train = X_train[X_train.index >= datastart]
                    y_train_cont = y_train_cont[y_train_cont.index >= datastart]
                if dataend is not None:
                    X_train = X_train[X_train.index <= dataend]
                    y_train_cont = y_train_cont[y_train_cont.index <= dataend]
                
                if len(X_train) == 0 or len(y_train_cont) == 0:
                    print(f"[AVISO] {crypto_name}: Nenhum dado no periodo {datastart} a {dataend}, pulando.")
                    continue
            else:
                import warnings
                warnings.warn(f"Dados para '{crypto_name}' nao tem indices datetime. Filtragem temporal ignorada.")
        
        y_train_values = y_train_cont.values if isinstance(y_train_cont, pd.Series) else y_train_cont
        
        if len(y_train_values) < 5:
            print(f"[AVISO] {crypto_name}: Dados insuficientes ({len(y_train_values)} amostras), pulando.")
            continue
        
        # ============================================================
        # FASE 1: Particionamento Supervisionado (APENAS RETORNOS)
        # ============================================================
        # Usamos o próprio retorno como "feature" para a árvore
        X_dummy = y_train_values.reshape(-1, 1)  # Formato (n_samples, 1)
        
        tree = DecisionTreeRegressor(
            max_leaf_nodes=5,
            random_state=42
        )
        tree.fit(X_dummy, y_train_values)
        
        # ============================================================
        # FASE 2: Extração e Ordenação dos Bins (As "Folhas")
        # ============================================================
        leaf_ids = tree.apply(X_dummy)
        
        df_temp = pd.DataFrame({
            'leaf_id': leaf_ids,
            'return': y_train_values
        })
        
        leaf_stats = df_temp.groupby('leaf_id')['return'].agg(['min', 'max', 'mean']).reset_index()
        leaf_stats_sorted = leaf_stats.sort_values('mean').reset_index(drop=True)
        
        num_leaves = len(leaf_stats_sorted)
        if num_leaves != 5:
            print(f"[AVISO] {crypto_name}: Arvore gerou {num_leaves} folhas em vez de 5, pulando.")
            continue
        
        # ============================================================
        # FASE 3: Definição dos Limites Estáticos (Os "Intervalos")
        # ============================================================
        cut_points = []
        for i in range(num_leaves - 1):
            max_current_bin = leaf_stats_sorted.iloc[i]['max']
            min_next_bin = leaf_stats_sorted.iloc[i + 1]['min']
            cut_point = (max_current_bin + min_next_bin) / 2.0
            cut_points.append(cut_point)
        
        # ============================================================
        # FASE 4: Saída Final
        # ============================================================
        bin_intervals = [-np.inf] + cut_points + [np.inf]
        bin_intervals_dict[crypto_name] = np.array(bin_intervals)
        
        # Armazenar dados para visualização
        visualization_data[crypto_name] = {
            'tree': tree,
            'y_train_values': y_train_values,
            'bin_intervals': bin_intervals,
            'leaf_stats': leaf_stats_sorted
        }
        
        print(f"[OK] {crypto_name} (Tree): Bins criados")
        
        # Gerar visualizações se solicitado
        if visualize:
            try:
                # Gerar visualização detalhada
                visualize_decision_tree_process(
                    crypto_name=crypto_name,
                    y_train_values=y_train_values,
                    tree=tree,
                    bin_intervals=bin_intervals,
                    output_dir=output_dir,
                    save_plots=True
                )
                # Gerar figura para artigo
                create_article_decision_tree_figure(
                    crypto_name=crypto_name,
                    y_train_values=y_train_values,
                    tree=tree,
                    bin_intervals=bin_intervals,
                    output_dir=output_dir,
                    language='pt'
                )
            except Exception as e:
                print(f"[AVISO] {crypto_name}: Erro ao gerar visualizacoes - {e}")
    
    return bin_intervals_dict, visualization_data


# ============================================================
# MÉTODO 2: BINNING POR OTIMIZAÇÃO (FORMULAÇÃO LATEX)
# ============================================================
def create_optimized_bins(
    Y_train_data: dict,
    datastart: Optional[Union[str, pd.Timestamp]] = None,
    dataend: Optional[Union[str, pd.Timestamp]] = None,
    n_min: int = 10,
    method: str = 'differential_evolution'
) -> dict:
    """
    Cria bins usando otimização para maximizar a variância entre grupos
    (Between-Group Variance), conforme formulação LaTeX.
    
    Formulação do problema:
    - Variáveis de decisão: 4 pontos de corte c1, c2, c3, c4
    - Objetivo: maximizar F(c1,...,c4) = sum_k n_k * (mu_k - mu)^2
      onde n_k é o número de observações no intervalo k,
      mu_k é a média dos retornos no intervalo k,
      mu é a média global dos retornos
    - Restrições:
      * r_min <= c1 < c2 < c3 < c4 <= r_max
      * n_k >= n_min para todo k
    
    Args:
        Y_train_data (dict): Dicionário de retornos contínuos (usado para criar os bins).
        datastart (str, datetime, optional): Data/hora de início do período de treino.
        dataend (str, datetime, optional): Data/hora de fim do período de treino.
        n_min (int): Mínimo de observações por intervalo.
        method (str): Método de otimização ('differential_evolution' ou 'minimize').
    
    Returns:
        dict: Dicionário com bins por criptomoeda (ex: [-inf, c1, c2, c3, c4, +inf])
    """
    # Converter datastart e dataend para datetime se forem strings
    if datastart is not None and isinstance(datastart, str):
        datastart = pd.to_datetime(datastart)
    if dataend is not None and isinstance(dataend, str):
        dataend = pd.to_datetime(dataend)
    
    bin_intervals_dict = {}
    
    for crypto_name, y_train_cont in Y_train_data.items():
        
        y_train_values_full = y_train_cont.values if isinstance(y_train_cont, pd.Series) else np.array(y_train_cont).flatten()
        y_train_index = y_train_cont.index if isinstance(y_train_cont, pd.Series) else None
        
        y_train_values = y_train_values_full.copy()
        
        # Aplicar filtragem temporal se necessário
        if (datastart is not None or dataend is not None) and \
           isinstance(y_train_index, pd.DatetimeIndex):
            
            mask = pd.Series(True, index=y_train_index)
            if datastart is not None:
                mask &= (y_train_index >= datastart)
            if dataend is not None:
                mask &= (y_train_index <= dataend)
            
            y_train_values = y_train_values[mask.values]
            
            if len(y_train_values) == 0:
                print(f"[AVISO] {crypto_name}: Nenhum dado no periodo {datastart} a {dataend}, pulando.")
                continue
        
        # Verificar dados suficientes
        if len(y_train_values) < 5 * n_min:
            print(f"[AVISO] {crypto_name}: Dados insuficientes ({len(y_train_values)} < {5 * n_min}), pulando.")
            continue
        
        # Calcular estatísticas globais
        r_min = np.min(y_train_values)
        r_max = np.max(y_train_values)
        mu_global = np.mean(y_train_values)
        
        # Função objetivo: maximizar variância entre grupos
        def objective(cuts):
            """Calcula -F(c1,...,c4) para minimização."""
            c1, c2, c3, c4 = cuts
            
            # Verificar ordem
            if not (r_min <= c1 < c2 < c3 < c4 <= r_max):
                return 1e10  # Penalidade alta para violação de ordem
            
            # Definir intervalos
            # I1 = [r_min, c1]
            # I2 = (c1, c2]
            # I3 = (c2, c3]
            # I4 = (c3, c4]
            # I5 = (c4, r_max]
            
            # Criar máscaras booleanas para cada intervalo
            mask1 = (y_train_values <= c1)
            mask2 = (y_train_values > c1) & (y_train_values <= c2)
            mask3 = (y_train_values > c2) & (y_train_values <= c3)
            mask4 = (y_train_values > c3) & (y_train_values <= c4)
            mask5 = (y_train_values > c4)
            
            masks = [mask1, mask2, mask3, mask4, mask5]
            
            F = 0.0
            for mask in masks:
                n_k = np.sum(mask)
                
                # Verificar mínimo de observações
                if n_k < n_min:
                    return 1e10  # Penalidade alta
                
                # n_k > 0 é garantido por n_k >= n_min (assumindo n_min > 0)
                mu_k = np.mean(y_train_values[mask])
                F += n_k * (mu_k - mu_global) ** 2
            
            # Retornar negativo para maximização (minimizamos -F)
            return -F
        
        # Bounds para os pontos de corte
        # Garantir que há espaço entre os cortes
        range_size = r_max - r_min
        if range_size == 0:
            print(f"[AVISO] {crypto_name}: Todos os retornos sao identicos ({r_min}), pulando.")
            continue
            
        margin = range_size * 0.01  # 1% de margem
        
        # Bounds mais realistas, baseados em percentis
        p10 = np.percentile(y_train_values, 10)
        p90 = np.percentile(y_train_values, 90)
        
        bounds = [
            (p10, np.percentile(y_train_values, 40)),  # c1
            (np.percentile(y_train_values, 20), np.percentile(y_train_values, 60)),  # c2
            (np.percentile(y_train_values, 40), np.percentile(y_train_values, 80)),  # c3
            (np.percentile(y_train_values, 60), p90)   # c4
        ]
        
        # Ponto inicial (quartis)
        initial_guess = [
            np.percentile(y_train_values, 20),
            np.percentile(y_train_values, 40),
            np.percentile(y_train_values, 60),
            np.percentile(y_train_values, 80)
        ]
        
        # Otimização
        try:
            if method == 'differential_evolution':
                result = differential_evolution(
                    objective,
                    bounds,
                    seed=42,
                    maxiter=1000,
                    popsize=15,
                    tol=1e-6
                )
            else:
                result = minimize(
                    objective,
                    initial_guess,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000}
                )
            
            if result.success or result.fun < 1e9:
                # Ordenar os cortes resultantes, pois o otimizador pode não garantir
                c_unsorted = result.x
                c1, c2, c3, c4 = sorted(c_unsorted)
                
                # Verificar se a solução é válida (mesmo após ordenação)
                if r_min <= c1 and c4 <= r_max:
                    bin_intervals = np.array([-np.inf, c1, c2, c3, c4, np.inf])
                    bin_intervals_dict[crypto_name] = bin_intervals
                    
                    # Verificar distribuição
                    mask1 = (y_train_values <= c1)
                    mask2 = (y_train_values > c1) & (y_train_values <= c2)
                    mask3 = (y_train_values > c2) & (y_train_values <= c3)
                    mask4 = (y_train_values > c3) & (y_train_values <= c4)
                    mask5 = (y_train_values > c4)
                    counts = [np.sum(mask1), np.sum(mask2), np.sum(mask3), np.sum(mask4), np.sum(mask5)]
                    
                    if any(c < n_min for c in counts):
                         print(f"[ERRO] {crypto_name} (Opt): Solucao final violou n_min. Dist: {counts}")
                    else:
                        print(f"[OK] {crypto_name} (Opt): Bins otimizados criados")
                        print(f"     Distribuicao: {counts}")
                        print(f"     Cortes: [{c1:.6f}, {c2:.6f}, {c3:.6f}, {c4:.6f}]")
                else:
                    print(f"[ERRO] {crypto_name} (Opt): Solucao invalida (fora de [r_min, r_max])")
            else:
                print(f"[ERRO] {crypto_name} (Opt): Otimizacao falhou - {result.message}")
        except Exception as e:
            print(f"[ERRO] {crypto_name} (Opt): Erro na otimizacao - {str(e)}")
    
    return bin_intervals_dict


# ============================================================
# Código executável principal
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("Pipeline de Bins Supervisionados para Criptomoedas")
    print("="*60)
    print(f"\nCriptomoedas configuradas: {CRYPTO_LIST}")
    print(f"Simbolos Binance: {BINANCE_SYMBOLS}")
    print(f"\nMetodos: AMBOS (Tree e Optimization)")
    print("  - Sempre executa ambos os metodos para comparacao")
    
    # Configurações
    start_time = '2020-01-01'
    end_time = '2024-12-31'
    limit = 5000 # Aumentar limite para buscar mais dados históricos
    
    # Estratégias: diária e semanal
    strategies = {
        'daily': {'interval': '1d', 'name': 'Estrategia Diaria'},
        'weekly': {'interval': '1w', 'name': 'Estrategia Semanal'}
    }
    
    # Dicionário para armazenar resultados
    all_results = {}
    
    try:
        for strategy_key, strategy_config in strategies.items():
            interval = strategy_config['interval']
            strategy_name = strategy_config['name']
            
            print("\n" + "="*60)
            print(f"{strategy_name.upper()}")
            print("="*60)
            print(f"Periodo: {start_time} a {end_time}")
            print(f"Intervalo: {interval}\n")
            
            # Buscar dados da Binance (usando cache se disponível)
            X_train_data, Y_train_data = prepare_binance_data_for_training(
                symbols=BINANCE_SYMBOLS,
                interval=interval,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
                use_cache=True,
                save_cache=True
            )
            
            print(f"\n[OK] Dados preparados para {len(X_train_data)} criptomoedas")
            for crypto, X in X_train_data.items():
                print(f"  {crypto}: {len(X_train_data[crypto])} amostras (X), {len(Y_train_data[crypto])} amostras (Y)")
            
            # Criar bins usando ambos os métodos
            results_for_strategy = {}
            
            # Método 1: DecisionTree
            print("\n" + "="*60)
            print("METODO 1: DecisionTree (Supervisionado)")
            print("="*60)
            
            # Opção para gerar visualizações (pode ser configurada)
            generate_visualizations = True  # Mudar para False para desabilitar visualizações
            
            if generate_visualizations:
                bin_intervals_tree, viz_data = create_supervised_static_bins_with_visualization(
                    X_train_data, Y_train_data, visualize=True, 
                    output_dir=f'decision_tree_graphs_{strategy_key}'
                )
            else:
                bin_intervals_tree = create_supervised_static_bins(X_train_data, Y_train_data)
            
            results_for_strategy['tree'] = bin_intervals_tree
            
            # Método 2: Otimização (sempre executado)
            print("\n" + "="*60)
            print("METODO 2: Otimizacao (Maximizar Variancia Entre Grupos)")
            print("="*60)
            bin_intervals_opt = create_optimized_bins(Y_train_data, n_min=10)
            results_for_strategy['optimization'] = bin_intervals_opt
            
            # Armazenar resultados
            all_results[strategy_key] = {
                'name': strategy_name,
                'results': results_for_strategy
            }
            
            # Visualizar resultados
            print("\n" + "="*60)
            print(f"RESULTADOS - {strategy_name.upper()}")
            print("="*60)
            
            for method_name, bin_intervals in results_for_strategy.items():
                print(f"\n--- Metodo: {method_name.upper()} ---")
                for crypto in sorted(bin_intervals.keys()):
                    intervals = bin_intervals[crypto]
                    print(f"\n{crypto}:")
                    print(f"  Numero de bins: {len(intervals) - 1}")
                    print(f"  Intervalos: {intervals}")
        
        # Resumo final
        print("\n" + "="*60)
        print("RESUMO FINAL - COMPARACAO DE METODOS")
        print("="*60)
        
        for strategy_key, result in all_results.items():
            print(f"\n{result['name']}:")
            print("-" * 60)
            
            # Coletar todas as criptos de ambos os métodos
            all_cryptos = set()
            if 'tree' in result['results']:
                all_cryptos.update(result['results']['tree'].keys())
            if 'optimization' in result['results']:
                all_cryptos.update(result['results']['optimization'].keys())
                
            for crypto in sorted(all_cryptos):
                print(f"  {crypto}:")
                
                if 'tree' in result['results'] and crypto in result['results']['tree']:
                    print(f"    Tree:         {result['results']['tree'][crypto]}")
                else:
                    print(f"    Tree:         NAO GERADO")
                    
                if 'optimization' in result['results'] and crypto in result['results']['optimization']:
                    print(f"    Optimization: {result['results']['optimization'][crypto]}")
                else:
                    print(f"    Optimization: NAO GERADO")

        
        print("\n" + "="*60)
        print("[OK] Processo concluido com sucesso!")
        print("[OK] Intervalos gerados usando ambos os metodos (Tree e Optimization)")
        print("="*60)
        
    except Exception as e:
        print(f"\n[ERRO] Erro durante a execucao: {e}")
        import traceback
        traceback.print_exc()