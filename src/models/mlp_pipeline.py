"""
Pipeline completo de MLP para classificação de retornos de criptomoedas.
Processa dados diários e semanais, aplica bins supervisionados e treina MLPClassifier.
"""

import os
import sys
from pathlib import Path

# Adicionar diretório raiz ao path para imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    classification_report
)

    # Importar funções de criação de bins
try:
    from src.data.data_fetcher import (
        create_supervised_static_bins,
        prepare_binance_data_for_training,
        BINANCE_SYMBOLS
    )
except ImportError:
    print("[AVISO] Nao foi possivel importar funcoes dos modulos")
    create_supervised_static_bins = None
    prepare_binance_data_for_training = None
    BINANCE_SYMBOLS = []


# ============================================================
# Configuração de Bins e Método
# ============================================================

# Método para criar bins: 'tree' (apenas)
BIN_METHOD = 'tree'  # Apenas Tree

# Bins serão sempre calculados dinamicamente usando os módulos de binning
# Os bins hardcoded abaixo são mantidos apenas como fallback em caso de erro
BINS_DIA = {}  # Será calculado dinamicamente
BINS_SEMANA = {}  # Será calculado dinamicamente


# ============================================================
# Funções Auxiliares
# ============================================================

def preparar_dados_para_classificacao(df, bins_dict, col_symbol='symbol'):
    """
    Prepara dados para classificação aplicando bins supervisionados.
    
    Args:
        df: DataFrame com indicadores
        bins_dict: Dicionário com bins por símbolo
        col_symbol: Nome da coluna de símbolo
    
    Returns:
        DataFrame com coluna 'target' adicionada
    """
    # 1. Identificar qual coluna de retorno está presente
    col_ret = None
    for c in ["ret_1", "ret_5", "ret_20", "vol_10"]:
        if c in df.columns:
            col_ret = c
            break
    
    # Se não encontrou, calcular ret_1 a partir de price_close
    if col_ret is None:
        if 'price_close' not in df.columns:
            raise ValueError("Nenhuma coluna de retorno encontrada e 'price_close' também não está presente.")
        print("[AVISO] Coluna de retorno não encontrada. Calculando ret_1 a partir de price_close...")
        # Calcular ret_1 agrupando por símbolo
        df = df.copy()
        df['ret_1'] = np.nan
        for sym, grupo in df.groupby(col_symbol):
            if len(grupo) > 1:
                df.loc[grupo.index, 'ret_1'] = np.log(grupo['price_close'] / grupo['price_close'].shift(1))
        col_ret = 'ret_1'
        # Remover primeira linha de cada grupo (que tem NaN)
        df = df.dropna(subset=[col_ret])

    # 2. Filtrar apenas as moedas válidas
    simbolos_validos = set(bins_dict.keys())
    df = df[df[col_symbol].isin(simbolos_validos)].copy()

    if len(df) == 0:
        raise ValueError(f"Nenhum dado encontrado para os símbolos válidos: {simbolos_validos}")

    # 3. Criar coluna target vazia
    df["target"] = np.nan

    # 4. Aplicar pd.cut por moeda
    for sym, grupo in df.groupby(col_symbol):
        if sym not in bins_dict:
            continue
        bins = bins_dict[sym]
        
        # Verificar se há valores válidos
        valores_validos = grupo[col_ret].dropna()
        if len(valores_validos) == 0:
            continue

        # Aplicar pd.cut com labels explícitos para garantir 5 classes (0-4)
        cut_result = pd.cut(
            grupo[col_ret],
            bins=bins,
            labels=False,
            include_lowest=True,
            duplicates='drop'
        )
        
        # Verificar se todas as classes foram geradas
        classes_geradas = cut_result.dropna().unique()
        if len(classes_geradas) < 5:
            print(f"[AVISO] {sym}: Apenas {len(classes_geradas)} classes geradas: {sorted(classes_geradas)}")
        
        df.loc[grupo.index, "target"] = cut_result

    # 5. Remover linhas sem target
    df = df.dropna(subset=["target"])

    # 6. Converter target para inteiro
    df["target"] = df["target"].astype(int)

    # 7. Remover a coluna do retorno original (evita leakage)
    if col_ret in df.columns:
        df = df.drop(columns=[col_ret])

    return df


def processar_dados(indicadores_treino, indicadores_teste, bins_dict, strategy_name):
    """
    Processa dados completos: aplica bins, onehot encoding, prepara X e y.
    
    Args:
        indicadores_treino: DataFrame de treino
        indicadores_teste: DataFrame de teste
        bins_dict: Dicionário de bins
        strategy_name: Nome da estratégia (para logs)
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test)
    """
    print(f"\n{'='*60}")
    print(f"Processando {strategy_name}")
    print(f"{'='*60}")
    
    # Aplicar bins
    print(f"\n1. Aplicando bins para {strategy_name}...")
    indicadores_treino = preparar_dados_para_classificacao(indicadores_treino.copy(), bins_dict)
    indicadores_teste = preparar_dados_para_classificacao(indicadores_teste.copy(), bins_dict)
    
    # OneHot encoding
    print(f"2. Aplicando OneHot encoding para {strategy_name}...")
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
    
    indicadores_treino_enc = enc.fit_transform(indicadores_treino[['symbol']])
    indicadores_teste_enc = enc.fit_transform(indicadores_teste[['symbol']])
    
    indicadores_treino = pd.concat([indicadores_treino, indicadores_treino_enc], axis=1).drop(columns=['symbol'])
    indicadores_teste = pd.concat([indicadores_teste, indicadores_teste_enc], axis=1).drop(columns=['symbol'])
    
    # Separar X e y
    print(f"3. Separando features e target para {strategy_name}...")
    X_train = indicadores_treino.drop(columns=['target'])
    y_train = indicadores_treino['target']
    
    X_test = indicadores_teste.drop(columns=['target'])
    y_test = indicadores_teste['target']
    
    # Verificações
    print(f"   Shape X_train: {X_train.shape}")
    print(f"   Shape X_test: {X_test.shape}")
    print(f"   NaN em X_train: {X_train.isna().sum().sum()}")
    print(f"   NaN em X_test: {X_test.isna().sum().sum()}")
    
    # Remover NaN se houver
    if X_train.isna().sum().sum() > 0 or X_test.isna().sum().sum() > 0:
        print("   [AVISO] Removendo linhas com NaN...")
        mask_train = ~X_train.isna().any(axis=1)
        mask_test = ~X_test.isna().any(axis=1)
        X_train = X_train[mask_train]
        y_train = y_train[mask_train]
        X_test = X_test[mask_test]
        y_test = y_test[mask_test]
    
    # Verificar classes
    print(f"\n   Classes no treino: {sorted(y_train.unique())}")
    print(f"   Classes no teste: {sorted(y_test.unique())}")
    print(f"   Distribuição no treino:\n{y_train.value_counts().sort_index()}")
    print(f"   Distribuição no teste:\n{y_test.value_counts().sort_index()}")
    
    return X_train, y_train, X_test, y_test


def treinar_mlp_arquitetura_unica(X_train, y_train, X_test, y_test, strategy_name, random_state=42):
    """
    Treina MLP com arquitetura única: 2 camadas (32, 64 neurônios).
    
    Args:
        X_train, y_train: Dados de treino
        X_test, y_test: Dados de teste
        strategy_name: Nome da estratégia
        random_state: Seed para reprodutibilidade
    
    Returns:
        dict: Dicionário com chaves:
            - 'mlp': modelo MLP treinado
            - 'y_pred': predições no teste
            - 'score_train': score no treino
            - 'score_test': score no teste
            - 'scaler': scaler aplicado
            - 'resultado': dict com todas as métricas
    """
    print(f"\n{'='*60}")
    print(f"TREINANDO MLP - {strategy_name}")
    print(f"{'='*60}")
    
    # Aplicar StandardScaler
    print(f"\n1. Aplicando StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Arquitetura única: 2 camadas (32, 64)
    hidden_sizes = (32, 64)
    descricao = "2 camadas (32, 64 neurônios)"
    
    print(f"\n2. Treinando MLP com arquitetura: {descricao}")
    print(f"   Configuração: {hidden_sizes}")
    
    # Treinar modelo
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_sizes,
        activation='relu',
        solver='adam',
        alpha=0.01,
        learning_rate_init=0.001,
        max_iter=200,
        tol=1e-4,
        n_iter_no_change=10,
        verbose=True,
        random_state=random_state
    )
    
    mlp.fit(X_train_scaled, y_train)
    
    # Métricas no treino
    score_train = mlp.score(X_train_scaled, y_train)
    y_pred_train = mlp.predict(X_train_scaled)
    f1_train_macro = f1_score(y_train, y_pred_train, average='macro')
    
    # Métricas no teste
    score_test = mlp.score(X_test_scaled, y_test)
    y_pred_test = mlp.predict(X_test_scaled)
    f1_test_macro = f1_score(y_test, y_pred_test, average='macro')
    
    # Diferença (overfitting)
    diff_train_test = score_train - score_test
    
    # Armazenar resultado
    resultado = {
        'arquitetura': hidden_sizes,
        'descricao': descricao,
        'score_train': score_train,
        'score_test': score_test,
        'f1_train_macro': f1_train_macro,
        'f1_test_macro': f1_test_macro,
        'diff_train_test': diff_train_test
    }
    
    print(f"\n{'='*60}")
    print(f"RESULTADOS - {strategy_name}")
    print(f"{'='*60}")
    print(f"Arquitetura: {descricao}")
    print(f"Score treino: {score_train:.4f}")
    print(f"Score teste:  {score_test:.4f}")
    print(f"Accuracy teste: {accuracy_score(y_test, y_pred_test):.4f}")
    print(f"F1 Macro teste: {f1_test_macro:.4f}")
    print(f"Overfitting: {diff_train_test:.4f}")
    
    if diff_train_test > 0.15:
        print(f"⚠ Diferença alta - possível overfitting")
    
    print(f"\n{'='*60}")
    print(f"[OK] MLP treinado com sucesso")
    print(f"{'='*60}")
    
    return {
        'mlp': mlp,
        'y_pred': y_pred_test,
        'score_train': score_train,
        'score_test': score_test,
        'scaler': scaler,
        'resultado': resultado
    }


def treinar_mlp(X_train, y_train, X_test, y_test, strategy_name, random_state=42, 
                hidden_layer_sizes=(100, 100)):
    """
    Treina MLPClassifier e retorna predições.
    
    Args:
        X_train, y_train: Dados de treino
        X_test, y_test: Dados de teste
        strategy_name: Nome da estratégia
        random_state: Seed para reprodutibilidade
        hidden_layer_sizes: Tupla com tamanho das camadas ocultas. Exemplos:
                           - (100,) -> 1 camada com 100 neurônios (simples)
                           - (100, 100) -> 2 camadas com 100 cada (atual)
                           - (50, 50) -> 2 camadas menores (mais simples)
                           - (200, 100) -> 2 camadas com diferentes tamanhos
    
    Returns:
        tuple: (mlp, y_pred, score_train, scaler)
    
    Nota sobre arquitetura:
    - Reduzir camadas pode reduzir overfitting, mas também pode limitar capacidade
    - Para problemas simples (poucas features, classes bem separadas), 1 camada pode ser suficiente
    - Para problemas complexos, mais camadas podem ser necessárias
    - Regra de ouro: use a arquitetura mais simples que ainda capture os padrões
    """
    print(f"\n4. Aplicando StandardScaler para {strategy_name}...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"5. Treinando MLP para {strategy_name}...")
    print(f"   Arquitetura: {hidden_layer_sizes} ({len(hidden_layer_sizes)} camada(s) oculta(s))")
    print(f"   Total de features: {X_train_scaled.shape[1]}")
    print(f"   Classes: {len(np.unique(y_train))}")
    
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=200,
        tol=1e-4,
        n_iter_no_change=10,
        verbose=True,
        random_state=random_state
    )
    
    mlp.fit(X_train_scaled, y_train)
    score_train = mlp.score(X_train_scaled, y_train)
    
    # Calcular score no teste também para detectar overfitting
    score_test = mlp.score(X_test_scaled, y_test)
    
    print(f"   [OK] MLP treinado")
    print(f"   - Score no treino: {score_train:.4f}")
    print(f"   - Score no teste:  {score_test:.4f}")
    print(f"   - Diferença (overfitting): {score_train - score_test:.4f}")
    
    # Aviso sobre possível overfitting
    if score_train - score_test > 0.15:  # Diferença > 15%
        print(f"   [AVISO] Diferença alta entre treino/teste - possível overfitting")
        print(f"   [DICA] Considere reduzir camadas ou aumentar regularização (alpha)")
    
    y_pred = mlp.predict(X_test_scaled)
    print(f"   [OK] Predicoes realizadas para {len(y_pred)} amostras")
    
    return mlp, y_pred, score_train, scaler


def plot_confusion_matrix(y_test, y_pred, strategy_name, save_path=None):
    """
    Plota matriz de confusão com todas as 5 classes.
    
    Args:
        y_test: Valores reais
        y_pred: Valores preditos
        strategy_name: Nome da estratégia
        save_path: Caminho para salvar figura (opcional)
    """
    all_classes = [0, 1, 2, 3, 4]
    cm = confusion_matrix(y_test, y_pred, labels=all_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=all_classes, yticklabels=all_classes)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusão {strategy_name.upper()} - 5 Classes (0-4)")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   [OK] Matriz de confusão salva em: {save_path}")
    else:
        plt.show()
    
    # Diagnóstico
    print(f"\n=== Distribuição de Classes - {strategy_name.upper()} ===")
    print(f"Classes no teste (real): {sorted(np.unique(y_test))}")
    print(f"Classes preditas: {sorted(np.unique(y_pred))}")
    print(f"Classes esperadas: {all_classes}")
    print(f"Classes ausentes no teste: {set(all_classes) - set(np.unique(y_test))}")
    print(f"Classes ausentes nas predições: {set(all_classes) - set(np.unique(y_pred))}")


def calcular_metricas(y_test, y_pred, strategy_name):
    """
    Calcula e exibe métricas de classificação.
    
    Args:
        y_test: Valores reais
        y_pred: Valores preditos
        strategy_name: Nome da estratégia
    """
    print(f"\n=== Métricas - {strategy_name.upper()} ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Macro: {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"F1 Micro: {f1_score(y_test, y_pred, average='micro'):.4f}")
    
    print(f"\n=== Classification Report - {strategy_name.upper()} ===")
    print(classification_report(y_test, y_pred, zero_division=0))


# ============================================================
# Função para Calcular Bins (usando módulos de binning)
# ============================================================

def compute_bins_from_main(interval='1d', use_both=True, bin_method=None):
    """
    Calcula bins usando os módulos de binning (decision_tree e optimization).
    
    Args:
        interval: '1d' para diário, '1w' para semanal
        use_both: Se True, calcula bins usando ambos os métodos (tree e optimization)
        bin_method: Método de bins a calcular ('tree', 'optimization', ou None para usar use_both)
    
    Returns:
        dict: Dicionário com chaves 'tree' e/ou 'optimization', cada uma contendo
              um dict de bins por criptomoeda
    """
    if bin_method is None:
        bin_method = BIN_METHOD
    if create_supervised_static_bins is None or prepare_binance_data_for_training is None:
        raise ImportError("Funcoes de binning nao disponiveis")
    
    print(f"\n{'='*60}")
    print(f"Calculando bins - Intervalo: {interval}")
    print(f"{'='*60}")
    
    # Buscar dados da Binance
    from src.data.data_fetcher import BINANCE_SYMBOLS
    symbols = BINANCE_SYMBOLS
    
    print(f"\n1. Buscando dados da Binance para intervalo {interval}...")
    X_train_data, Y_train_data = prepare_binance_data_for_training(
        symbols=symbols,
        interval=interval,
        start_time='2020-01-01',
        end_time='2024-12-31',
        limit=2000,
        use_cache=True,
        save_cache=True
    )
    
    print(f"[OK] Dados preparados para {len(X_train_data)} criptomoedas")
    
    results = {}
    
    # Método 1: DecisionTree (sempre calcular se use_both=True ou se bin_method for 'tree' ou 'both')
    if use_both or bin_method in ['tree', 'both']:
        print(f"\n2. Calculando bins usando metodo DecisionTree...")
        bins_tree = create_supervised_static_bins(X_train_data, Y_train_data)
        results['tree'] = bins_tree
        print(f"[OK] Bins Tree calculados para {len(bins_tree)} criptomoedas")
    
    # Método 2: Optimization (se solicitado ou se for o método escolhido)
    if use_both or bin_method == 'optimization':
        print(f"\n3. Calculando bins usando metodo Optimization...")
        if create_optimized_bins is None:
            raise ImportError("create_optimized_bins nao disponivel")
        bins_opt = create_optimized_bins(Y_train_data, n_min=10)
        results['optimization'] = bins_opt
        print(f"[OK] Bins Optimization calculados para {len(bins_opt)} criptomoedas")
    
    return results


# ============================================================
# Pipeline Principal
# ============================================================

def main():
    """Pipeline principal de treinamento e avaliação.
    
    Executa apenas:
    - Weekly (semanal)
    - Tree (apenas)
    """
    
    print("="*60)
    print("Pipeline MLP para Classificação de Criptomoedas")
    print("="*60)
    print("Executa apenas: Weekly (semanal) | Tree")
    
    # Diretório de dados e outputs
    data_dir = "data/processed"
    output_dir = Path("results/plots")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Outputs serão salvos em: {output_dir}")
    
    # Calcular bins usando apenas Tree
    print("\n" + "="*60)
    print("CALCULANDO BINS (TREE)")
    print("="*60)
    
    bins_semana_tree = None
    
    # Calcular bins semanais (apenas Tree)
    print("\n--- BINS SEMANAIS (TREE) ---")
    try:
        bins_semana_results = compute_bins_from_main('1w', use_both=False, bin_method='tree')
        
        bins_semana_tree = bins_semana_results.get('tree')
        
        if bins_semana_tree is None:
            raise ValueError("Bins de tree não foram calculados para semanal")
            
        print(f"[OK] Bins semanais calculados - Método Tree: {len(bins_semana_tree)} criptos")
    except Exception as e:
        print(f"[ERRO] Falha ao calcular bins semanais: {e}")
        raise
    
    print("\n[INFO] Bins calculados dinamicamente usando módulos de binning")
    print("[INFO] Treinando modelos para: Weekly (semanal) | Tree")
    
    # ============================================================
    # 1. Carregar Dados
    # ============================================================
    print("\n" + "="*60)
    print("1. Carregando dados...")
    print("="*60)
    
    # Dados semanais (treino: 2020-2023, teste: 2024-2025)
    indicadores_semana_treino = pd.read_csv(f"{data_dir}/indicators_weekly_train_2020_2023.csv")
    indicadores_semana_teste = pd.read_csv(f"{data_dir}/indicators_weekly_test_2024_2025.csv")
    
    print("[OK] Dados carregados")
    
    # Remover timestamp (não usado no modelo)
    indicadores_semana_treino = indicadores_semana_treino.drop(columns=['timestamp'], errors='ignore')
    indicadores_semana_teste = indicadores_semana_teste.drop(columns=['timestamp'], errors='ignore')
    
    # ============================================================
    # 2. Processar e Treinar com Bins Tree (SEMANAL) - SEMPRE
    # ============================================================
    print("\n" + "="*60)
    print("TREINANDO MLP SEMANAL - METODO TREE")
    print("="*60)
    
    X_train_semana_tree, y_train_semana_tree, X_test_semana_tree, y_test_semana_tree = processar_dados(
        indicadores_semana_treino.copy(),
        indicadores_semana_teste.copy(),
        bins_semana_tree,
        "SEMANAL - TREE"
    )
    
    # Treinar MLP com arquitetura única (32, 64)
    modelo_semana_tree = treinar_mlp_arquitetura_unica(
        X_train_semana_tree, y_train_semana_tree, X_test_semana_tree, y_test_semana_tree,
        "SEMANAL - TREE"
    )
    
    # Calcular métricas e plotar
    calcular_metricas(y_test_semana_tree, modelo_semana_tree['y_pred'], "SEMANAL - TREE")
    plot_confusion_matrix(y_test_semana_tree, modelo_semana_tree['y_pred'], "SEMANAL - TREE",
                         save_path=str(output_dir / "confusion_matrix_weekly_tree.png"))
    
    mlp_semana_tree = modelo_semana_tree['mlp']
    y_semana_pred_tree = modelo_semana_tree['y_pred']
    score_semana_tree = modelo_semana_tree['score_train']
    scaler_semana_tree = modelo_semana_tree['scaler']
    resultado_semana_tree = modelo_semana_tree['resultado']
    
    # Variáveis de compatibilidade
    X_train_semana = X_train_semana_tree
    y_train_semana = y_train_semana_tree
    X_test_semana = X_test_semana_tree
    y_test_semana = y_test_semana_tree
    mlp_semana = mlp_semana_tree
    y_semana_pred = y_semana_pred_tree
    score_semana = score_semana_tree
    scaler_semana = scaler_semana_tree
    
    # ============================================================
    # 3. Resumo Final
    # ============================================================
    print("\n" + "="*60)
    print("RESUMO FINAL")
    print("="*60)
    
    print(f"\n{'='*60}")
    print("SEMANAL - METODO TREE")
    print(f"{'='*60}")
    print(f"\n{resultado_semana_tree['descricao']}:")
    print(f"  - Score treino: {resultado_semana_tree['score_train']:.4f}")
    print(f"  - Score teste:  {resultado_semana_tree['score_test']:.4f}")
    print(f"  - Accuracy teste: {accuracy_score(y_test_semana_tree, y_semana_pred_tree):.4f}")
    print(f"  - F1 Macro: {resultado_semana_tree['f1_test_macro']:.4f}")
    print(f"  - Overfitting: {resultado_semana_tree['diff_train_test']:.4f}")
    
    # Mostrar bins calculados
    print("\n" + "="*60)
    print("BINS CALCULADOS (TREE)")
    print("="*60)
    
    print("\n--- BINS SEMANAIS ---")
    for crypto in sorted(bins_semana_tree.keys()):
        print(f"\n{crypto}: {bins_semana_tree[crypto]}")
    
    print("\n" + "="*60)
    print("[OK] Pipeline concluido com sucesso!")
    print("="*60)
    print(f"\nResultados salvos em: {output_dir.absolute()}")
    print("\nArquivos gerados:")
    print(f"  - confusion_matrix_weekly_tree.png")
    print(f"\nTotal: 1 matriz de confusão (Tree)")
    
    return {
        # Modelos completos
        'modelos_semana_tree': modelo_semana_tree,
        # Modelos individuais
        'mlp_semana_tree': mlp_semana_tree,
        'scaler_semana_tree': scaler_semana_tree,
        'y_semana_pred_tree': y_semana_pred_tree,
        'bins_semana_tree': bins_semana_tree,
        # Resultados
        'resultados_semana_tree': resultado_semana_tree,
        # Compatibilidade
        'mlp_semana': mlp_semana_tree,
        'scaler_semana': scaler_semana_tree,
        'y_semana_pred': y_semana_pred_tree
    }


def main_semanal():
    """Pipeline apenas para dados semanais.
    
    OBS: Esta função foi mantida para compatibilidade. Use main() que executa
    apenas semanal com Tree.
    """
    print("[AVISO] main_semanal() está deprecada. Use main() que executa apenas semanal com Tree.")
    print("Redirecionando para main()...\n")
    return main()


if __name__ == "__main__":
    """
    Pipeline executa apenas:
    - Weekly (semanal)
    - Tree (apenas)
    
    Argumentos de linha de comando são ignorados (sempre executa semanal).
    """
    print("="*60)
    print("Pipeline MLP - Executando apenas semanal")
    print("="*60)
    print("Executa apenas: Weekly (semanal) | Tree")
    print("="*60)
    
    # Sempre executar main() que faz tudo
    results = main()

