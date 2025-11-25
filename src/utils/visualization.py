"""
Script para gerar figura da árvore de decisão para artigo acadêmico.

Este script demonstra como gerar uma figura profissional e explicativa
do processo de binning usando árvore de decisão, adequada para publicação.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from src.data.data_fetcher import (
    create_article_decision_tree_figure,
    create_supervised_static_bins_with_visualization,
    prepare_binance_data_for_training,
    BINANCE_SYMBOLS
)

def generate_article_figure_example():
    """
    Exemplo de como gerar a figura para artigo usando dados reais ou sintéticos.
    """
    print("="*60)
    print("GERAÇÃO DE FIGURA PARA ARTIGO")
    print("="*60)
    
    # Opção 1: Usar dados sintéticos para demonstração rápida
    use_synthetic = True  # Mude para False para usar dados reais
    
    if use_synthetic:
        print("\n[INFO] Usando dados sintéticos para demonstração...")
        
        # Criar dados sintéticos
        np.random.seed(42)
        n_samples = 1000
        returns = np.random.normal(0.001, 0.02, n_samples)
        
        # Criar estrutura de dados
        X_train_data = {
            'BTC': pd.DataFrame(index=pd.date_range('2020-01-01', periods=n_samples, freq='D'))
        }
        Y_train_data = {
            'BTC': pd.Series(returns, index=pd.date_range('2020-01-01', periods=n_samples, freq='D'))
        }
        
        # Criar bins e obter dados de visualização
        bin_intervals, viz_data = create_supervised_static_bins_with_visualization(
            X_train_data=X_train_data,
            Y_train_data=Y_train_data,
            visualize=False  # Não gerar visualizações detalhadas agora
        )
        
        # Gerar apenas a figura para artigo
        if 'BTC' in viz_data:
            data = viz_data['BTC']
            filepath = create_article_decision_tree_figure(
                crypto_name='BTC',
                y_train_values=data['y_train_values'],
                tree=data['tree'],
                bin_intervals=data['bin_intervals'],
                output_dir='article_figures',
                filename='decision_tree_article_BTC.png',
                language='pt'  # 'pt' para português, 'en' para inglês
            )
            print(f"\n[OK] Figura para artigo gerada: {filepath}")
    
    else:
        print("\n[INFO] Usando dados reais da Binance...")
        
        # Buscar dados reais
        X_train_data, Y_train_data = prepare_binance_data_for_training(
            symbols=BINANCE_SYMBOLS[:3],  # Usar apenas 3 para exemplo rápido
            interval='1d',
            start_time='2020-01-01',
            end_time='2023-12-31',
            limit=2000,
            use_cache=True,
            save_cache=False
        )
        
        # Gerar figura para cada criptomoeda
        bin_intervals, viz_data = create_supervised_static_bins_with_visualization(
            X_train_data=X_train_data,
            Y_train_data=Y_train_data,
            visualize=False
        )
        
        for crypto_name, data in viz_data.items():
            filepath = create_article_decision_tree_figure(
                crypto_name=crypto_name,
                y_train_values=data['y_train_values'],
                tree=data['tree'],
                bin_intervals=data['bin_intervals'],
                output_dir='article_figures',
                language='pt'
            )
            print(f"[OK] Figura para {crypto_name} gerada: {filepath}")


def generate_figure_for_specific_crypto(crypto_name: str, language: str = 'pt'):
    """
    Gera figura para uma criptomoeda específica.
    
    Args:
        crypto_name (str): Nome da criptomoeda (ex: 'BTC', 'ETH')
        language (str): Idioma ('pt' ou 'en')
    """
    print(f"\n{'='*60}")
    print(f"Gerando figura para {crypto_name}")
    print(f"{'='*60}")
    
    # Buscar dados
    symbol = f'{crypto_name}USDT' if crypto_name != 'USDC' else 'USDCUSDT'
    
    X_train_data, Y_train_data = prepare_binance_data_for_training(
        symbols=[symbol],
        interval='1d',
        start_time='2020-01-01',
        end_time='2023-12-31',
        limit=2000,
        use_cache=True,
        save_cache=False
    )
    
    # Criar bins
    bin_intervals, viz_data = create_supervised_static_bins_with_visualization(
        X_train_data=X_train_data,
        Y_train_data=Y_train_data,
        visualize=False
    )
    
    # Gerar figura
    if crypto_name in viz_data:
        data = viz_data[crypto_name]
        filepath = create_article_decision_tree_figure(
            crypto_name=crypto_name,
            y_train_values=data['y_train_values'],
            tree=data['tree'],
            bin_intervals=data['bin_intervals'],
            output_dir='article_figures',
            language=language
        )
        print(f"\n[OK] Figura gerada: {filepath}")
        return filepath
    else:
        print(f"[ERRO] {crypto_name} não encontrado nos dados")
        return None


if __name__ == "__main__":
    # Exemplo 1: Gerar figura com dados sintéticos (rápido)
    generate_article_figure_example()
    
    # Exemplo 2: Gerar figura para uma criptomoeda específica
    # generate_figure_for_specific_crypto('BTC', language='pt')
    
    print("\n" + "="*60)
    print("Processo concluído!")
    print("="*60)
    print("\nAs figuras foram salvas no diretório 'article_figures/'")
    print("Formato: PNG, 300 DPI, adequado para publicação")

