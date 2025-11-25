"""
Pipeline completo que executa todos os passos necessários em sequência:
1. indicators.py - Gera indicadores técnicos (treino: 2020-2023, teste: 2024-2025)
2. mlp_pipeline.py - Treina modelos MLP (usa bins de main.py)
3. rl_pipeline.py - Treina e testa modelos RL (diário e semanal)

Períodos:
- Treino: 2020-01-01 até 2023-12-31 (incluso)
- Teste: 2024-01-01 até 2025-12-31 (incluso)
"""

import sys
import subprocess
from pathlib import Path
import time

# Adicionar src ao path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_command(description: str, script: str, args: list = None):
    """
    Executa um script Python e imprime o resultado.
    
    Args:
        description: Descrição do passo que está sendo executado
        script: Nome do script Python a ser executado
        args: Lista de argumentos adicionais para o script
    """
    print("\n" + "=" * 80)
    print(f" {description}")
    print("=" * 80)
    print(f"Executando: python {script} {' '.join(args) if args else ''}")
    print("=" * 80)
    
    cmd = [sys.executable, script]
    if args:
        cmd.extend(args)
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Mostrar output em tempo real
            text=True
        )
        elapsed = time.time() - start_time
        print(f"\n[OK] {description} concluído em {elapsed:.1f} segundos")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERRO] Falha ao executar {description}")
        print(f"Código de saída: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n[INTERROMPIDO] Execução de {description} foi interrompida pelo usuário")
        return False


def main():
    """
    Executa todos os pipelines em sequência.
    """
    print("=" * 80)
    print(" PIPELINE COMPLETO - DATATHON")
    print("=" * 80)
    print("\nPeríodos configurados:")
    print("  Treino: 2020-01-01 até 2023-12-31 (incluso)")
    print("  Teste:  2024-01-01 até 2025-12-31 (incluso)")
    print("\n" + "=" * 80)
    
    steps_completed = 0
    steps_total = 3
    
    # ============================================================
    # PASSO 1: Gerar indicadores técnicos
    # ============================================================
    print("\n" + "=" * 80)
    print(" PASSO 1/3: GERANDO INDICADORES TÉCNICOS")
    print("=" * 80)
    print("Arquivo: src/data/indicators.py")
    print("Saída: data/processed/")
    print("  - indicators_weekly_train_2020_2023.csv")
    print("  - indicators_weekly_test_2024_2025.csv")
    
    if not run_command(
        "PASSO 1: Gerando indicadores técnicos",
        "src/data/indicators.py"
    ):
        print("\n[ERRO] Falha no passo 1. Abortando pipeline.")
        return
    
    steps_completed += 1
    
    # Verificar se os arquivos foram criados
    data_dir = Path("data/processed")
    required_files = [
        "indicators_weekly_train_2020_2023.csv",
        "indicators_weekly_test_2024_2025.csv"
    ]
    
    missing_files = [f for f in required_files if not (data_dir / f).exists()]
    if missing_files:
        print(f"\n[ERRO] Arquivos faltando: {missing_files}")
        print("[AVISO] Continuando mesmo assim...")
    else:
        print("\n[OK] Todos os arquivos de indicadores foram gerados")
    
    # ============================================================
    # PASSO 2: Treinar modelos MLP
    # ============================================================
    print("\n" + "=" * 80)
    print(" PASSO 2/3: TREINANDO MODELOS MLP")
    print("=" * 80)
    print("Arquivo: src/models/mlp_pipeline.py")
    print("Saída: results/plots/")
    print("  - Confusion matrix (Tree semanal)")
    print("  - Métricas de classificação")
    
    if not run_command(
        "PASSO 2: Treinando modelos MLP",
        "src/models/mlp_pipeline.py"
    ):
        print("\n[ERRO] Falha no passo 2. Abortando pipeline.")
        return
    
    steps_completed += 1
    
    # ============================================================
    # PASSO 3: Treinar e testar modelos RL
    # ============================================================
    print("\n" + "=" * 80)
    print(" PASSO 3/3: TREINANDO E TESTANDO MODELOS RL")
    print("=" * 80)
    print("Arquivo: src/models/rl_pipeline.py")
    print("Saída: results/")
    print("  - Modelos PPO (semanal)")
    print("  - Backtests e comparações com benchmarks")
    print("  - Visualizações e métricas")
    
    # rl_pipeline.py já roda ambos os intervalos automaticamente
    # train_end padrão agora é 2023-12-31
    if not run_command(
        "PASSO 3: Treinando e testando modelos RL (semanal)",
        "src/models/rl_pipeline.py",
        ["--train_end", "2023-12-31"]  # Garantir período correto
    ):
        print("\n[ERRO] Falha no passo 3.")
        print("[AVISO] Pipeline completo parcialmente executado.")
        return
    
    steps_completed += 1
    
    # ============================================================
    # RESUMO FINAL
    # ============================================================
    print("\n" + "=" * 80)
    print(" PIPELINE COMPLETO EXECUTADO COM SUCESSO!")
    print("=" * 80)
    print(f"\nPassos concluídos: {steps_completed}/{steps_total}")
    print("\nArquivos gerados:")
    print("\n1. Indicadores Técnicos (data/processed/):")
    print("   - indicators_weekly_train_2020_2023.csv")
    print("   - indicators_weekly_test_2024_2025.csv")
    print("\n2. Modelos MLP (results/plots/):")
    print("   - confusion_matrix_weekly_tree.png")
    print("\n3. Modelos RL (results/):")
    print("   - ppo_model_1w_*.zip (semanal)")
    print("   - backtest_results_*.csv")
    print("   - dashboard_*.html")
    print("   - comparative_*.png (6 gráficos)")
    print("   - comparative_metrics_*.csv e .tex")
    print("\nPeríodos utilizados:")
    print("  Treino: 2020-01-01 até 2023-12-31")
    print("  Teste:  2024-01-01 até 2025-12-31")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERROMPIDO] Pipeline foi interrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERRO FATAL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
