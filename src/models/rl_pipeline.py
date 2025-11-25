"""
Pipeline de Reinforcement Learning para Alocação de Portfólio de Criptomoedas

Este módulo implementa um ambiente de Eurytion (RL) para alocação de portfólio usando
PPO (Proximal Policy Optimization) do Stable-Baselines3.

Uso:
    python rl_pipeline.py --interval 1d --total_steps 150000 --seed 42
    python rl_pipeline.py --interval 1w --total_steps 150000 --no-plots
"""

from __future__ import annotations

import argparse
import os
import sys
import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Adicionar diretório raiz ao path para imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ============================================================
# VERIFICAÇÃO DE IMPORTS
# ============================================================

def check_imports():
    """Verifica se todos os imports necessários estão instalados."""
    missing_packages = []
    
    try:
        import gymnasium as gym
    except ImportError:
        missing_packages.append("gymnasium")
    
    try:
        import numpy as np
    except ImportError:
        missing_packages.append("numpy")
    
    try:
        import pandas as pd
    except ImportError:
        missing_packages.append("pandas")
    
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        missing_packages.append("plotly")
    
    try:
        from gymnasium import spaces
    except ImportError:
        if "gymnasium" not in missing_packages:
            missing_packages.append("gymnasium")
    
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.utils import set_random_seed
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        missing_packages.append("stable-baselines3")
    
    if missing_packages:
        print("=" * 60)
        print("ERRO: Pacotes faltando!")
        print("=" * 60)
        print(f"Os seguintes pacotes não estão instalados: {', '.join(missing_packages)}")
        print("\nPara instalar, execute:")
        print(f"  pip install {' '.join(missing_packages)}")
        print("\nOu instale todos de uma vez:")
        print("  pip install gymnasium numpy pandas plotly stable-baselines3")
        print("=" * 60)
        raise ImportError(f"Pacotes faltando: {', '.join(missing_packages)}")
    
    print("[OK] Todos os imports necessários estão instalados")
    return True

# Verificar imports no início do módulo
check_imports()

import gymnasium as gym
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import warnings
from gymnasium import spaces
from plotly.subplots import make_subplots
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

# ============================================================
# UTILIDADES BÁSICAS
# ============================================================


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Softmax estável numericamente: mapeia logits em probabilidades no simplex.
    Usado para transformar a ação contínua (logits) em pesos (somam 1).
    """
    x = x - np.max(x, axis=axis, keepdims=True)  # estabilidade numérica
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def ewma_update(prev_var: float, x: float, lam: float) -> float:
    """
    Atualiza a variância EWMA (RiskMetrics). 'lam' ~ 0.94 diário é comum.
    Aqui usamos 'x' como retorno líquido; mantemos uma proxy de vol local.
    """
    return lam * prev_var + (1 - lam) * (x ** 2)


def renorm_with_mask(w: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Aplica a máscara 0/1 (zera ativos não elegíveis) e renormaliza
    preservando apenas os ativos com mask=1. Se tudo zera, distribui
    uniformemente nos permitidos (ou zero se ninguém é permitido).
    """
    w = w * mask
    s = w.sum()
    if s <= 1e-12:
        cnt = int(mask.sum())
        if cnt == 0:
            return np.zeros_like(w)
        w = mask / cnt
    else:
        w /= s
    return w


# ============================================================
# CONFIGURAÇÃO DO AMBIENTE
# ============================================================


@dataclass
class EnvConfig:
    """
    Hiperparâmetros do ambiente:
      - cost_bps: custo por lado (bps) aplicado ao turnover (L1).
      - turnover_penalty: penalidade extra no reward para giro elevado.
      - include_cash: se True, adiciona 'caixa (USD)' como ativo extra.
      - risk_free_daily: retorno diário do caixa (ex.: 0.05/252 ~ 5% a.a.).
      - ewma_lambda: fator da EWMA para vol local.
      - lookback: empilha K dias de [S e P] no estado (memória curta).
      - clip_turnover: teto L1 de variação diária (se quiser limitar).
      - wmax: limite por ativo (cap máximo por cripto).
      - reward_mode: "local_sharpe" (padrão) ou "mean_variance".
      - lam_risk: aversão ao risco se utilizar "mean_variance".
      - eps: epsilon numérico para evitar divisão por zero.
      - logit_scale: escala aplicada aos logits antes do softmax.
    """

    cost_bps: float = 0.0
    turnover_penalty: float = 0.0
    include_cash: bool = True
    risk_free_daily: float = 0.03/252 # 3% a.a. USD
    ewma_lambda: float = 0.94
    lookback: int = 1
    clip_turnover: Optional[float] = None
    wmax: Optional[float] = None
    reward_mode: str = "local_sharpe"
    lam_risk: float = 0.0
    eps: float = 1e-8
    logit_scale: float = 5.0


# ============================================================
# AMBIENTE DE PORTFÓLIO (Gymnasium)
# ============================================================


class CryptoPortfolioEnv(gym.Env):
    """
    Ambiente de Eurytion (RL) para alocação diária:

    ENTRADAS (por dia t):
      - S[t]  : vetor (3) de sentimento [positivo, neutro, negativo]
      - P[t]  : matriz (N,5) de probabilidades em 5 bins de retorno por cripto
      - R[t+1]: retorno realizado das N criptos de t→t+1 (usado p/ calcular r_p)
      - M[t]  : máscara (N) 0/1 indicando quais criptos podem ser operadas no dia t

    AÇÃO:
      - vetor de logits com dimensão (N + 1 se include_cash=True, senão N).
      - mapeado via softmax (depois de escalar por logit_scale) para pesos.

    RECOMPENSA (padrão):
      - "local_sharpe": r_net / vol_EWMA - turnover_penalty * turnover
        (r_net já tem custo de transação deduzido)

    OBSERVAÇÃO (state):
      - Empilha 'lookback' dias de [S(t), P(t,:,:).reshape(-1)]
      - Acrescenta pesos prévios, vol_EWMA^0.5 e retorno líquido anterior.

    TERMINAÇÃO:
      - Quando t chega a end_index (janela definida na criação do env).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        returns: np.ndarray,  # shape (T, N) - retornos das criptos
        sentiments: np.ndarray,  # shape (T, 3) - sentimentos diários
        probs: np.ndarray,  # shape (T, N, 5) - probs por faixa de retorno
        mask: Optional[np.ndarray] = None,  # shape (T, N) 0/1 - elegibilidade
        config: EnvConfig = EnvConfig(),
        start_index: int = 0,  # início (inclusive) do episódio
        end_index: Optional[int] = None,  # fim (inclusive) do episódio
        seed: int = 42,
    ):
        super().__init__()
        # ------------ Checks de shape ------------
        assert returns.ndim == 2, "returns deve ser (T,N)"
        assert sentiments.shape[0] == returns.shape[0], "S e R devem ter mesmo T"
        assert probs.shape == (returns.shape[0], returns.shape[1], 5), "P deve ser (T,N,5)"
        if mask is None:
            mask = np.ones_like(returns, dtype=float)  # sem restrição de universo por padrão

        # ------------ Dados principais ------------
        self.R = returns.astype(np.float64)  # retornos
        self.S = sentiments.astype(np.float64)  # sentimentos
        self.P = probs.astype(np.float64)  # probabilidades por bin
        self.M = mask.astype(np.float64)  # máscara 0/1
        self.cfg = config
        self.N = self.R.shape[1]
        self.include_cash = config.include_cash
        self.dim_action = self.N + (1 if self.include_cash else 0)

        # ------------ Janela temporal do episódio ------------
        self.start_index = max(config.lookback, start_index)
        self.end_index = end_index if end_index is not None else self.R.shape[0] - 1
        if self.end_index <= self.start_index:
            raise ValueError("Intervalo inválido para episódio.")

        # ------------ Espaços Gym ------------
        # Estado empilha K dias de features (3 + 5*N), depois concatena pesos_prev e [vol, ret_prev]
        base_feat = 3 + 5 * self.N
        self.obs_per_t = base_feat
        self.obs_dim = self.obs_per_t * self.cfg.lookback + self.dim_action + 2

        high = np.ones(self.obs_dim, dtype=np.float64) * np.inf
        self.observation_space = spaces.Box(-high, high, dtype=np.float64)
        # Stable-Baselines3 exige bounds finitos no action space
        self.action_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.dim_action,), dtype=np.float64
        )

        # ------------ Estado interno dinâmico ------------
        self._t = None
        self._w_prev = None
        self._ewma_var = None
        self._ret_prev = 0.0

    # --------- Helpers para construir o estado ---------

    def _get_features_t(self, t: int) -> np.ndarray:
        """Concatena S[t] (3) e P[t] reshape (N*5)."""
        return np.concatenate([self.S[t], self.P[t].reshape(-1)])

    def _stack_lookback(self, t: int) -> np.ndarray:
        """Empilha 'lookback' dias de features até t (inclui o próprio t)."""
        feats = [
            self._get_features_t(tt)
            for tt in range(t - self.cfg.lookback + 1, t + 1)
        ]
        return np.concatenate(feats)

    def _build_obs(self, t: int) -> np.ndarray:
        """Monta a observação completa do passo t."""
        x = self._stack_lookback(t)
        obs = np.concatenate(
            [
                x,
                self._w_prev,
                np.array([np.sqrt(self._ewma_var + self.cfg.eps), self._ret_prev]),
            ]
        )
        return obs

    # --------- Transformação de ação em pesos válidos ---------

    def _apply_limits_and_mask(self, w_raw: np.ndarray, t: int) -> np.ndarray:
        """
        1) Escala logits e aplica softmax → pesos no simplex
        2) Separa pesos de cripto vs. caixa
        3) Aplica máscara de elegibilidade (zera ativos proibidos) e renormaliza
        4) Aplica limite por ativo (wmax) e renormaliza
        5) Se include_cash, renormaliza junto com caixa para somar 1
        6) (Opcional) Clipa turnover se solicitado em cfg.clip_turnover
        """
        w = softmax(self.cfg.logit_scale * w_raw, axis=-1)

        # separa parte de cripto e de caixa (se existir)
        if self.include_cash:
            w_assets, w_cash = w[:-1], w[-1]
        else:
            w_assets, w_cash = w, 0.0

        # aplica máscara 0/1 do dia t (top-N dinâmico, suspensões, etc.)
        w_assets = renorm_with_mask(w_assets, self.M[t])

        # aplica limite por ativo (cap) e renormaliza
        if self.cfg.wmax is not None:
            w_assets = np.minimum(w_assets, self.cfg.wmax)
            w_assets = renorm_with_mask(w_assets, self.M[t])

        # recompõe com caixa e garante soma=1
        if self.include_cash:
            s = w_assets.sum() + w_cash
            if s <= 1e-12:
                # se tudo sumiu, fica 100% caixa
                w_assets = np.zeros_like(w_assets)
                w_cash = 1.0
            else:
                w_assets = w_assets / s
                w_cash = w_cash / s
            
            # IMPORTANTE: Após normalizar com cash, garantir que os pesos ainda respeitem wmax
            # Isso pode ser necessário se a normalização com cash fez algum peso ultrapassar o limite
            if self.cfg.wmax is not None:
                w_assets_capped = np.minimum(w_assets, self.cfg.wmax)
                # Se algum peso foi limitado, renormalizar apenas os assets (sem cash)
                if not np.allclose(w_assets, w_assets_capped):
                    w_assets = w_assets_capped
                    w_assets = renorm_with_mask(w_assets, self.M[t])
                    # Recalcular cash para manter soma=1
                    w_cash = 1.0 - w_assets.sum()
                    if w_cash < 0:
                        w_cash = 0.0
                        # Se cash ficou negativo, renormalizar tudo novamente
                        total = w_assets.sum()
                        if total > 1e-12:
                            w_assets = w_assets / total
            
            w = np.concatenate([w_assets, np.array([w_cash])])
        else:
            w = w_assets / (w_assets.sum() + 1e-12)
            
            # Garantir que wmax ainda é respeitado (sem cash, não deve ser necessário, mas por segurança)
            if self.cfg.wmax is not None:
                w = np.minimum(w, self.cfg.wmax)
                w = renorm_with_mask(w, self.M[t])

        # limita giro diário (projeção L1 simples) — opcional
        if self.cfg.clip_turnover is not None and self._w_prev is not None:
            delta = w - self._w_prev
            l1 = np.abs(delta).sum()
            if l1 > self.cfg.clip_turnover + 1e-12:
                # Separar assets e cash para aplicar turnover limit corretamente
                if self.include_cash:
                    w_assets_prev = self._w_prev[:-1]
                    w_cash_prev = self._w_prev[-1]
                    w_assets_curr = w[:-1]
                    w_cash_curr = w[-1]
                    
                    delta_assets = w_assets_curr - w_assets_prev
                    delta_cash = w_cash_curr - w_cash_prev
                    l1_total = np.abs(delta_assets).sum() + abs(delta_cash)
                    
                    if l1_total > self.cfg.clip_turnover + 1e-12:
                        scale = self.cfg.clip_turnover / l1_total
                        w_assets = w_assets_prev + delta_assets * scale
                        w_cash = w_cash_prev + delta_cash * scale
                        w_assets = np.clip(w_assets, 0, 1)
                        w_cash = np.clip(w_cash, 0, 1)
                        
                        # Reaplicar wmax após turnover limit
                        if self.cfg.wmax is not None:
                            w_assets = np.minimum(w_assets, self.cfg.wmax)
                            w_assets = renorm_with_mask(w_assets, self.M[t])
                        
                        # Recalcular cash para manter soma=1
                        w_cash = 1.0 - w_assets.sum()
                        if w_cash < 0:
                            w_cash = 0.0
                            total = w_assets.sum()
                            if total > 1e-12:
                                w_assets = w_assets / total
                        
                        w = np.concatenate([w_assets, np.array([w_cash])])
                else:
                    w = self._w_prev + delta * (self.cfg.clip_turnover / l1)
                    w = np.clip(w, 0, 1)
                    # Reaplicar wmax após turnover limit
                    if self.cfg.wmax is not None:
                        w = np.minimum(w, self.cfg.wmax)
                    w = renorm_with_mask(w, self.M[t])
        
        # Verificação final: garantir que nenhum peso de asset ultrapasse wmax
        # (pode acontecer devido a erros numéricos ou normalizações)
        if self.cfg.wmax is not None:
            if self.include_cash:
                w_assets_final = w[:-1]
                if np.any(w_assets_final > self.cfg.wmax + 1e-6):  # Tolerância numérica
                    w_assets_final = np.minimum(w_assets_final, self.cfg.wmax)
                    w_assets_final = renorm_with_mask(w_assets_final, self.M[t])
                    w_cash_final = 1.0 - w_assets_final.sum()
                    if w_cash_final < 0:
                        w_cash_final = 0.0
                        total = w_assets_final.sum()
                        if total > 1e-12:
                            w_assets_final = w_assets_final / total
                    w = np.concatenate([w_assets_final, np.array([w_cash_final])])
            else:
                if np.any(w > self.cfg.wmax + 1e-6):
                    w = np.minimum(w, self.cfg.wmax)
                    w = renorm_with_mask(w, self.M[t])
        
        return w

    # --------- Gym API ---------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        """
        Inicia episódio na data start_index. Começa 100% em caixa (se include_cash),
        ou uniforme nas criptos caso não exista caixa.
        """
        super().reset(seed=seed)
        self._t = self.start_index
        self._w_prev = np.zeros(self.dim_action, dtype=np.float64)
        if self.include_cash:
            self._w_prev[-1] = 1.0  # 100% caixa no início por padrão
        else:
            self._w_prev[:] = 1.0 / self.dim_action
        self._ewma_var = 0.0
        self._ret_prev = 0.0
        obs = self._build_obs(self._t)
        return obs, {}

    def step(self, action: np.ndarray):
        """
        Converte ação→pesos, aplica custos, computa retorno líquido,
        atualiza EWMA e avança um dia. Retorna (obs, reward, done, truncated, info).
        """
        t = self._t
        w = self._apply_limits_and_mask(action, t)

        # se já estamos no final, não há t+1 para realizar retorno
        if t + 1 > self.end_index:
            return self._build_obs(t), 0.0, True, False, {"done_reason": "end"}

        # retorno dos ativos de t→t+1
        r_assets = self.R[t + 1]

        # custo de transação proporcional ao giro (L1)
        turnover = np.abs(w - self._w_prev).sum()
        cost = (self.cfg.cost_bps / 1e4) * turnover

        # retorno bruto do portfólio (com caixa opcional)
        if self.include_cash:
            r_p = (w[:-1] * r_assets).sum() + w[-1] * self.cfg.risk_free_daily
        else:
            r_p = (w * r_assets).sum()

        # retorno líquido (deduz custo)
        r_net = r_p - cost

        # update da variância EWMA (para Sharpe local)
        self._ewma_var = ewma_update(self._ewma_var, r_net, self.cfg.ewma_lambda)

        # recompensa
        if self.cfg.reward_mode == "local_sharpe":
            vol = np.sqrt(self._ewma_var + self.cfg.eps)
            reward = r_net / (vol + self.cfg.eps) - self.cfg.turnover_penalty * turnover
        elif self.cfg.reward_mode == "mean_variance":
            reward = (
                r_net
                - self.cfg.lam_risk * self._ewma_var
                - self.cfg.turnover_penalty * turnover
            )
        else:
            raise ValueError("reward_mode inválido")

        # avança o relógio e o estado
        self._t += 1
        self._w_prev = w
        self._ret_prev = r_net

        terminated = self._t >= self.end_index  # true no último passo da janela
        truncated = False  # não usamos truncation aqui
        obs = self._build_obs(self._t)

        # info útil para monitorar/plotar/backtest
        info = {
            "w": w.copy(),  # pesos aplicados
            "r_p": float(r_p),  # retorno bruto
            "r_net": float(r_net),  # retorno líquido (após custos)
            "turnover": float(turnover),
            "cost": float(cost),
            "reward": float(reward),
        }
        return obs, float(reward), terminated, truncated, info


# ============================================================
# SPLIT POR DATA E CRIAÇÃO DOS AMBIENTES (TRAIN / TEST)
# ============================================================


def split_indices_by_date(dates: np.ndarray, cutoff_train_end: np.datetime64) -> Tuple[int, int]:
    """
    Recebe 'dates' (array com T datas) e um cutoff (data final do treino).
    Retorna (t_train_end, t_test_end) como índices inclusivos.
    - treino = [0 .. t_train_end]
    - teste  = [t_train_end .. t_test_end]
    """
    dts = pd.to_datetime(dates).to_numpy(dtype="datetime64[D]")  # garante dia
    cutoff = np.datetime64(cutoff_train_end, "D")
    idx_le = np.where(dts <= cutoff)[0]
    if len(idx_le) == 0:
        raise ValueError("Nenhuma data <= cutoff para treino.")
    t_train_end = int(idx_le.max())
    t_test_end = len(dts) - 1
    if not (0 < t_train_end < t_test_end):
        raise ValueError("Corte de datas gera janelas inválidas.")
    return t_train_end, t_test_end


def make_env_train_test(
    R: np.ndarray,
    S: np.ndarray,
    P: np.ndarray,
    M: Optional[np.ndarray],
    cfg: EnvConfig,
    t_train_end: int,
    t_test_end: int,
) -> Tuple[CryptoPortfolioEnv, CryptoPortfolioEnv]:
    """
    Cria dois ambientes:
      - env_train: do início até t_train_end (inclusive)
      - env_test : de t_train_end até t_test_end (inclusive)
    """
    T = R.shape[0]
    if not (0 < t_train_end < t_test_end <= T - 1):
        raise ValueError("Índices inválidos para treino/teste.")
    env_train = CryptoPortfolioEnv(
        R, S, P, M, cfg, start_index=0, end_index=t_train_end
    )
    env_test = CryptoPortfolioEnv(
        R, S, P, M, cfg, start_index=t_train_end, end_index=t_test_end
    )
    return env_train, env_test


# ============================================================
# TREINO COM PPO (SEM VALIDAÇÃO)
# ============================================================


class TrainMetrics:
    """
    Estrutura para coletar métricas durante o treino (exploração):
      - reward médio do rollout amostrado
      - retorno líquido médio
      - turnover médio
      - Sharpe móvel (janela 252 dias) dos retornos líquidos vistos
    """

    def __init__(self):
        self.steps: List[int] = []
        self.rew_mean: List[float] = []
        self.rnet_mean: List[float] = []
        self.turn_mean: List[float] = []
        self.sharpe: List[float] = []
        self._buf = deque(maxlen=252)  # buffer para Sharpe móvel anualizado
        self._t = 0

    def push_rollout(self, infos_rollout: List[dict]):
        """
        Recebe uma lista de dicts 'info' (coletados do env.step) e atualiza as médias.
        """
        if not infos_rollout:
            return
        rnets = [d.get("r_net", 0.0) for d in infos_rollout]
        turns = [d.get("turnover", 0.0) for d in infos_rollout]
        rews = [d.get("reward", 0.0) for d in infos_rollout]
        self._buf.extend(rnets)
        self._t += len(infos_rollout)
        self.steps.append(self._t)
        self.rew_mean.append(float(np.mean(rews)))
        self.rnet_mean.append(float(np.mean(rnets)))
        self.turn_mean.append(float(np.mean(turns)))
        arr = np.array(self._buf, dtype=float)
        sharpe = 0.0 if arr.std() == 0 else arr.mean() / arr.std() * np.sqrt(252)
        self.sharpe.append(float(sharpe))


def train_ppo_no_val(
    env_train: gym.Env, total_steps: int = 300_000, seed: int = 42
) -> Tuple[PPO, TrainMetrics]:
    """
    Laço de treino com PPO (Stable-Baselines3) SEM validação.
    A cada 'n_steps' do PPO, coletamos um pequeno rollout no próprio env para
    registrar métricas de exploração (tm).
    """
    set_random_seed(seed)
    venv = DummyVecEnv([lambda: env_train])

    # Hiperparâmetros padrões (ajuste conforme necessário)
    model = PPO(
        policy="MlpPolicy",
        env=venv,
        n_steps=2048,  # tamanho do rollout por atualização
        batch_size=256,  # minibatch SGD
        learning_rate=3e-4,  # taxa de aprendizado
        gamma=0.99,  # desconto
        gae_lambda=0.95,  # GAE
        clip_range=0.2,  # PPO clipping
        ent_coef=0.0,  # entropia (exploração adicional)
        vf_coef=0.5,  # loss do value function
        seed=seed,
        verbose=0,
    )

    tm = TrainMetrics()
    steps_done = 0
    while steps_done < total_steps:
        # Atualiza o agente por 'n_steps'
        model.learn(
            total_timesteps=model.n_steps, reset_num_timesteps=False, progress_bar=False
        )
        steps_done += model.n_steps

        # Amostra um rollout "analítico" estocástico para medir métricas recentes
        tmp_obs, _ = env_train.reset()
        infos_tmp = []
        for _ in range(256):  # 256 passos são suficientes para médias estáveis
            a, _ = model.predict(tmp_obs, deterministic=False)  # estocástico = exploração
            tmp_obs, _, done, _, info = env_train.step(a)
            infos_tmp.append(info)
            if done:
                break
        tm.push_rollout(infos_tmp)

        print(
            f"[steps={steps_done}] train: reward_mean={tm.rew_mean[-1]:+.5f} "
            f"rnet_mean={tm.rnet_mean[-1]:+.5f} turn_mean={tm.turn_mean[-1]:.3f} "
            f"Sharpe_mov={tm.sharpe[-1]:+.2f}"
        )

    return model, tm


# ============================================================
# PLOTLY BACKTEST: utilitários, DataFrame e dashboards
# ============================================================


def compute_drawdown(nav: pd.Series) -> pd.Series:
    """
    Underwater: dd_t = (NAV_t / cummax(NAV)) - 1
    """
    peak = nav.cummax()
    dd = (nav / peak) - 1.0
    return dd


def rolling_sharpe(ret: pd.Series, window: int = 63, ann: int = 252, risk_free_rate: float = 0.0) -> pd.Series:
    """
    Sharpe móvel anualizado (média/vol * sqrt(ann)) em janela deslizante.
    
    Args:
        ret: Série de retornos
        window: Tamanho da janela móvel
        ann: Fator de anualização (252 para diários, 52 para semanais)
        risk_free_rate: Taxa livre de risco anualizada (padrão: 0.0)
    """
    mu = ret.rolling(window).mean()
    sd = ret.rolling(window).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        # Retorno médio anualizado menos risk_free (taxa livre de risco já está em base anual)
        excess_return = (mu * ann) - risk_free_rate
        shp = excess_return / (sd * np.sqrt(ann))
    return shp.replace([np.inf, -np.inf], np.nan)


def rolling_vol(ret: pd.Series, window: int = 63, ann: int = 252) -> pd.Series:
    """
    Vol móvel anualizada (std * sqrt(252)) em janela deslizante.
    """
    return ret.rolling(window).std() * np.sqrt(ann)


def run_backtest_df(
    model, env, dates: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Roda a política de forma DETERMINÍSTICA (exploit) do começo ao fim do env.
    Retorna DataFrame com colunas: date, ret_gross, ret_net, turnover, cost, nav, w_*
    Index será a própria 'date' (se fornecida) ou o passo 't'.
    """
    obs, _ = env.reset()
    rows, nav = [], 1.0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)
        nav *= 1.0 + info["r_net"]
        row = {
            "t": env._t,
            "date": (None if dates is None else pd.to_datetime(dates[env._t])),
            "ret_gross": info["r_p"],
            "ret_net": info["r_net"],
            "turnover": info["turnover"],
            "cost": info["cost"],
            "nav": nav,
        }
        # pesos
        for i, wi in enumerate(info["w"]):
            row[f"w_{i}"] = wi
        rows.append(row)
        if done:
            break
    df = pd.DataFrame(rows)
    # Index: datas se existir; senão, t
    if df["date"].notna().all():
        df.set_index("date", inplace=True)
        df.index.name = "date"
    else:
        df.set_index("t", inplace=True)
    return df


def plotly_backtest_dashboard(
    df_bt: pd.DataFrame,
    title: str = "Backtest (exploit)",
    ret_roll: int = 21,
    theme: str = "plotly_white",
) -> go.Figure:
    """
    4 subplots (compartilham eixo-x):
      1) NAV (linha)
      2) Underwater (drawdown em área)
      3) Retorno líquido diário (barras) + média móvel (linha)
      4) Turnover (linha) + média móvel (linha)
    """
    idx = df_bt.index
    ret_ma = df_bt["ret_net"].rolling(ret_roll).mean()
    dd = compute_drawdown(df_bt["nav"])
    turn_ma = df_bt["turnover"].rolling(ret_roll).mean()

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.35, 0.20, 0.25, 0.20]
    )

    # 1) NAV
    fig.add_trace(
        go.Scatter(x=idx, y=df_bt["nav"], mode="lines", name="NAV (líquido)"), row=1, col=1
    )

    # 2) Drawdown (underwater)
    fig.add_trace(
        go.Scatter(x=idx, y=dd, mode="lines", name="Drawdown", fill="tozeroy"), row=2, col=1
    )

    # 3) Retorno diário (barras) + média móvel
    fig.add_trace(
        go.Bar(x=idx, y=df_bt["ret_net"], name="Ret. líquido (diário)"), row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=idx, y=ret_ma, mode="lines", name=f"Média {ret_roll}d (ret_net)"),
        row=3,
        col=1,
    )

    # 4) Turnover (linha) + média móvel
    fig.add_trace(
        go.Scatter(x=idx, y=df_bt["turnover"], mode="lines", name="Turnover"), row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=idx, y=turn_ma, mode="lines", name=f"Média {ret_roll}d (turnover)"),
        row=4,
        col=1,
    )

    fig.update_layout(template=theme, title=title, hovermode="x unified", height=900)
    fig.update_yaxes(tickformat=".2%", row=2, col=1)  # drawdown em %
    fig.update_yaxes(tickformat=".2%", row=3, col=1)  # retorno % aproximado
    fig.update_yaxes(tickformat=".2f", row=4, col=1)  # turnover em fração (0-2)
    fig.update_xaxes(title="Data", row=4, col=1)
    return fig


def plotly_allocations(
    df_bt: pd.DataFrame,
    labels: Optional[Sequence[str]] = None,
    title: str = "Alocações (pesos)",
) -> go.Figure:
    """
    Área empilhada com w_i ao longo do tempo (se existir caixa, ela será a última coluna).
    Se 'labels' for fornecido, usa como nome das séries.
    """
    idx = df_bt.index
    w_cols = [c for c in df_bt.columns if c.startswith("w_")]
    W = df_bt[w_cols]
    if labels is None:
        labels = [c for c in w_cols]  # nomes padrão "w_0", ...
    else:
        assert len(labels) == len(w_cols), "labels deve ter mesmo comprimento que as colunas w_*"

    fig = go.Figure()
    for col, name in zip(w_cols, labels):
        fig.add_trace(
            go.Scatter(x=idx, y=W[col], mode="lines", stackgroup="one", name=name)
        )
    fig.update_layout(template="plotly_white", title=title, hovermode="x unified")
    fig.update_yaxes(tickformat=".0%", range=[0, 1], title="Peso")
    fig.update_xaxes(title="Data")
    return fig


def plotly_returns_distribution(
    df_bt: pd.DataFrame,
    bins: int = 60,
    title: str = "Distribuição de Retornos Diários (net)",
) -> go.Figure:
    """
    Histograma dos retornos diários líquidos, com linhas para média e mediana.
    """
    r = df_bt["ret_net"].dropna()
    mu, med = float(r.mean()), float(r.median())

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(x=r, nbinsx=bins, name="ret_net", histnorm="probability")
    )
    fig.add_vline(
        x=mu,
        line_dash="dash",
        annotation_text=f"média={mu:.4f}",
        annotation_position="top right",
    )
    fig.add_vline(
        x=med,
        line_dash="dot",
        annotation_text=f"mediana={med:.4f}",
        annotation_position="bottom right",
    )
    fig.update_layout(template="plotly_white", title=title)
    fig.update_xaxes(title="ret_net")
    fig.update_yaxes(title="Frequência")
    return fig


def plotly_rolling_metrics(
    df_bt: pd.DataFrame,
    window: int = 63,
    title: str = "Métricas Móveis (63d ~ 3 meses)",
    is_weekly: bool = False,
    risk_free_rate: float = 0.03,
) -> go.Figure:
    """
    Mostra Sharpe móvel anualizado e Vol móvel anualizada.
    
    Args:
        df_bt: DataFrame com retornos
        window: Tamanho da janela móvel
        title: Título do gráfico
        is_weekly: Se True, retornos são semanais (anualiza com 52), senão diários (anualiza com 252)
        risk_free_rate: Taxa livre de risco anualizada (padrão: 0.03 = 3% a.a.)
    """
    r = df_bt["ret_net"].astype(float)
    ann_factor = 52 if is_weekly else 252
    shp = rolling_sharpe(r, window=window, ann=ann_factor, risk_free_rate=risk_free_rate)
    vol = rolling_vol(r, window=window, ann=ann_factor)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    window_label = f"{window}{'w' if is_weekly else 'd'}"
    fig.add_trace(
        go.Scatter(x=df_bt.index, y=shp, name=f"Sharpe {window_label}", mode="lines", line=dict(width=2)), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_bt.index, y=vol, name=f"Vol {window_label} (ann.)", mode="lines", line=dict(width=2)), row=2, col=1
    )
    fig.update_layout(template="plotly_white", title=title, hovermode="x unified", height=600)
    fig.update_yaxes(title="Sharpe Ratio", row=1, col=1)
    fig.update_yaxes(title="Vol (ann.)", tickformat=".2%", row=2, col=1)
    fig.update_xaxes(title="Data", row=2, col=1)
    return fig


def plotly_monthly_heatmap(
    df_bt: pd.DataFrame, title: str = "Retornos Mensais (net)"
) -> go.Figure:
    """
    Agrega retornos diários em retornos mensais (soma dos diários) e mostra heatmap.
    """
    df = df_bt.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Para o heatmap mensal, o índice do DataFrame precisa ser DatetimeIndex.")
    monthly = df["ret_net"].resample("M").sum()
    pivot = (
        monthly.to_frame("ret_m")
        .assign(year=lambda x: x.index.year, month=lambda x: x.index.month)
        .pivot_table(index="year", columns="month", values="ret_m", aggfunc="sum")
        .sort_index()
    )

    # cria matriz ordenada 1..12 colunas
    pivot = pivot.reindex(columns=range(1, 13))

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[
                pd.Timestamp(month=m, year=2000, day=1).strftime("%b")
                for m in pivot.columns
            ],
            y=pivot.index.astype(str),
            zmid=0,
            colorscale="RdBu",
        )
    )
    fig.update_layout(template="plotly_white", title=title)
    fig.update_xaxes(title="Mês")
    fig.update_yaxes(title="Ano")
    return fig


# ============================================================
# CARREGAR DADOS REAIS
# ============================================================


def load_mlp_model_arch1_tree(interval='1d'):
    """
    Carrega ou treina o modelo MLP arquitetura única (32, 64) tree do mlp_pipeline.py.
    
    Args:
        interval: '1d' para diário ou '1w' para semanal
    
    Returns:
        tuple: (mlp_model, scaler, encoder, bins_dict) - modelo MLP, scaler, encoder e bins usados
    """
    if interval == '1d':
        print("  Carregando/treinando modelo MLP arquitetura (32, 64) tree (daily)...")
        interval_name = 'daily'
        data_dir = Path("data/processed")
        train_file = "indicators_daily_train_2020_2023.csv"
        test_file = "indicators_daily_test_2024_2025.csv"
    elif interval == '1w':
        print("  Carregando/treinando modelo MLP arquitetura (32, 64) tree (weekly)...")
        interval_name = 'weekly'
        data_dir = Path("data/processed")
        train_file = "indicators_weekly_train_2020_2023.csv"
        test_file = "indicators_weekly_test_2024_2025.csv"
    else:
        raise ValueError(f"Intervalo inválido: {interval}. Use '1d' ou '1w'")
    
    try:
        # Forçar recarregamento do módulo para evitar cache
        import sys
        if 'mlp_pipeline' in sys.modules:
            del sys.modules['mlp_pipeline']
        
        # Importar funções necessárias do mlp_pipeline
        from src.models.mlp_pipeline import (
            processar_dados, 
            treinar_mlp_arquitetura_unica,
            compute_bins_from_main,
            preparar_dados_para_classificacao
        )
        from sklearn.preprocessing import OneHotEncoder
        import pandas as pd
        
        # 1. Calcular bins tree para o intervalo especificado
        print(f"    - Calculando bins tree ({interval_name})...")
        bins_results = compute_bins_from_main(interval, use_both=True, bin_method='tree')
        bins_tree = bins_results.get('tree')
        
        if bins_tree is None:
            raise ValueError(f"Bins tree {interval_name} não foram calculados")
        
        # 2. Carregar dados de indicadores
        print(f"    - Carregando dados de indicadores ({interval_name})...")
        if interval == "1d":
            train_file_mlp = "indicators_daily_train_2020_2023.csv"
            test_file_mlp = "indicators_daily_test_2024_2025.csv"
        else:  # interval == "1w"
            train_file_mlp = "indicators_weekly_train_2020_2023.csv"
            test_file_mlp = "indicators_weekly_test_2024_2025.csv"
        
        indicadores_treino = pd.read_csv(data_dir / train_file_mlp)
        indicadores_teste = pd.read_csv(data_dir / test_file_mlp)
        
        # Remover timestamp
        indicadores_treino = indicadores_treino.drop(columns=['timestamp'], errors='ignore')
        indicadores_teste = indicadores_teste.drop(columns=['timestamp'], errors='ignore')
        
        # 3. Processar dados
        print("    - Processando dados...")
        X_train, y_train, X_test, y_test = processar_dados(
            indicadores_treino.copy(),
            indicadores_teste.copy(),
            bins_tree,
            f"{interval_name.upper()} - TREE (para Eurytion)"
        )
        
        # 4. Treinar modelo com arquitetura única (32, 64)
        print("    - Treinando modelo MLP arquitetura (32, 64)...")
        modelo = treinar_mlp_arquitetura_unica(
            X_train, y_train, X_test, y_test,
            f"{interval_name.upper()} - TREE (para Eurytion)",
            random_state=42
        )
        
        # Retornar modelo
        mlp_model = modelo['mlp']
        scaler = modelo['scaler']
        
        # Recriar encoder (necessário para transformar novos dados)
        print("    - Preparando encoder...")
        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
        # Fit encoder com dados de treino
        indicadores_enc_train = enc.fit_transform(indicadores_treino[['symbol']])
        
        print(f"    [OK] Modelo MLP arquitetura (32, 64) tree ({interval_name}) carregado/treinado")
        
        return mlp_model, scaler, enc, bins_tree
        
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar/treinar modelo MLP: {e}")


def load_real_data_for_eurytion(
    interval: str = "1d",  # '1d' ou '1w'
    train_end: str = "2023-12-31",  # Treino: 2020-2023, Teste: 2024-2025
    use_pickle_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Carrega dados reais dos arquivos gerados e converte para formato Eurytion:
    - R: retornos (T, N)
    - S: sentimentos (T, 3) - [negativo, neutro, positivo] (de daily_sentiment_full.csv)
    - P: probabilidades (T, N, 5) - probs dos 5 bins por cripto (do MLP arquitetura 1 tree daily)
    - M: máscara (T, N) - 1 se pode operar, 0 caso contrário
    - dates: array de datas (T,)
    
    Args:
        interval: '1d' para diário ou '1w' para semanal (NOTA: atualmente apenas '1d' é suportado)
        train_end: data final do treino (str)
        use_pickle_cache: se True, tenta carregar de arquivos pickle
    
    Returns:
        Tuple com (R, S, P, M, dates, crypto_list)
    """
    if interval not in ["1d", "1w"]:
        raise ValueError(f"Intervalo inválido: {interval}. Use '1d' para diário ou '1w' para semanal.")
    
    # Lista de criptos (sem USD e sem USDC, apenas risk-on)
    crypto_list = ["ADA", "BNB", "BTC", "DOGE", "ETH", "SOL", "TRX", "XRP"]
    N = len(crypto_list)

    interval_name = "daily" if interval == "1d" else "weekly"
    print(f"Carregando dados reais para intervalo {interval} ({interval_name})...")
    
    if interval == "1d":
        print("  Usando: MLP arquitetura (32, 64) tree (daily) + daily_sentiment_full.csv")
        sentiment_file = "data/raw/daily_sentiment_full.csv"
    else:
        print("  Usando: MLP arquitetura (32, 64) tree (weekly) + daily_sentiment_full.csv (agregado semanalmente)")
        sentiment_file = "data/raw/daily_sentiment_full.csv"  # Usaremos o mesmo arquivo e agregaremos

    # 1) Carregar modelo MLP arquitetura (32, 64) tree para o intervalo especificado
    print(f"\n1. Carregando modelo MLP arquitetura (32, 64) tree ({interval_name})...")
    mlp_model, scaler_mlp, encoder, bins_dict = load_mlp_model_arch1_tree(interval=interval)

    # 2) Carregar indicadores (treino + teste) para obter features
    data_dir = Path("data/processed")
    if interval == "1d":
        train_path = data_dir / "indicators_daily_train_2020_2023.csv"
        test_path = data_dir / "indicators_daily_test_2024_2025.csv"
    else:  # interval == "1w"
        train_path = data_dir / "indicators_weekly_train_2020_2023.csv"
        test_path = data_dir / "indicators_weekly_test_2024_2025.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Arquivos de indicadores não encontrados. Execute indicators.py primeiro.\n"
            f"Procurando: {train_path} e {test_path}"
        )

    print(f"\n2. Carregando dados de indicadores ({interval_name})...")
    print(f"  Carregando {train_path}...")
    df_train = pd.read_csv(train_path)
    print(f"  Carregando {test_path}...")
    df_test = pd.read_csv(test_path)

    # Concatenar treino e teste
    df_full = pd.concat([df_train, df_test], ignore_index=True)
    df_full["timestamp"] = pd.to_datetime(df_full["timestamp"])
    df_full = df_full.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    # Filtrar apenas as criptos válidas
    df_full = df_full[df_full["symbol"].isin(crypto_list)].copy()

    # Calcular retornos se não existirem
    if "ret_1" not in df_full.columns:
        print("  Calculando ret_1 a partir de price_close...")
        df_full["ret_1"] = np.nan
        for sym in crypto_list:
            mask = df_full["symbol"] == sym
            group = df_full.loc[mask].copy().sort_values("timestamp")
            if len(group) > 1:
                prices = group["price_close"].values
                returns = np.log(prices[1:] / prices[:-1])
                indices = group.index[1:]
                df_full.loc[indices, "ret_1"] = returns

    # Remover NaN
    df_full = df_full.dropna(subset=["ret_1", "timestamp"]).copy()

    # 3) Criar pivot tables para alinhamento temporal
    # Retornos (R)
    df_returns = df_full.pivot_table(
        index="timestamp", columns="symbol", values="ret_1", aggfunc="first"
    )

    # Garantir ordem das colunas
    df_returns = df_returns[crypto_list]

    # Remover linhas com NaN
    df_returns = df_returns.dropna()

    # 4) Calcular probabilidades dos bins (P) usando MLP arquitetura (32, 64) tree
    print("\n3. Calculando probabilidades dos bins (P) usando MLP arquitetura (32, 64) tree...")
    
    # Usar a mesma função processar_dados do mlp_pipeline para garantir compatibilidade
    # Mas precisamos preparar os dados de forma diferente - usar apenas para obter o formato de features
    # Processar dados do treino uma vez para obter as colunas esperadas
    print("    - Preparando features no formato esperado pelo modelo...")
    from src.models.mlp_pipeline import processar_dados as processar_dados_mlp
    
    # Carregar dados de treino temporariamente para obter estrutura de features
    # IMPORTANTE: Usar o mesmo intervalo que o modelo foi treinado (interval)
    if interval == "1d":
        temp_train_file = "indicators_daily_train_2020_2023.csv"
        temp_test_file = "indicators_daily_test_2024_2025.csv"
    else:  # interval == "1w"
        temp_train_file = "indicators_weekly_train_2020_2023.csv"
        temp_test_file = "indicators_weekly_test_2024_2025.csv"
    
    temp_train = pd.read_csv(data_dir / temp_train_file)
    temp_test = pd.read_csv(data_dir / temp_test_file)
    temp_train = temp_train.drop(columns=['timestamp'], errors='ignore')
    temp_test = temp_test.drop(columns=['timestamp'], errors='ignore')
    
    # Processar temporariamente para obter X_train e descobrir as colunas esperadas
    X_train_temp, _, _, _ = processar_dados_mlp(
        temp_train.copy(),
        temp_test.copy(),
        bins_dict,
        f"TEMPORÁRIO ({interval_name} - para obter formato)"
    )
    
    # Obter lista de colunas esperadas pelo modelo
    expected_columns = list(X_train_temp.columns)
    print(f"    - Features esperadas pelo modelo: {len(expected_columns)} colunas")
    print(f"    - Colunas: {list(expected_columns)}")
    
    # Preparar dados completos usando a mesma função processar_dados
    # Mas precisamos fazer isso por timestamp
    # Criar um DataFrame temporário com target dummy para processar_dados funcionar
    df_full_processed = df_full.copy()
    
    # Adicionar target dummy (não será usado, mas é necessário para processar_dados)
    # Aplicar bins temporariamente apenas para obter estrutura correta
    from src.models.mlp_pipeline import preparar_dados_para_classificacao
    
    df_full_with_target = preparar_dados_para_classificacao(df_full.copy(), bins_dict)
    
    # Remover timestamp
    df_full_with_target = df_full_with_target.drop(columns=['timestamp'], errors='ignore')
    
    # Aplicar OneHot encoding (usando encoder já treinado)
    if 'symbol' in df_full_with_target.columns:
        df_symbol_enc_full = encoder.transform(df_full_with_target[['symbol']])
        df_full_with_target = pd.concat([
            df_full_with_target.drop(columns=['symbol']), 
            df_symbol_enc_full
        ], axis=1)
    
    # Separar X (sem target) - isso deve ter as mesmas colunas que X_train
    df_features_processed = df_full_with_target.drop(columns=['target'], errors='ignore')
    
    # Garantir que as colunas estão na mesma ordem que o modelo espera
    # Reordenar colunas para corresponder ao modelo
    missing_cols = set(expected_columns) - set(df_features_processed.columns)
    extra_cols = set(df_features_processed.columns) - set(expected_columns)
    
    if missing_cols:
        print(f"    [AVISO] Colunas faltando: {missing_cols}")
        # Adicionar colunas faltantes com zeros
        for col in missing_cols:
            df_features_processed[col] = 0.0
    
    if extra_cols:
        print(f"    [AVISO] Colunas extras: {extra_cols}")
        # Remover colunas extras
        df_features_processed = df_features_processed.drop(columns=list(extra_cols))
    
    # Reordenar colunas para corresponder ao modelo
    df_features_processed = df_features_processed[expected_columns]
    
    # Preencher NaN
    df_features_processed = df_features_processed.fillna(0)
    
    # Processar por timestamp
    P_list = []
    df_full_with_timestamp = df_full.copy()
    
    for t, timestamp in enumerate(df_returns.index):
        probs_t = np.zeros((N, 5), dtype=np.float64)
        
        # Obter índices das linhas deste timestamp
        mask_timestamp = df_full_with_timestamp["timestamp"] == timestamp
        
        if not mask_timestamp.any():
            probs_t[:] = 0.2
            P_list.append(probs_t)
            continue
        
        # Obter features para este timestamp (já processadas e na ordem correta)
        indices_timestamp = df_full_with_timestamp[mask_timestamp].index
        X_timestamp = df_features_processed.loc[indices_timestamp].copy()
        
        # Garantir ordem das criptos
        symbols_timestamp = df_full_with_timestamp.loc[indices_timestamp, 'symbol'].values
        symbol_to_order = {sym: i for i, sym in enumerate(crypto_list)}
        order_list = [symbol_to_order.get(sym, 999) for sym in symbols_timestamp]
        sorted_indices = sorted(range(len(order_list)), key=lambda i: order_list[i])
        
        X_timestamp_sorted = X_timestamp.iloc[sorted_indices].copy()
        
        # Aplicar scaler (garantir ordem das colunas)
        X_timestamp_sorted = X_timestamp_sorted[expected_columns]
        X_scaled = scaler_mlp.transform(X_timestamp_sorted.values)
        
        # Prever probabilidades usando MLP
        probs_mlp = mlp_model.predict_proba(X_scaled)
        
        # Função auxiliar para garantir 5 classes nas probabilidades
        def ensure_5_classes(probs_array, n_classes_expected=5):
            """
            Garante que o array de probabilidades tenha exatamente n_classes_expected classes.
            Se tiver menos, preenche com zeros e renormaliza.
            """
            if len(probs_array.shape) == 1:
                # Array 1D (uma predição)
                n_classes = probs_array.shape[0]
                if n_classes < n_classes_expected:
                    # Criar array com 5 classes, preencher com zeros e renormalizar
                    probs_5 = np.zeros(n_classes_expected, dtype=np.float64)
                    probs_5[:n_classes] = probs_array
                    # Renormalizar
                    probs_5 = probs_5 / (probs_5.sum() + 1e-10)
                    return probs_5
                elif n_classes > n_classes_expected:
                    # Se tiver mais classes, pegar apenas as primeiras n_classes_expected
                    return probs_array[:n_classes_expected]
                else:
                    return probs_array
            else:
                # Array 2D (múltiplas predições)
                return np.array([ensure_5_classes(p, n_classes_expected) for p in probs_array])
        
        # Garantir que todas as probabilidades tenham 5 classes
        probs_mlp = ensure_5_classes(probs_mlp, n_classes_expected=5)
        
        # Mapear para ordem correta das criptos
        symbols_sorted = [symbols_timestamp[i] for i in sorted_indices]
        
        for j, sym in enumerate(crypto_list):
            if sym in symbols_sorted:
                idx_in_sorted = symbols_sorted.index(sym)
                if idx_in_sorted < len(probs_mlp):
                    probs_array = probs_mlp[idx_in_sorted]
                    # Garantir que tem exatamente 5 elementos
                    if len(probs_array) == 5:
                        probs_t[j] = probs_array.astype(np.float64)
                    else:
                        # Se não tiver 5, usar distribuição uniforme
                        probs_t[j] = np.full(5, 0.2, dtype=np.float64)
                else:
                    probs_t[j] = np.full(5, 0.2, dtype=np.float64)
            else:
                probs_t[j] = np.full(5, 0.2, dtype=np.float64)
        
        P_list.append(probs_t)

    P = np.array(P_list, dtype=np.float64)

    # 5) Carregar sentimentos (S) de data/raw/daily_sentiment_full.csv
    print(f"\n4. Carregando sentimentos (S) de {sentiment_file}...")
    sentiment_path = Path(sentiment_file)
    
    if not sentiment_path.exists():
        raise FileNotFoundError(
            f"Arquivo de sentimentos não encontrado: {sentiment_path}"
        )
    
    df_sentiment = pd.read_csv(sentiment_path)
    df_sentiment["date"] = pd.to_datetime(df_sentiment["date"])
    df_sentiment = df_sentiment.sort_values("date").reset_index(drop=True)
    
    # Se for semanal, agregar sentimentos por semana
    if interval == "1w":
        print("    - Agregando sentimentos diários para semanais...")
        # Agrupar por semana (início da semana = segunda-feira)
        df_sentiment["week"] = df_sentiment["date"].dt.to_period("W-MON")
        df_sentiment_weekly = df_sentiment.groupby("week").agg({
            "positive": "mean",
            "negative": "mean",
            "neutral": "mean"
        }).reset_index()
        # Converter semana para data (início da semana)
        df_sentiment_weekly["date"] = df_sentiment_weekly["week"].dt.start_time
        df_sentiment_weekly = df_sentiment_weekly.drop(columns=["week"])
        df_sentiment = df_sentiment_weekly
    
    # Mapear sentimentos para datas do df_returns
    # O formato é: [negative, neutral, positive] mas o CSV tem: [positive, negative, neutral]
    S = np.zeros((len(df_returns), 3), dtype=np.float64)
    
    for t, timestamp in enumerate(df_returns.index):
        # Buscar sentimento para esta data
        timestamp_pd = pd.Timestamp(timestamp)
        sentiment_row = df_sentiment[df_sentiment["date"].dt.date == timestamp_pd.date()]
        
        if len(sentiment_row) > 0:
            # CSV tem: positive, negative, neutral
            # Eurytion precisa: [negative, neutral, positive]
            S[t, 0] = float(sentiment_row.iloc[0]["negative"])   # negativo
            S[t, 1] = float(sentiment_row.iloc[0]["neutral"])    # neutro
            S[t, 2] = float(sentiment_row.iloc[0]["positive"])   # positivo
        else:
            # Se não encontrar, usar valores padrão (uniforme)
            S[t] = np.array([0.33, 0.34, 0.33])
        
        # Normalizar para somar 1
        S[t] = S[t] / (S[t].sum() + 1e-10)

    # 6) Retornos (R)
    R = df_returns.values.astype(np.float64)

    # 7) Máscara (M) - tudo habilitado por padrão (1)
    M = np.ones((len(df_returns), N), dtype=np.float64)

    # 8) Datas
    dates = df_returns.index.values.astype("datetime64[D]")

    # 9) Sanity checks
    T = len(df_returns)
    assert S.shape == (T, 3), f"S shape: {S.shape}, esperado: ({T}, 3)"
    assert P.shape == (T, N, 5), f"P shape: {P.shape}, esperado: ({T}, {N}, 5)"
    assert R.shape == (T, N), f"R shape: {R.shape}, esperado: ({T}, {N})"
    assert M.shape == (T, N), f"M shape: {M.shape}, esperado: ({T}, {N})"
    assert len(dates) == T, f"dates length: {len(dates)}, esperado: {T}"

    print(f"  [OK] Dados carregados:")
    print(f"    - Período: {dates[0]} a {dates[-1]}")
    print(f"    - T (dias): {T}")
    print(f"    - N (criptos): {N}")
    print(f"    - Criptos: {crypto_list}")

    return R, S, P, M, dates, crypto_list


# ============================================================
# BENCHMARKS E COMPARAÇÃO
# ============================================================


def compute_buy_and_hold_equal_weight(
    returns_df: pd.DataFrame,
    initial_capital: float = 1.0,
) -> pd.DataFrame:
    """
    Calcula performance de Buy & Hold Equal Weight (pesos iguais em todas as criptos).
    
    Args:
        returns_df: DataFrame com retornos diários (index: datas, columns: criptos)
        initial_capital: Capital inicial (padrão: 1.0)
    
    Returns:
        DataFrame com colunas: date, value, return, strategy='Buy & Hold EW'
    """
    n_assets = len(returns_df.columns)
    weights = np.ones(n_assets) / n_assets  # Pesos iguais
    
    portfolio_values = []
    portfolio_value = initial_capital
    
    for date, daily_returns in returns_df.iterrows():
        # Retorno do portfólio
        port_return = (weights * daily_returns.values).sum()
        portfolio_value *= (1 + port_return)
        
        portfolio_values.append({
            'date': date,
            'value': portfolio_value,
            'return': port_return,
            'strategy': 'Buy & Hold EW'
        })
    
    df = pd.DataFrame(portfolio_values)
    df.set_index('date', inplace=True)
    return df


def optimize_markowitz_sharpe(
    returns_df: pd.DataFrame,
    lookback_days: int = 252,
    rebal_freq: int = 63,  # Rebalanceamento a cada ~3 meses
    initial_capital: float = 1.0,
    max_weight: float = 0.35,
    min_weight: float = 0.0,
    is_weekly: bool = False,  # Se True, retornos são semanais
) -> pd.DataFrame:
    """
    Otimização Markowitz para maximizar Sharpe Ratio.
    Rebalancea periodicamente usando janela móvel de retornos históricos.
    
    Args:
        returns_df: DataFrame com retornos (index: datas, columns: criptos)
        lookback_days: Janela de retornos históricos para estimar média e covariância
        rebal_freq: Frequência de rebalanceamento (dias ou semanas conforme is_weekly)
        initial_capital: Capital inicial
        max_weight: Peso máximo por ativo
        min_weight: Peso mínimo por ativo (0 = sem short)
        is_weekly: Se True, retornos são semanais (anualiza com 52), senão diários (anualiza com 252)
    
    Returns:
        DataFrame com colunas: date, value, return, strategy='Markowitz'
    """
    from scipy.optimize import minimize
    
    assets = returns_df.columns.tolist()
    n_assets = len(assets)
    
    # Fator de anualização baseado na frequência
    ann_factor = 52 if is_weekly else 252
    min_data_points = 13 if is_weekly else 63  # Mínimo de dados necessário
    
    portfolio_values = []
    portfolio_value = initial_capital
    current_weights = np.ones(n_assets) / n_assets  # Começa com peso igual
    last_rebal_date = None
    
    for i, (date, period_returns) in enumerate(returns_df.iterrows()):
        # Verifica se deve rebalancear
        if last_rebal_date is None or i == 0:
            should_rebal = True
        else:
            if is_weekly:
                # Para semanais, verifica diferença em semanas
                weeks_diff = (date - last_rebal_date).days / 7
                should_rebal = weeks_diff >= rebal_freq
            else:
                # Para diários, verifica diferença em dias
                should_rebal = (date - last_rebal_date).days >= rebal_freq
        
        if should_rebal and i >= lookback_days:
            # Usa apenas dados históricos (evita lookahead bias)
            hist_returns = returns_df.iloc[:i].tail(lookback_days)
            
            # Sempre garantir que temos pelo menos 63 dias de histórico
            # Para semanais, isso significa 9 semanas
            min_required = 9 if is_weekly else 63
            if len(hist_returns) >= min_required:  # Mínimo de 63 dias
                # Estima média e covariância anualizados
                mu_annual = hist_returns.mean() * ann_factor
                Sigma_annual = hist_returns.cov() * ann_factor
                
                # Função objetivo: minimizar -Sharpe (maximizar Sharpe)
                def objective(w):
                    port_return = w @ mu_annual.values
                    port_variance = w @ Sigma_annual.values @ w
                    port_std = np.sqrt(port_variance)
                    sharpe = port_return / port_std if port_std > 0 else 0
                    return -sharpe  # Minimizar negativo = maximizar positivo
                
                # Restrições
                constraints = [
                    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Totalmente investido
                ]
                
                # Limites
                bounds = [(min_weight, max_weight) for _ in range(n_assets)]
                
                # Otimiza
                try:
                    result = minimize(
                        objective,
                        current_weights,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints,
                        options={'maxiter': 1000, 'ftol': 1e-9}
                    )
                    
                    if result.success:
                        current_weights = result.x
                        last_rebal_date = date
                except Exception as e:
                    # Em caso de erro, mantém pesos anteriores
                    pass
        
        # Calcula retorno do portfólio com pesos atuais
        port_return = (current_weights * period_returns.values).sum()
        portfolio_value *= (1 + port_return)
        
        portfolio_values.append({
            'date': date,
            'value': portfolio_value,
            'return': port_return,
            'strategy': 'Markowitz'
        })
    
    df = pd.DataFrame(portfolio_values)
    df.set_index('date', inplace=True)
    return df


def calculate_strategy_metrics(
    values_df: pd.DataFrame,
    initial_capital: float = 1.0,
    is_weekly: bool = False,
    risk_free_rate: float = 0.03,  # 3% a.a.
) -> Dict[str, float]:
    """
    Calcula métricas de performance para uma estratégia.
    
    Args:
        values_df: DataFrame com colunas: value, return (e opcionalmente index com datas)
        initial_capital: Capital inicial
        is_weekly: Se True, retornos são semanais (anualiza com 52), senão diários (anualiza com 252)
        risk_free_rate: Taxa livre de risco anualizada (padrão: 3% a.a.)
    
    Returns:
        Dicionário com métricas
    """
    returns = values_df['return'].copy()
    values = values_df['value'].copy()
    
    # Remover NaN e inf
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(returns) == 0 or len(values) == 0:
        # Retornar métricas zeradas se não houver dados válidos
        return {
            'Initial Capital': initial_capital,
            'Final Value': initial_capital,
            'Total Return': 0.0,
            'CAGR': 0.0,
            'Annual Volatility': 0.0,
            'Sharpe Ratio': 0.0,
            'Sortino Ratio': 0.0,
            'Max Drawdown': 0.0,
            'Win Rate': 0.0,
            'Backtest Period (years)': 0.0,
        }
    
    # Determinar período e fator de anualização
    ann_factor = 52 if is_weekly else 252  # Dias/semanas de trading por ano
    periods_per_year = ann_factor
    
    # Calcular duração do período em anos
    # Tentar usar DatetimeIndex se disponível, senão usar len
    years = None
    
    # Verificar se há uma coluna 'date' no DataFrame (pode estar após reset_index)
    if 'date' in values_df.columns:
        dates_col = pd.to_datetime(values_df['date'])
        if len(dates_col) > 1:
            first_date = dates_col.min()
            last_date = dates_col.max()
            days_diff = (last_date - first_date).days
            if days_diff > 0:
                years = days_diff / 365.25  # Usar 365.25 para considerar anos bissextos
    elif isinstance(values_df.index, pd.DatetimeIndex) and len(values_df) > 1:
        # Calcular anos reais baseado nas datas do índice
        first_date = values_df.index[0]
        last_date = values_df.index[-1]
        days_diff = (last_date - first_date).days
        if days_diff > 0:
            years = days_diff / 365.25  # Usar 365.25 para considerar anos bissextos
    
    # Se não conseguiu calcular por datas, usar número de períodos
    if years is None or years <= 0:
        n_periods = len(values_df)
        years = n_periods / periods_per_year
    
    # Garantir que years seja pelo menos um período mínimo razoável
    # Mínimo de ~1 semana para evitar CAGR extremo (anos muito pequenos)
    min_years_reasonable = max(7.0 / 365.25, 1.0 / periods_per_year)  # ~1 semana mínimo
    years = max(years, min_years_reasonable)
    
    # Métricas básicas
    # Usar o parâmetro initial_capital fornecido para garantir consistência entre estratégias
    # O primeiro valor do DataFrame pode ter pequenas diferenças numéricas
    initial_value = initial_capital  # Usar o capital inicial fornecido (padrão: 1_000_000)
    final_value = values.iloc[-1] if len(values) > 0 else initial_capital
    
    # Retorno total: (valor_final - valor_inicial) / valor_inicial
    if initial_value > 1e-12:
        total_return = (final_value - initial_value) / initial_value
    else:
        total_return = 0.0
    
    # CAGR (Compound Annual Growth Rate)
    # Usar proteção contra valores extremos
    if years > 0 and total_return > -1 and years >= min_years_reasonable:
        # Limitar o total_return para evitar overflow
        # Se total_return for muito grande, usar aproximação
        if total_return > 100:  # Se retorno total > 10000%, usar aproximação
            # Usar retorno médio anualizado como aproximação
            mean_return = returns.mean()
            cagr = mean_return * periods_per_year
        else:
            # Fórmula padrão do CAGR
            try:
                cagr = (1 + total_return) ** (1 / years) - 1
                # Verificar se não é um valor extremo
                if abs(cagr) > 10.0:  # Se CAGR > 1000%, algo está errado
                    # Usar aproximação baseada em retorno médio
                    mean_return = returns.mean()
                    cagr = mean_return * periods_per_year
            except (OverflowError, ZeroDivisionError):
                # Em caso de overflow, usar retorno médio anualizado
                mean_return = returns.mean()
                cagr = mean_return * periods_per_year
    else:
        cagr = 0.0
    
    # Retorno médio periódico
    mean_return = returns.mean()
    
    # Volatilidade anualizada
    std_return = returns.std()
    annual_vol = std_return * np.sqrt(periods_per_year)
    
    # Sharpe Ratio: (retorno médio anualizado - risk_free) / volatilidade anualizada
    mean_return_annual = mean_return * periods_per_year
    excess_return_annual = mean_return_annual - risk_free_rate
    sharpe = excess_return_annual / annual_vol if annual_vol > 0 else 0.0
    
    # Sortino Ratio: usa apenas downside deviation
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_std = downside_returns.std()
        downside_vol_annual = downside_std * np.sqrt(periods_per_year)
        sortino = excess_return_annual / downside_vol_annual if downside_vol_annual > 0 else 0.0
    else:
        sortino = sharpe  # Se não houver retornos negativos, Sortino = Sharpe
    
    # Drawdown: baseado nos valores do portfólio
    if len(values) > 0:
        running_max = values.expanding().max()
        drawdown = (values - running_max) / running_max
        max_drawdown = drawdown.min()
    else:
        max_drawdown = 0.0
    
    # Win rate
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0.0
    
    return {
        'Initial Capital': initial_capital,  # Usar o capital inicial fornecido para consistência
        'Final Value': final_value,
        'Total Return': total_return,
        'CAGR': cagr,
        'Annual Volatility': annual_vol,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_drawdown,
        'Win Rate': win_rate,
        'Backtest Period (years)': years,
    }


def create_comparative_visualizations(
    eurytion_df: pd.DataFrame,
    bh_df: pd.DataFrame,
    mkv_df: pd.DataFrame,
    output_dir: Path,
    interval: str,
    seed: int,
    initial_capital: float = 1_000_000.0,
    risk_free_rate: float = 0.03,
) -> None:
    """
    Cria visualizações comparativas entre Eurytion, Buy & Hold e Markowitz.
    
    Args:
        eurytion_df: DataFrame de resultados do Eurytion (com colunas: nav, ret_net)
        bh_df: DataFrame de resultados do Buy & Hold
        mkv_df: DataFrame de resultados do Markowitz
        output_dir: Diretório para salvar arquivos
        interval: Intervalo usado (1d, 1w)
        seed: Seed usado
    """
    print("\n" + "="*60)
    print("CRIANDO VISUALIZAÇÕES COMPARATIVAS")
    print("="*60)
    
    # Preparar dados do Eurytion (assumindo que tem 'nav' e 'ret_net')
    if 'nav' not in eurytion_df.columns:
        # Se não tiver nav, calcular a partir de ret_net
        eurytion_df = eurytion_df.copy()
        eurytion_df['nav'] = (1 + eurytion_df['ret_net']).cumprod()
    
    # Garantir que o valor usa o capital inicial correto
    # Se value não existe ou não começa com initial_capital, recalcular
    if 'value' not in eurytion_df.columns or (len(eurytion_df) > 0 and abs(eurytion_df['value'].iloc[0] - initial_capital) > 1e-3):
        eurytion_df = eurytion_df.copy()
        eurytion_df['value'] = eurytion_df['nav'] * initial_capital
    
    if 'return' not in eurytion_df.columns:
        eurytion_df['return'] = eurytion_df.get('ret_net', eurytion_df.get('return', 0.0))
    
    # Cores
    eurytion_color = '#2E86AB'  # Azul
    bh_color = '#F18F01'  # Laranja
    mkv_color = '#A23B72'  # Roxo
    
    # Calcular métricas com o mesmo capital inicial para todas as estratégias
    is_weekly_interval = (interval == "1w")
    eurytion_metrics = calculate_strategy_metrics(
        eurytion_df[['value', 'return']].reset_index(),
        initial_capital=initial_capital,
        is_weekly=is_weekly_interval,
        risk_free_rate=risk_free_rate
    )
    bh_metrics = calculate_strategy_metrics(
        bh_df.reset_index() if isinstance(bh_df.index, pd.DatetimeIndex) else bh_df,
        initial_capital=initial_capital,
        is_weekly=is_weekly_interval,
        risk_free_rate=risk_free_rate
    )
    mkv_metrics = calculate_strategy_metrics(
        mkv_df.reset_index() if isinstance(mkv_df.index, pd.DatetimeIndex) else mkv_df,
        initial_capital=initial_capital,
        is_weekly=is_weekly_interval,
        risk_free_rate=risk_free_rate
    )
    
    # === 1. Portfolio Value Over Time ===
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    
    ax1.plot(eurytion_df.index, eurytion_df['value'], 
            label=f'Eurytion (Sharpe: {eurytion_metrics["Sharpe Ratio"]:.2f} | Retorno Total: {eurytion_metrics["Total Return"]*100:.1f}%)', 
            color=eurytion_color, linewidth=2)
    ax1.plot(bh_df.index, bh_df['value'], 
            label=f'Buy & Hold EW (Sharpe: {bh_metrics["Sharpe Ratio"]:.2f} | Retorno Total: {bh_metrics["Total Return"]*100:.1f}%)', 
            color=bh_color, linewidth=2, linestyle='--', alpha=0.8)
    ax1.plot(mkv_df.index, mkv_df['value'], 
            label=f'Markowitz (Sharpe: {mkv_metrics["Sharpe Ratio"]:.2f} | Retorno Total: {mkv_metrics["Total Return"]*100:.1f}%)', 
            color=mkv_color, linewidth=2, linestyle='-.', alpha=0.8)
    
    ax1.set_title('Portfolio Value Over Time - Comparative Analysis', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.2f}'))
    plt.tight_layout()
    
    output_path = output_dir / f"comparative_portfolio_value_{interval}_{seed}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: {output_path}")
    plt.close()
    
    # === 2. Cumulative Returns ===
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    
    eurytion_cum_returns = (1 + eurytion_df['return']).cumprod() - 1
    bh_cum_returns = (1 + bh_df['return']).cumprod() - 1
    mkv_cum_returns = (1 + mkv_df['return']).cumprod() - 1
    
    ax2.plot(eurytion_cum_returns.index, eurytion_cum_returns * 100, 
            label='Eurytion', color=eurytion_color, linewidth=2)
    ax2.plot(bh_cum_returns.index, bh_cum_returns * 100, 
            label='Buy & Hold EW', color=bh_color, linewidth=2, linestyle='--', alpha=0.8)
    ax2.plot(mkv_cum_returns.index, mkv_cum_returns * 100, 
            label='Markowitz', color=mkv_color, linewidth=2, linestyle='-.', alpha=0.8)
    
    ax2.set_title('Cumulative Returns - Comparative Analysis', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / f"comparative_cumulative_returns_{interval}_{seed}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: {output_path}")
    plt.close()
    
    # === 3. Drawdown ===
    fig3, ax3 = plt.subplots(figsize=(14, 8))
    
    def calc_drawdown(returns):
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        return drawdown
    
    eurytion_dd = calc_drawdown(eurytion_df['return'])
    bh_dd = calc_drawdown(bh_df['return'])
    mkv_dd = calc_drawdown(mkv_df['return'])
    
    ax3.plot(eurytion_dd.index, eurytion_dd, color=eurytion_color, linewidth=2, label='Eurytion')
    ax3.plot(bh_dd.index, bh_dd, color=bh_color, linewidth=2, linestyle='--', alpha=0.8, label='Buy & Hold EW')
    ax3.plot(mkv_dd.index, mkv_dd, color=mkv_color, linewidth=2, linestyle='-.', alpha=0.8, label='Markowitz')
    ax3.fill_between(eurytion_dd.index, eurytion_dd, 0, color=eurytion_color, alpha=0.1)
    ax3.fill_between(bh_dd.index, bh_dd, 0, color=bh_color, alpha=0.1)
    ax3.fill_between(mkv_dd.index, mkv_dd, 0, color=mkv_color, alpha=0.1)
    
    ax3.set_title('Drawdown Over Time - Comparative Analysis', fontsize=16, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('Drawdown (%)', fontsize=12)
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / f"comparative_drawdown_{interval}_{seed}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: {output_path}")
    plt.close()
    
    # === 4. Rolling Sharpe Ratio ===
    fig4, ax4 = plt.subplots(figsize=(14, 8))
    
    # Ajustar janela e fator de anualização baseado no intervalo
    if interval == "1d":
        window = 63  # ~3 meses em dias
        ann_factor = 252
        window_label = "63-day"
    else:  # interval == "1w"
        window = 13  # ~3 meses em semanas (13 semanas ≈ 63 dias)
        ann_factor = 52
        window_label = "13-week"
    
    def rolling_sharpe_comparative(returns, window=window, ann=ann_factor, rf_rate=risk_free_rate):
        """Calcula Sharpe móvel com risk-free rate."""
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        # Retorno médio anualizado menos risk_free
        excess_return_annual = (rolling_mean * ann) - rf_rate
        # Volatilidade anualizada
        vol_annual = rolling_std * np.sqrt(ann)
        # Sharpe Ratio
        with np.errstate(divide="ignore", invalid="ignore"):
            sharpe = excess_return_annual / vol_annual
        return sharpe.replace([np.inf, -np.inf], np.nan)
    
    eurytion_roll_sharpe = rolling_sharpe_comparative(eurytion_df['return'], window=window, ann=ann_factor, rf_rate=risk_free_rate)
    bh_roll_sharpe = rolling_sharpe_comparative(bh_df['return'], window=window, ann=ann_factor, rf_rate=risk_free_rate)
    mkv_roll_sharpe = rolling_sharpe_comparative(mkv_df['return'], window=window, ann=ann_factor, rf_rate=risk_free_rate)
    
    # Remover NaN para evitar gráficos vazios
    eurytion_roll_sharpe = eurytion_roll_sharpe.dropna()
    bh_roll_sharpe = bh_roll_sharpe.dropna()
    mkv_roll_sharpe = mkv_roll_sharpe.dropna()
    
    if len(eurytion_roll_sharpe) > 0:
        ax4.plot(eurytion_roll_sharpe.index, eurytion_roll_sharpe, 
                label='Eurytion', color=eurytion_color, linewidth=2)
    if len(bh_roll_sharpe) > 0:
        ax4.plot(bh_roll_sharpe.index, bh_roll_sharpe, 
                label='Buy & Hold EW', color=bh_color, linewidth=2, linestyle='--', alpha=0.8)
    if len(mkv_roll_sharpe) > 0:
        ax4.plot(mkv_roll_sharpe.index, mkv_roll_sharpe, 
                label='Markowitz', color=mkv_color, linewidth=2, linestyle='-.', alpha=0.8)
    ax4.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    
    ax4.set_title(f'Rolling Sharpe Ratio ({window_label} window) - Comparative Analysis', fontsize=16, fontweight='bold')
    ax4.set_xlabel('Date', fontsize=12)
    ax4.set_ylabel('Sharpe Ratio', fontsize=12)
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / f"comparative_rolling_sharpe_{interval}_{seed}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: {output_path}")
    plt.close()
    
    # === 5. Returns Distribution ===
    fig5, ax5 = plt.subplots(figsize=(14, 8))
    
    ax5.hist(eurytion_df['return'] * 100, bins=50, alpha=0.5, 
            label='Eurytion', color=eurytion_color, density=True)
    ax5.hist(bh_df['return'] * 100, bins=50, alpha=0.5, 
            label='Buy & Hold EW', color=bh_color, density=True)
    ax5.hist(mkv_df['return'] * 100, bins=50, alpha=0.5, 
            label='Markowitz', color=mkv_color, density=True)
    ax5.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    ax5.set_title('Daily Returns Distribution - Comparative Analysis', fontsize=16, fontweight='bold')
    ax5.set_xlabel('Daily Return (%)', fontsize=12)
    ax5.set_ylabel('Density', fontsize=12)
    ax5.legend(loc='best', fontsize=10)
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / f"comparative_returns_distribution_{interval}_{seed}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: {output_path}")
    plt.close()
    
    # === 6. Performance Metrics Comparison ===
    fig6, ax6 = plt.subplots(figsize=(14, 8))
    
    metrics_to_plot = ['CAGR', 'Sharpe Ratio', 'Sortino Ratio']
    eurytion_vals = [eurytion_metrics['CAGR'] * 100, eurytion_metrics['Sharpe Ratio'], eurytion_metrics['Sortino Ratio']]
    bh_vals = [bh_metrics['CAGR'] * 100, bh_metrics['Sharpe Ratio'], bh_metrics['Sortino Ratio']]
    mkv_vals = [mkv_metrics['CAGR'] * 100, mkv_metrics['Sharpe Ratio'], mkv_metrics['Sortino Ratio']]
    
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    
    ax6.bar(x - width, eurytion_vals, width, label='Eurytion', color=eurytion_color, alpha=0.8)
    ax6.bar(x, bh_vals, width, label='Buy & Hold EW', color=bh_color, alpha=0.8)
    ax6.bar(x + width, mkv_vals, width, label='Markowitz', color=mkv_color, alpha=0.8)
    
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics_to_plot)
    ax6.set_title('Performance Metrics Comparison', fontsize=16, fontweight='bold')
    ax6.set_ylabel('Value', fontsize=12)
    ax6.legend(loc='best', fontsize=10)
    ax6.grid(True, alpha=0.3, axis='y')
    
    ax6.text(0, max(max(eurytion_vals), max(bh_vals), max(mkv_vals)) * 1.05, 
            'CAGR in %', ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    
    output_path = output_dir / f"comparative_metrics_{interval}_{seed}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: {output_path}")
    plt.close()
    
    print("\n  Total: 6 gráficos comparativos criados!")


def export_comparative_metrics_table(
    eurytion_metrics: Dict,
    bh_metrics: Dict,
    mkv_metrics: Dict,
    output_dir: Path,
    interval: str,
    seed: int,
) -> None:
    """
    Exporta tabela comparativa de métricas em formato LaTeX e CSV.
    
    Args:
        eurytion_metrics: Métricas do Eurytion
        bh_metrics: Métricas do Buy & Hold
        mkv_metrics: Métricas do Markowitz
        output_dir: Diretório para salvar
        interval: Intervalo usado
        seed: Seed usado
    """
    print("\n" + "="*60)
    print("EXPORTANDO TABELA COMPARATIVA DE MÉTRICAS")
    print("="*60)
    
    # Salvar CSV
    metrics_df = pd.DataFrame([
        {**eurytion_metrics, 'strategy': 'Eurytion'},
        {**bh_metrics, 'strategy': 'Buy & Hold EW'},
        {**mkv_metrics, 'strategy': 'Markowitz'}
    ])
    
    csv_path = output_dir / f"comparative_metrics_{interval}_{seed}.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"  [OK] Saved CSV: {csv_path}")
    
    # Salvar LaTeX
    latex_content = r"""\begin{table}[htbp]
\centering
\caption{Comparação de Performance das Estratégias (Teste Out-of-Sample)}
\label{tab:strategies_comparison}
\begin{tabular}{lrrr}
\toprule
\textbf{Métrica} & \textbf{Eurytion} & \textbf{Buy \& Hold EW} & \textbf{Markowitz} \\
\midrule
\multicolumn{4}{l}{\textit{\textbf{Desempenho}}} \\
Capital Inicial (\$) & """ + f"{eurytion_metrics['Initial Capital']:,.2f}" + r""" & """ + f"{bh_metrics['Initial Capital']:,.2f}" + r""" & """ + f"{mkv_metrics['Initial Capital']:,.2f}" + r""" \\
Valor Final (\$) & """ + f"{eurytion_metrics['Final Value']:,.2f}" + r""" & """ + f"{bh_metrics['Final Value']:,.2f}" + r""" & """ + f"{mkv_metrics['Final Value']:,.2f}" + r""" \\
Retorno Total (\%) & """ + f"{eurytion_metrics['Total Return']*100:.2f}" + r""" & """ + f"{bh_metrics['Total Return']*100:.2f}" + r""" & """ + f"{mkv_metrics['Total Return']*100:.2f}" + r""" \\
CAGR (\%) & """ + f"{eurytion_metrics['CAGR']*100:.2f}" + r""" & """ + f"{bh_metrics['CAGR']*100:.2f}" + r""" & """ + f"{mkv_metrics['CAGR']*100:.2f}" + r""" \\
\midrule
\multicolumn{4}{l}{\textit{\textbf{Métricas Ajustadas ao Risco}}} \\
Volatilidade Anual (\%) & """ + f"{eurytion_metrics['Annual Volatility']*100:.2f}" + r""" & """ + f"{bh_metrics['Annual Volatility']*100:.2f}" + r""" & """ + f"{mkv_metrics['Annual Volatility']*100:.2f}" + r""" \\
Razão de Sharpe & """ + f"{eurytion_metrics['Sharpe Ratio']:.2f}" + r""" & """ + f"{bh_metrics['Sharpe Ratio']:.2f}" + r""" & """ + f"{mkv_metrics['Sharpe Ratio']:.2f}" + r""" \\
Razão de Sortino & """ + f"{eurytion_metrics['Sortino Ratio']:.2f}" + r""" & """ + f"{bh_metrics['Sortino Ratio']:.2f}" + r""" & """ + f"{mkv_metrics['Sortino Ratio']:.2f}" + r""" \\
Máximo Drawdown (\%) & """ + f"{eurytion_metrics['Max Drawdown']*100:.2f}" + r""" & """ + f"{bh_metrics['Max Drawdown']*100:.2f}" + r""" & """ + f"{mkv_metrics['Max Drawdown']*100:.2f}" + r""" \\
\midrule
\multicolumn{4}{l}{\textit{\textbf{Estatísticas Operacionais}}} \\
Taxa de Vitória (\%) & """ + f"{eurytion_metrics['Win Rate']*100:.2f}" + r""" & """ + f"{bh_metrics['Win Rate']*100:.2f}" + r""" & """ + f"{mkv_metrics['Win Rate']*100:.2f}" + r""" \\
\midrule
Período (anos) & """ + f"{eurytion_metrics['Backtest Period (years)']:.1f}" + r""" & """ + f"{bh_metrics['Backtest Period (years)']:.1f}" + r""" & """ + f"{mkv_metrics['Backtest Period (years)']:.1f}" + r""" \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textbf{Eurytion}: Reinforcement Learning (PPO) com sinais MLP e sentimentos
\item \textbf{Buy \& Hold EW}: Portfolio equal-weight com rebalanceamento inicial
\item \textbf{Markowitz}: Otimização média-variância (maximizar Sharpe) com rebalanceamento trimestral
\item Período de teste: out-of-sample
\end{tablenotes}
\end{table}"""
    
    tex_path = output_dir / f"comparative_metrics_{interval}_{seed}.tex"
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    print(f"  [OK] Saved LaTeX: {tex_path}")


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================


def run_pipeline_for_interval(interval: str, total_steps: int, seed: int, train_end: str, 
                               output_dir: Path, no_plots: bool):
    """
    Executa o pipeline completo de Eurytion para um intervalo específico.
    
    Args:
        interval: '1d' para diário ou '1w' para semanal
        total_steps: Número total de steps de treinamento
        seed: Seed para reprodutibilidade
        train_end: Data final do conjunto de treino
        output_dir: Diretório para salvar resultados
        no_plots: Se True, não gera gráficos Plotly
    """
    interval_name = "DIÁRIO" if interval == "1d" else "SEMANAL"
    period_name = "dias" if interval == "1d" else "semanas"
    
    print("\n" + "=" * 80)
    print(f" EXECUTANDO PIPELINE PARA INTERVALO {interval.upper()} ({interval_name})")
    print("=" * 80)
    
    # 1) Carregar dados reais
    print("\n" + "=" * 60)
    print(f"CARREGANDO DADOS REAIS - {interval_name}")
    print("=" * 60)
    R, S, P, M, dates, crypto_list = load_real_data_for_eurytion(
        interval=interval, train_end=train_end
    )

    # 2) Split temporal
    print("\n" + "=" * 60)
    print("SPLIT TEMPORAL")
    print("=" * 60)
    t_train_end, t_test_end = split_indices_by_date(dates, np.datetime64(train_end))
    print(f"Treino: índices 0 até {t_train_end} (data: {dates[t_train_end]})")
    print(f"Teste: índices {t_train_end+1} até {t_test_end} (data: {dates[t_test_end]})")

    # 3) Configurar ambiente
    # Ajustar risk_free_daily baseado no intervalo
    if interval == "1d":
        risk_free_daily = 0.05 / 252  # ~5% a.a. USD (diário)
    else:  # interval == "1w"
        risk_free_daily = 0.05 / 52   # ~5% a.a. USD (semanal)
    
    cfg = EnvConfig(
        cost_bps=0.0,  # Backtest sem custo de transação
        turnover_penalty=0.0,
        include_cash=True,
        risk_free_daily=risk_free_daily,
        ewma_lambda=0.94,
        lookback=3,
        clip_turnover=None,
        wmax=0.25,  # Máximo de 25% por ativo
        reward_mode="local_sharpe",
        lam_risk=0.0,
        logit_scale=5.0,
    )

    # 4) Criar ambientes
    print("\n" + "=" * 60)
    print("CRIANDO AMBIENTES")
    print("=" * 60)
    env_train, env_test = make_env_train_test(R, S, P, M, cfg, t_train_end, t_test_end)
    print("[OK] Ambientes criados com sucesso!")

    # 5) Treinar modelo
    print("\n" + "=" * 60)
    print("TREINANDO PPO")
    print("=" * 60)
    model, tm = train_ppo_no_val(env_train, total_steps=total_steps, seed=seed)

    # Salvar modelo
    model_path = output_dir / f"ppo_model_{interval}_{seed}.zip"
    model.save(str(model_path))
    print(f"[OK] Modelo salvo em: {model_path}")

    # 6) Backtest
    print("\n" + "=" * 60)
    print("EXECUTANDO BACKTEST")
    print("=" * 60)
    df_test = run_backtest_df(model, env_test, dates=dates)

    # Métricas (anualizar baseado no intervalo)
    ann_factor = 252 if interval == "1d" else 52
    risk_free_rate = 0.03  # 3% a.a.
    
    # Sharpe Ratio correto: (retorno médio anualizado - risk_free) / volatilidade anualizada
    mean_return = df_test["ret_net"].mean()
    std_return = df_test["ret_net"].std()
    mean_return_annual = mean_return * ann_factor
    annual_vol = std_return * np.sqrt(ann_factor)
    excess_return_annual = mean_return_annual - risk_free_rate
    sharpe_test = excess_return_annual / annual_vol if annual_vol > 0 else 0.0
    
    print(f"\n[OK] Test Sharpe: {sharpe_test:+.3f} (retorno médio anual: {mean_return_annual*100:.2f}%, vol anual: {annual_vol*100:.2f}%)")

    # Salvar resultados
    results_path = output_dir / f"backtest_results_{interval}_{seed}.csv"
    df_test.to_csv(results_path)
    print(f"[OK] Resultados salvos em: {results_path}")

    # 6b) Calcular benchmarks (Buy & Hold e Markowitz)
    print("\n" + "=" * 60)
    print("CALCULANDO BENCHMARKS (Buy & Hold EW e Markowitz)")
    print("=" * 60)
    
    # Obter retornos do período de teste
    R_test = R[t_train_end+1:t_test_end+1]
    dates_test = dates[t_train_end+1:t_test_end+1]
    returns_df_test = pd.DataFrame(R_test, index=pd.to_datetime(dates_test), columns=crypto_list)
    
    # Buy & Hold Equal Weight
    print("\nCalculando Buy & Hold Equal Weight...")
    initial_capital = 1_000_000.0  # $1 milhão USD
    bh_df = compute_buy_and_hold_equal_weight(returns_df_test, initial_capital=initial_capital)
    print(f"  [OK] Buy & Hold EW calculado: {len(bh_df)} {period_name}")
    
    # Markowitz (Sharpe optimization)
    print("\nCalculando Markowitz (Sharpe optimization)...")
    # Markowitz sempre usa 63 dias de lookback (fixo), independente da frequência
    # Para semanais, 63 dias = 9 semanas (63 / 7 = 9)
    lookback_days_fixed = 63  # Sempre 63 dias
    if interval == "1d":
        lookback_periods = 63  # 63 dias
        rebal_freq = 63  # Rebalanceamento trimestral (~3 meses)
    else:  # interval == "1w"
        lookback_periods = 9  # 63 dias = 9 semanas exatas (63 / 7 = 9)
        rebal_freq = 13  # Rebalanceamento trimestral (~3 meses = 13 semanas)
    
    print(f"  Lookback: {lookback_days_fixed} dias (fixo) = {lookback_periods} {period_name}")
    
    mkv_df = optimize_markowitz_sharpe(
        returns_df_test,
        lookback_days=lookback_periods,
        rebal_freq=rebal_freq,
        initial_capital=initial_capital,
        max_weight=0.25,  # Máximo de 25% por ativo (mesmo limite do Eurytion)
        min_weight=0.0,
        is_weekly=(interval == "1w"),
    )
    print(f"  [OK] Markowitz calculado: {len(mkv_df)} {period_name}")

    # 7) Visualizações (opcional)
    if not no_plots:
        print("\n" + "=" * 60)
        print("GERANDO VISUALIZAÇÕES")
        print("=" * 60)

        # Dashboard principal
        fig_dash = plotly_backtest_dashboard(
            df_test, title=f"Teste (exploit OOS) - {interval.upper()}"
        )
        dash_path = output_dir / f"dashboard_{interval}_{seed}.html"
        fig_dash.write_html(str(dash_path))
        print(f"[OK] Dashboard salvo em: {dash_path}")

        # Alocações
        labels = crypto_list + ["USD (cash)"]
        fig_w = plotly_allocations(
            df_test,
            labels=labels,
            title=f"Alocações (pesos) — Teste OOS - {interval.upper()}",
        )
        alloc_path = output_dir / f"allocations_{interval}_{seed}.html"
        fig_w.write_html(str(alloc_path))
        print(f"[OK] Alocações salvas em: {alloc_path}")

        # Distribuição de retornos
        fig_hist = plotly_returns_distribution(
            df_test,
            bins=60,
            title=f"Distribuição de Retornos (net) — Teste OOS - {interval.upper()}",
        )
        hist_path = output_dir / f"returns_dist_{interval}_{seed}.html"
        fig_hist.write_html(str(hist_path))
        print(f"[OK] Distribuição de retornos salva em: {hist_path}")

        # Métricas móveis
        window_size = 63 if interval == "1d" else 13  # ~3 meses
        window_name = f"{window_size}{'d' if interval == '1d' else 'w'}"
        risk_free_rate_plotly = 0.03  # 3% a.a.
        fig_roll = plotly_rolling_metrics(
            df_test,
            window=window_size,
            title=f"Métricas Móveis ({window_name} ~ 3 meses) — Teste OOS - {interval.upper()}",
            is_weekly=(interval == "1w"),
            risk_free_rate=risk_free_rate_plotly
        )
        roll_path = output_dir / f"rolling_metrics_{interval}_{seed}.html"
        fig_roll.write_html(str(roll_path))
        print(f"[OK] Métricas móveis salvas em: {roll_path}")

        # Heatmap mensal
        try:
            fig_heat = plotly_monthly_heatmap(
                df_test, title=f"Retornos Mensais (net) — Teste OOS - {interval.upper()}"
            )
            heat_path = output_dir / f"monthly_heatmap_{interval}_{seed}.html"
            fig_heat.write_html(str(heat_path))
            print(f"[OK] Heatmap mensal salvo em: {heat_path}")
        except Exception as e:
            print(f"[AVISO] Não foi possível gerar heatmap mensal: {e}")

    # 8) Visualizações comparativas (Eurytion vs B&H vs Markowitz)
    if not no_plots:
        print("\n" + "=" * 60)
        print("GERANDO VISUALIZAÇÕES COMPARATIVAS")
        print("=" * 60)
        
        # Preparar DataFrame do Eurytion para comparação
        eurytion_df_comp = df_test.copy()
        if 'nav' not in eurytion_df_comp.columns:
            eurytion_df_comp['nav'] = (1 + eurytion_df_comp['ret_net']).cumprod()
        eurytion_df_comp['value'] = eurytion_df_comp['nav'] * initial_capital
        eurytion_df_comp['return'] = eurytion_df_comp['ret_net']
        
        # Criar visualizações comparativas
        risk_free_rate = 0.03  # 3% a.a.
        create_comparative_visualizations(
            eurytion_df_comp,
            bh_df,
            mkv_df,
            output_dir,
            interval,
            seed,
            initial_capital=initial_capital,
            risk_free_rate=risk_free_rate
        )
        
        # Calcular e exportar métricas comparativas
        print("\n" + "=" * 60)
        print("CALCULANDO MÉTRICAS COMPARATIVAS")
        print("=" * 60)
        
        # Calcular métricas com o intervalo correto
        is_weekly_interval = (interval == "1w")
        
        # Preparar DataFrames mantendo o índice de datas para cálculo correto de anos
        eurytion_df_for_metrics = eurytion_df_comp[['value', 'return']].copy()
        if isinstance(eurytion_df_for_metrics.index, pd.DatetimeIndex):
            eurytion_df_for_metrics = eurytion_df_for_metrics.reset_index()
        else:
            eurytion_df_for_metrics = eurytion_df_for_metrics.reset_index()
        
        bh_df_for_metrics = bh_df.copy()
        if isinstance(bh_df_for_metrics.index, pd.DatetimeIndex):
            bh_df_for_metrics = bh_df_for_metrics.reset_index()
        else:
            bh_df_for_metrics = bh_df_for_metrics.reset_index()
        
        mkv_df_for_metrics = mkv_df.copy()
        if isinstance(mkv_df_for_metrics.index, pd.DatetimeIndex):
            mkv_df_for_metrics = mkv_df_for_metrics.reset_index()
        else:
            mkv_df_for_metrics = mkv_df_for_metrics.reset_index()
        
        eurytion_metrics = calculate_strategy_metrics(
            eurytion_df_for_metrics, 
            initial_capital,
            is_weekly=is_weekly_interval,
            risk_free_rate=risk_free_rate
        )
        bh_metrics = calculate_strategy_metrics(
            bh_df_for_metrics, 
            initial_capital,
            is_weekly=is_weekly_interval,
            risk_free_rate=risk_free_rate
        )
        mkv_metrics = calculate_strategy_metrics(
            mkv_df_for_metrics, 
            initial_capital,
            is_weekly=is_weekly_interval,
            risk_free_rate=risk_free_rate
        )
        
        # Exportar tabela comparativa
        export_comparative_metrics_table(
            eurytion_metrics,
            bh_metrics,
            mkv_metrics,
            output_dir,
            interval,
            seed
        )
        
        # Imprimir resumo comparativo
        print("\n" + "=" * 100)
        print(f" " * 30 + f"RESULTADOS COMPARATIVOS - {interval_name}")
        print("=" * 100)
        print(f"\n{'Métrica':<30} {'Eurytion':>22} {'Buy & Hold EW':>22} {'Markowitz':>22}")
        print("-" * 100)
        print("\n Desempenho do Portfólio:")
        print(f"{'  Capital Inicial':<30} ${eurytion_metrics['Initial Capital']:>20,.2f} ${bh_metrics['Initial Capital']:>20,.2f} ${mkv_metrics['Initial Capital']:>20,.2f}")
        print(f"{'  Valor Final':<30} ${eurytion_metrics['Final Value']:>20,.2f} ${bh_metrics['Final Value']:>20,.2f} ${mkv_metrics['Final Value']:>20,.2f}")
        print(f"{'  Retorno Total':<30} {eurytion_metrics['Total Return']:>21.2%} {bh_metrics['Total Return']:>21.2%} {mkv_metrics['Total Return']:>21.2%}")
        print(f"{'  CAGR':<30} {eurytion_metrics['CAGR']:>21.2%} {bh_metrics['CAGR']:>21.2%} {mkv_metrics['CAGR']:>21.2%}")
        print("\n Métricas de Risco:")
        print(f"{'  Volatilidade Anual':<30} {eurytion_metrics['Annual Volatility']:>21.2%} {bh_metrics['Annual Volatility']:>21.2%} {mkv_metrics['Annual Volatility']:>21.2%}")
        print(f"{'  Razão de Sharpe':<30} {eurytion_metrics['Sharpe Ratio']:>21.2f} {bh_metrics['Sharpe Ratio']:>21.2f} {mkv_metrics['Sharpe Ratio']:>21.2f}")
        print(f"{'  Razão de Sortino':<30} {eurytion_metrics['Sortino Ratio']:>21.2f} {bh_metrics['Sortino Ratio']:>21.2f} {mkv_metrics['Sortino Ratio']:>21.2f}")
        print(f"{'  Máximo Drawdown':<30} {eurytion_metrics['Max Drawdown']:>21.2%} {bh_metrics['Max Drawdown']:>21.2%} {mkv_metrics['Max Drawdown']:>21.2%}")
        print("\n Estatísticas de Trading:")
        print(f"{'  Taxa de Vitória':<30} {eurytion_metrics['Win Rate']:>21.2%} {bh_metrics['Win Rate']:>21.2%} {mkv_metrics['Win Rate']:>21.2%}")
        print("\n Período:")
        print(f"{'  Duração do Backtest':<30} {eurytion_metrics['Backtest Period (years)']:>18.1f} anos {bh_metrics['Backtest Period (years)']:>18.1f} anos {mkv_metrics['Backtest Period (years)']:>18.1f} anos")
    
    print("\n" + "=" * 80)
    print(f" PIPELINE CONCLUÍDO PARA INTERVALO {interval.upper()} ({interval_name})")
    print("=" * 80)
    print(f"\nResultados salvos em: {output_dir.absolute()}")
    print(f"\nArquivos gerados para {interval.upper()}:")
    print(f"  - ppo_model_{interval}_{seed}.zip")
    print(f"  - backtest_results_{interval}_{seed}.csv")
    if not no_plots:
        print(f"  - dashboard_{interval}_{seed}.html")
        print(f"  - allocations_{interval}_{seed}.html")
        print(f"  - returns_dist_{interval}_{seed}.html")
        print(f"  - rolling_metrics_{interval}_{seed}.html")
        print(f"  - monthly_heatmap_{interval}_{seed}.html")
        print(f"  - comparative_*.png (6 gráficos)")
        print(f"  - comparative_metrics_{interval}_{seed}.csv")
        print(f"  - comparative_metrics_{interval}_{seed}.tex")
        print(f"\nTotal: 1 modelo + 1 CSV + 5 HTML + 6 PNG + 1 CSV comparativo + 1 LaTeX")
    else:
        print(f"\nTotal: 1 modelo + 1 CSV (visualizações desabilitadas)")


def main():
    """Pipeline principal de treinamento e avaliação RL.
    
    Executa apenas o intervalo semanal (1w).
    """
    parser = argparse.ArgumentParser(
        description="Pipeline de RL para alocação de portfólio de criptomoedas"
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1w",
        choices=["1w"],
        help="Intervalo a ser usado (apenas 1w - semanal)",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=150000,
        help="Número total de steps de treinamento (padrão: 150000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed para reprodutibilidade (padrão: 42)"
    )
    parser.add_argument(
        "--train_end",
        type=str,
        default="2023-12-31",
        help="Data final do conjunto de treino (padrão: 2023-12-31, treino: 2020-2023)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Não gerar gráficos Plotly (apenas métricas)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Diretório para salvar resultados (padrão: results)",
    )

    args = parser.parse_args()

    # Criar diretório de saída
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("PIPELINE DE RL PARA ALOCAÇÃO DE PORTFÓLIO")
    print("=" * 80)
    print("EXECUTA APENAS: Intervalo SEMANAL (1w)")
    print("=" * 80)
    print(f"Total steps: {args.total_steps}")
    print(f"Seed: {args.seed}")
    print(f"Data final treino: {args.train_end}")
    print(f"Output dir: {output_dir}")
    print(f"Plots: {'Desabilitados' if args.no_plots else 'Habilitados'}")
    print("=" * 80)
    
    # Executar pipeline apenas para intervalo semanal
    intervals_to_run = ["1w"]
    
    for interval in intervals_to_run:
        try:
            run_pipeline_for_interval(
                interval=interval,
                total_steps=args.total_steps,
                seed=args.seed,
                train_end=args.train_end,
                output_dir=output_dir,
                no_plots=args.no_plots
            )
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"ERRO ao executar pipeline para intervalo {interval.upper()}")
            print(f"{'='*80}")
            print(f"Erro: {e}")
            import traceback
            traceback.print_exc()
            print(f"\nContinuando com próximo intervalo...")
            continue
    
    print("\n" + "=" * 80)
    print(" PIPELINE COMPLETO CONCLUÍDO!")
    print("=" * 80)
    print(f"\nResultados salvos em: {output_dir.absolute()}")
    print("\nArquivos gerados para intervalo semanal (1w):")
    print("  - Modelos PPO")
    print("  - Resultados de backtest")
    if not args.no_plots:
        print("  - Visualizações Plotly (HTML)")
        print("  - Gráficos comparativos (PNG)")
        print("  - Tabelas comparativas (CSV + LaTeX)")
    print("\nTotal de execuções: 1 intervalo (semanal)")


if __name__ == "__main__":
    main()

