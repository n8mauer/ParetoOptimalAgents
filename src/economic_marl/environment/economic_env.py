from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ..utils.logging import get_logger
from .data_seeder import DataSeeder

log = get_logger(__name__)

@dataclass
class MacroState:
    tariff_rate: float = 0.05
    commodity_price: float = 100.0
    gold_price: float = 1900.0
    inflation: float = 0.02
    fiat_demand: float = 1.0  # normalized index
    trade_balance: float = 0.0  # surplus(+)/deficit(-)

@dataclass
class EconomicEnvConfig:
    n_agents: int = 5
    max_steps: int = 200
    seed: int = 42
    tariff_sensitivity: float = 0.6
    commodity_pass_through: float = 0.4
    gold_hedge_beta: float = 0.35
    inflation_beta: float = 0.25
    fiat_substitution_beta: float = 0.30
    use_real_data: bool = True  # Use LandingAI ADE data for seeding
    ade_data_path: Optional[str] = None  # Path to ADE parquet file

class EconomicEnv(gym.Env):
    """Gymnasium environment for multi-agent economic simulation.

    Agents act on reserve allocation and tariff stance:
        action[i] = [delta_tariff, w_fiat, w_gold, w_commodity]
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, cfg: Optional[EconomicEnvConfig] = None):
        super().__init__()
        self.cfg = cfg or EconomicEnvConfig()
        self.np_random = np.random.default_rng(self.cfg.seed)
        self.n_agents = self.cfg.n_agents

        # Initialize DataSeeder for real-world data
        if self.cfg.use_real_data:
            self.data_seeder = DataSeeder(ade_data_path=self.cfg.ade_data_path)
            if self.data_seeder.has_real_data():
                log.info("data_seeder_enabled", status="using_real_ade_data",
                        summary=self.data_seeder.get_summary_stats())
            else:
                log.info("data_seeder_fallback", status="ade_data_not_found", using="synthetic_defaults")
        else:
            self.data_seeder = None
            log.info("data_seeder_disabled", using="synthetic_only")

        self.state = MacroState()
        self.obs_dim = 6
        self.action_dim = 4
        low = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, shape=(self.action_dim,), dtype=np.float32)
        self.steps = 0

    def _observe(self) -> np.ndarray:
        s = self.state
        return np.array([
            s.tariff_rate, s.commodity_price, s.gold_price, s.inflation, s.fiat_demand, s.trade_balance
        ], dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # Use real data if available, otherwise fallback to synthetic
        if self.data_seeder is not None and self.data_seeder.has_real_data():
            initial_state = self.data_seeder.get_initial_state(self.np_random)
            self.state = MacroState(**initial_state)
            log.debug("reset_with_real_data", state=initial_state)
        else:
            # Fallback to synthetic data
            self.state = MacroState(
                tariff_rate=float(self.np_random.uniform(0.0, 0.15)),
                commodity_price=float(self.np_random.normal(100.0, 5.0)),
                gold_price=float(self.np_random.normal(1900.0, 30.0)),
                inflation=float(self.np_random.uniform(0.0, 0.04)),
                fiat_demand=1.0,
                trade_balance=float(self.np_random.normal(0.0, 0.5)),
            )
            log.debug("reset_with_synthetic_data")

        self.steps = 0
        obs = np.stack([self._observe() for _ in range(self.n_agents)], axis=0)
        return obs, {}

    def step(self, actions: np.ndarray):
        """Compute macro transitions and rewards.

        actions: shape (n_agents, 4) with values in [low, high].
        """
        assert actions.shape == (self.n_agents, self.action_dim)
        self.steps += 1
        cfg = self.cfg
        s = self.state

        # Aggregated policy signals
        delta_tariff = float(np.clip(np.mean(actions[:, 0]), -0.1, 0.1))
        avg_w_fiat = float(np.clip(np.mean(actions[:, 1]), 0.0, 1.0))
        avg_w_gold = float(np.clip(np.mean(actions[:, 2]), 0.0, 1.0))
        avg_w_cmdty = float(np.clip(np.mean(actions[:, 3]), 0.0, 1.0))

        # Tariff dynamics -> inflation and commodity prices
        s.tariff_rate = float(np.clip(s.tariff_rate + delta_tariff, 0.0, 1.0))
        commodity_shock = cfg.tariff_sensitivity * delta_tariff + self.np_random.normal(0, 0.01)
        s.commodity_price = max(1.0, s.commodity_price * (1.0 + cfg.commodity_pass_through * commodity_shock))
        s.inflation = max(0.0, s.inflation + cfg.inflation_beta * commodity_shock)

        # Gold hedge response to inflation & fiat weakness
        fiat_weakness = max(0.0, cfg.fiat_substitution_beta * (s.inflation - 0.02))
        s.fiat_demand = max(0.1, s.fiat_demand * (1.0 - fiat_weakness))
        s.gold_price = max(100.0, s.gold_price * (1.0 + cfg.gold_hedge_beta * (s.inflation + fiat_weakness)))

        # Trade balance deteriorates with tariffs (simplified)
        s.trade_balance -= delta_tariff * 0.5 + self.np_random.normal(0, 0.02)

        # Rewards: each agent prefers stability and return on allocations
        rewards = np.zeros((self.n_agents,), dtype=np.float32)
        gold_ret = (s.gold_price / 1900.0) - 1.0
        fiat_ret = -fiat_weakness
        cmdty_ret = (s.commodity_price / 100.0) - 1.0

        for i in range(self.n_agents):
            w_fiat, w_gold, w_cmdty = avg_w_fiat, avg_w_gold, avg_w_cmdty
            port_ret = 0.4*w_gold*gold_ret + 0.3*w_cmdty*cmdty_ret + 0.3*w_fiat*fiat_ret
            stability_penalty = abs(delta_tariff) * 0.2 + abs(s.trade_balance) * 0.05
            rewards[i] = float(port_ret - stability_penalty)

        terminated = self.steps >= self.cfg.max_steps
        obs = np.stack([self._observe() for _ in range(self.n_agents)], axis=0)
        info = {"gold_price": s.gold_price, "tariff_rate": s.tariff_rate, "inflation": s.inflation,
                "commodity_price": s.commodity_price, "fiat_demand": s.fiat_demand, "trade_balance": s.trade_balance}
        return obs, rewards, np.array([terminated]*self.n_agents), np.array([False]*self.n_agents), info

    def apply_event(self, kind: str, **kwargs):
        if kind == "tariff":
            delta = float(kwargs.get("delta", 0.05))
            self.state.tariff_rate = float(np.clip(self.state.tariff_rate + delta, 0.0, 1.0))
        elif kind == "sanction":
            self.state.inflation += 0.01
            self.state.fiat_demand = max(0.1, self.state.fiat_demand * 0.98)
        else:
            log.warning("unknown_event", kind=kind)
