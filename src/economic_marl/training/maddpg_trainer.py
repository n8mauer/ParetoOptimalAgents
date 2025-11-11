from __future__ import annotations
from typing import Dict, Any
import os
import torch
import numpy as np
import pandas as pd
from tqdm import trange
from ..environment.economic_env import EconomicEnv, EconomicEnvConfig
from ..environment.agents import MADDPGAgent, ReplayBuffer, DEVICE
from ..environment.qmix import MixingNetwork
from ..utils.logging import get_logger

log = get_logger(__name__)

class MADDPGTrainer:
    def __init__(self, cfg: Dict[str, Any], output_dir: str = "./outputs"):
        self.env = EconomicEnv(
            EconomicEnvConfig(
                n_agents=5,
                max_steps=cfg.get("max_steps_per_episode", 200),
                seed=cfg.get("seed", 42),
            )
        )
        self.n_agents = self.env.n_agents
        self.state_dim = self.env.obs_dim
        self.action_dim = self.env.action_dim
        self.gamma = cfg.get("gamma", 0.99)
        self.batch_size = cfg.get("batch_size", 256)
        self.use_qmix = cfg.get("use_qmix", True)

        self.buffer = ReplayBuffer(
            cfg.get("buffer_size", 200_000),
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            n_agents=self.n_agents,
        )

        # Create agents with agent IDs
        self.agents = [
            MADDPGAgent(
                self.state_dim,
                self.action_dim,
                self.n_agents,
                lr_actor=cfg.get("lr_actor", 1e-4),
                lr_critic=cfg.get("lr_critic", 1e-3),
                tau=cfg.get("tau", 0.01),
                agent_id=i,
            )
            for i in range(self.n_agents)
        ]

        # QMIX integration for Pareto coordination
        if self.use_qmix:
            mixer_embed_dim = cfg.get("qmix_embed_dim", 32)
            self.mixer = MixingNetwork(
                n_agents=self.n_agents,
                state_dim=self.state_dim,
                embed_dim=mixer_embed_dim
            ).to(DEVICE)
            self.target_mixer = MixingNetwork(
                n_agents=self.n_agents,
                state_dim=self.state_dim,
                embed_dim=mixer_embed_dim
            ).to(DEVICE)
            self.target_mixer.load_state_dict(self.mixer.state_dict())

            # Optimizer for mixer network
            self.mixer_opt = torch.optim.Adam(
                self.mixer.parameters(),
                lr=cfg.get("lr_mixer", 1e-3)
            )

            # Share mixer with all agents
            for agent in self.agents:
                agent.mixer = self.mixer
                agent.target_mixer = self.target_mixer

            log.info("qmix_enabled", n_agents=self.n_agents, embed_dim=mixer_embed_dim)
        else:
            self.mixer = None
            self.target_mixer = None
            self.mixer_opt = None
            log.info("qmix_disabled", mode="standard_maddpg")

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _select_actions(self, obs: np.ndarray, noise_scale: float = 0.2) -> np.ndarray:
        actions = np.stack([self.agents[i].act(obs[i], noise_scale) for i in range(self.n_agents)], axis=0)
        actions[:, 1:] = np.clip(actions[:, 1:], 0.0, 1.0)
        row_sums = actions[:, 1:].sum(axis=1, keepdims=True) + 1e-8
        actions[:, 1:] = actions[:, 1:] / row_sums
        return actions

    def train(self, episodes: int = 1000, max_steps: int = 200, save_every: int = 50) -> str:
        metrics = []
        for ep in trange(episodes, desc="Training"):
            obs, _ = self.env.reset()
            ep_reward = 0.0
            t = 0
            for t in range(max_steps):
                actions = self._select_actions(obs, noise_scale=max(0.05, 0.3*(1 - ep/episodes)))
                next_obs, reward, terminated, truncated, info = self.env.step(actions)
                done = np.logical_or(terminated, truncated)
                self.buffer.add(obs, actions, reward, next_obs, done)
                ep_reward += float(np.mean(reward))
                obs = next_obs

                if self.buffer.size >= self.batch_size:
                    batch = self.buffer.sample(self.batch_size)
                    for ag in self.agents:
                        ag.update(batch, self.agents, gamma=self.gamma, use_qmix=self.use_qmix)

                    # Update target mixer if using QMIX
                    if self.use_qmix and self.mixer is not None:
                        tau = self.agents[0].tau  # Use same tau as agents
                        for p, tp in zip(self.mixer.parameters(), self.target_mixer.parameters()):
                            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

                if np.all(done):
                    break

            metrics.append({
                "episode": ep,
                "avg_reward": ep_reward / (t+1),
                "gold_price": info["gold_price"],
                "tariff_rate": info["tariff_rate"],
                "inflation": info["inflation"],
                "commodity_price": info["commodity_price"],
                "fiat_demand": info["fiat_demand"],
                "trade_balance": info["trade_balance"],
            })

            if (ep+1) % save_every == 0:
                self._save_checkpoint(ep)

        df = pd.DataFrame(metrics)
        out_path = os.path.join(self.output_dir, "metrics.parquet")
        df.to_parquet(out_path, index=False)
        log.info("training_complete", metrics_path=out_path, episodes=episodes)
        return out_path

    def _save_checkpoint(self, ep: int):
        ckpt_dir = os.path.join(self.output_dir, "checkpoints", f"ep_{ep+1:05d}")
        os.makedirs(ckpt_dir, exist_ok=True)

        # Save agent networks
        for i, ag in enumerate(self.agents):
            torch.save(ag.actor.state_dict(), os.path.join(ckpt_dir, f"agent_{i}_actor.pt"))
            torch.save(ag.critic.state_dict(), os.path.join(ckpt_dir, f"agent_{i}_critic.pt"))

        # Save QMIX mixer if enabled
        if self.use_qmix and self.mixer is not None:
            torch.save(self.mixer.state_dict(), os.path.join(ckpt_dir, "mixer.pt"))
            torch.save(self.target_mixer.state_dict(), os.path.join(ckpt_dir, "target_mixer.pt"))

        log.info("checkpoint_saved", episode=ep+1, path=ckpt_dir, qmix=self.use_qmix)
