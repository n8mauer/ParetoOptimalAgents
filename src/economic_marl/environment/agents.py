from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .qmix import MixingNetwork

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Tanh()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, total_state_dim: int, total_action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_state_dim + total_action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, actions], dim=-1)
        return self.net(x)

class OUNoise:
    def __init__(self, dim: int, mu: float=0.0, theta: float=0.15, sigma: float=0.2):
        self.dim = dim; self.mu = mu; self.theta = theta; self.sigma = sigma
        self.state = np.ones(self.dim) * self.mu
    def reset(self): self.state = np.ones(self.dim) * self.mu
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.dim)
        self.state += dx; return self.state

@dataclass
class ReplayBuffer:
    capacity: int
    state_dim: int
    action_dim: int
    n_agents: int
    ptr: int = 0
    size: int = 0

    def __post_init__(self):
        self.states = np.zeros((self.capacity, self.n_agents, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.n_agents, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, self.n_agents), dtype=np.float32)
        self.next_states = np.zeros_like(self.states)
        self.dones = np.zeros((self.capacity, self.n_agents), dtype=np.bool_)

    def add(self, s, a, r, s2, d):
        self.states[self.ptr] = s; self.actions[self.ptr] = a; self.rewards[self.ptr] = r
        self.next_states[self.ptr] = s2; self.dones[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (self.states[idxs], self.actions[idxs], self.rewards[idxs],
                self.next_states[idxs], self.dones[idxs])

class MADDPGAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_agents: int,
        lr_actor: float,
        lr_critic: float,
        tau: float,
        agent_id: int = 0
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.agent_id = agent_id
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.critic = Critic(state_dim*n_agents, action_dim*n_agents).to(DEVICE)
        self.target_actor = Actor(state_dim, action_dim).to(DEVICE)
        self.target_critic = Critic(state_dim*n_agents, action_dim*n_agents).to(DEVICE)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.tau = tau
        self.noise = OUNoise(action_dim)

        # QMIX integration - set externally by trainer
        self.mixer: Optional[MixingNetwork] = None
        self.target_mixer: Optional[MixingNetwork] = None

    def act(self, obs: np.ndarray, noise_scale: float = 0.2) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            act = self.actor(x).cpu().numpy()[0]
        self.actor.train()
        return np.clip(act + noise_scale * self.noise.sample(), -1.0, 1.0)

    def soft_update(self, src: nn.Module, dst: nn.Module):
        for p, tp in zip(src.parameters(), dst.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def update(self, batch, agents: List['MADDPGAgent'], gamma: float, use_qmix: bool = True):
        """Update agent using MADDPG algorithm, optionally with QMIX mixing.

        Args:
            batch: Tuple of (states, actions, rewards, next_states, dones)
            agents: List of all agents in the environment
            gamma: Discount factor
            use_qmix: If True and mixer is available, use QMIX for Pareto coordination

        Returns:
            Dictionary with loss metrics
        """
        states, actions, rewards, next_states, dones = batch
        B = states.shape[0]

        states_t = torch.tensor(states.reshape(B, -1), dtype=torch.float32, device=DEVICE)
        actions_t = torch.tensor(actions.reshape(B, -1), dtype=torch.float32, device=DEVICE)
        next_states_t = torch.tensor(next_states.reshape(B, -1), dtype=torch.float32, device=DEVICE)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=DEVICE)

        # Target actions from target actors
        next_actions = []
        for i, ag in enumerate(agents):
            o = torch.tensor(next_states[:, i, :], dtype=torch.float32, device=DEVICE)
            next_actions.append(ag.target_actor(o))
        next_actions_t = torch.cat(next_actions, dim=-1)

        # Compute target Q values with optional QMIX mixing
        with torch.no_grad():
            if use_qmix and self.target_mixer is not None:
                # QMIX: Compute individual Q values and mix them
                individual_q_targets = []
                for ag in agents:
                    q_i = ag.target_critic(next_states_t, next_actions_t).squeeze(-1)
                    individual_q_targets.append(q_i)
                q_agents = torch.stack(individual_q_targets, dim=1)  # [B, n_agents]

                # Get global state for mixer (use mean of agent observations)
                global_state = next_states_t[:, :states.shape[2]]  # [B, state_dim]
                target_q_tot = self.target_mixer(q_agents, global_state)

                # Use agent-specific rewards for this agent
                y = rewards_t[:, self.agent_id] + gamma * (1.0 - dones_t[:, self.agent_id].float()) * target_q_tot
            else:
                # Standard MADDPG: Use this agent's critic directly
                target_q = self.target_critic(next_states_t, next_actions_t).squeeze(-1)
                y = rewards_t[:, self.agent_id] + gamma * (1.0 - dones_t[:, self.agent_id].float()) * target_q

        # Critic update
        if use_qmix and self.mixer is not None:
            # QMIX critic update
            individual_q_values = []
            for ag in agents:
                q_i = ag.critic(states_t, actions_t).squeeze(-1)
                individual_q_values.append(q_i)
            q_agents = torch.stack(individual_q_values, dim=1)  # [B, n_agents]

            # Get global state for mixer
            global_state = states_t[:, :states.shape[2]]  # [B, state_dim]
            q_tot = self.mixer(q_agents, global_state)

            critic_loss = torch.nn.MSELoss()(q_tot, y)
        else:
            # Standard MADDPG critic update
            q_val = self.critic(states_t, actions_t).squeeze(-1)
            critic_loss = torch.nn.MSELoss()(q_val, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # Actor update (deterministic policy gradient)
        cur_actions = []
        for i, ag in enumerate(agents):
            o = torch.tensor(states[:, i, :], dtype=torch.float32, device=DEVICE)
            if ag is self:
                cur_actions.append(self.actor(o))
            else:
                with torch.no_grad():
                    cur_actions.append(ag.actor(o))
        cur_actions_t = torch.cat(cur_actions, dim=-1)

        # Use mixer for actor gradient if available
        if use_qmix and self.mixer is not None:
            individual_q_values = []
            for ag in agents:
                if ag is self:
                    q_i = ag.critic(states_t, cur_actions_t).squeeze(-1)
                else:
                    with torch.no_grad():
                        q_i = ag.critic(states_t, cur_actions_t).squeeze(-1)
                individual_q_values.append(q_i)
            q_agents = torch.stack(individual_q_values, dim=1)
            global_state = states_t[:, :states.shape[2]]
            q_tot = self.mixer(q_agents, global_state)
            actor_loss = -q_tot.mean()
        else:
            actor_loss = -self.critic(states_t, cur_actions_t).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # Target updates
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

        return {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "qmix_enabled": use_qmix and self.mixer is not None,
        }
