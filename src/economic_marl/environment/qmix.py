from __future__ import annotations
import torch
import torch.nn as nn

class MixingNetwork(nn.Module):
    """QMIX mixing network: enforces monotonic relation between per-agent Q and global Q_tot.

    Note: In a full QMIX implementation, each agent would have its own Q_net producing
    q_i, combined here into Q_tot via state-conditioned hypernetworks. For brevity,
    this mixing network can be used to regularize the MADDPG critic target as a
    Pareto-coordination signal.
    """
    def __init__(self, n_agents: int, state_dim: int, embed_dim: int = 32):
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        self.hyper_w1 = nn.Linear(state_dim, n_agents * embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Linear(state_dim, 1)
        self.elu = nn.ELU()

    def forward(self, q_agents: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """q_agents: [B, n_agents]; state: [B, state_dim]"""
        B = q_agents.size(0)
        w1 = torch.abs(self.hyper_w1(state)).view(B, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(B, 1, self.embed_dim)
        y = torch.bmm(q_agents.unsqueeze(1), w1) + b1  # [B, 1, embed_dim]
        y = self.elu(y)
        w2 = torch.abs(self.hyper_w2(state)).view(B, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(B, 1, 1)
        q_tot = torch.bmm(y, w2) + b2  # [B, 1, 1]
        return q_tot.squeeze(-1).squeeze(-1)  # [B]
