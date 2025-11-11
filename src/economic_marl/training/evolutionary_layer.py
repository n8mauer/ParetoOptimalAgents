from __future__ import annotations
from typing import List
import numpy as np
import torch
from copy import deepcopy

class EvolutionaryLayer:
    """Genetic-style meta-optimizer that mutates agent actor weights and hyperparameters."""
    def __init__(self, mutation_rate: float = 0.03, elite_fraction: float = 0.2, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.mutation_rate = mutation_rate
        self.elite_fraction = elite_fraction

    def evolve(self, population: List, fitness: List[float]) -> List:
        n = len(population); k = max(1, int(n * self.elite_fraction))
        elite_idx = np.argsort(fitness)[-k:]
        new_pop = [deepcopy(population[i]) for i in elite_idx]
        while len(new_pop) < n:
            parent = deepcopy(self.rng.choice(new_pop))
            with torch.no_grad():
                for p in parent.actor.parameters():
                    noise = self.mutation_rate * torch.randn_like(p)
                    p.add_(noise)
            new_pop.append(parent)
        return new_pop
