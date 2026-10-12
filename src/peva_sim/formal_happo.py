"""HAPPO primitives for heterogeneous recurrent decentralized policies."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .formal_mappo import RecurrentActor


class HeterogeneousRecurrentActors(nn.Module):
    """One independent recurrent policy per fixed mission role."""

    def __init__(self, n_agents: int, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        if min(n_agents, obs_dim, hidden_dim) < 1:
            raise ValueError("actor dimensions must be positive")
        self.n_agents = int(n_agents)
        self.obs_dim = int(obs_dim)
        self.hidden_dim = int(hidden_dim)
        self.actors = nn.ModuleList([
            RecurrentActor(obs_dim, hidden_dim, learned_comm=False, role_dim=0)
            for _ in range(n_agents)])

    def initial_state(self, batch_size=None, device=None):
        shape = ((self.n_agents, self.hidden_dim) if batch_size is None else
                 (int(batch_size), self.n_agents, self.hidden_dim))
        return torch.zeros(*shape, device=device)

    def step(self, observations, hidden, masks):
        if observations.shape != (self.n_agents, self.obs_dim):
            raise ValueError("step observations must be [agent,feature]")
        if hidden.shape != (self.n_agents, self.hidden_dim):
            raise ValueError("step hidden state shape mismatch")
        if masks.shape not in ((self.n_agents,), (self.n_agents, 1)):
            raise ValueError("step recurrent mask shape mismatch")
        means, logstds, next_hidden = [], [], []
        for agent, actor in enumerate(self.actors):
            mean, logstd, state = actor.step(
                observations[agent:agent + 1], hidden[agent:agent + 1],
                masks[agent:agent + 1], None)
            means.append(mean[0])
            logstds.append(logstd)
            next_hidden.append(state[0])
        return (torch.stack(means), torch.stack(logstds),
                torch.stack(next_hidden))

    def sequence(self, observations, initial_hidden, masks):
        if observations.ndim != 4:
            raise ValueError("sequence observations must be [time,batch,agent,feature]")
        time_steps, batch, agents, features = observations.shape
        if (agents, features) != (self.n_agents, self.obs_dim):
            raise ValueError("sequence observation shape mismatch")
        if initial_hidden.shape != (batch, self.n_agents, self.hidden_dim):
            raise ValueError("sequence initial hidden shape mismatch")
        if masks.shape != (time_steps, batch, self.n_agents, 1):
            raise ValueError("sequence recurrent mask shape mismatch")
        means, logstds, final_hidden = [], [], []
        for agent, actor in enumerate(self.actors):
            mean, logstd, state = actor.sequence(
                observations[:, :, agent:agent + 1],
                initial_hidden[:, agent:agent + 1],
                masks[:, :, agent:agent + 1], None)
            means.append(mean)
            logstds.append(logstd)
            final_hidden.append(state[:, 0])
        return (torch.cat(means, dim=2), torch.stack(logstds),
                torch.stack(final_hidden, dim=1))


class FormalHAPPO(nn.Module):
    """Heterogeneous actors with one privileged training-only value critic."""

    def __init__(self, n_agents: int, obs_dim: int, state_dim: int,
                 hidden_dim: int = 128):
        super().__init__()
        self.actors = HeterogeneousRecurrentActors(
            n_agents, obs_dim, hidden_dim)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, np.sqrt(2.0))
                nn.init.zeros_(layer.bias)
        for actor in self.actors.actors:
            nn.init.orthogonal_(actor.head.weight, 0.01)
        nn.init.orthogonal_(self.critic[-1].weight, 1.0)


def happo_surrogate_loss(ratio, factor, advantage, clip=0.2):
    """Clipped HAPPO objective for one agent and a frozen prefix factor."""
    if ratio.shape != factor.shape or ratio.shape != advantage.shape:
        raise ValueError("HAPPO ratio/factor/advantage shapes differ")
    if not 0 < clip < 1:
        raise ValueError("invalid HAPPO clipping coefficient")
    weighted_advantage = factor * advantage
    surrogate = torch.minimum(
        ratio * weighted_advantage,
        ratio.clamp(1.0 - clip, 1.0 + clip) * weighted_advantage)
    return -surrogate.mean()


@torch.no_grad()
def update_advantage_factor(factor, new_logprob, old_logprob):
    """Append one updated agent's likelihood ratio to the HAPPO prefix."""
    if factor.shape != new_logprob.shape or factor.shape != old_logprob.shape:
        raise ValueError("HAPPO factor/log-probability shapes differ")
    ratio = torch.exp(new_logprob - old_logprob)
    if not torch.isfinite(ratio).all():
        raise FloatingPointError("non-finite HAPPO likelihood ratio")
    return factor * ratio
