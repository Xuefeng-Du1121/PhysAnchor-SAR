"""IC3Net-style recurrent communication with explicit Bernoulli gates."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .communication import MESSAGE_DIM, encode_message
from .joint_transport import quantize_message


class GatedRecurrentActor(nn.Module):
    """Shared actor whose binary send decision is part of the policy action."""

    def __init__(self, obs_dim: int, hidden_dim: int = 128, role_dim: int = 0):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.hidden_dim = int(hidden_dim)
        self.role_dim = int(role_dim)
        if self.role_dim < 0 or self.role_dim > self.obs_dim:
            raise ValueError("role_dim must be between zero and obs_dim")
        self.message_head = nn.Linear(obs_dim, MESSAGE_DIM)
        self.gate_head = nn.Linear(obs_dim + hidden_dim, 1)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + MESSAGE_DIM, hidden_dim), nn.Tanh())
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, 2)
        if self.role_dim:
            self.role_bias = nn.Parameter(torch.zeros(self.role_dim, 2))
        else:
            self.register_parameter("role_bias", None)
        self.logstd = nn.Parameter(torch.full((2,), -.5))

    def initial_state(self, n_agents, device=None):
        return torch.zeros(n_agents, self.hidden_dim, device=device)

    def encoded_messages(self, observations):
        return quantize_message(torch.tanh(self.message_head(observations)))

    def gate_distribution(self, observations, hidden, recurrent_mask):
        recurrent_mask = torch.as_tensor(
            recurrent_mask, dtype=observations.dtype, device=observations.device)
        if recurrent_mask.ndim == observations.ndim - 1:
            recurrent_mask = recurrent_mask[..., None]
        if hidden.shape[:-1] != observations.shape[:-1]:
            raise ValueError("gate hidden/observation shape mismatch")
        if recurrent_mask.shape != observations.shape[:-1] + (1,):
            raise ValueError("gate recurrent mask shape mismatch")
        inputs = torch.cat((observations, hidden * recurrent_mask), dim=-1)
        return torch.distributions.Bernoulli(logits=self.gate_head(inputs).squeeze(-1))

    def _fuse_messages(self, observations, delivery_mask):
        if delivery_mask is None:
            raise ValueError("IC3Net requires a delivery mask")
        squeeze = observations.ndim == 2
        batched = observations[None] if squeeze else observations
        delivery = (torch.as_tensor(
            delivery_mask, dtype=observations.dtype,
            device=observations.device)[None] if squeeze else
            torch.as_tensor(delivery_mask, dtype=observations.dtype,
                            device=observations.device))
        expected = (batched.shape[0], batched.shape[1], batched.shape[1])
        if batched.ndim != 3 or delivery.shape != expected:
            raise ValueError("delivery mask must be [batch,receiver,sender]")
        messages = self.encoded_messages(batched)
        denominator = delivery.sum(-1, keepdim=True).clamp_min(1.)
        aggregate = torch.einsum("brs,bsd->brd", delivery, messages)
        fused = torch.cat((batched, aggregate / denominator), dim=-1)
        return fused[0] if squeeze else fused

    def _action_mean(self, hidden, observations):
        mean = self.action_head(hidden)
        if not self.role_dim:
            return mean
        roles = observations[..., -self.role_dim:]
        if not torch.isfinite(roles).all():
            raise ValueError("non-finite role features")
        if not torch.allclose(roles.sum(-1), torch.ones_like(roles[..., 0]),
                              atol=1e-5, rtol=0.):
            raise ValueError("role suffix must be one-hot")
        if bool(((roles < -1e-6) | (roles > 1. + 1e-6)).any()):
            raise ValueError("role suffix must be one-hot")
        return mean + roles @ self.role_bias

    def step(self, observations, hidden, recurrent_mask, delivery_mask):
        recurrent_mask = torch.as_tensor(
            recurrent_mask, dtype=observations.dtype, device=observations.device)
        if observations.ndim == 2 and recurrent_mask.ndim == 1:
            recurrent_mask = recurrent_mask[:, None]
        fused = self._fuse_messages(observations, delivery_mask)
        shape = fused.shape[:-1]
        encoded = self.encoder(fused.reshape(-1, fused.shape[-1]))
        flat_hidden = hidden.reshape(-1, self.hidden_dim)
        flat_mask = recurrent_mask.reshape(-1, 1)
        flat_hidden = self.gru(encoded, flat_hidden * flat_mask)
        next_hidden = flat_hidden.reshape(*shape, self.hidden_dim)
        return (self._action_mean(next_hidden, observations),
                self.logstd.clamp(-4, 1), next_hidden)

    def sequence(self, observations, initial_hidden, recurrent_masks,
                 delivery_masks):
        if observations.ndim != 4:
            raise ValueError("expected [time,batch,agent,feature] observations")
        if delivery_masks is None:
            raise ValueError("IC3Net sequence requires delivery masks")
        hidden = initial_hidden
        means, gate_logits = [], []
        for step in range(observations.shape[0]):
            gate = self.gate_distribution(
                observations[step], hidden, recurrent_masks[step])
            mean, _, hidden = self.step(
                observations[step], hidden, recurrent_masks[step],
                delivery_masks[step])
            means.append(mean)
            gate_logits.append(gate.logits)
        return (torch.stack(means), self.logstd.clamp(-4, 1), hidden,
                torch.stack(gate_logits))


class FormalIC3Net(nn.Module):
    def __init__(self, obs_dim, state_dim, hidden_dim=128, role_dim=0):
        super().__init__()
        self.actor = GatedRecurrentActor(
            obs_dim, hidden_dim, role_dim=role_dim)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, np.sqrt(2.0))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.message_head.weight, .01)
        nn.init.orthogonal_(self.actor.gate_head.weight, .01)
        nn.init.orthogonal_(self.actor.action_head.weight, .01)
        if self.actor.role_bias is not None:
            nn.init.normal_(self.actor.role_bias, mean=0., std=.01)
        nn.init.orthogonal_(self.critic[-1].weight, 1.0)


@torch.no_grad()
def exchange_gated_messages(actor, observations, gate_actions, env):
    """Send only gate-open packets and return actual receiver/sender delivery."""
    gates = torch.as_tensor(gate_actions).reshape(-1)
    if gates.shape != (observations.shape[-2],):
        raise ValueError("one communication gate is required per agent")
    if not bool(((gates == 0) | (gates == 1)).all()):
        raise ValueError("communication gates must be binary")
    messages = actor.encoded_messages(observations).detach().cpu().numpy()
    packets = [
        encode_message(message, relevance=1.) if bool(gate) else b""
        for message, gate in zip(messages, gates.cpu().numpy(), strict=True)]
    inboxes = env.exchange_messages(packets)
    agents = len(packets)
    delivery = np.zeros((agents, agents), dtype=np.float32)
    for receiver, inbox in enumerate(inboxes):
        for sender, payload in inbox:
            if not packets[sender] or payload != packets[sender]:
                raise ValueError("delivered gated payload is invalid")
            delivery[receiver, sender] = 1.
    return torch.as_tensor(
        delivery, dtype=observations.dtype, device=observations.device), packets


def joint_policy_log_prob(mean, logstd, raw_action, gate_distribution,
                          gate_action):
    action_logprob = torch.distributions.Normal(
        mean, logstd.exp()).log_prob(raw_action).sum(-1)
    gate_logprob = gate_distribution.log_prob(gate_action)
    if action_logprob.shape != gate_logprob.shape:
        raise ValueError("action and communication log-probability shapes differ")
    return action_logprob + gate_logprob


def joint_policy_entropy(mean, logstd, gate_distribution):
    action_entropy = torch.distributions.Normal(
        mean, logstd.exp()).entropy().sum(-1)
    gate_entropy = gate_distribution.entropy()
    if action_entropy.shape != gate_entropy.shape:
        raise ValueError("action and communication entropy shapes differ")
    return action_entropy + gate_entropy
