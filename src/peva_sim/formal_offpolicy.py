"""CTDE primitives for formal continuous-action multi-agent baselines.

Actors consume only the same per-agent feature rows used by MAPPO. Critics may
consume the joint feature/action tuple during training and are never required
at deployment. Actions use the simulator's open speed disk instead of clipping.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from .formal_mappo import physics_anchor_term


def radial_squash(raw: torch.Tensor) -> torch.Tensor:
    """Biject R^D to the open unit ball."""
    return raw / torch.sqrt(1.0 + raw.square().sum(dim=-1, keepdim=True))


def radial_log_abs_det_jacobian(raw: torch.Tensor) -> torch.Tensor:
    """Log determinant of :func:`radial_squash` for arbitrary action D."""
    dimension = raw.shape[-1]
    radius_squared = raw.square().sum(dim=-1)
    return -0.5 * (dimension + 2) * torch.log1p(radius_squared)


class DecentralizedActor(nn.Module):
    """One parameter-shared actor evaluated independently by every agent."""

    def __init__(self, obs_dim: int, hidden_dim: int = 256,
                 stochastic: bool = False, anchor_strength: float = 0.0,
                 anchor_angle_deg: float = 15.0, anchor_targets: int = 3,
                 role_dim: int = 0):
        super().__init__()
        self.anchor_strength = float(anchor_strength)
        self.anchor_angle_deg = float(anchor_angle_deg)
        self.anchor_targets = int(anchor_targets)
        self.role_dim = int(role_dim)
        if not np.isfinite(self.anchor_strength) or self.anchor_strength < 0:
            raise ValueError("anchor_strength must be finite and nonnegative")
        if self.anchor_strength > 0 and self.role_dim <= 0:
            raise ValueError("physics anchor requires role features")
        self.obs_dim = int(obs_dim)
        self.hidden_dim = int(hidden_dim)
        self.stochastic = bool(stochastic)
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.mean = nn.Linear(hidden_dim, 2)
        self.logstd = nn.Linear(hidden_dim, 2) if stochastic else None
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, np.sqrt(2.0))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.mean.weight, 0.01)
        if self.logstd is not None:
            nn.init.orthogonal_(self.logstd.weight, 0.01)

    def _latent_mean(self, observations: torch.Tensor, hidden: torch.Tensor):
        """Learned head is a residual on the physics anchor (if enabled)."""
        mean = self.mean(hidden)
        if self.anchor_strength > 0:
            mean = mean + physics_anchor_term(
                observations, self.role_dim, self.anchor_targets,
                self.anchor_angle_deg, self.anchor_strength)
        return mean

    def distribution(self, observations: torch.Tensor):
        if not self.stochastic:
            raise RuntimeError("deterministic actor has no action distribution")
        hidden = self.trunk(observations)
        mean = self._latent_mean(observations, hidden)
        logstd = self.logstd(hidden).clamp(-5.0, 2.0)
        return torch.distributions.Normal(mean, logstd.exp())

    def deterministic(self, observations: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(observations)
        return radial_squash(self._latent_mean(observations, hidden))

    def sample(self, observations: torch.Tensor):
        distribution = self.distribution(observations)
        raw = distribution.rsample()
        action = radial_squash(raw)
        log_prob = (distribution.log_prob(raw).sum(dim=-1)
                    - radial_log_abs_det_jacobian(raw))
        return action, log_prob


class CentralizedCritic(nn.Module):
    """Q(s_joint, a_joint), used only by the centralized trainer."""

    def __init__(self, n_agents: int, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.n_agents = int(n_agents)
        self.obs_dim = int(obs_dim)
        input_dim = n_agents * (obs_dim + 2)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, np.sqrt(2.0))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.network[-1].weight, 1.0)

    def forward(self, observations: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        if observations.shape[-2:] != (self.n_agents, self.obs_dim):
            raise ValueError("joint observation shape mismatch")
        if actions.shape[-2:] != (self.n_agents, 2):
            raise ValueError("joint action shape mismatch")
        if observations.shape[:-2] != actions.shape[:-2]:
            raise ValueError("joint observation/action batch mismatch")
        inputs = torch.cat((observations.flatten(-2), actions.flatten(-2)), -1)
        return self.network(inputs).squeeze(-1)


class JointReplayBuffer:
    """Fixed-size replay with checkpointable sampling and insertion state."""

    def __init__(self, capacity: int, n_agents: int, obs_dim: int, seed: int):
        if min(capacity, n_agents, obs_dim) < 1:
            raise ValueError("replay dimensions must be positive")
        self.capacity = int(capacity)
        self.n_agents = int(n_agents)
        self.obs_dim = int(obs_dim)
        # Zero initialization keeps unused checkpoint slots deterministic and
        # avoids serializing unrelated bytes from the allocator.
        self.observations = np.zeros(
            (capacity, n_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents, 2), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros_like(self.observations)
        self.bootstrap = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, observation, action, reward, next_observation, bootstrap):
        index = self.position
        self.observations[index] = observation
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_observations[index] = next_observation
        self.bootstrap[index] = bootstrap
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_indices(self, batch_size: int) -> np.ndarray:
        if batch_size < 1 or batch_size > self.size:
            raise ValueError("invalid replay batch size")
        return self.rng.integers(0, self.size, size=batch_size)

    def sample(self, batch_size: int, device: torch.device):
        indices = self.sample_indices(batch_size)
        tensor = lambda value: torch.as_tensor(
            value[indices], dtype=torch.float32, device=device)
        return {
            "observations": tensor(self.observations),
            "actions": tensor(self.actions),
            "rewards": tensor(self.rewards),
            "next_observations": tensor(self.next_observations),
            "bootstrap": tensor(self.bootstrap),
        }

    def state_dict(self):
        return {
            "capacity": self.capacity, "n_agents": self.n_agents,
            "obs_dim": self.obs_dim, "position": self.position,
            "size": self.size, "observations": self.observations.copy(),
            "actions": self.actions.copy(), "rewards": self.rewards.copy(),
            "next_observations": self.next_observations.copy(),
            "bootstrap": self.bootstrap.copy(),
            "rng": deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state):
        contract = (state["capacity"], state["n_agents"], state["obs_dim"])
        if contract != (self.capacity, self.n_agents, self.obs_dim):
            raise ValueError("replay checkpoint contract mismatch")
        for name in ("observations", "actions", "rewards",
                     "next_observations", "bootstrap"):
            target = getattr(self, name)
            source = np.asarray(state[name])
            if target.shape != source.shape or target.dtype != source.dtype:
                raise ValueError("replay array contract mismatch: " + name)
            target[...] = source
        self.position = int(state["position"])
        self.size = int(state["size"])
        self.rng.bit_generator.state = deepcopy(state["rng"])


@dataclass
class OffPolicyNetworks:
    actor: DecentralizedActor
    actor_target: DecentralizedActor
    critic1: CentralizedCritic
    critic1_target: CentralizedCritic
    critic2: CentralizedCritic | None = None
    critic2_target: CentralizedCritic | None = None


def build_networks(algorithm: str, n_agents: int, obs_dim: int,
                   hidden_dim: int = 256, anchor_strength: float = 0.0,
                   anchor_angle_deg: float = 15.0,
                   anchor_targets: int = 3) -> OffPolicyNetworks:
    if algorithm not in ("maddpg", "matd3", "masac"):
        raise ValueError("unknown off-policy algorithm")
    stochastic = algorithm == "masac"
    actor = DecentralizedActor(obs_dim, hidden_dim, stochastic,
                               anchor_strength, anchor_angle_deg,
                               anchor_targets, role_dim=n_agents)
    critic1 = CentralizedCritic(n_agents, obs_dim, hidden_dim)
    critic2 = (CentralizedCritic(n_agents, obs_dim, hidden_dim)
               if algorithm in ("matd3", "masac") else None)
    return OffPolicyNetworks(
        actor=actor, actor_target=deepcopy(actor),
        critic1=critic1, critic1_target=deepcopy(critic1),
        critic2=critic2,
        critic2_target=(deepcopy(critic2) if critic2 is not None else None))


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float):
    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must be in (0,1]")
    for target_parameter, source_parameter in zip(
            target.parameters(), source.parameters(), strict=True):
        target_parameter.mul_(1.0 - tau).add_(source_parameter, alpha=tau)


def target_actions(algorithm: str, networks: OffPolicyNetworks,
                   next_observations: torch.Tensor,
                   policy_noise: float = 0.2, noise_clip: float = 0.5):
    if algorithm == "masac":
        return networks.actor.sample(next_observations)
    actions = networks.actor_target.deterministic(next_observations)
    if algorithm == "matd3":
        noise = torch.randn_like(actions).mul(policy_noise).clamp(
            -noise_clip, noise_clip)
        actions = actions + noise
        norm = actions.norm(dim=-1, keepdim=True)
        actions = actions / norm.clamp_min(1.0 + 1e-6)
    return actions, None


def _critic_parameters(networks: OffPolicyNetworks):
    parameters = list(networks.critic1.parameters())
    if networks.critic2 is not None:
        parameters += list(networks.critic2.parameters())
    return parameters


def update_offpolicy(
        algorithm: str, networks: OffPolicyNetworks, batch: dict,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer, update_index: int,
        gamma: float = 0.99, tau: float = 0.005,
        policy_delay: int = 2, policy_noise: float = 0.2,
        noise_clip: float = 0.5, alpha: torch.Tensor | float = 0.2,
        log_alpha: torch.Tensor | None = None,
        alpha_optimizer: torch.optim.Optimizer | None = None,
        target_entropy: float | None = None):
    """Apply one MADDPG, MATD3, or MASAC minibatch update.

    Rewards and bootstrap masks are mission-level scalars. The decentralized
    policy factorizes over agents, so MASAC's joint log probability is the sum
    of the per-agent terms.
    """
    if algorithm not in ("maddpg", "matd3", "masac"):
        raise ValueError("unknown off-policy algorithm")
    if not 0.0 < gamma <= 1.0 or not 0.0 < tau <= 1.0:
        raise ValueError("invalid discount or target-update coefficient")
    if policy_delay < 1:
        raise ValueError("policy_delay must be positive")
    required = {"observations", "actions", "rewards",
                "next_observations", "bootstrap"}
    if set(batch) != required:
        raise ValueError("off-policy batch contract mismatch")
    observations = batch["observations"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    next_observations = batch["next_observations"]
    bootstrap = batch["bootstrap"]
    if rewards.shape != bootstrap.shape or rewards.ndim != 1:
        raise ValueError("reward/bootstrap shape mismatch")

    alpha_value = (alpha.detach() if isinstance(alpha, torch.Tensor)
                   else torch.as_tensor(alpha, dtype=rewards.dtype,
                                        device=rewards.device))
    with torch.no_grad():
        next_actions, next_log_prob = target_actions(
            algorithm, networks, next_observations,
            policy_noise=policy_noise, noise_clip=noise_clip)
        target_q1 = networks.critic1_target(next_observations, next_actions)
        if networks.critic2_target is not None:
            target_q2 = networks.critic2_target(next_observations, next_actions)
            target_q = torch.minimum(target_q1, target_q2)
        else:
            target_q = target_q1
        if algorithm == "masac":
            target_q = target_q - alpha_value * next_log_prob.sum(dim=-1)
        target = rewards + gamma * bootstrap * target_q

    q1 = networks.critic1(observations, actions)
    critic_loss = (q1 - target).square().mean()
    q2 = None
    if networks.critic2 is not None:
        q2 = networks.critic2(observations, actions)
        critic_loss = critic_loss + (q2 - target).square().mean()
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(_critic_parameters(networks), 10.0)
    critic_optimizer.step()

    update_actor = algorithm != "matd3" or update_index % policy_delay == 0
    actor_loss_value = None
    entropy_value = None
    alpha_loss_value = None
    if update_actor:
        for parameter in _critic_parameters(networks):
            parameter.requires_grad_(False)
        if algorithm == "masac":
            policy_actions, log_prob = networks.actor.sample(observations)
            joint_log_prob = log_prob.sum(dim=-1)
            policy_q1 = networks.critic1(observations, policy_actions)
            policy_q2 = networks.critic2(observations, policy_actions)
            policy_q = torch.minimum(policy_q1, policy_q2)
            actor_loss = (alpha_value * joint_log_prob - policy_q).mean()
            entropy_value = float((-joint_log_prob).mean().detach())
        else:
            policy_actions = networks.actor.deterministic(observations)
            actor_loss = -networks.critic1(
                observations, policy_actions).mean()
            joint_log_prob = None
        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(networks.actor.parameters(), 10.0)
        actor_optimizer.step()
        actor_loss_value = float(actor_loss.detach())
        for parameter in _critic_parameters(networks):
            parameter.requires_grad_(True)

        if algorithm == "masac" and log_alpha is not None:
            if alpha_optimizer is None or target_entropy is None:
                raise ValueError("automatic entropy tuning contract incomplete")
            alpha_loss = -(
                log_alpha * (joint_log_prob.detach() + target_entropy)).mean()
            alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            alpha_optimizer.step()
            alpha_loss_value = float(alpha_loss.detach())

        soft_update(networks.actor_target, networks.actor, tau)
        soft_update(networks.critic1_target, networks.critic1, tau)
        if networks.critic2 is not None:
            soft_update(networks.critic2_target, networks.critic2, tau)

    result = {
        "critic_loss": float(critic_loss.detach()),
        "actor_loss": actor_loss_value,
        "target_mean": float(target.mean()),
        "q1_mean": float(q1.mean().detach()),
        "actor_updated": update_actor,
    }
    if q2 is not None:
        result["q2_mean"] = float(q2.mean().detach())
    if entropy_value is not None:
        result["joint_entropy"] = entropy_value
        result["alpha"] = float(alpha_value)
    if alpha_loss_value is not None:
        result["alpha_loss"] = alpha_loss_value
    return result
