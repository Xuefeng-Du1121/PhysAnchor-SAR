import unittest

import numpy as np
import torch

from peva_sim.evaluate_formal_offpolicy import load_actor
from peva_sim.formal_offpolicy import (
    CentralizedCritic, DecentralizedActor, JointReplayBuffer, build_networks,
    radial_log_abs_det_jacobian, radial_squash, soft_update, target_actions,
    update_offpolicy,
)


class FormalOffPolicyTests(unittest.TestCase):
    def test_evaluator_restores_each_actor_architecture(self):
        for algorithm in ("maddpg", "matd3", "masac"):
            networks = build_networks(algorithm, 3, 8, hidden_dim=32)
            saved = {
                "version": "formal_ctde_continuous_offpolicy_v1",
                "metadata": {
                    "version": "formal_ctde_continuous_offpolicy_v1",
                    "algorithm": algorithm,
                    "config": {"n_uavs": 3},
                },
                "actor": networks.actor.state_dict(),
            }
            restored, metadata = load_actor(saved, n_agents=3)
            observations = torch.randn(5, 3, 8)
            torch.testing.assert_close(
                restored.deterministic(observations),
                networks.actor.deterministic(observations))
            self.assertEqual(metadata["algorithm"], algorithm)
            self.assertEqual(restored.stochastic, algorithm == "masac")

    def test_evaluator_rejects_team_size_mismatch(self):
        networks = build_networks("matd3", 3, 8, hidden_dim=32)
        saved = {
            "version": "formal_ctde_continuous_offpolicy_v1",
            "metadata": {
                "version": "formal_ctde_continuous_offpolicy_v1",
                "algorithm": "matd3", "config": {"n_uavs": 3},
            },
            "actor": networks.actor.state_dict(),
        }
        with self.assertRaisesRegex(ValueError, "team size"):
            load_actor(saved, n_agents=4)

    def test_radial_actions_stay_in_open_unit_disk(self):
        raw = torch.randn(1000, 2) * 20
        action = radial_squash(raw)
        self.assertTrue(bool((action.norm(dim=-1) < 1).all()))

    def test_radial_jacobian_matches_autograd(self):
        raw = torch.tensor([0.4, -1.2], dtype=torch.float64,
                           requires_grad=True)
        jacobian = torch.autograd.functional.jacobian(radial_squash, raw)
        expected = torch.logdet(jacobian)
        actual = radial_log_abs_det_jacobian(raw)
        torch.testing.assert_close(actual, expected)

    def test_actor_is_per_agent_and_critic_is_joint(self):
        actor = DecentralizedActor(9, hidden_dim=32)
        critic = CentralizedCritic(6, 9, hidden_dim=32)
        observations = torch.randn(7, 6, 9)
        actions = actor.deterministic(observations)
        self.assertEqual(actions.shape, (7, 6, 2))
        self.assertEqual(critic(observations, actions).shape, (7,))
        changed = observations.clone()
        changed[:, 1:] = torch.randn_like(changed[:, 1:])
        torch.testing.assert_close(
            actor.deterministic(observations)[:, 0],
            actor.deterministic(changed)[:, 0])

    def test_masac_log_probability_and_twin_critics(self):
        networks = build_networks("masac", 4, 11, hidden_dim=32)
        observations = torch.randn(5, 4, 11)
        actions, log_prob = networks.actor.sample(observations)
        self.assertEqual(actions.shape, (5, 4, 2))
        self.assertEqual(log_prob.shape, (5, 4))
        self.assertTrue(torch.isfinite(log_prob).all())
        self.assertIsNotNone(networks.critic2)

    def test_matd3_target_smoothing_respects_action_disk(self):
        networks = build_networks("matd3", 3, 8, hidden_dim=32)
        observations = torch.randn(50, 3, 8)
        actions, log_prob = target_actions("matd3", networks, observations)
        self.assertIsNone(log_prob)
        self.assertTrue(bool((actions.norm(dim=-1) <= 1).all()))

    def test_soft_update(self):
        source = DecentralizedActor(5, hidden_dim=16)
        target = DecentralizedActor(5, hidden_dim=16)
        with torch.no_grad():
            for parameter in source.parameters():
                parameter.fill_(2.0)
            for parameter in target.parameters():
                parameter.zero_()
        soft_update(target, source, 0.25)
        for parameter in target.parameters():
            torch.testing.assert_close(parameter, torch.full_like(parameter, .5))

    def test_replay_round_trip_preserves_next_sample(self):
        first = JointReplayBuffer(16, 3, 4, seed=19)
        for step in range(12):
            observation = np.full((3, 4), step, dtype=np.float32)
            action = np.full((3, 2), step / 20, dtype=np.float32)
            first.add(observation, action, step, observation + 1, step % 2)
        state = first.state_dict()
        second = JointReplayBuffer(16, 3, 4, seed=999)
        second.load_state_dict(state)
        np.testing.assert_array_equal(first.sample_indices(7),
                                      second.sample_indices(7))

    def test_shape_contract_rejects_non_joint_input(self):
        critic = CentralizedCritic(3, 4, hidden_dim=16)
        with self.assertRaises(ValueError):
            critic(torch.randn(8, 4), torch.randn(8, 3, 2))

    def make_batch(self, batch=16, agents=3, obs_dim=8):
        return {
            "observations": torch.randn(batch, agents, obs_dim),
            "actions": radial_squash(torch.randn(batch, agents, 2)),
            "rewards": torch.randn(batch),
            "next_observations": torch.randn(batch, agents, obs_dim),
            "bootstrap": torch.randint(0, 2, (batch,)).float(),
        }

    def test_maddpg_update_changes_actor_and_single_critic(self):
        networks = build_networks("maddpg", 3, 8, hidden_dim=32)
        actor_optimizer = torch.optim.Adam(networks.actor.parameters(), lr=1e-3)
        critic_optimizer = torch.optim.Adam(networks.critic1.parameters(), lr=1e-3)
        before = [parameter.detach().clone() for parameter in networks.actor.parameters()]
        result = update_offpolicy(
            "maddpg", networks, self.make_batch(), actor_optimizer,
            critic_optimizer, update_index=1)
        self.assertTrue(result["actor_updated"])
        self.assertIsNone(networks.critic2)
        self.assertTrue(any(not torch.equal(old, new) for old, new in
                            zip(before, networks.actor.parameters())))

    def test_matd3_delays_policy_update(self):
        networks = build_networks("matd3", 3, 8, hidden_dim=32)
        actor_optimizer = torch.optim.Adam(networks.actor.parameters(), lr=1e-3)
        critic_optimizer = torch.optim.Adam(
            list(networks.critic1.parameters()) +
            list(networks.critic2.parameters()), lr=1e-3)
        result = update_offpolicy(
            "matd3", networks, self.make_batch(), actor_optimizer,
            critic_optimizer, update_index=1, policy_delay=2)
        self.assertFalse(result["actor_updated"])
        result = update_offpolicy(
            "matd3", networks, self.make_batch(), actor_optimizer,
            critic_optimizer, update_index=2, policy_delay=2)
        self.assertTrue(result["actor_updated"])

    def test_masac_update_tunes_entropy_temperature(self):
        networks = build_networks("masac", 3, 8, hidden_dim=32)
        actor_optimizer = torch.optim.Adam(networks.actor.parameters(), lr=1e-3)
        critic_optimizer = torch.optim.Adam(
            list(networks.critic1.parameters()) +
            list(networks.critic2.parameters()), lr=1e-3)
        log_alpha = torch.tensor(np.log(.2), requires_grad=True)
        alpha_optimizer = torch.optim.Adam([log_alpha], lr=1e-3)
        result = update_offpolicy(
            "masac", networks, self.make_batch(), actor_optimizer,
            critic_optimizer, update_index=1, alpha=log_alpha.exp(),
            log_alpha=log_alpha, alpha_optimizer=alpha_optimizer,
            target_entropy=-6.0)
        self.assertTrue(np.isfinite(result["joint_entropy"]))
        self.assertIn("alpha_loss", result)


if __name__ == "__main__":
    unittest.main()
