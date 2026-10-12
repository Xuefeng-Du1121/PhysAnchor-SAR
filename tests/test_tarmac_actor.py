import unittest

import torch

from peva_sim.formal_mappo import FormalMAPPO, TargetedRecurrentActor


class TarMACActorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.actor = TargetedRecurrentActor(
            obs_dim=12, hidden_dim=32, role_dim=3, key_dim=8)
        self.observation = torch.randn(3, 12)
        self.observation[:, -3:] = torch.eye(3)

    def test_packet_budget_is_split_into_key_and_value(self):
        payload = self.actor.encoded_messages(self.observation)
        self.assertEqual(payload.shape, (3, 32))
        self.assertEqual(self.actor.key_dim, 8)
        self.assertEqual(self.actor.value_dim, 24)

    def test_delivery_mask_changes_only_affected_receiver(self):
        hidden = self.actor.initial_state(3)
        recurrent_mask = torch.zeros(3, 1)
        identity = torch.eye(3)
        directed = identity.clone()
        directed[0, 1] = 1
        mean_identity, _, _ = self.actor.step(
            self.observation, hidden, recurrent_mask, identity)
        mean_directed, _, _ = self.actor.step(
            self.observation, hidden, recurrent_mask, directed)
        self.assertFalse(torch.allclose(mean_identity[0], mean_directed[0]))
        torch.testing.assert_close(mean_identity[1:], mean_directed[1:])

    def test_attention_path_is_differentiable(self):
        delivery = torch.ones(3, 3)
        mean, _, _ = self.actor.step(
            self.observation, self.actor.initial_state(3),
            torch.zeros(3, 1), delivery)
        mean.square().sum().backward()
        for layer in (self.actor.message_head, self.actor.query_head):
            self.assertIsNotNone(layer.weight.grad)
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(float(layer.weight.grad.abs().sum()), 0.0)

    def test_every_receiver_requires_local_self_loop(self):
        delivery = torch.eye(3)
        delivery[1] = 0
        with self.assertRaisesRegex(ValueError, "self-loop"):
            self.actor.step(
                self.observation, self.actor.initial_state(3),
                torch.zeros(3, 1), delivery)

    def test_formal_model_selects_targeted_actor(self):
        model = FormalMAPPO(
            12, 20, hidden_dim=32, learned_comm=True, role_dim=3,
            communication_type="tarmac")
        self.assertIsInstance(model.actor, TargetedRecurrentActor)
        with self.assertRaisesRegex(ValueError, "disagree"):
            FormalMAPPO(12, 20, learned_comm=False,
                        communication_type="tarmac")


if __name__ == "__main__":
    unittest.main()
