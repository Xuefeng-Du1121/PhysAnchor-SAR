import unittest

import numpy as np
import torch

from peva_sim.formal_ic3net import (
    GatedRecurrentActor, exchange_gated_messages, joint_policy_entropy,
    joint_policy_log_prob)


class FakeRadio:
    def __init__(self, agents):
        self.agents = agents
        self.messages = None

    def exchange_messages(self, messages):
        self.messages = messages
        return [[(sender, payload) for sender, payload in enumerate(messages)
                 if sender != receiver and payload]
                for receiver in range(self.agents)]


class IC3NetTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.actor = GatedRecurrentActor(10, hidden_dim=16, role_dim=3)
        self.observations = torch.randn(3, 10)
        self.observations[:, -3:] = torch.eye(3)
        self.hidden = self.actor.initial_state(3)
        self.recurrent_mask = torch.zeros(3, 1)

    def test_gate_is_a_bernoulli_policy_action(self):
        gate = self.actor.gate_distribution(
            self.observations, self.hidden, self.recurrent_mask)
        samples = gate.sample()
        self.assertEqual(samples.shape, (3,))
        self.assertTrue(bool(((samples == 0) | (samples == 1)).all()))
        self.assertTrue(torch.isfinite(gate.log_prob(samples)).all())

    def test_closed_gate_sends_no_packet_or_header_candidate(self):
        radio = FakeRadio(3)
        delivery, packets = exchange_gated_messages(
            self.actor, self.observations, torch.tensor([1., 0., 1.]), radio)
        self.assertEqual(len(packets[0]), 42)
        self.assertEqual(packets[1], b"")
        self.assertEqual(len(packets[2]), 42)
        self.assertTrue(bool((delivery[:, 1] == 0).all()))
        self.assertEqual(int(delivery.sum()), 4)

    def test_joint_logprob_and_entropy_include_gate_action(self):
        gate = self.actor.gate_distribution(
            self.observations, self.hidden, self.recurrent_mask)
        gate_action = gate.sample()
        delivery = torch.zeros(3, 3)
        mean, logstd, _ = self.actor.step(
            self.observations, self.hidden, self.recurrent_mask, delivery)
        raw = torch.randn(3, 2)
        logprob = joint_policy_log_prob(
            mean, logstd, raw, gate, gate_action)
        entropy = joint_policy_entropy(mean, logstd, gate)
        self.assertEqual(logprob.shape, (3,))
        self.assertEqual(entropy.shape, (3,))
        self.assertTrue(torch.isfinite(logprob).all())
        self.assertTrue(torch.isfinite(entropy).all())

    def test_delivered_message_path_has_gradient(self):
        delivery = torch.tensor([
            [0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
        mean, _, _ = self.actor.step(
            self.observations, self.hidden, self.recurrent_mask, delivery)
        mean.square().sum().backward()
        gradient = self.actor.message_head.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_gate_gradient_comes_from_policy_logprob(self):
        gate = self.actor.gate_distribution(
            self.observations, self.hidden, self.recurrent_mask)
        loss = -gate.log_prob(torch.tensor([1., 0., 1.])).mean()
        loss.backward()
        gradient = self.actor.gate_head.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
