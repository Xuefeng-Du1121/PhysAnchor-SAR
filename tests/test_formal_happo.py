import unittest
from types import SimpleNamespace

import torch

from peva_sim.evaluate_formal_happo import load_model
from peva_sim.formal_happo import (
    FormalHAPPO, HeterogeneousRecurrentActors, happo_surrogate_loss,
    update_advantage_factor)
from peva_sim.ppo_math import latent_log_prob
from peva_sim.train_formal_happo import agent_logprob


class FormalHAPPOTests(unittest.TestCase):
    def test_each_agent_has_independent_parameters(self):
        actors = HeterogeneousRecurrentActors(3, 9, hidden_dim=16)
        pointers = [next(actor.parameters()).data_ptr()
                    for actor in actors.actors]
        self.assertEqual(len(set(pointers)), 3)

    def test_step_and_sequence_contracts_match(self):
        torch.manual_seed(5)
        actors = HeterogeneousRecurrentActors(3, 9, hidden_dim=16)
        observations = torch.randn(4, 2, 3, 9)
        masks = torch.ones(4, 2, 3, 1)
        masks[0].zero_()
        initial = actors.initial_state(batch_size=2)
        sequence_mean, logstd, final = actors.sequence(
            observations, initial, masks)
        self.assertEqual(sequence_mean.shape, (4, 2, 3, 2))
        self.assertEqual(logstd.shape, (3, 2))
        self.assertEqual(final.shape, (2, 3, 16))
        hidden = actors.initial_state()
        outputs = []
        for step in range(4):
            mean, step_logstd, hidden = actors.step(
                observations[step, 0], hidden, masks[step, 0])
            outputs.append(mean)
        torch.testing.assert_close(torch.stack(outputs), sequence_mean[:, 0])
        torch.testing.assert_close(step_logstd, logstd)
        torch.testing.assert_close(hidden, final[0])

    def test_centralized_critic_is_separate_from_deployment_actors(self):
        model = FormalHAPPO(3, 9, 21, hidden_dim=16)
        self.assertEqual(model.critic(torch.randn(7, 21)).shape, (7, 1))
        actor_parameter_ids = {
            id(parameter) for parameter in model.actors.parameters()}
        self.assertFalse(any(id(parameter) in actor_parameter_ids
                             for parameter in model.critic.parameters()))

    def test_surrogate_applies_prefix_factor_before_clipping(self):
        ratio = torch.tensor([1.5, 0.5])
        factor = torch.tensor([2.0, 3.0])
        advantage = torch.tensor([1.0, -1.0])
        loss = happo_surrogate_loss(ratio, factor, advantage, clip=.2)
        expected = -torch.tensor([1.2 * 2.0, .8 * -3.0]).mean()
        torch.testing.assert_close(loss, expected)

    def test_factor_accumulates_updated_agent_ratios_without_gradient(self):
        factor = torch.ones(4)
        old = torch.zeros(4)
        first = torch.log(torch.tensor([1.1, .9, 1.2, .8], requires_grad=True))
        second = torch.log(torch.tensor([.7, 1.3, 1.0, 1.1], requires_grad=True))
        updated = update_advantage_factor(factor, first, old)
        updated = update_advantage_factor(updated, second, old)
        torch.testing.assert_close(updated, first.exp() * second.exp())
        self.assertFalse(updated.requires_grad)

    def test_chunked_agent_logprob_replays_behavior_policy(self):
        torch.manual_seed(31)
        model = FormalHAPPO(3, 9, 21, hidden_dim=16)
        length = 8
        observations = torch.randn(length, 3, 9)
        observations[:, :, -3:] = torch.eye(3)[None]
        raw = torch.randn(length, 3, 2)
        masks = torch.ones(length, 3, 1)
        masks[0].zero_()
        hidden_before = torch.zeros(length, 3, 16)
        old = []
        hidden = model.actors.initial_state()
        for step in range(length):
            mean, logstd, next_hidden = model.actors.step(
                observations[step], hidden, masks[step])
            hidden_before[step] = hidden
            old.append(latent_log_prob(mean, logstd, raw[step]))
            hidden = next_hidden
        old = torch.stack(old)
        for agent in range(3):
            replayed = agent_logprob(
                model, observations, raw, hidden_before, masks,
                agent, chunk_length=4)
            torch.testing.assert_close(replayed, old[:, agent])

    def test_evaluator_restores_heterogeneous_checkpoint(self):
        model = FormalHAPPO(3, 9, 21, hidden_dim=16)
        saved = {
            "version": "formal_recurrent_happo_v1",
            "metadata": {"version": "formal_recurrent_happo_v1"},
            "model": model.state_dict(),
        }
        restored, _ = load_model(
            saved, SimpleNamespace(n_uavs=3), obs_dim=9, state_dim=21)
        for left, right in zip(
                model.state_dict().values(), restored.state_dict().values(),
                strict=True):
            torch.testing.assert_close(left, right)


if __name__ == "__main__":
    unittest.main()
