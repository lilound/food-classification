import copy
import unittest

import torch

from src.lr_validation import ValidationLossRate


class ValidationLossRateTest(unittest.TestCase):
    def make(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.Adam([parameter], lr=1e-4)
        return parameter, optimizer, ValidationLossRate(optimizer, batches_per_epoch=3)

    def train_batch(self, parameter, optimizer, policy):
        optimizer.zero_grad()
        parameter.square().sum().backward()
        policy.before_step()
        optimizer.step()
        policy.after_step(None)

    def test_warmup_keeps_all_adam_steps(self):
        parameter, optimizer, policy = self.make()
        for _ in range(3):
            self.train_batch(parameter, optimizer, policy)
        for actual, expected in zip(policy.rates, (1e-4, 2e-4, 3e-4), strict=True):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(optimizer.state[parameter]["step"].item(), 3)

    def test_plateau_reduces_only_after_two_bad_epochs_and_survives_resume(self):
        parameter, optimizer, policy = self.make()
        for _ in range(3):
            self.train_batch(parameter, optimizer, policy)
        policy.validate(0.8)
        policy.validate(0.79)
        policy.validate(0.8)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 3e-4)
        new_parameter, new_optimizer, restored = self.make()
        new_parameter.data.copy_(parameter.detach())
        new_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        restored.load_state_dict(copy.deepcopy(policy.state_dict()))
        restored.validate(0.8)
        self.train_batch(new_parameter, new_optimizer, restored)
        self.assertAlmostEqual(restored.rates[-1], 1.5e-4)
        self.assertEqual(new_optimizer.state[new_parameter]["step"].item(), 4)
