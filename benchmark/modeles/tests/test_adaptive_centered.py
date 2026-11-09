import copy
import unittest

import torch

from benchmark.modeles.benchmark_lr import CONFIG, AdaptiveRate, CenteredAdaptiveRate, read


class CenteredAdaptiveRateTest(unittest.TestCase):
    def make_controller(self, kind):
        head = torch.nn.Parameter(torch.ones(2))
        optimizer = torch.optim.Adam([head], lr=1e-4)
        return optimizer, kind(optimizer, head, read(CONFIG)["scheduler"])

    def test_stable_positive_signal_does_not_pin_centered_rate(self):
        _, original = self.make_controller(AdaptiveRate)
        _, centered = self.make_controller(CenteredAdaptiveRate)
        for _ in range(250):
            original.adjust(0.6)
            centered.adjust(0.6)
        self.assertEqual(original.group["lr"], original.ceiling)
        self.assertAlmostEqual(centered.group["lr"], 1e-4)
        centered.adjust(0.2)
        self.assertLess(centered.group["lr"], 1e-4)
        self.assertGreaterEqual(centered.group["lr"], 1e-4 / 1.05)
        for _ in range(10):
            centered.adjust(0.9)
        self.assertGreater(centered.group["lr"], 1e-4)

    def test_plateau_and_resume_preserve_the_controller(self):
        optimizer, control = self.make_controller(CenteredAdaptiveRate)
        for signal in (0.4, 0.7, 0.1):
            control.adjust(signal)
        for _ in range(5):
            control.validate(0.6)
        self.assertAlmostEqual(control.base_lr, 5e-5)
        next_optimizer, restored = self.make_controller(CenteredAdaptiveRate)
        next_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        restored.load_state_dict(copy.deepcopy(control.state_dict()))
        for signal in (0.2, 0.9, 0.6):
            control.adjust(signal)
            restored.adjust(signal)
            self.assertEqual(control.group["lr"], restored.group["lr"])
