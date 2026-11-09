import contextlib
import io
import unittest

import torch

from src.lr_probe import ProbeRate


class ProbeRateTest(unittest.TestCase):
    def trial(self, probe_label):
        model = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.ones_(model.weight)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        controller = ProbeRate(opt)
        controller.load_state_dict({"lots": 19})
        x = torch.ones(2, 1)
        model(x).square().mean().backward()
        with contextlib.redirect_stdout(io.StringIO()):
            controller.step(model, torch.nn.MSELoss(), lambda: (x, torch.full_like(x, probe_label)))
        self.assertTrue(model.training)
        self.assertEqual(controller.state_dict(), {"lots": 20})
        self.assertEqual(len(controller.events[0]["candidats"]), 3)
        return model, opt, controller

    def test_accepts_one_adam_step(self):
        model, opt, control = self.trial(0.95)
        self.assertTrue(control.events[0]["accepte"])
        self.assertAlmostEqual(model.weight.item(), 1 - 1.5e-4, places=6)
        self.assertEqual(opt.state[model.weight]["step"].item(), 1)
        self.assertAlmostEqual(control.group["lr"], 1.5e-4)

    def test_rejection_restores_weights_and_optimizer(self):
        model, opt, control = self.trial(2)
        self.assertFalse(control.events[0]["accepte"])
        self.assertEqual(model.weight.item(), 1.0)
        self.assertFalse(opt.state)
        self.assertAlmostEqual(control.group["lr"], 5e-5)
