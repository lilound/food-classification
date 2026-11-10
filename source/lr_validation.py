"""Montée progressive du taux, puis adaptation sur la perte de validation."""

import torch


LOSS_PLATEAU_POLICY = {
    "name": "loss-plateau-v1",
    "warmup_epochs": 1,
    "peak_lr": 3e-4,
    "scheduler": {
        "mode": "min", "factor": 0.5, "patience": 1,
        "threshold": 0.002, "threshold_mode": "rel", "min_lr": 1e-5,
    },
}


class ValidationLossRate:
    def __init__(self, optimizer, batches_per_epoch):
        self.optimizer = optimizer
        self.start = optimizer.param_groups[0]["lr"]
        self.ceiling = LOSS_PLATEAU_POLICY["peak_lr"]
        self.warmup_steps = batches_per_epoch * LOSS_PLATEAU_POLICY["warmup_epochs"]
        self.steps = 0
        self.rates = []
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **LOSS_PLATEAU_POLICY["scheduler"])

    def before_step(self):
        group = self.optimizer.param_groups[0]
        if self.steps < self.warmup_steps:
            fraction = self.steps / (self.warmup_steps - 1) if self.warmup_steps > 1 else 1.0
            group["lr"] = (self.ceiling if fraction == 1.0
                           else self.start + (self.ceiling - self.start) * fraction)
        self.rates.append(group["lr"])

    def after_step(self, _):
        self.steps += 1

    def validate(self, loss):
        self.scheduler.step(loss)

    def state_dict(self):
        return {"steps": self.steps, "scheduler": self.scheduler.state_dict()}

    def load_state_dict(self, state):
        self.steps = state["steps"]
        self.scheduler.load_state_dict(state["scheduler"])
