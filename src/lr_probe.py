"""Recherche périodique du taux sur un lot d'apprentissage distinct."""

import copy
import math

import torch

PROBE_POLICY = {
    "name": "probe-v1", "interval": 20, "factors": [0.5, 1.0, 1.5],
    "relative_tolerance": 0.001, "min_lr": 1e-6, "max_lr": 3e-4,
}


class ProbeRate:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.floor = PROBE_POLICY["min_lr"]
        self.ceiling = PROBE_POLICY["max_lr"]
        self.lots = 0
        self.rates = []
        self.events = []

    @property
    def group(self):
        return self.optimizer.param_groups[0]

    def state_dict(self):
        return {"lots": self.lots}

    def load_state_dict(self, state):
        self.lots = state["lots"]

    def validate(self, score):
        # La validation choisit le checkpoint, pas le taux de ce contrôleur.
        pass

    @torch.no_grad()
    def step(self, model, criterion, probe_batch):
        self.lots += 1
        current = self.group["lr"]
        if self.lots % PROBE_POLICY["interval"] or probe_batch is None:
            self.optimizer.step()
            self.rates.append(current)
            return

        images, labels = probe_batch()
        params = [p for p in self.group["params"] if p.grad is not None]
        originals = [p.detach().clone() for p in params]
        saved_optimizer = copy.deepcopy(self.optimizer.state_dict())
        training_modes = [(module, module.training) for module in model.modules()]
        model.eval()
        try:
            before = criterion(model(images), labels).item()
            self.optimizer.step()
            direction = [p.detach() - w for p, w in zip(params, originals, strict=True)]

            def apply(rate):
                for p, w, delta in zip(params, originals, direction, strict=True):
                    p.copy_(w + (rate / current) * delta)

            candidates = sorted({min(self.ceiling, max(self.floor, current * f))
                                 for f in PROBE_POLICY["factors"]})
            losses = {}
            for rate in candidates:
                apply(rate)
                loss = criterion(model(images), labels).item()
                losses[rate] = loss if math.isfinite(loss) else None
            tolerance = PROBE_POLICY["relative_tolerance"] * max(abs(before), 1e-12)
            good = {rate: loss for rate, loss in losses.items()
                    if loss is not None and math.isfinite(before) and before - loss >= tolerance}
            if good:
                best = min(good, key=good.__getitem__)
                chosen = current if current in good and good[current] - good[best] <= tolerance else best
                apply(chosen)
                self.rates.append(chosen)
            else:
                for p, w in zip(params, originals, strict=True):
                    p.copy_(w)
                self.optimizer.load_state_dict(saved_optimizer)
                chosen = max(self.floor, current * 0.5)
            self.group["lr"] = chosen
        except Exception:
            for p, w in zip(params, originals, strict=True):
                p.copy_(w)
            self.optimizer.load_state_dict(saved_optimizer)
            raise
        finally:
            for module, training in training_modes:
                module.training = training
        self.events.append({
            "lot": self.lots, "perte_initiale": before if math.isfinite(before) else None,
            "candidats": [{"lr": rate, "perte": loss} for rate, loss in losses.items()],
            "lr_suivant": chosen, "accepte": bool(good),
        })
        tested = ", ".join(f"{lr:.2g}:{loss:.4f}" if loss is not None else f"{lr:.2g}:invalide"
                           for lr, loss in losses.items())
        print(f"Contrôle lot {self.lots} | perte {before:.4f} | {tested} | "
              f"{'accepté' if good else 'annulé'} | LR {chosen:.3g}", flush=True)
