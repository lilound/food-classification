import unittest
import json
import tempfile
import argparse
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from benchmark.modeles import benchmark_lr
from benchmark.gel_couches.benchmark import compare
from models.compact import load_model


class FreezeTest(unittest.TestCase):
    def test_maximal_shuffle_freeze_preserves_trainable_head(self):
        model = load_model("shufflenet05", 11, pretrained=False)
        modules = benchmark_lr.freeze_early_layers(model, "shufflenet05", level="more")
        self.assertIn(model.stage4, modules)
        self.assertTrue(all(not p.requires_grad for module in modules for p in module.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.conv5.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.fc.parameters()))

    def test_stronger_freeze_keeps_more_parameters_and_batchnorm_fixed(self):
        torch.set_num_threads(1)
        for name in ("shufflenet05", "shufflenet10", "mobilenet_v2"):
            with self.subTest(model=name):
                model = load_model(name, 11, pretrained=False)
                normal = benchmark_lr.freeze_early_layers(model, name)
                normal_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
                model = load_model(name, 11, pretrained=False)
                frozen = benchmark_lr.freeze_early_layers(model, name, level="strong")
                self.assertGreater(sum(p.numel() for p in model.parameters() if not p.requires_grad), normal_count)
                self.assertTrue(any(p.requires_grad for p in model.parameters()))
                before = [{n: p.clone() for n, p in module.state_dict().items()} for module in frozen]
                data = TensorDataset(torch.randn(2, 3, 128, 128), torch.tensor([0, 1]))
                loader = DataLoader(data, batch_size=2, sampler=torch.utils.data.RandomSampler(data, generator=torch.Generator()))
                optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-4)
                benchmark_lr.run_epoch(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"), optimizer, 42, frozen_modules=frozen)
                for module, saved in zip(frozen, before):
                    for key, value in module.state_dict().items():
                        self.assertTrue(torch.equal(value, saved[key]), f"{name}: {key}")

    def test_frozen_benchmark_saves_and_resumes(self):
        protocol = benchmark_lr.read(benchmark_lr.CONFIG)
        protocol.update(models={"shufflenet05": protocol["models"]["shufflenet05"]},
                        batch_size=2, max_epochs=1, min_epochs=1)
        dataset = TensorDataset(torch.rand(2, 3, 128, 128), torch.tensor([0, 1]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, split = root / "protocol.json", root / "split.json"
            config.write_text(json.dumps(protocol))
            split.write_text('{}')
            args = argparse.Namespace(mode="fixed", models=["shufflenet05"], data_dir=root, output=root / "run",
                                      resume=False, device="cpu", workers=0, threads=1, freeze_early=True)
            with patch.object(benchmark_lr, "CONFIG", config), patch.object(benchmark_lr, "SPLIT", split), \
                 patch.object(benchmark_lr, "check_dependencies", return_value={}), \
                 patch.object(benchmark_lr, "verified_split", return_value=(dataset, dataset)), \
                 patch.object(benchmark_lr, "load_model", side_effect=lambda name, classes, **kwargs: load_model(name, classes, pretrained=False)):
                benchmark_lr.run(args)
                history = benchmark_lr.read(args.output / "history.json")
                self.assertEqual(len(history), 1)
                self.assertIn("freeze_early", benchmark_lr.read(args.output / "run.json")["signature"])
                args.resume = True
                benchmark_lr.run(args)
                self.assertEqual(history, benchmark_lr.read(args.output / "history.json"))

    def test_comparison_checks_protocol_and_plots_both_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, signature in (("old", {"mode": "fixed", "protocol_sha256": "abc"}),
                                    ("new", {"mode": "fixed", "protocol_sha256": "abc", "freeze_early": "v1"})):
                path = root / name
                path.mkdir()
                (path / "run.json").write_text(json.dumps({"signature": signature, "models": ["shufflenet05"]}))
                (path / "history.json").write_text(json.dumps([{"model": "shufflenet05", "epoch": 1, "frugality": 0.5}]))
            compare("shufflenet05", root / "old", root / "new", root / "charts")
            self.assertTrue((root / "charts/frugalite_shufflenet05.png").exists())
            run = json.loads((root / "new/run.json").read_text())
            run["signature"]["protocol_sha256"] = "different"
            (root / "new/run.json").write_text(json.dumps(run))
            with self.assertRaisesRegex(ValueError, "incompatibles"):
                compare("shufflenet05", root / "old", root / "new", root / "charts")

    def test_frozen_layers_and_batchnorm_remain_unchanged(self):
        torch.set_num_threads(1)
        for name in ("shufflenet05", "shufflenet10", "mobilenet_v2"):
            with self.subTest(model=name):
                model = load_model(name, 11, pretrained=False)
                frozen = benchmark_lr.freeze_early_layers(model, name)
                before = {id(module): {n: p.clone() for n, p in module.state_dict().items()} for module in frozen}
                self.assertTrue(any(not p.requires_grad for p in model.parameters()))
                self.assertTrue(any(p.requires_grad for p in model.parameters()))
                data = TensorDataset(torch.randn(2, 3, 128, 128), torch.tensor([0, 1]))
                loader = DataLoader(data, batch_size=2, sampler=torch.utils.data.RandomSampler(data, generator=torch.Generator()))
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
                benchmark_lr.run_epoch(model, loader, nn.CrossEntropyLoss(), torch.device('cpu'), optimizer, 42, frozen_modules=frozen)
                for module in frozen:
                    for key, value in module.state_dict().items():
                        self.assertTrue(torch.equal(value, before[id(module)][key]), f"{name}: {key}")


if __name__ == '__main__':
    unittest.main()
