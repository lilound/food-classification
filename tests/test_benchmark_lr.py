import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

import benchmark_lr
import export_benchmark


class BenchmarkTest(unittest.TestCase):
    def test_fixed_and_adaptive_training_and_resume(self):
        protocol = benchmark_lr.read(benchmark_lr.CONFIG)
        protocol.update(models={"toy": ["Toy", 143, "none"]}, batch_size=4,
                        max_epochs=2, min_epochs=1)
        dataset = TensorDataset(torch.rand(8, 3, 2, 2), torch.arange(8) % 11)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, split = root / "protocol.json", root / "split.json"
            config.write_text(json.dumps(protocol))
            split.write_text('{}')
            with patch.object(benchmark_lr, "CONFIG", config), patch.object(benchmark_lr, "SPLIT", split), \
                 patch.object(benchmark_lr, "check_dependencies", return_value={}), \
                 patch.object(benchmark_lr, "verified_split", return_value=(dataset, dataset)), \
                 patch.object(benchmark_lr, "load_model", side_effect=lambda *a, **kw: nn.Sequential(nn.Flatten(), nn.Linear(12, 11))):
                for mode in ("fixed", "adaptive"):
                    args = argparse.Namespace(mode=mode, models=None, data_dir=root, output=root / mode,
                                              resume=False, device="cpu", workers=0, threads=1)
                    original = benchmark_lr.run_epoch
                    calls = 0

                    def interrupt(*args, **kwargs):
                        nonlocal calls
                        calls += 1
                        if calls == 3:
                            raise RuntimeError("interruption")
                        return original(*args, **kwargs)

                    with patch.object(benchmark_lr, "run_epoch", side_effect=interrupt):
                        with self.assertRaisesRegex(RuntimeError, "interruption"):
                            benchmark_lr.run(args)
                    self.assertEqual(len(benchmark_lr.read(args.output / "history.json")), 1)
                    args.resume = True
                    benchmark_lr.run(args)
                    history = benchmark_lr.read(args.output / "history.json")
                    self.assertEqual(len(history), 2)
                    if mode == "fixed":
                        self.assertTrue(all(row["lr"] == 0.0001 and row["next_lr"] == 0.0001 for row in history))
                    else:
                        self.assertNotEqual(history[-1]["next_lr"], protocol["initial_lr"])
                    benchmark_lr.run(args)
                    self.assertEqual(benchmark_lr.read(args.output / "history.json"), history)
                snapshots = [export_benchmark.read_run(root / mode) for mode in ("fixed", "adaptive")]
                export_benchmark.render(root / "export", snapshots, protocol, {}, [])
                self.assertTrue((root / "export/comparaison/scores.csv").is_file())
                self.assertTrue((root / "export/courbes_par_modele/frugalite_toy.svg").is_file())
                snapshots[0]["run"]["signature"]["fixed_lr"] = 0.001
                with self.assertRaisesRegex(ValueError, "taux fixe"):
                    export_benchmark.render(root / "export", snapshots, protocol, {}, [])


if __name__ == "__main__":
    unittest.main()
