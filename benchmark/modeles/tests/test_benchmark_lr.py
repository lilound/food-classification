import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

from benchmark.modeles import benchmark_lr, export_benchmark, lancer_benchmarks


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
                for mode in ("fixed", "adaptive", "cosine"):
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
                    elif mode == "adaptive":
                        self.assertNotEqual(history[-1]["next_lr"], protocol["initial_lr"])
                    else:
                        self.assertGreater(history[0]["lr"], history[0]["next_lr"])
                        self.assertEqual(history[1]["lr"], history[0]["next_lr"])
                        self.assertAlmostEqual(history[1]["next_lr"], protocol["scheduler"]["min_lr"])
                    benchmark_lr.run(args)
                    self.assertEqual(benchmark_lr.read(args.output / "history.json"), history)
                snapshots = [export_benchmark.read_run(root / mode) for mode in ("fixed", "adaptive", "cosine")]
                export_benchmark.render(root / "export", snapshots, protocol, {}, [])
                self.assertTrue((root / "export/comparaison/scores.csv").is_file())
                self.assertTrue((root / "export/courbes_par_modele/frugalite_toy.svg").is_file())
                snapshots[0]["run"]["signature"]["fixed_lr"] = 0.001
                with self.assertRaisesRegex(ValueError, "taux fixe"):
                    export_benchmark.render(root / "export", snapshots, protocol, {}, [])

    def test_launcher_commands_target_real_script_for_all_modes(self):
        self.assertEqual(set(export_benchmark.MODES), {"fixed", "adaptive", "cosine"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = argparse.Namespace(data_dir=root, threads=1, workers=0)
            for mode in export_benchmark.MODES:
                cmd = lancer_benchmarks.command(mode, root / mode, args)
                self.assertEqual(Path(cmd[2]), benchmark_lr.KIT / "benchmark_lr.py")
                self.assertEqual(cmd[3:5], ["--mode", mode])

    def test_launcher_runs_all_modes_and_exports(self):
        protocol = benchmark_lr.read(benchmark_lr.CONFIG)
        protocol.update(models={"toy": ["Toy", 143, "none"]}, batch_size=4,
                        max_epochs=2, min_epochs=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, split = root / "protocol.json", root / "split.json"
            config.write_text(json.dumps(protocol))
            split.write_text('{}')
            (root / "benchmark_lr.py").write_text(f'''
import importlib.util, sys
from pathlib import Path
sys.path.insert(0, {str(benchmark_lr.ROOT)!r})
import torch
from torch import nn
from torch.utils.data import TensorDataset
spec = importlib.util.spec_from_file_location("real_benchmark", {str(benchmark_lr.KIT / "benchmark_lr.py")!r})
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
bench.CONFIG, bench.SPLIT = Path({str(config)!r}), Path({str(split)!r})
bench.check_dependencies = lambda: {{}}
dataset = TensorDataset(torch.rand(8, 3, 2, 2), torch.arange(8) % 11)
bench.verified_split = lambda *args: (dataset, dataset)
bench.load_model = lambda *args, **kwargs: nn.Sequential(nn.Flatten(), nn.Linear(12, 11))
bench.main()
''')
            args = argparse.Namespace(jobs=1, modes=list(export_benchmark.MODES), device="cpu",
                                      runs=root / "runs", output=root / "report", resume=False,
                                      data_dir=root, threads=1, workers=0)
            with patch.object(lancer_benchmarks, "ROOT", root), patch.object(lancer_benchmarks, "KIT", root), patch.object(lancer_benchmarks, "CONFIG", config), \
                 patch.object(lancer_benchmarks, "SPLIT", split):
                self.assertEqual(lancer_benchmarks.run(args), 0)
                results = json.loads((args.output / "execution.json").read_text())
                self.assertEqual({mode: result["etat"] for mode, result in results.items()},
                                 {mode: "terminé" for mode in args.modes})
                self.assertTrue((root / "report.zip").is_file())
                args.resume = True
                self.assertEqual(lancer_benchmarks.run(args), 0)


if __name__ == "__main__":
    unittest.main()
