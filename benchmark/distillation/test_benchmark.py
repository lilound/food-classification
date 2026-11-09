import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from benchmark.distillation import benchmark


class DistillationTest(unittest.TestCase):
    def test_script_starts_directly(self):
        result = subprocess.run([sys.executable, str(Path(benchmark.__file__)), "--help"],
                                cwd=benchmark.ROOT, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--resume", result.stdout)

    def test_loss_combines_labels_and_teacher_distribution(self):
        student = torch.tensor([[0.0, 0.0]], requires_grad=True)
        target = torch.tensor([0])
        same = benchmark.distillation_loss(student, torch.zeros_like(student), target)
        self.assertAlmostEqual(same.item(), 0.75 * math.log(2), places=6)
        disagree = benchmark.distillation_loss(student, torch.tensor([[4.0, -4.0]]), target)
        self.assertGreater(disagree.item(), same.item())
        disagree.backward()
        self.assertIsNotNone(student.grad)

    def test_epoch_updates_student_without_changing_teacher(self):
        torch.manual_seed(3)
        student = nn.Sequential(nn.Flatten(), nn.Linear(48, 11))
        teacher = nn.Sequential(nn.Flatten(), nn.Linear(48, 11))
        teacher_before = {k: v.clone() for k, v in teacher.state_dict().items()}
        student_before = student[1].weight.detach().clone()
        dataset = TensorDataset(torch.rand(8, 3, 4, 4), torch.arange(8) % 11)
        loader = DataLoader(dataset, batch_size=4, sampler=RandomSampler(dataset, generator=torch.Generator()))
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
        policy = {"mode": "max", "factor": 0.5, "patience": 3, "threshold": 0.001,
                  "threshold_mode": "abs", "cooldown": 0, "min_lr": 1e-6}
        controller = benchmark.base.AdaptiveRate(optimizer, student[1].weight, policy)
        loss = benchmark.train_distillation_epoch(student, teacher, loader, optimizer, controller,
                                                  torch.device("cpu"), seed=42, frozen_modules=())
        self.assertTrue(math.isfinite(loss))
        self.assertFalse(torch.equal(student_before, student[1].weight))
        self.assertTrue(all(torch.equal(value, teacher.state_dict()[key]) for key, value in teacher_before.items()))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertFalse(teacher.training)
        self.assertTrue(controller.rates)

    def test_real_student_keeps_frozen_stage_unchanged(self):
        torch.set_num_threads(1)
        student = benchmark.base.load_model("shufflenet05", 11, pretrained=False)
        frozen = benchmark.base.freeze_early_layers(student, "shufflenet05", level="strong")
        before = {k: v.clone() for k, v in student.stage3.state_dict().items()}
        teacher = benchmark.make_teacher(True)
        dataset = TensorDataset(torch.rand(2, 3, 128, 128), torch.tensor([0, 1]))
        loader = DataLoader(dataset, batch_size=2, sampler=RandomSampler(dataset, generator=torch.Generator()))
        optimizer = torch.optim.Adam((p for p in student.parameters() if p.requires_grad), lr=1e-4)
        protocol = benchmark.base.read(benchmark.base.CONFIG)
        controller = benchmark.base.AdaptiveRate(optimizer, student.fc.weight, protocol["scheduler"])
        benchmark.train_distillation_epoch(student, teacher, loader, optimizer, controller,
                                          torch.device("cpu"), seed=42, frozen_modules=frozen)
        self.assertTrue(all(torch.equal(value, student.stage3.state_dict()[key]) for key, value in before.items()))
        self.assertIsNotNone(student.fc.weight.grad)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_checkpoint_stages_and_resume(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Identity()
                self.stage2 = nn.Identity()
                self.stage3 = nn.Identity()
                self.fc = nn.Linear(48, 11)

            def forward(self, x):
                return self.fc(x.flatten(1))

        torch.set_num_threads(1)
        protocol = benchmark.base.read(benchmark.base.CONFIG)
        protocol.update(batch_size=4, max_epochs=2, min_epochs=1)
        dataset = TensorDataset(torch.rand(8, 3, 4, 4), torch.arange(8) % 11)
        train = DataLoader(dataset, batch_size=4, sampler=RandomSampler(dataset, generator=torch.Generator()))
        val = DataLoader(dataset, batch_size=4)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(benchmark, "make_teacher", side_effect=lambda exists: Toy()), \
             patch.object(benchmark.base, "load_model", side_effect=lambda *a, **kw: Toy()):
            output = Path(directory)
            teacher, teacher_best = benchmark.train_stage("teacher", output, protocol, train, val, torch.device("cpu"))
            student, student_best = benchmark.train_stage("student", output, protocol, train, val, torch.device("cpu"), teacher)
            self.assertTrue((output / "teacher/checkpoint.pth").is_file())
            self.assertTrue((output / "student/checkpoint.pth").is_file())
            _, resumed = benchmark.train_stage("student", output, protocol, train, val, torch.device("cpu"), teacher)
            self.assertEqual(student_best, resumed)
            self.assertIn(teacher_best["epoch"], (1, 2))
            self.assertEqual(student.fc.weight.shape, (11, 48))
            with (output / "teacher/checkpoint.pth").open("ab") as stream:
                stream.write(b"modified")
            with self.assertRaisesRegex(ValueError, "enseignant"):
                benchmark.train_stage("student", output, protocol, train, val, torch.device("cpu"), teacher)

    def test_resume_rejects_changed_protocol(self):
        protocol = benchmark.base.read(benchmark.base.CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            first = benchmark.prepare_run(output, protocol, {}, False)
            self.assertEqual(first, benchmark.prepare_run(output, protocol, {}, True))
            changed = dict(protocol, image_size=256)
            with self.assertRaisesRegex(ValueError, "Protocole différent"):
                benchmark.prepare_run(output, changed, {}, True)
            (output / "student").mkdir()
            (output / "student/checkpoint.pth").touch()
            with self.assertRaisesRegex(ValueError, "enseignant"):
                benchmark.prepare_run(output, protocol, {}, True)

    def test_record_results_does_not_duplicate_a_completed_run(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(benchmark, "ROOT", Path(directory)):
            root = Path(directory)
            (root / "RESULTS.md").write_text("# Résultats\n")
            teacher = {"epoch": 2, "f1": 0.8, "frugality": 0.6, "params": 11_000_000}
            student = {"epoch": 3, "f1": 0.7, "frugality": 0.62, "params": 353_067}
            run = {"id": "example"}
            benchmark.record_results(run, root / "run", teacher, student, 10, 4)
            benchmark.record_results(run, root / "run", teacher, student, 10, 4)
            content = (root / "RESULTS.md").read_text()
            self.assertEqual(content.count("distillation-run: example"), 1)
            self.assertIn("0.6200", content)


if __name__ == "__main__":
    unittest.main()
