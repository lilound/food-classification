import tempfile
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from creer_dataset_equilibre import benchmark_protocol, build, file_digest, files_digest, read_json, verify


class BalancedDatasetTest(unittest.TestCase):
    def test_determinism_provenance_and_split_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            train = ["bread/0.jpg", "bread/1.jpg", "rice/0.jpg"]
            validation = ["bread/2.jpg", "rice/1.jpg"]
            images = [f"train/{p}" for p in train + validation] + ["test/0.jpg"]
            for i, name in enumerate(images):
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                image = Image.new("RGB", (45, 40), (50 + i * 20, 110, 50))
                ImageDraw.Draw(image).ellipse((4, 6, 24 + i, 28), fill=(222, 180, 60))
                image.save(path)
            source_hash = files_digest((name, file_digest(source / "train" / name))
                                       for name in train + validation)
            split = dict(train=train, validation=validation, classes=["bread", "rice"])
            a, b = root / "a", root / "b"
            out_a = build(source, a, split, source_hash, 3, 42, test_count=1)
            out_b = build(source, b, split, source_hash, 3, 42, test_count=1)
            self.assertEqual(out_a["dataset_sha256"], out_b["dataset_sha256"])
            self.assertEqual(verify(a)["train_augmented"], 3)
            rows = read_json(a / "manifest.json")
            variants = [r for r in rows if r["transform"] is not None]
            self.assertEqual(Counter(r["path"].split("/")[1] for r in variants), {"bread": 1, "rice": 2})
            self.assertTrue(all(r["source"].removeprefix("train/") in train for r in variants))
            for row in rows:
                if row["transform"] is None:
                    self.assertEqual((a / row["path"]).read_bytes(), (source / row["source"]).read_bytes())
            protocol, _ = benchmark_protocol(a, {"dataset_sha256": source_hash}, split)
            self.assertEqual(protocol["train_augmentation"], "none")
            with self.assertRaises(FileExistsError):
                build(source, a, split, source_hash, 3, 42, test_count=1)
            (a / variants[0]["path"]).write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "Image différente"):
                verify(a)
