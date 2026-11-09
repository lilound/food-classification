"""Matérialise un jeu Food-11 équilibré sans modifier les données officielles."""

import argparse
import hashlib
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

import PIL
from PIL import Image, ImageEnhance, ImageOps, features

ROOT = Path(__file__).resolve().parent
KIT = ROOT / "benchmark/modeles/protocole"
DEFAULT_SOURCE = ROOT / "data/food-11"
DEFAULT_OUTPUT = ROOT / "data/food-11-balanced"
KIND = "food11-balanced-v1"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".pgm", ".tif", ".tiff", ".webp"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def image_paths(root):
    return {p.relative_to(root).as_posix() for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS}


def relative_path(name):
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"Chemin invalide dans le manifeste : {name}")
    if path.as_posix() != name:
        raise ValueError(f"Chemin non canonique : {name}")
    return path


def files_digest(paths_and_hashes):
    digest = hashlib.sha256()
    for name, sha in sorted(paths_and_hashes):
        digest.update(f"{name}\0{sha}\n".encode())
    return digest.hexdigest()


def recipe_for(seed, label, number):
    rng = random.Random(f"{KIND}:{seed}:{label}:{number}")
    return {
        "angle": rng.choice((-1, 1)) * rng.randint(2, 10),
        "brightness_percent": rng.randint(90, 110),
        "contrast_percent": rng.randint(90, 110),
        "crop_percent": rng.randint(94, 100),
        "mirror": rng.choice((False, True)),
    }


def augment(source, output, recipe):
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    w, h = image.size
    angle = math.radians(recipe["angle"])
    cos, sin = math.cos(angle), math.sin(angle)
    # Le zoom intérieur laisse les coins de la sortie dans l'image source.
    scale = min(w / (w * cos + h * abs(sin)), h / (h * cos + w * abs(sin)))
    scale *= recipe["crop_percent"] / 100
    a, b = scale * cos, scale * sin
    cx, cy = w / 2, h / 2
    image = image.transform((w, h), Image.Transform.AFFINE,
                            (a, b, cx - a * cx - b * cy, -b, a, cy + b * cx - a * cy),
                            resample=Image.Resampling.BICUBIC)
    if recipe["mirror"]:
        image = ImageOps.mirror(image)
    image = ImageEnhance.Brightness(image).enhance(recipe["brightness_percent"] / 100)
    image = ImageEnhance.Contrast(image).enhance(recipe["contrast_percent"] / 100)
    image.save(output, "JPEG", quality=92, subsampling=0, optimize=False, progressive=False)


def build(source, output, split, expected_sha, per_class, seed, test_count=3097):
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Le dossier existe déjà : {output}")
    if source in output.parents or output in source.parents:
        raise ValueError("Le jeu généré doit être à côté du jeu source, dans un dossier séparé.")
    train, validation = split["train"], split["validation"]
    classes = split["classes"]
    all_paths = train + validation
    if (len(set(all_paths)) != len(all_paths) or set(all_paths) != image_paths(source / "train")
            or set(classes) != {relative_path(name).parts[0] for name in all_paths}):
        raise ValueError("Les fichiers étiquetés ne correspondent pas au découpage figé.")
    grouped = {label: sorted(name for name in train if relative_path(name).parts[0] == label)
               for label in classes}
    if per_class < 1 or any(not group or len(group) > per_class for group in grouped.values()):
        raise ValueError("Le quota doit permettre de conserver toutes les images d'entraînement.")
    source_hashes = {name: file_digest(source / "train" / relative_path(name)) for name in all_paths}
    if files_digest(source_hashes.items()) != expected_sha:
        raise ValueError("Le contenu de train ne correspond pas au jeu Food-11 attendu.")
    val_hashes = {source_hashes[name] for name in validation}
    if val_hashes & {source_hashes[name] for name in train}:
        raise ValueError("Une copie exacte traverse le découpage.")
    test = sorted(image_paths(source / "test"))
    if len(test) != test_count or len({PurePosixPath(name).stem for name in test}) != len(test):
        raise ValueError(f"Le dossier test doit contenir {test_count} images aux identifiants uniques.")

    output.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as tmp:
        stage = Path(tmp) / "dataset"
        stage.mkdir()

        def add(name, source_name, recipe=None, sha=None):
            dst = stage / relative_path(name)
            original = source / relative_path(source_name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if recipe is None:
                shutil.copyfile(original, dst)
            else:
                augment(original, dst, recipe)
            entries.append({"path": name, "source": source_name, "transform": recipe,
                            "sha256": sha or file_digest(dst)})

        for label, group in grouped.items():
            for name in group:
                add(f"train/{name}", f"train/{name}", sha=source_hashes[name])
            order = list(group)
            random.Random(f"{KIND}:{seed}:{label}").shuffle(order)
            missing = per_class - len(group)
            for i in range(missing):
                name = order[i % len(order)]
                recipe = recipe_for(seed, label, i)
                add(f"train/{label}/__augmented_v1__/{i:04d}.jpg", f"train/{name}", recipe)
                if entries[-1]["sha256"] in val_hashes:
                    raise ValueError("Une variante est identique à une image de validation.")
            print(f"{label}: {len(group)} originaux + {missing} variantes", flush=True)
        for name in sorted(validation):
            add(f"validation/{name}", f"train/{name}", sha=source_hashes[name])
        for name in test:
            add(f"test/{name}", f"test/{name}")

        entries.sort(key=lambda row: row["path"])
        write_json(stage / "manifest.json", entries)
        metadata = {
            "kind": KIND, "seed": seed, "classes": classes, "train_per_class": per_class,
            "train_original": len(train), "train_augmented": per_class * len(classes) - len(train),
            "validation": len(validation), "test": len(test),
            "source_train_sha256": expected_sha, "source_split_sha256": json_digest(split),
            "dataset_sha256": files_digest((row["path"], row["sha256"]) for row in entries),
            "manifest_sha256": json_digest(entries),
            "encoder": {"pillow": PIL.__version__, "jpeg": features.version_codec("jpg")},
        }
        write_json(stage / "dataset.json", metadata)
        verify(stage)
        if output.exists():
            raise FileExistsError(f"Le dossier existe déjà : {output}")
        stage.rename(output)
    return metadata


def verify(root):
    metadata = read_json(root / "dataset.json")
    entries = read_json(root / "manifest.json")
    if metadata["kind"] != KIND or json_digest(entries) != metadata["manifest_sha256"]:
        raise ValueError("Manifeste équilibré invalide.")
    paths = {row["path"] for row in entries}
    if len(paths) != len(entries) or paths != image_paths(root):
        raise ValueError("Les fichiers présents ne correspondent pas au manifeste.")
    counts, train_hashes, val_hashes = Counter(), set(), set()
    for row in entries:
        path = relative_path(row["path"])
        if file_digest(root / path) != row["sha256"]:
            raise ValueError(f"Image différente du manifeste : {path}")
        if path.parts[0] == "train":
            counts[path.parts[1]] += 1
            train_hashes.add(row["sha256"])
        elif path.parts[0] == "validation":
            val_hashes.add(row["sha256"])
    expected = {label: metadata["train_per_class"] for label in metadata["classes"]}
    if counts != expected or train_hashes & val_hashes:
        raise ValueError("Quotas de train ou séparation de la validation invalides.")
    splits = Counter(relative_path(name).parts[0] for name in paths)
    if splits != {"train": sum(expected.values()), "validation": metadata["validation"], "test": metadata["test"]}:
        raise ValueError("Nombre de fichiers inattendu par sous-ensemble.")
    if files_digest((row["path"], row["sha256"]) for row in entries) != metadata["dataset_sha256"]:
        raise ValueError("Empreinte du jeu équilibré invalide.")
    return metadata


def benchmark_protocol(root, protocol, split):
    """Identifie une expérience équilibrée sans changer la signature du jeu original."""
    marker = root / "dataset.json"
    if not marker.is_file():
        return protocol, split
    metadata = read_json(marker)
    if (metadata["kind"] != KIND or metadata["source_split_sha256"] != json_digest(split)
            or metadata["source_train_sha256"] != protocol["dataset_sha256"]):
        raise ValueError("Le jeu équilibré ne dérive pas du découpage officiel de ce benchmark.")
    configured = {**protocol, "dataset_sha256": metadata["dataset_sha256"], "train_augmentation": "none"}
    description = {key: metadata[key] for key in (
        "kind", "classes", "train_per_class", "train_original", "train_augmented",
        "validation", "test", "source_split_sha256", "manifest_sha256")}
    return configured, description


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-class", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify", type=Path, metavar="DATASET", help="Vérifie un jeu déjà généré")
    args = parser.parse_args()
    try:
        if args.verify:
            result = verify(args.verify)
        else:
            result = build(args.source, args.output, read_json(KIT / "split.json"),
                           read_json(KIT / "protocol.json")["dataset_sha256"], args.per_class, args.seed)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    target = args.verify or args.output
    print(f"{target}: {result['train_per_class']} images par classe, "
          f"{result['train_augmented']} variantes, {result['validation']} validation, {result['test']} test")
    print(f"SHA-256 : {result['dataset_sha256']}")


if __name__ == "__main__":
    main()
