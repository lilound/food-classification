"""Entraîne ResNet18 puis distille ses sorties vers ShuffleNetV2 ×0,5."""

import argparse
import hashlib
import json
import math
import sys
import uuid
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[2]
script_dir = str(Path(__file__).resolve().parent)
if script_dir in sys.path:
    sys.path.remove(script_dir)
sys.path.insert(0, str(ROOT))

from benchmark.modeles import benchmark_lr as base
from src.dataset import find_data_folder, get_transforms
from src.evaluate import predict_test

TEMPERATURE = 2.0
DISTILLATION_WEIGHT = 0.25


def distillation_loss(student_logits, teacher_logits, labels):
    soft = F.kl_div(F.log_softmax(student_logits / TEMPERATURE, dim=1),
                    F.softmax(teacher_logits / TEMPERATURE, dim=1),
                    reduction="batchmean") * TEMPERATURE ** 2
    return (1 - DISTILLATION_WEIGHT) * F.cross_entropy(student_logits, labels) + DISTILLATION_WEIGHT * soft


def train_distillation_epoch(student, teacher, loader, optimizer, controller, device, *, seed, frozen_modules):
    student.train()
    teacher.eval()
    for module in frozen_modules:
        module.eval()
    loader.sampler.generator.manual_seed(seed)
    flips = torch.Generator().manual_seed(seed + 1)
    total = 0.0
    for images, labels in loader:
        mask = torch.rand(len(images), generator=flips) < 0.5
        images[mask] = images[mask].flip(-1)
        images = images.to(device, memory_format=torch.channels_last)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_logits = teacher(images)
        loss = distillation_loss(student(images), teacher_logits, labels)
        loss.backward()
        before = controller.before_step()
        optimizer.step()
        controller.after_step(before)
        total += loss.detach().item() * len(labels)
    result = total / len(loader.dataset)
    if not math.isfinite(result):
        raise RuntimeError("Perte non finie : la dernière époque valide est conservée.")
    return result


def prepare_run(output, protocol, split, resume):
    sources = (Path(__file__), Path(base.__file__), ROOT / "models/compact.py",
               ROOT / "src/dataset.py", ROOT / "src/evaluate.py")
    signature = {
        "protocol_sha256": hashlib.sha256(base.fingerprint(protocol) + base.fingerprint(split)).hexdigest(),
        "code_sha256": hashlib.sha256(b"".join(path.read_bytes() for path in sources)).hexdigest(),
        "dataset_sha256": protocol["dataset_sha256"],
        "seed": protocol["seed"], "image_size": protocol["image_size"],
        "teacher": "resnet18_imagenet1k_v1", "teacher_lr": protocol["initial_lr"],
        "student": "shufflenet05_imagenet1k_v1", "freeze_level": "strong",
        "policy": base.POLICY, "temperature": TEMPERATURE, "distillation_weight": DISTILLATION_WEIGHT,
    }
    run_path = output / "run.json"
    if resume:
        if not run_path.is_file():
            raise ValueError(f"Aucune exécution à reprendre dans {output}.")
        run = base.read(run_path)
        if run["signature"] != signature:
            raise ValueError("Protocole différent : reprise impossible.")
        student_path = output / "student/checkpoint.pth"
        teacher_path = output / "teacher/checkpoint.pth"
        if student_path.exists():
            if not teacher_path.is_file():
                raise ValueError("Checkpoint de l'enseignant manquant : reprise impossible.")
            saved = torch.load(student_path, map_location="cpu", weights_only=True)
            if saved.get("teacher_sha256") != file_hash(teacher_path):
                raise ValueError("Checkpoint de l'enseignant modifié : reprise impossible.")
    else:
        output.mkdir(parents=True, exist_ok=False)
        run = {"id": uuid.uuid4().hex, "signature": signature}
        base.write(run_path, run)
    return run


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def make_teacher(checkpoint_exists):
    weights = None if checkpoint_exists else ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 11)
    return model


def train_stage(stage, output, protocol, train_loader, val_loader, device, teacher=None):
    folder = output / stage
    folder.mkdir(exist_ok=True)
    path = folder / "checkpoint.pth"
    teacher_hash = file_hash(output / "teacher/checkpoint.pth") if stage == "student" else None
    torch.manual_seed(protocol["seed"])
    if stage == "teacher":
        model = make_teacher(path.exists())
        frozen = ()
    else:
        model = base.load_model("shufflenet05", 11, pretrained=not path.exists(), weight_id="IMAGENET1K_V1")
        frozen = base.freeze_early_layers(model, "shufflenet05", level="strong")
    model.to(device, dtype=torch.float32, memory_format=torch.channels_last)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=protocol["initial_lr"])
    controller = base.AdaptiveRate(optimizer, model.fc.weight, protocol["scheduler"]) if stage == "student" else None
    history, best_weights, reference, stale, stop = [], None, -math.inf, 0, None
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if stage == "student" and saved.get("teacher_sha256") != teacher_hash:
            raise ValueError("Checkpoint de l'enseignant modifié : reprise impossible.")
        model.load_state_dict(saved["weights"])
        optimizer.load_state_dict(saved["optimizer"])
        if controller:
            controller.load_state_dict(saved["scheduler"])
        history, best_weights, reference, stale, stop = (saved[k] for k in ("history", "best_weights", "reference", "stale", "stop"))
    best_score = max((row["f1"] for row in history), default=-math.inf)
    if stop:
        print(f"{stage}: étape déjà terminée ({stop}).", flush=True)
    else:
        for epoch in range(len(history) + 1, protocol["max_epochs"] + 1):
            seed = protocol["seed"] + 3 * epoch
            torch.manual_seed(seed)
            started = perf_counter()
            print(f"{stage} | début époque {epoch}/{protocol['max_epochs']}", flush=True)
            if controller:
                controller.rates.clear()
                train_loss = train_distillation_epoch(model, teacher, train_loader, optimizer, controller,
                                                      device, seed=seed, frozen_modules=frozen)
            else:
                train_loss = base.run_epoch(model, train_loader, nn.CrossEntropyLoss(), device, optimizer, seed)
            val_loss, accuracy, f1 = base.run_epoch(model, val_loader, nn.CrossEntropyLoss(), device)
            frugality = f1 - 0.3 * params / (1_000_000 + params)
            if f1 > best_score:
                best_score = f1
                best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            reference, stale, stop = base.progress(protocol, f1, reference, stale, epoch)
            if controller:
                controller.validate(frugality)
            rates = controller.rates if controller else [optimizer.param_groups[0]["lr"]]
            row = dict(epoch=epoch, params=params, train_loss=train_loss, val_loss=val_loss, accuracy=accuracy,
                       f1=f1, frugality=frugality, min_lr=min(rates), max_lr=max(rates), seconds=perf_counter()-started)
            history.append(row)
            base.save_checkpoint(path, dict(weights=model.state_dict(), best_weights=best_weights,
                                            optimizer=optimizer.state_dict(), scheduler=controller.state_dict() if controller else None,
                                            history=history, reference=reference, stale=stale, stop=stop,
                                            teacher_sha256=teacher_hash))
            base.write(folder / "history.json", history)
            base.write(folder / "summary.json", {**max(history, key=lambda r: r["f1"]), "last_epoch": epoch, "stop": stop})
            print(f"{stage} | époque {epoch:2} | F1 {f1:.4f} | frugalité {frugality:.4f} | "
                  f"LR {row['min_lr']:.6g} à {row['max_lr']:.6g} | {row['seconds']:.0f} s", flush=True)
            if stop:
                break
    base.write(folder / "history.json", history)
    base.write(folder / "summary.json", {**max(history, key=lambda r: r["f1"]), "last_epoch": history[-1]["epoch"], "stop": stop})
    model.load_state_dict(best_weights)
    model.eval()
    return model, max(history, key=lambda row: row["f1"])


def record_results(run, output, teacher, student, train_count, val_count):
    path = ROOT / "RESULTS.md"
    content = path.read_text(encoding="utf-8") if path.exists() else "# Résultats\n"
    marker = f"<!-- distillation-run: {run['id']} -->"
    if marker in content:
        return
    relative = output.relative_to(ROOT) if output.is_relative_to(ROOT) else output
    text = (f"\n\n## Distillation ResNet18 vers ShuffleNetV2 ×0,5\n\n{marker}\n\n"
            f"Food-11 : {train_count} images d'entraînement et {val_count} de validation ; "
            "découpage figé, graine 42, 128 × 128, poids ImageNet et Adam. "
            "Enseignant ResNet18 entraîné sur ce découpage, étudiant avec gel renforcé et taux adaptatif ; "
            f"température {TEMPERATURE:g}, poids de distillation {DISTILLATION_WEIGHT:g}.\n\n"
            f"| Modèle | Meilleure époque | F1 macro validation | Frugalité validation | Paramètres |\n"
            f"| --- | ---: | ---: | ---: | ---: |\n"
            f"| ResNet18 enseignant | {teacher['epoch']} | {teacher['f1']:.4f} | {teacher['frugality']:.4f} | {teacher['params']:,} |\n"
            f"| ShuffleNetV2 ×0,5 distillé | {student['epoch']} | {student['f1']:.4f} | {student['frugality']:.4f} | {student['params']:,} |\n\n"
            f"Résultats d'une seule exécution sur la validation locale, utilisée pour sélectionner les deux checkpoints ; "
            "le score peut être optimiste et aucun score du test officiel n'est connu. "
            f"Historiques et checkpoints : `{relative}`.\n")
    path.write_text(content.rstrip() + text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/food-11")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark/distillation/execution")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 0 or args.threads < 1:
        parser.error("Workers ou threads invalides.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA demandé mais indisponible.")
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    base.check_dependencies()
    protocol, split = base.read(base.CONFIG), base.read(base.SPLIT)
    if protocol["image_size"] != 128:
        raise ValueError("Ce benchmark exige la résolution 128 × 128.")
    train, val = base.verified_split(args.data_dir, protocol)
    output = args.output.resolve()
    run = prepare_run(output, protocol, split, args.resume)
    options = dict(batch_size=protocol["batch_size"], num_workers=args.workers, persistent_workers=args.workers > 0)
    train_loader = DataLoader(train, sampler=RandomSampler(train, generator=torch.Generator()),
                              generator=torch.Generator().manual_seed(protocol["seed"]), **options)
    val_loader = DataLoader(val, generator=torch.Generator().manual_seed(protocol["seed"]), **options)
    print(f"Données : {len(train)} entraînement, {len(val)} validation ; appareil : {device} ; sortie : {output}", flush=True)
    teacher, teacher_best = train_stage("teacher", output, protocol, train_loader, val_loader, device)
    teacher.requires_grad_(False)
    student, student_best = train_stage("student", output, protocol, train_loader, val_loader, device, teacher)
    record_results(run, output, teacher_best, student_best, len(train), len(val))
    test_dir = find_data_folder(args.data_dir, ("test", "testing"))
    predict_test(student, test_dir, get_transforms(size=128), split["classes"], device,
                 output / "predictions_shufflenet05_distillation.json")
    print(f"Terminé : meilleur F1 {student_best['f1']:.4f}, frugalité {student_best['frugality']:.4f}.", flush=True)


if __name__ == "__main__":
    main()
