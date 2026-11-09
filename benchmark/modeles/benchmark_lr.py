"""Compare les taux d'apprentissage fixe et variable sur le protocole Food-11 figé."""

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader, RandomSampler, Subset
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).resolve().parents[2]
KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.compact import load_model
from creer_dataset_equilibre import benchmark_protocol, verify as verify_balanced
from src.dataset import find_data_folder, get_transforms

CONFIG = KIT / "protocole/protocol.json"
SPLIT = KIT / "protocole/split.json"
MODES = ("fixed", "adaptive", "cosine")

POLICY = {"gain": 0.02, "smoothing": 0.95, "max_lr_factor": 3.0}


class AdaptiveRate:
    """Approximation normalisée de l'hypergradient sur la tête de classification."""

    def __init__(self, optimizer, head, plateau_options):
        self.head = head
        self.direction = None
        self.alignment = 0.0
        self.rates = []
        self.group = optimizer.param_groups[0]
        self.floor = plateau_options["min_lr"]
        self.ceiling = POLICY["max_lr_factor"] * self.group["lr"]
        self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **plateau_options)

    @torch.no_grad()
    def before_step(self):
        if self.direction is not None and self.head.grad is not None:
            cosine = cosine_similarity(self.head.grad.reshape(-1), self.direction.reshape(-1),
                                       dim=0, eps=1e-12).item()
            if math.isfinite(cosine):
                smoothing = POLICY["smoothing"]
                self.alignment = smoothing * self.alignment + (1 - smoothing) * cosine
                lr = self.group["lr"] * math.exp(POLICY["gain"] * self.alignment)
                self.group["lr"] = min(self.ceiling, max(self.floor, lr))
        self.rates.append(self.group["lr"])
        return self.head.detach().clone()

    @torch.no_grad()
    def after_step(self, before):
        # Le déplacement effectif d'Adam est -LR × direction.
        self.direction = before - self.head.detach()

    def validate(self, score):
        # ReduceLROnPlateau pilote le plafond, l'hypergradient pilote le taux courant.
        lr, old_ceiling = self.group["lr"], self.ceiling
        self.group["lr"] = self.ceiling
        self.plateau.step(score)
        self.ceiling = self.group["lr"]
        self.group["lr"] = min(self.ceiling, max(self.floor, lr * (self.ceiling / old_ceiling)))

    def state_dict(self):
        return dict(direction=self.direction, alignment=self.alignment,
                    ceiling=self.ceiling, plateau=self.plateau.state_dict())

    def load_state_dict(self, state):
        # Adam remplace ses groupes lors du chargement de son état.
        self.group = self.plateau.optimizer.param_groups[0]
        direction = state["direction"]
        self.direction = direction.to(self.head.device) if direction is not None else None
        self.alignment = state["alignment"]
        self.ceiling = state["ceiling"]
        self.plateau.load_state_dict(state["plateau"])


def fingerprint(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, payload):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def save_checkpoint(path, payload):
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def verified_split(data_dir, protocol):
    root = find_data_folder(data_dir, ("train", "training"))
    dataset = ImageFolder(root, transform=get_transforms(size=protocol["image_size"]))
    split = read(SPLIT)
    if dataset.classes != split["classes"] or len(dataset.classes) != 11:
        raise ValueError("Les classes du dataset diffèrent du découpage figé.")
    paths = {Path(p).relative_to(root).as_posix(): idx for idx, (p, _) in enumerate(dataset.samples)}
    train, val = split["train"], split["validation"]
    if set(train) & set(val) or len(train) + len(val) != len(paths) or set(train + val) != set(paths):
        raise ValueError("Les fichiers du dataset ne correspondent pas au découpage figé.")
    digest = hashlib.sha256()
    hashes = {}
    for key in sorted(paths):
        hashes[key] = hashlib.sha256((root / key).read_bytes()).hexdigest()
        digest.update(f"{key}\0{hashes[key]}\n".encode())
    if {hashes[p] for p in train} & {hashes[p] for p in val}:
        raise ValueError("Des fichiers identiques traversent le découpage.")
    if digest.hexdigest() != protocol["dataset_sha256"]:
        raise ValueError("Le contenu du dataset ne correspond pas à l'empreinte du kit.")
    return Subset(dataset, [paths[p] for p in train]), Subset(dataset, [paths[p] for p in val])


def freeze_early_layers(model, name, level="early"):
    if level not in ("early", "strong", "more") or (level == "more" and name != "shufflenet05"):
        raise ValueError(f"Niveau de gel inconnu : {level}")
    if name in ("shufflenet05", "shufflenet10"):
        modules = [model.conv1, model.stage2]
        if level in ("strong", "more"):
            modules.append(model.stage3)
        if level == "more":
            modules.append(model.stage4)
    elif name == "mobilenet_v2":
        modules = list(model.features[:14 if level == "strong" else 7])
    else:
        raise ValueError(f"Gel non défini pour {name}.")
    for module in modules:
        module.requires_grad_(False)
        module.eval()
    return modules


def run_epoch(model, loader, criterion, device, optimizer=None, seed=None, controller=None,
              frozen_modules=(), *, augment=True):
    training = optimizer is not None
    model.train(training)
    for module in frozen_modules:
        module.eval()
    loss_sum = torch.zeros((), device=device)
    predictions, targets = [], []
    if training:
        loader.sampler.generator.manual_seed(seed)
        if augment:
            flips = torch.Generator().manual_seed(seed + 1)
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            if training and augment:
                mask = torch.rand(len(images), generator=flips) < 0.5
                images[mask] = images[mask].flip(-1)
            images = images.to(device, memory_format=torch.channels_last)
            device_labels = labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, device_labels)
            if training:
                loss.backward()
                before = controller.before_step() if controller else None
                optimizer.step()
                if controller:
                    controller.after_step(before)
            else:
                predictions.extend(logits.argmax(dim=1).tolist())
                targets.extend(labels.tolist())
            loss_sum += loss.detach() * len(labels)
    mean_loss = loss_sum.item() / len(loader.dataset)
    if not math.isfinite(mean_loss):
        raise RuntimeError("Perte non finie : la dernière époque valide reste disponible.")
    if training:
        return mean_loss
    return mean_loss, float(accuracy_score(targets, predictions)), float(
        f1_score(targets, predictions, labels=list(range(11)), average="macro", zero_division=0))


def check_dependencies():
    installed = {}
    for pin in (KIT / "protocole/requirements.txt").read_text().splitlines():
        name, required = pin.split("==")
        installed[name] = version(name)
        if installed[name].split("+")[0] != required:
            raise RuntimeError(f"{name} : version {installed[name]}, le protocole exige {required}.")
    return installed


def update_reports(out, states):
    write(out / "history.json", [row for state in states.values() for row in state["history"]])
    summary = []
    for name, state in states.items():
        best = max(state["history"], key=lambda row: row["frugality"])
        summary.append({**best, "last_epoch": state["history"][-1]["epoch"], "stop": state["stop"]})
    write(out / "summary.json", sorted(summary, key=lambda row: -row["frugality"]))


def progress(protocol, score, reference, stale, epoch):
    if score >= reference + protocol["min_delta"]:
        reference, stale = score, 0
    else:
        stale += 1
    stop = None
    if epoch >= protocol["min_epochs"] and stale >= protocol["early_stopping_patience"]:
        stop = "plateau"
    elif epoch == protocol["max_epochs"]:
        stop = "plafond"
    return reference, stale, stop


def run(args):
    protocol, split = benchmark_protocol(args.data_dir, read(CONFIG), read(SPLIT))
    selected = args.models or list(protocol["models"])
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    use_cuda = args.device != "cpu" and torch.cuda.is_available()
    if args.device == "cuda" and not use_cuda:
        raise RuntimeError("CUDA demandé mais indisponible.")
    device = torch.device("cuda" if use_cuda else "cpu")
    dependencies = check_dependencies()
    balanced = protocol.get("train_augmentation") == "none"
    if balanced:
        metadata = verify_balanced(args.data_dir)
        if metadata["dataset_sha256"] != protocol["dataset_sha256"]:
            raise ValueError("Le jeu équilibré a changé pendant sa vérification.")
        transform = get_transforms(size=protocol["image_size"])
        train = ImageFolder(args.data_dir / "train", transform=transform)
        val = ImageFolder(args.data_dir / "validation", transform=transform)
        if train.classes != val.classes or train.classes != split["classes"] or len(train.classes) != 11:
            raise ValueError("Classes incompatibles entre l'entraînement et la validation.")
    else:
        train, val = verified_split(args.data_dir, protocol)
    split_bytes = fingerprint(split)
    signature = {
        "mode": args.mode,
        "protocol_sha256": hashlib.sha256(fingerprint(protocol) + split_bytes).hexdigest(),
        "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
    }
    freeze_level = getattr(args, "freeze_level", "early")
    freeze = getattr(args, "freeze_early", False)
    if freeze:
        signature["freeze_early"] = {
            "early": "shufflenet-conv1-stage2_mobilenet-features-0-6-v1",
            "strong": "shufflenet-conv1-stage2-stage3_mobilenet-features-0-13-v1",
            "more": "shufflenet05-conv1-stage2-stage3-stage4-v1",
        }[freeze_level]
    if args.mode == "fixed":
        signature["fixed_lr"] = protocol["initial_lr"]
    elif args.mode == "adaptive":
        signature["policy_sha256"] = hashlib.sha256(fingerprint(POLICY)).hexdigest()
    machine = dict(device=str(device), gpu=torch.cuda.get_device_name() if use_cuda else None,
                   platform=platform.platform(), python=platform.python_version(), dependencies=dependencies,
                   threads=args.threads, workers=args.workers)
    run_name = {
        "fixed": "benchmark_lr_fixed_0.0001",
        "adaptive": "benchmark_lr_adaptive",
        "cosine": "benchmark_lr_cosine",
    }[args.mode]
    out = args.output or KIT / "executions" / (run_name + ("_balanced" if balanced else ""))
    states = {}
    if args.resume:
        metadata = read(out / "run.json")
        if metadata["signature"] != signature or metadata["models"] != selected:
            raise ValueError("Protocole ou sélection de modèles incompatible avec la reprise.")
        metadata["machines"].append(machine)
        for name in selected:
            path = out / f"{name}.pth"
            if path.exists():
                saved = torch.load(path, map_location="cpu", weights_only=True)
                states[name] = {key: saved[key] for key in ("history", "stop")}
                del saved
        update_reports(out, states)
    else:
        out.mkdir(parents=True, exist_ok=False)
        metadata = dict(signature=signature, models=selected, policy=POLICY, machines=[machine])
    write(out / "run.json", metadata)
    print(f"{args.mode}: {len(train)} train, {len(val)} validation, {device}, {out}", flush=True)

    options = dict(batch_size=protocol["batch_size"], num_workers=args.workers,
                   persistent_workers=args.workers > 0)
    train_loader = DataLoader(train, sampler=RandomSampler(train, generator=torch.Generator()),
                              generator=torch.Generator().manual_seed(protocol["seed"]), **options)
    val_loader = DataLoader(val, generator=torch.Generator().manual_seed(protocol["seed"]), **options)
    criterion = nn.CrossEntropyLoss()
    for index, (name, (_, expected, weights)) in enumerate(protocol["models"].items()):
        if name not in selected or (name in states and states[name]["stop"]):
            continue
        path = out / f"{name}.pth"
        torch.manual_seed(protocol["seed"])
        model = load_model(name, 11, pretrained=not path.exists(), weight_id=weights)
        frozen_modules = freeze_early_layers(model, name, freeze_level) if freeze else ()
        params = sum(p.numel() for p in model.parameters())
        if params != expected:
            raise ValueError(f"{name} : attendu {expected} paramètres, obtenu {params}.")
        model.to(device, dtype=torch.float32, memory_format=torch.channels_last)
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=protocol["initial_lr"])
        if frozen_modules:
            print(f"{name}: {sum(p.numel() for p in model.parameters() if p.requires_grad)} "
                  f"paramètres entraînables / {params} au total", flush=True)
        controller = None
        scheduler = None
        if args.mode == "adaptive":
            head = [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))][-1]
            if head.weight.shape[0] != 11:
                raise ValueError(f"{name} : tête de classification non reconnue.")
            controller = AdaptiveRate(optimizer, head.weight, protocol["scheduler"])
        elif args.mode == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=protocol["max_epochs"], eta_min=protocol["scheduler"]["min_lr"])
        history, best_weights, reference, stale = [], None, -math.inf, 0
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(saved["weights"])
            optimizer.load_state_dict(saved["optimizer"])
            if controller is not None:
                controller.load_state_dict(saved["scheduler"])
            elif scheduler is not None:
                scheduler.load_state_dict(saved["scheduler"])
            history, best_weights, reference, stale = (
                saved["history"], saved["best_weights"], saved["reference"], saved["stale"])
            del saved
        best = max((row["frugality"] for row in history), default=-math.inf)
        for epoch in range(len(history) + 1, protocol["max_epochs"] + 1):
            seed = protocol["seed"] + 10_000 * index + 3 * epoch
            torch.manual_seed(seed)
            started = perf_counter()
            if controller is not None:
                controller.rates.clear()
            epoch_lr = optimizer.param_groups[0]["lr"]
            train_loss = run_epoch(model, train_loader, criterion, device, optimizer, seed, controller,
                                   frozen_modules=frozen_modules, augment=not balanced)
            val_loss, accuracy, f1 = run_epoch(model, val_loader, criterion, device)
            frugality = f1 - 0.3 * params / (1_000_000 + params)
            if frugality > best:
                best = frugality
                best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            reference, stale, stop = progress(protocol, frugality, reference, stale, epoch)
            if controller is not None:
                controller.validate(frugality)
            elif scheduler is not None:
                scheduler.step()
            rates = controller.rates if controller is not None else [epoch_lr]
            row = dict(model=name, epoch=epoch, params=params, lr=rates[0],
                       min_lr=min(rates), max_lr=max(rates), mean_lr=sum(rates) / len(rates),
                       next_lr=optimizer.param_groups[0]["lr"],
                       train_loss=train_loss, val_loss=val_loss, accuracy=accuracy, f1=f1,
                       frugality=frugality, seconds=perf_counter() - started)
            if controller is not None:
                row["lr_ceiling"] = controller.ceiling
            history.append(row)
            save_checkpoint(path, dict(weights=model.state_dict(), best_weights=best_weights,
                                       optimizer=optimizer.state_dict(), scheduler=(controller or scheduler).state_dict() if controller or scheduler else None,
                                       history=history, reference=reference, stale=stale, stop=stop))
            states[name] = dict(history=history, stop=stop)
            update_reports(out, states)
            print(f"{name:19} époque {epoch:2} | F1 {f1:.4f} | frugalité {frugality:.4f} | "
                  f"LR {row['min_lr']:.6g} à {row['max_lr']:.6g} | {row['seconds']:.0f}s"
                  + (f" | arrêt : {stop}" if stop else ""), flush=True)
            if stop:
                break
        del controller, scheduler, optimizer, model, best_weights


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--models", nargs="+", choices=tuple(read(CONFIG)["models"]))
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/food-11")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--freeze-early", action="store_true", help="Geler les premiers blocs des trois modèles retenus.")
    parser.add_argument("--freeze-level", choices=("early", "strong", "more"), default="early")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 0 or args.threads < 1 or (args.models and len(args.models) != len(set(args.models))):
        parser.error("Nombre de threads, de workers ou sélection de modèles invalide.")
    if args.freeze_early and (not args.output or not args.models or any(
            name not in ("shufflenet05", "shufflenet10", "mobilenet_v2") for name in args.models)):
        parser.error("Le gel exige --output et une sélection des trois modèles retenus.")
    if args.freeze_level != "early" and not args.freeze_early:
        parser.error("--freeze-level exige --freeze-early.")
    if args.freeze_level == "more" and args.models != ["shufflenet05"]:
        parser.error("Le gel more est défini uniquement pour shufflenet05.")
    run(args)


if __name__ == "__main__":
    main()
