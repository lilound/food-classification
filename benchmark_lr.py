"""Compare les taux d'apprentissage fixe et variable sur le protocole Food-11 figé."""

import argparse
import hashlib
import json
import math
import os
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, Subset
from torchvision.datasets import ImageFolder

from models.compact import load_model
from src.dataset import find_data_folder, get_transforms

ROOT = Path(__file__).resolve().parent
KIT = ROOT / "benchmarks/lr"
CONFIG = KIT / "protocol.json"
SPLIT = KIT / "split.json"
MODES = ("fixed", "plateau")
PATIENCE10_FINGERPRINT = "20859ef94e17dd6a7b804db956ba49cba788d50306a08474a8b9f9dcf715adb9"


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


def run_epoch(model, loader, criterion, device, optimizer=None, seed=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = torch.zeros((), device=device)
    predictions, targets = [], []
    if training:
        loader.sampler.generator.manual_seed(seed)
        flips = torch.Generator().manual_seed(seed + 1)
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            if training:
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
                optimizer.step()
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
    for pin in (KIT / "requirements.txt").read_text().splitlines():
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


def import_patience10(source, out, protocol, mode):
    """Rejoue l'arrêt sans modifier le run d'origine ni inventer des checkpoints."""
    old_config = CONFIG.read_bytes().replace(b"\r\n", b"\n").replace(
        b'"early_stopping_patience": 6', b'"early_stopping_patience": 10')
    split = SPLIT.read_bytes().replace(b"\r\n", b"\n")
    old = read(source / "run.json")
    valid = any(
        old["signature"] == {
            "mode": mode,
            "protocol_sha256": hashlib.sha256(c + s).hexdigest(),
            "split_sha256": hashlib.sha256(s).hexdigest(),
        }
        for c in (old_config, old_config.replace(b"\n", b"\r\n"))
        for s in (split, split.replace(b"\n", b"\r\n"))
    )
    if hashlib.sha256(old_config + split).hexdigest() != PATIENCE10_FINGERPRINT or not valid:
        raise ValueError("Le run source doit utiliser exactement l'ancien protocole à patience 10 et le même mode.")

    states = {}
    for name in protocol["models"]:
        path = source / f"{name}.pth"
        if not path.is_file():
            continue
        saved = torch.load(path, map_location="cpu", weights_only=True)
        original = saved["history"]
        kept = []
        reference, stale, stop = -math.inf, 0, None
        for row in original:
            kept.append(row)
            reference, stale, stop = progress(
                protocol, row["frugality"], reference, stale, row["epoch"])
            if stop:
                break
        if len(kept) != len(original):
            best_kept = max(kept, key=lambda row: row["frugality"])["epoch"]
            best_original = max(original, key=lambda row: row["frugality"])["epoch"]
            if best_kept != best_original:
                print(f"{name}: les poids du meilleur état admissible manquent ; réentraînement.", flush=True)
                continue
            # Les poids de dernière époque et Adam sont postérieurs à l'arrêt rejoué.
            saved.update(weights=None, optimizer=None, scheduler=None)
        saved.update(history=kept, reference=reference, stale=stale, stop=stop)
        save_checkpoint(out / path.name, saved)
        states[name] = {"history": kept, "stop": stop}
        print(f"{name}: importé jusqu'à l'époque {kept[-1]['epoch']}"
              + (f" (arrêt : {stop})" if stop else " (à reprendre)"), flush=True)
    return states, old["machines"]


def run(args):
    protocol = read(CONFIG)
    dependencies = check_dependencies()
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    use_cuda = args.device != "cpu" and torch.cuda.is_available()
    if args.device == "cuda" and not use_cuda:
        raise RuntimeError("CUDA demandé mais indisponible.")
    device = torch.device("cuda" if use_cuda else "cpu")
    train, val = verified_split(args.data_dir, protocol)
    out = args.output or ROOT / "results" / (
        "benchmark_lr_fixed" if args.mode == "fixed" else "benchmark_lr_variable"
    )
    config_bytes = json.dumps(protocol, ensure_ascii=False, separators=(",", ":")).encode()
    split_bytes = json.dumps(read(SPLIT), ensure_ascii=False, separators=(",", ":")).encode()
    signature = {"mode": args.mode, "protocol_sha256": hashlib.sha256(config_bytes + split_bytes).hexdigest(),
                 "split_sha256": hashlib.sha256(split_bytes).hexdigest()}
    machine = {"platform": platform.platform(), "python": platform.python_version(),
               "device": str(device), "gpu": torch.cuda.get_device_name() if use_cuda else None,
               "dependencies": dependencies, "threads": args.threads, "workers": args.workers}
    states = {}
    if args.resume:
        metadata = read(out / "run.json")
        if metadata["signature"] != signature:
            raise ValueError("Le mode ou le protocole ne correspond pas aux sauvegardes.")
        metadata["machines"].append(machine)
    else:
        out.mkdir(parents=True, exist_ok=False)
        metadata = {"signature": signature, "machines": [machine]}
        if args.import_patience10:
            states, previous_machines = import_patience10(args.import_patience10, out, protocol, args.mode)
            metadata["machines"] = [*previous_machines, machine]
            metadata["imported_patience10"] = True
            update_reports(out, states)
    write(out / "run.json", metadata)
    print(f"{args.mode}: {len(train)} train, {len(val)} validation, {device}, {out}", flush=True)
    options = dict(batch_size=protocol["batch_size"], num_workers=args.workers,
                   persistent_workers=args.workers > 0)
    train_loader = DataLoader(train, sampler=RandomSampler(train, generator=torch.Generator()),
                              generator=torch.Generator().manual_seed(protocol["seed"]), **options)
    val_loader = DataLoader(val, generator=torch.Generator().manual_seed(protocol["seed"]), **options)
    criterion = nn.CrossEntropyLoss()
    if args.resume:
        for name in protocol["models"]:
            path = out / f"{name}.pth"
            if path.exists():
                saved = torch.load(path, map_location="cpu", weights_only=True)
                states[name] = {k: saved[k] for k in ("history", "stop")}
        update_reports(out, states)

    for model_index, (name, (_, expected, weights)) in enumerate(protocol["models"].items()):
        path = out / f"{name}.pth"
        if name in states and states[name]["stop"]:
            continue
        torch.manual_seed(protocol["seed"])
        model = load_model(name, 11, pretrained=not path.exists(), weight_id=weights)
        params = sum(p.numel() for p in model.parameters())
        if params != expected or not 100_000 <= params <= 8_000_000:
            raise ValueError(f"{name} : attendu {expected} paramètres, obtenu {params}.")
        model.to(device, dtype=torch.float32, memory_format=torch.channels_last)
        optimizer = torch.optim.Adam(model.parameters(), lr=protocol["initial_lr"])
        scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **protocol["scheduler"])
                     if args.mode == "plateau" else None)
        history, best_weights, reference, stale = [], None, -math.inf, 0
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(saved["weights"])
            optimizer.load_state_dict(saved["optimizer"])
            if scheduler is not None:
                scheduler.load_state_dict(saved["scheduler"])
            history, best_weights, reference, stale = (
                saved["history"], saved["best_weights"], saved["reference"], saved["stale"])
            del saved
        best = max((r["frugality"] for r in history), default=-math.inf)
        for epoch in range(len(history) + 1, protocol["max_epochs"] + 1):
            # Les tirages sont reconstruits à partir du modèle et de l'époque à la reprise.
            seed = protocol["seed"] + 10_000 * model_index + 3 * epoch
            torch.manual_seed(seed)
            started = perf_counter()
            lr = optimizer.param_groups[0]["lr"]
            train_loss = run_epoch(model, train_loader, criterion, device, optimizer, seed)
            val_loss, accuracy, f1 = run_epoch(model, val_loader, criterion, device)
            frugality = f1 - 0.3 * params / (1_000_000 + params)
            if frugality > best:
                best = frugality
                best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            reference, stale, stop = progress(protocol, frugality, reference, stale, epoch)
            if scheduler is not None:
                scheduler.step(frugality)
            row = dict(model=name, epoch=epoch, params=params, lr=lr, next_lr=optimizer.param_groups[0]["lr"],
                       train_loss=train_loss, val_loss=val_loss, accuracy=accuracy, f1=f1,
                       frugality=frugality, seconds=perf_counter() - started)
            history.append(row)
            save_checkpoint(path, dict(weights=model.state_dict(), best_weights=best_weights,
                                       optimizer=optimizer.state_dict(),
                                       scheduler=scheduler.state_dict() if scheduler is not None else None,
                                       history=history, reference=reference, stale=stale, stop=stop))
            states[name] = {"history": history, "stop": stop}
            update_reports(out, states)
            print(f"{name:19} époque {epoch:2} | F1 {f1:.4f} | frugalité {frugality:.4f} | "
                  f"LR {lr:.2g} → {row['next_lr']:.2g} | {row['seconds']:.0f}s"
                  + (f" | arrêt : {stop}" if stop else ""), flush=True)
            if stop:
                break
        # Libère la mémoire du réseau précédent avant de construire le suivant.
        del model, optimizer, scheduler, best_weights
    from plot_benchmark_lr import plot_runs
    plot_runs([out], out)


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/food-11")
    parser.add_argument("--output", type=Path)
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument("--resume", action="store_true")
    continuation.add_argument("--import-patience10", type=Path, metavar="ANCIEN_DOSSIER")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 0 or args.threads < 1:
        parser.error("Nombre de workers ou threads invalide.")
    run(args)


if __name__ == "__main__":
    main()
