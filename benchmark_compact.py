"""Compare plusieurs réseaux Food-11 sur une même validation sans doublons exacts."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from shutil import copyfile
from time import perf_counter

import sklearn
import torch
import torchvision
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from models.compact import BUILDERS, load_model
from src.dataset import find_data_folder, get_transforms
from src.evaluate import evaluate, predict_test


def split_by_content(dataset, seed):
    """Place les fichiers identiques dans le même sous-ensemble, quel que soit leur nom."""
    hashes = [hashlib.sha256(Path(path).read_bytes()).digest() for path, _ in dataset.samples]
    groups = [digest.hex() for digest in hashes]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_ids, val_ids = next(splitter.split(dataset.targets, dataset.targets, groups))
    return train_ids.tolist(), val_ids.tolist(), sum(n > 1 for n in Counter(groups).values())


def train_epoch(model, loader, optimizer, criterion, device, seed):
    model.train()
    # Réinitialise les tirages de l'ordre des lots et des retournements à chaque passage.
    loader.generator.manual_seed(seed)
    flips = torch.Generator().manual_seed(seed)
    loss_total = torch.zeros((), device=device)
    for images, labels in loader:
        mask = torch.rand(len(images), generator=flips) < 0.5
        images[mask] = images[mask].flip(-1)
        images = images.to(device, memory_format=torch.channels_last)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        loss_total += loss.detach() * len(labels)
    return loss_total.item() / len(loader.dataset)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=root / "data/food-11")
    parser.add_argument("--output", type=Path, required=True, help="Nouveau dossier de résultats")
    parser.add_argument("--models", nargs="+", choices=tuple(BUILDERS), default=list(BUILDERS))
    parser.add_argument("--epochs", type=int, default=5, help="Nombre total d'époques, reprise comprise")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume-from", type=Path, help="Reprend une comparaison depuis ses sauvegardes")
    parser.add_argument("--patience", type=int, default=0, help="Arrête après ce nombre de tours sans gain (0 : désactivé)")
    parser.add_argument("--stop-on-drop", action="store_true",
                        help="Arrête chaque modèle dès qu'il n'améliore plus son record de frugalité")
    parser.add_argument("--min-delta", type=float, default=0.002, help="Gain minimal de frugalité pour prolonger")
    parser.add_argument("--predict", action="store_true", help="Exporte le JSON du meilleur candidat compact")
    args = parser.parse_args()
    if min(args.epochs, args.size, args.batch_size, args.threads) < 1 or args.lr <= 0 or args.workers < 0:
        parser.error("Paramètres d'entraînement invalides.")
    if len(set(args.models)) != len(args.models):
        parser.error("Chaque modèle doit être indiqué une seule fois.")
    if args.patience < 0 or args.min_delta <= 0:
        parser.error("La patience doit être positive ou nulle et le gain minimal strictement positif.")
    if args.patience and args.stop_on_drop:
        parser.error("Choisissez l'arrêt individuel ou la patience globale.")
    if args.predict and not any(name != "mobilenet" for name in args.models):
        parser.error("--predict nécessite au moins un candidat compact.")
    return args


def write_json(path, contents):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(contents, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def write_scores(output, history, summary):
    write_json(output / "history.json", history)
    write_json(output / "summary.json", sorted(
        summary.values(), key=lambda row: row["frugality"], reverse=True))


def read_previous_run(args, split):
    if args.resume_from is None:
        return {}, [], {}

    previous = args.resume_from
    config = json.loads((previous / "config.json").read_text(encoding="utf-8"))
    if (not set(args.models) <= set(config["models"]) or
            any(config[key] != getattr(args, key) for key in ("seed", "size", "batch_size", "lr"))):
        raise ValueError("La reprise exige des modèles déjà présents et les mêmes réglages d'entraînement.")
    if json.loads((previous / "split.json").read_text(encoding="utf-8")) != split:
        raise ValueError("Le découpage ne correspond pas à l'expérience précédente.")

    history = json.loads((previous / "history.json").read_text(encoding="utf-8"))
    history = [row for row in history if row["model"] in args.models]
    last_epochs = {name: max((row["epoch"] for row in history if row["model"] == name), default=0)
                   for name in args.models}
    if min(last_epochs.values()) < 1 or args.epochs <= min(last_epochs.values()):
        raise ValueError("Chaque modèle doit avoir une époque sauvegardée et le budget doit augmenter.")
    summary = {
        row["model"]: row
        for row in json.loads((previous / "summary.json").read_text(encoding="utf-8"))
        if row["model"] in args.models
    }
    for name, last_epoch in last_epochs.items():
        if not (previous / f"last_{name}.pth").exists() and summary[name]["epoch"] != last_epoch:
            raise ValueError(f"Les poids de la dernière époque de {name} sont indisponibles.")
    return last_epochs, history, summary


def has_plateaued(history, summary, epoch, patience, min_delta):
    if not patience or epoch <= patience:
        return False
    cutoff = epoch - patience
    old_best = {name: max(row["frugality"] for row in history
                         if row["model"] == name and row["epoch"] <= cutoff)
                for name in summary}
    return all(row["frugality"] - old_best[name] < min_delta
               for name, row in summary.items())


def run(args):
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = find_data_folder(args.data_dir, ("train", "training"))
    transform = get_transforms(size=args.size)
    dataset = ImageFolder(train_dir, transform=transform)
    train_ids, val_ids, duplicate_groups = split_by_content(dataset, args.seed)
    split = {
        "classes": dataset.classes,
        "duplicate_groups": duplicate_groups,
        "train": [Path(dataset.samples[i][0]).relative_to(train_dir).as_posix() for i in train_ids],
        "validation": [Path(dataset.samples[i][0]).relative_to(train_dir).as_posix() for i in val_ids],
    }
    last_epochs, history, summary = read_previous_run(args, split)
    args.output.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), "data_dir": str(args.data_dir), "output": str(args.output)}
    config["resume_from"] = str(args.resume_from) if args.resume_from else None
    config.update(device=str(device), torch=torch.__version__, torchvision=torchvision.__version__,
                  sklearn=sklearn.__version__)
    write_json(args.output / "config.json", config)
    write_json(args.output / "split.json", split)
    print(f"{device}: {len(train_ids)} train, {len(val_ids)} validation, "
          f"{duplicate_groups} groupes de doublons, images de {args.size} px", flush=True)

    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers,
                          persistent_workers=args.workers > 0)
    train_loader = DataLoader(Subset(dataset, train_ids), shuffle=True,
                              generator=torch.Generator(), **loader_options)
    val_loader = DataLoader(Subset(dataset, val_ids), **loader_options)
    criterion = nn.CrossEntropyLoss()
    runs = {}
    active = set(args.models)
    last_scores = {}
    for name in args.models:
        previous = [row["frugality"] for row in history if row["model"] == name]
        if previous:
            last_scores[name] = previous[-1]
        if args.stop_on_drop and len(previous) >= 2 and previous[-1] <= max(previous[:-1]):
            active.remove(name)
            print(f"{name}: arrêté à l'époque {last_epochs[name]}, "
                  f"frugalité {previous[-1]:.4f} sous son record de {max(previous[:-1]):.4f}.", flush=True)

    for name in args.models:
        torch.manual_seed(args.seed)
        last_epoch = last_epochs.get(name, 0)
        model = load_model(name, len(dataset.classes), pretrained=last_epoch == 0)
        model = model.to(device, memory_format=torch.channels_last)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        if last_epoch:
            best_path = args.resume_from / f"{name}.pth"
            last_path = args.resume_from / f"last_{name}.pth"
            saved = torch.load(last_path if last_path.exists() else best_path,
                               map_location=device, weights_only=True)
            if saved["model"] != name or saved["metrics"]["epoch"] != last_epoch:
                raise ValueError(f"Sauvegarde inattendue pour {name}, époque {last_epoch}.")
            model.load_state_dict(saved["state_dict"])
            if "optimizer_state_dict" in saved:
                optimizer.load_state_dict(saved["optimizer_state_dict"])
            else:
                print(f"{name}: poids de l'époque {last_epoch} rechargés, Adam réinitialisé.", flush=True)
            copyfile(best_path, args.output / best_path.name)
            if last_path.exists():
                copyfile(last_path, args.output / last_path.name)
        runs[name] = model, optimizer, sum(p.numel() for p in model.parameters())

    if history:
        write_scores(args.output, history, summary)

    # Alterne les architectures pour obtenir une comparaison à budget égal dès la première époque.
    stop_reason = "budget"
    for epoch in range(min(last_epochs.values(), default=0) + 1, args.epochs + 1):
        if not active:
            stop_reason = "no_gain"
            break
        for name, (model, optimizer, params) in runs.items():
            if name not in active or epoch <= last_epochs.get(name, 0):
                continue
            start = perf_counter()
            torch.manual_seed(args.seed + epoch)
            loss = train_epoch(model, train_loader, optimizer, criterion, device, args.seed + epoch)
            scores = evaluate(model, val_loader, device)
            row = dict(model=name, epoch=epoch, params=params, train_loss=loss,
                       seconds=perf_counter() - start, **scores)
            history.append(row)
            checkpoint = {
                "model": name, "state_dict": model.state_dict(), "classes": dataset.classes,
                "image_size": args.size, "metrics": row,
            }
            if name not in summary or scores["frugality"] > summary[name]["frugality"]:
                summary[name] = row
                torch.save(checkpoint, args.output / f"{name}.pth")
            torch.save({**checkpoint, "optimizer_state_dict": optimizer.state_dict()},
                       args.output / f"last_{name}.pth")
            write_scores(args.output, history, summary)
            print(f"{name:14s} {epoch}/{args.epochs} | loss {loss:.4f} | "
                  f"acc {scores['accuracy']:.4f} | F1 {scores['f1']:.4f} | "
                  f"frug {scores['frugality']:.4f} | {row['seconds']:.0f} s", flush=True)
            previous_score = last_scores.get(name)
            last_scores[name] = row["frugality"]
            last_epochs[name] = epoch
            if args.stop_on_drop and previous_score is not None and row["frugality"] <= previous_score:
                active.remove(name)
                print(f"{name}: arrêté après l'époque {epoch} "
                      f"({previous_score:.4f} → {row['frugality']:.4f}).", flush=True)

        if has_plateaued(history, summary, epoch, args.patience, args.min_delta):
            stop_reason = "plateau"
            print(f"Arrêt à l'époque {epoch}: aucun gain de {args.min_delta} sur "
                  f"les {args.patience} derniers tours.", flush=True)
            break

    if not active:
        stop_reason = "no_gain"
    config.update(completed_epochs=max(last_epochs.values()), last_epoch_by_model=last_epochs,
                  stop_reason=stop_reason)
    write_json(args.output / "config.json", config)

    if args.predict:
        winner = max((row for name, row in summary.items() if name != "mobilenet"),
                     key=lambda row: row["frugality"])["model"]
        model = runs[winner][0]
        saved = torch.load(args.output / f"{winner}.pth", map_location=device, weights_only=True)
        model.load_state_dict(saved["state_dict"])
        test_dir = find_data_folder(args.data_dir, ("test", "testing"))
        predict_test(model, test_dir, transform, saved["classes"], device,
                     args.output / f"predictions_{winner}.json", batch_size=args.batch_size)


if __name__ == "__main__":
    run(parse_args())
