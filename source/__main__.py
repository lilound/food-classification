"""Entraîne le modèle Food-11 du rendu et exporte ses prédictions."""

import argparse
import math
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from torchvision.datasets import ImageFolder

from .creer_dataset_equilibre import DEFAULT_OUTPUT, KIT, benchmark_protocol, read_json, verify, write_json
from .shufflenet05 import load_model
from .dataset import get_transforms
from .evaluate import predict_test
from .lr_validation import ValidationLossRate

PROTOCOL = read_json(KIT / "protocol.json") # hyperparamètres figs
CLASSES = read_json(KIT / "split.json")["classes"]


def run_epoch(model, loader, loss_fn, frozen, optimizer=None, rate=None):
    training = optimizer is not None
    model.train(training)
    for block in frozen:
        # Fige aussi les statistiques BatchNorm des blocs préentraînés.
        block.eval()

    loss_sum = torch.zeros(())
    targets, predictions = [], []
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.contiguous(memory_format=torch.channels_last)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                rate.before_step() # hook du contrôleur de LR
                optimizer.step()
                rate.after_step(None)
            else:
                targets.extend(labels.tolist())
                predictions.extend(logits.argmax(dim=1).tolist())
            loss_sum += loss.detach() * len(labels)

    loss = loss_sum.item() / len(loader.dataset)
    if not math.isfinite(loss):
        raise RuntimeError("Perte non finie. Le dernier checkpoint est disponible.")
    if training:
        return loss
    # Validation
    return loss, accuracy_score(targets, predictions), f1_score(
        targets, predictions, labels=list(range(len(CLASSES))), average="macro", zero_division=0)


def train(data, output, resume):
    split = read_json(KIT / "split.json")
    protocol, _ = benchmark_protocol(data, PROTOCOL, split)
    # Vérifie que le jeu équilibré est bien celui du protocole
    metadata = verify(data)
    if (metadata["dataset_sha256"] != protocol["dataset_sha256"]
            or metadata["train_per_class"] != 1600 or metadata["seed"] != PROTOCOL["seed"]):
        raise ValueError("Le jeu équilibré doit correspondre au protocole Food-11 du rendu.")
    # Reproductibilité
    seed = PROTOCOL["seed"]
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(seed)
    # Data : dossiers train/ et validation/ du jeu équilibré, avec le même prétraitement
    transform = get_transforms(PROTOCOL["image_size"])
    training = ImageFolder(data / "train", transform=transform)
    validation = ImageFolder(data / "validation", transform=transform)
    if training.classes != CLASSES or validation.classes != CLASSES:
        raise ValueError("Les classes diffèrent du découpage du rendu.")
    batch_size = PROTOCOL["batch_size"]
    # Le sampler mélange le train à chaque époque ; son générateur est réinitialisé plus bas
    sampler = RandomSampler(training, generator=torch.Generator())
    train_loader = DataLoader(training, batch_size=batch_size, sampler=sampler,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(validation, batch_size=batch_size,
                            generator=torch.Generator().manual_seed(seed))
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "shufflenet05.pth"
    # Protection : on n'écrase jamais un entraînement existant sans --resume
    if checkpoint.exists() and not resume:
        raise FileExistsError(f"{checkpoint} existe déjà. Utilisez --resume ou un autre dossier.")
    # Modèle : poids préentraînés au départ, poids du checkpoint en cas de reprise
    model = load_model(len(CLASSES), pretrained=not resume)
    model.to(memory_format=torch.channels_last)
    # Gel
    # Seuls stage4 et la tête de classification sont entraînés
    frozen = model.conv1, model.stage2, model.stage3
    for block in frozen:
        block.requires_grad_(False)
    params = sum(p.numel() for p in model.parameters()) # total, gelés inclus (sert à la frugalité)
    # L'optimiseur Adam ne reçoit que les paramètres non gelés
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=PROTOCOL["initial_lr"])
    rate = ValidationLossRate(optimizer, len(train_loader)) # warmup puis réduction du LR sur plateau de la perte de validation
    criterion = nn.CrossEntropyLoss()
    history, best_weights = [], None
    # Reprise : on recharge poids, optimiseur, état du LR et historique
    if resume:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["protocol"] != protocol:
            raise ValueError("Le checkpoint a été créé avec un autre protocole ou jeu de données.")
        model.load_state_dict(saved["weights"])
        optimizer.load_state_dict(saved["optimizer"])
        rate.load_state_dict(saved["scheduler"])
        history, best_weights = saved["history"], saved["best_weights"]

    best = max((row["frugality"] for row in history), default=-math.inf)
    # On repart de la dernière époque terminée
    for epoch in range(len(history) + 1, PROTOCOL["epochs"] + 1):
        # Graine propre à chaque époque 
        epoch_seed = seed + 3 * epoch
        torch.manual_seed(epoch_seed)
        sampler.generator.manual_seed(epoch_seed)
        rate.rates.clear() # liste des LR de l'époque, remise à zéro
        train_loss = run_epoch(model, train_loader, criterion, frozen, optimizer, rate)
        val_loss, accuracy, f1 = run_epoch(model, val_loader, criterion, frozen)
        rate.validate(val_loss) # le scheduler décide de réduire ou non le LR selon la perte de validation

        # Calcul de frugalité
        # F1 pénalisé par la taille du modèle ; c'est ce score qui choisit le meilleur modèle
        score = f1 - 0.3 * params / (1_000_000 + params)
        if score > best:
            best = score
            best_weights = {key: value.detach().clone() for key, value in model.state_dict().items()}
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                            accuracy=float(accuracy), f1=float(f1), frugality=float(score),
                            next_lr=optimizer.param_groups[0]["lr"]))
        # Checkpoint écrit dans un fichier temporaire puis renommé :
        # un plantage en pleine écriture ne corrompt pas le checkpoint précédent
        temp = checkpoint.with_suffix(".tmp")
        torch.save(dict(weights=model.state_dict(), best_weights=best_weights, history=history,
                        optimizer=optimizer.state_dict(), scheduler=rate.state_dict(), protocol=protocol), temp)
        temp.replace(checkpoint)
        write_json(output / "history.json", history)
        print(f"Époque {epoch:2d} | F1 {f1:.4f} | frugalité {score:.4f} | "
              f"perte val {val_loss:.4f} | prochain LR {optimizer.param_groups[0]['lr']:.6g}", flush=True)


def main():
    # Arguments 
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    # ---- Sous-commande train : python script.py train [options] ----
    training = commands.add_parser("train", help="Entraîne et choisit le meilleur checkpoint en validation.")
    training.add_argument("--data-dir", type=Path, default=DEFAULT_OUTPUT) # jeu équilibré (train/validation)
    training.add_argument("--output", type=Path, default=Path("results")) # dossier des checkpoints et de l'historique
    training.add_argument("--resume", action="store_true") # reprendre depuis le dernier checkpoint

    # ---- Sous-commande predict : python script.py predict [options] ----
    prediction = commands.add_parser("predict", help="Exporte les prédictions du meilleur checkpoint.")
    prediction.add_argument("--checkpoint", type=Path, default=Path("results/shufflenet05.pth")) # checkpoint produit par train
    prediction.add_argument("--test-dir", type=Path, default=DEFAULT_OUTPUT / "test") # images de test à prédire
    prediction.add_argument("--output", type=Path, default=Path("results/predictions.json")) # fichier de prédictions
args = parser.parse_args()
    args = parser.parse_args()

    torch.set_num_threads(4)
    if args.command == "train":
        train(args.data_dir, args.output, args.resume)
    else:
        # Prediction
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model = load_model(len(CLASSES), pretrained=False)
        model.load_state_dict(saved["best_weights"])
        model.to(memory_format=torch.channels_last)
        predict_test(model, args.test_dir, get_transforms(PROTOCOL["image_size"]),
                     CLASSES, torch.device("cpu"), args.output, batch_size=16)


if __name__ == "__main__":
    main()
