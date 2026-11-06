"""Prépare les JSON et les champs du formulaire Food-11 pour les meilleurs modèles."""

import argparse
import json
from pathlib import Path

import torch

from models.compact import load_model
from src.dataset import find_data_folder, get_transforms
from src.evaluate import predict_test


NAMES = {
    "mobilenet": "MobileNetV3 Small",
    "shufflenet05": "ShuffleNetV2 x0.5",
    "shufflenet10": "ShuffleNetV2 x1.0",
    "squeezenet": "SqueezeNet 1.1",
    "mnasnet05": "MNASNet 0.5",
}
FORM_URL = "https://www.creatis.insa-lyon.fr/kechichian/food11-classif-leaderboard.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Dossier de résultats du benchmark")
    parser.add_argument("--top", type=int, default=3, help="Nombre de modèles à exporter")
    parser.add_argument("--data-dir", type=Path, help="Dossier Food-11 si différent du benchmark")
    args = parser.parse_args()

    config = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    rows = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))
    if not 1 <= args.top <= len(rows):
        parser.error(f"--top doit être compris entre 1 et {len(rows)}.")

    data_dir = args.data_dir or Path(config["data_dir"])
    test_dir = find_data_folder(data_dir, ("test", "testing"))
    output = args.run / "submission"
    output.mkdir(exist_ok=False)
    torch.set_num_threads(config["threads"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    form = [
        "# Évaluation Food-11",
        "",
        f"[Ouvrir le formulaire officiel]({FORM_URL}).",
        "",
        "Utilisez le nom déjà convenu par le groupe. Pour comparer les modèles, "
        "laissez **Update leaderboard ?** décoché. Cochez **Report class metrics ?** "
        "si vous voulez aussi les scores par classe. Reportez le score de test "
        "renvoyé par le serveur avant de choisir votre unique soumission finale.",
        "",
        "| Modèle | JSON | Model parameter count (Millions) | Method keywords | Frugalité locale |",
        "| --- | --- | ---: | --- | ---: |",
    ]

    for row in rows[:args.top]:
        name = row["model"]
        saved = torch.load(args.run / f"{name}.pth", map_location=device, weights_only=True)
        if saved["model"] != name or saved["metrics"] != row:
            raise ValueError(f"Les poids sauvegardés ne correspondent pas au classement pour {name}.")
        model = load_model(name, len(saved["classes"]), pretrained=False)
        model.load_state_dict(saved["state_dict"])
        model.to(device, memory_format=torch.channels_last)
        params = sum(p.numel() for p in model.parameters())
        if params != row["params"]:
            raise ValueError(f"Nombre de paramètres incohérent pour {name}.")

        size = saved["image_size"]
        path = output / f"predictions_{name}.json"
        predictions = predict_test(model, test_dir, get_transforms(size=size), saved["classes"],
                                   device, path, batch_size=config["batch_size"])
        if len(predictions) != 3097 or set(predictions.values()) - set(saved["classes"]):
            raise ValueError(f"Le JSON de {name} ne correspond pas au test officiel.")
        keywords = (f"{NAMES[name]}, ImageNet pretrained, Food-11 fine-tuning, "
                    f"{size}x{size}, Adam lr={config['lr']:g}, "
                    f"epoch {row['epoch']}, horizontal flip")
        form.append(f"| {NAMES[name]} | [{path.name}]({path.name}) | {params / 1e6:.6f} | "
                    f"{keywords} | {row['frugality']:.4f} |")

    (output / "formulaire.md").write_text("\n".join(form) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
