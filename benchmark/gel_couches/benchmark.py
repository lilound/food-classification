"""Lance les trois essais avec gel et compare aux historiques du benchmark modèle."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmark/modeles/benchmark_lr.py"
MODELS = {"shufflenet05": "fixed", "shufflenet10": "adaptive", "mobilenet_v2": "adaptive"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare(model, reference, frozen, output):
    old, new = read(reference / "run.json"), read(frozen / "run.json")
    old_sig, new_sig = old["signature"], new["signature"]
    if old_sig != {k: v for k, v in new_sig.items() if k != "freeze_early"} or not new_sig.get("freeze_early") or model not in old["models"] or model not in new["models"]:
        raise ValueError(f"Protocoles incompatibles pour {model}.")
    if old_sig["mode"] != MODELS[model]:
        raise ValueError(f"Mauvais taux de référence pour {model}.")
    fig, ax = plt.subplots()
    for label, directory in (("Sans gel", reference), ("Avec gel", frozen)):
        rows = [row for row in read(directory / "history.json") if row["model"] == model]
        if not rows:
            raise ValueError(f"Historique absent : {model} dans {directory}")
        ax.plot([r["epoch"] for r in rows], [r["frugality"] for r in rows], label=label)
    ax.set(xlabel="Époque", ylabel="Frugalité", title=model)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"frugalite_{model}.png", dpi=160)
    fig.savefig(output / f"frugalite_{model}.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="Dossier benchmark_nuit contenant fixed/ et adaptive/.")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark/gel_couches/executions")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--freeze-level", choices=("early", "strong"), default="early")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir in sys.path:
        sys.path.remove(script_dir)
    sys.path.insert(0, str(ROOT))
    from benchmark.modeles import benchmark_lr
    protocol = benchmark_lr.read(benchmark_lr.CONFIG)
    split = benchmark_lr.fingerprint(benchmark_lr.read(benchmark_lr.SPLIT))
    protocol_hash = hashlib.sha256(benchmark_lr.fingerprint(protocol) + split).hexdigest()
    split_hash = hashlib.sha256(split).hexdigest()
    for model, mode in MODELS.items():
        reference = args.reference / mode
        if not (reference / "run.json").is_file() or not (reference / "history.json").is_file():
            parser.error(f"Référence manquante : {reference}")
        metadata = read(reference / "run.json")
        signature = metadata["signature"]
        if (signature.get("mode") != mode or signature.get("protocol_sha256") != protocol_hash
                or signature.get("split_sha256") != split_hash or model not in metadata["models"]
                or not any(row["model"] == model for row in read(reference / "history.json"))
                or (mode == "fixed" and signature.get("fixed_lr") != protocol["initial_lr"])
                or (mode == "adaptive" and signature.get("policy_sha256") != hashlib.sha256(benchmark_lr.fingerprint(benchmark_lr.POLICY)).hexdigest())):
            parser.error(f"Référence incompatible : {model} dans {reference}")
    for model, mode in MODELS.items():
        result = args.output / model
        command = [sys.executable, "-u", str(RUNNER), "--mode", mode, "--models", model,
                   "--freeze-early", "--output", str(result), "--data-dir", str(args.data_dir),
                   "--device", args.device, "--workers", str(args.workers), "--threads", str(args.threads)]
        if args.freeze_level == "strong":
            command.extend(["--freeze-level", "strong"])
        if args.resume and result.exists():
            command.append("--resume")
        subprocess.run(command, check=True)
        compare(model, args.reference / mode, result, args.output / "graphiques")


if __name__ == "__main__":
    main()
