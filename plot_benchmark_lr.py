"""Trace un benchmark ou compare les deux modes sur des axes communs."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

PROTOCOL = Path(__file__).resolve().parent / "benchmarks/lr/protocol.json"
TITLES = {"fixed": "taux fixe", "plateau": "taux variable"}
NAMES = {"fixed": "frugalite_taux_fixe", "plateau": "frugalite_taux_variable"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def plot_runs(dirs, output):
    models = read(PROTOCOL)["models"]
    records = {}
    signature = None
    for directory in dirs:
        metadata = read(directory / "run.json")["signature"]
        mode = metadata["mode"]
        if mode in records:
            raise ValueError(f"Deux résultats pour le mode {mode}.")
        current = (metadata["protocol_sha256"], metadata["split_sha256"])
        if signature is not None and signature != current:
            raise ValueError("Les résultats ne partagent pas le même protocole ou découpage.")
        signature = current
        rows = read(directory / "history.json")
        if not rows or any(row["model"] not in models for row in rows):
            raise ValueError(f"Historique vide ou modèle inconnu dans {directory}.")
        records[mode] = {name: sorted((r for r in rows if r["model"] == name),
                                      key=lambda row: row["epoch"]) for name in models}
    all_rows = [row for series in records.values() for rows in series.values() for row in rows]
    max_epoch = max(row["epoch"] for row in all_rows)
    low, high = (f(row["frugality"] for row in all_rows) for f in (min, max))
    pad = max(0.01, (high - low) * 0.05)
    output.mkdir(parents=True, exist_ok=True)

    for mode, series in records.items():
        fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
        for i, (name, (label, _, _)) in enumerate(models.items()):
            rows = series[name]
            if not rows:
                continue
            color = plt.get_cmap("tab10")(i)
            ax.plot([r["epoch"] for r in rows], [r["frugality"] for r in rows], label=label,
                    color=color, linestyle=("-", "--")[i % 2], linewidth=1.8)
            best = max(rows, key=lambda r: r["frugality"])
            ax.scatter(best["epoch"], best["frugality"], color=color, s=28, zorder=3)
        ax.set(title=f"Food-11 · Frugalité par époque — {TITLES[mode]}",
               xlabel="Époque", ylabel="Frugalité en validation",
               xlim=(0.8, max(2, max_epoch + 0.2)), ylim=(low - pad, high + pad))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
        for extension in ("png", "svg"):
            fig.savefig(output / f"{NAMES[mode]}.{extension}", dpi=165)
        plt.close(fig)

    if set(records) == set(TITLES):
        comparison = []
        for name in models:
            fixed, variable = (records[mode][name] for mode in ("fixed", "plateau"))
            if not fixed or not variable:
                continue
            fixed_by_epoch = {row["epoch"]: row["frugality"] for row in fixed}
            variable_by_epoch = {row["epoch"]: row["frugality"] for row in variable}
            common = set(fixed_by_epoch) & set(variable_by_epoch)
            if not common:
                continue
            epoch = max(common)
            best_fixed, best_variable = (max(rows, key=lambda row: row["frugality"])
                                         for rows in (fixed, variable))
            comparison.append(dict(model=name, best_fixed=best_fixed["frugality"],
                                   best_fixed_epoch=best_fixed["epoch"],
                                   best_variable=best_variable["frugality"],
                                   best_variable_epoch=best_variable["epoch"],
                                   best_delta=best_variable["frugality"] - best_fixed["frugality"],
                                   common_epoch=epoch,
                                   fixed_at_common_epoch=fixed_by_epoch[epoch],
                                   variable_at_common_epoch=variable_by_epoch[epoch],
                                   common_delta=variable_by_epoch[epoch] - fixed_by_epoch[epoch]))
        (output / "comparaison.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.runs) not in (1, 2):
        parser.error("Indiquez un ou deux dossiers de résultats.")
    plot_runs(args.runs, args.output)
