"""Archive les historiques Food-11 et trace la frugalité par époque."""

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
KIT = Path(__file__).resolve().parent
CONFIG = KIT / "protocole/protocol.json"
SPLIT = KIT / "protocole/split.json"
MODES = {
    "fixed": ("Taux fixe", "#3165A7", "-"),
    "adaptive": ("Taux adaptatif · par lot", "#187455", "-."),
    "cosine": ("Taux cosinus", "#B55400", "--"),
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + ".tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(path)


def write_json(path, payload):
    write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    temp = path.with_name("." + path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def read_run(source):
    paths = [source / f"{key}.json" for key in ("run", "history", "summary")]
    if not all(path.is_file() for path in paths):
        return None
    # Chaque fichier est remplacé atomiquement, mais pas forcément au même instant.
    for _ in range(5):
        run, history, summary = map(read, paths)
        latest = {row["model"]: row["epoch"] for row in history}
        if latest == {row["model"]: row["last_epoch"] for row in summary}:
            return {"run": run, "history": history, "summary": summary} if history else None
        time.sleep(0.1)
    raise RuntimeError(f"Historique et classement désynchronisés dans {source}.")


def chart_style(ax):
    ax.set(xlabel="Époque", ylabel="Frugalité en validation")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", color="#e2e7ec")
    ax.set_axisbelow(True)
    ax.spines[["right", "top"]].set_visible(False)


def draw_line(ax, rows, label, color, style):
    """Trace une courbe de frugalité et marque la meilleure époque d'un point."""
    if not rows:
        return
    best = max(rows, key=lambda row: row["frugality"])
    line, = ax.plot([r["epoch"] for r in rows], [r["frugality"] for r in rows],
                    color=color, linestyle=style, linewidth=1.8, label=label)
    ax.scatter(best["epoch"], best["frugality"], color=color, s=30, zorder=3)
    return line


def save_figure(fig, path):
    """Sauvegarde la figure en PNG et SVG, puis ferme la figure"""
    path.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        target = path.with_suffix(f".{extension}")
        temp = target.with_name("." + target.name + ".tmp")
        fig.savefig(temp, format=extension, dpi=180, facecolor="white")
        temp.replace(target)
    plt.close(fig)


def render(output, snapshots, protocol, split, pending):
    names = protocol["models"]
    split_hash = hashlib.sha256(encode(split)).hexdigest()
    protocol_hash = hashlib.sha256(encode(protocol) + encode(split)).hexdigest()
    # Validation : on vérifie que chaque run corresponde au protocole figé
    runs = {}
    for snapshot in snapshots:
        signature = snapshot["run"]["signature"]
        mode = signature["mode"]
        if mode not in MODES or mode in runs:
            raise ValueError(f"Mode inconnu ou présent deux fois : {mode}.")
        if mode == "fixed" and signature.get("fixed_lr") != protocol["initial_lr"]:
            raise ValueError("Le taux fixe ne correspond pas au protocole.")
        if (signature["protocol_sha256"], signature["split_sha256"]) != (protocol_hash, split_hash):
            raise ValueError(f"{mode}: résultat incompatible avec le protocole figé. Utilisez le run à patience 6.")
        if any(row["model"] not in names for row in snapshot["history"]):
            raise ValueError(f"{mode}: modèle absent du protocole.")
        runs[mode] = snapshot
    runs = {mode: runs[mode] for mode in MODES if mode in runs}
    curves = {mode: {model: sorted((row for row in snapshot["history"] if row["model"] == model),
                                  key=lambda row: row["epoch"]) for model in names}
              for mode, snapshot in runs.items()}
    # Archivage
    write_json(output / "protocole" / "protocol.json", protocol)
    write_json(output / "protocole" / "split.json", split)
    for mode, snapshot in runs.items():
        raw = output / "experiences" / mode / "donnees_brutes"
        for key, payload in snapshot.items():
            write_json(raw / f"{key}.json", payload)
        for key in ("history", "summary"):
            write_csv(raw / f"{key}.csv", snapshot[key])
    # Graphique individuel
    for model, (label, _, _) in names.items():
        if not any(curves[mode][model] for mode in runs):
            continue
        fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
        for mode in runs:
            draw_line(ax, curves[mode][model], *MODES[mode])
        chart_style(ax)
        ax.set_title(f"Food-11 · {label}")
        ax.legend(frameon=False, loc="lower right")
        save_figure(fig, output / "courbes_par_modele" / f"frugalite_{model}")
    # Graphique d'ensemble 
    if runs:
        fig, axes = plt.subplots(1, len(runs), figsize=(max(8, 5.1 * len(runs)), 5.8),
                                 sharex=True, sharey=True, squeeze=False, layout="constrained")
        handles = {}
        for i_panel, (ax, mode) in enumerate(zip(axes[0], runs, strict=True)):
            for i, (model, (label, _, _)) in enumerate(names.items()):
                line = draw_line(ax, curves[mode][model], label, plt.get_cmap("tab10")(i), "-")
                if line is not None:
                    handles[model] = line
            chart_style(ax)
            if i_panel:
                ax.set_ylabel("")
            ax.set_title(MODES[mode][0])
        fig.suptitle("Food-11 · Évolution de la frugalité par modèle et par mode")
        fig.legend(handles=[handles[model] for model in names if model in handles],
                   loc="outside lower center", ncols=min(5, len(handles)), frameon=False)
        save_figure(fig, output / "comparaison" / "frugalite_toutes_courbes")
    # Tableau des scores
    comparison = []
    for model, (label, _, _) in names.items():
        available = {mode: {row["epoch"]: row for row in curves[mode][model]}
                     for mode in runs if curves[mode][model]}
        common = set.intersection(*(set(epochs) for epochs in available.values())) if len(available) >= 2 else set()
        epoch = max(common) if common else None
        for mode, epochs in available.items():
            best = max(epochs.values(), key=lambda row: row["frugality"])
            summary = next(r for r in runs[mode]["summary"] if r["model"] == model)
            comparison.append(dict(model=model, label=label, mode=mode, params=best["params"],
                                   meilleure_epoque=best["epoch"], meilleure_frugalite=best["frugality"],
                                   derniere_epoque=summary["last_epoch"], arret=summary["stop"] or "en cours",
                                   epoque_commune=epoch, frugalite_epoque_commune=epochs[epoch]["frugality"] if epoch else None))
    if comparison:
        write_csv(output / "comparaison" / "scores.csv", comparison)
    #  Création de index.md (page d'accueil du dossier de résultats)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = ["# Food-11 — dossier de résultats", "", f"Dernière actualisation : {date}.", "",
             "Chaque point est une époque mesurée en validation. Le point mis en évidence marque le meilleur score disponible.",
             "Les courbes et classements restent provisoires tant que des essais sont en cours.", ""]
    for mode, snapshot in runs.items():
        rows = snapshot["summary"]
        expected = snapshot["run"].get("models", list(names))
        done = sum(bool(row["stop"]) for row in rows)
        epochs = len(snapshot["history"])
        count = "1 époque enregistrée" if epochs == 1 else f"{epochs} époques enregistrées"
        lines.append(f"- **{MODES[mode][0]}** : {done}/{len(expected)} modèles terminés, {count} — "
                     f"[données brutes](experiences/{mode}/donnees_brutes/history.json) · "
                     f"[classement](experiences/{mode}/donnees_brutes/summary.csv).")
    for mode in MODES:
        if mode not in runs:
            lines.append(f"- **{MODES[mode][0]}** : aucun historique importé.")
    if pending:
        lines.extend(["", "Dossiers fournis sans résultats disponibles : " +
                      ", ".join(f"`{path.name}`" for path in pending) + "."])
    if comparison:
        lines.extend(["", "[Diagramme d’ensemble](comparaison/frugalite_toutes_courbes.svg) · "
                      "[Tableau des scores](comparaison/scores.csv)", "", "Courbes par modèle :", ""])
        lines.extend(f"- [{names[model][0]}](courbes_par_modele/frugalite_{model}.svg)"
                     for model in names if any(curves[mode][model] for mode in runs))
    write(output / "index.md", "\n".join(lines) + "\n")


def alive(pid):
    status = Path(f"/proc/{pid}/stat")
    try:
        return status.read_text().rsplit(") ", 1)[1].split()[0] not in {"Z", "X"}
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=KIT / "resultats")
    parser.add_argument("--watch-pid", action="append", type=int, default=[],
                        help="Sur Linux, actualise tant qu'un des processus indiqués tourne")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    if args.interval < 5 or any(pid <= 0 for pid in args.watch_pid):
        parser.error("Intervalle minimal de 5 secondes et PID positifs.")
    if args.watch_pid and not Path("/proc/self/stat").is_file():
        parser.error("--watch-pid requiert Linux.")
    protocol, split = read(CONFIG), read(SPLIT)
    last = None
    while True:
        snapshots, pending = [], []
        for path in args.runs:
            snapshot = read_run(path)
            if snapshot:
                snapshots.append(snapshot)
            else:
                pending.append(path) # dossier sans résultats exploitables
        current = (snapshots, pending)
        if current != last: # regénération si les données ont changé
            render(args.output, snapshots, protocol, split, pending)
            print(f"Export : {sum(len(s['history']) for s in snapshots)} époques → {args.output}", flush=True)
            last = current
        if not any(map(alive, args.watch_pid)):
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
