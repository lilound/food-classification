"""Lance les expériences Food-11 et prépare le dossier de rendu, même après une interruption."""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark.modeles.export_benchmark import CONFIG, KIT, MODES, ROOT, SPLIT, read, read_run, render, write_json


def devices(kind, jobs):
    """Liste les "emplacements" de calcul : un GPU par job, complété par des CPU si besoin."""
    count = torch.cuda.device_count() if kind != "cpu" else 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    tokens = visible.split(",") if visible is not None else list(map(str, range(count)))
    if kind == "cuda" and count < jobs:
        raise ValueError(f"{count} GPU CUDA visibles pour {jobs} essais simultanés. "
                         "Réduisez --jobs ou utilisez --device auto.")
    return [("cuda", token) for token in tokens[:min(count, jobs)]] + [
        ("cpu", None)] * max(0, jobs - count)


def command(mode, source, args):
    """Construit la commande qui lance benchmark_lr.py pour un mode."""
    base = [sys.executable, "-u", str(KIT / "benchmark_lr.py"), "--mode", mode]
    base += ["--data-dir", str(args.data_dir.resolve()), "--output", str(source.resolve()),
             "--threads", str(args.threads), "--workers", str(args.workers)]
    if source.exists():
        base += ["--resume"]
    return base


def complete(snapshot, expected):
    """Vérifie que tous les modèles attendus ont fini leur entraînement"""
    return bool(snapshot and
                {row["model"] for row in snapshot["summary"] if row["stop"]} == set(expected))


def stop(children):
    def send(proc, sig):
        if proc.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(proc.pid, sig)
            elif sig == signal.SIGTERM:
                proc.terminate()
            else:
                proc.kill()
        except ProcessLookupError:
            pass

    for proc, _ in children.values():
        send(proc, signal.SIGTERM)
    deadline = time.monotonic() + 8
    for proc, _ in children.values():
        try:
            proc.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            send(proc, getattr(signal, "SIGKILL", 9))
            proc.wait()


def run(args):
    protocol, split = read(CONFIG), read(SPLIT)
    jobs = min(args.jobs, len(args.modes))
    slots = devices(args.device, jobs)
    root, output = args.runs.resolve(), args.output.resolve()
    if root == output or root in output.parents or output in root.parents:
        raise ValueError("Les dossiers de checkpoints et de rendu doivent être séparés.")
    sources = {mode: root / mode for mode in args.modes}
    if not args.resume and (any(path.exists() for path in sources.values()) or output.exists()):
        raise FileExistsError("Dossier déjà présent : utilisez --resume ou choisissez de nouveaux chemins.")
    root.mkdir(parents=True, exist_ok=True)
    # Ajout d'un verrou
    lock = root / ".lancement.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(f"{os.getpid()}\n")
    results = {}
    children = {}
    queue = []
    last = None
    output_ready = False
    plan = {"runs": str(root), "modes": args.modes}

    def report():
        """Régénère le dossier de rendu depuis les historiques."""
        nonlocal last
        try:
            snapshots = {mode: read_run(source) for mode, source in sources.items()}
            if any(s and s["run"]["signature"]["mode"] != m for m, s in snapshots.items()):
                raise ValueError("Un dossier de résultats ne correspond pas au mode attendu.")
            current = list(snapshots.values())
            if current != last:
                render(output, [s for s in current if s], protocol, split,
                       [sources[m] for m, s in snapshots.items() if s is None])
                last = current
            write_json(output / "execution.json", results)
            return True
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"Export temporairement indisponible : {exc}", flush=True)
            return False

    try:
        plan_path = output / "plan.json"
        if output.exists() and (not plan_path.is_file() or read(plan_path) != plan):
            raise ValueError("Dossier de rendu déjà associé à un autre lancement.")
        output.mkdir(parents=True, exist_ok=True)
        write_json(plan_path, plan)
        output_ready = True
        logdir = output / "journaux"
        logdir.mkdir(exist_ok=True)
        for mode, source in sources.items():
            snapshot = read_run(source)
            if complete(snapshot, protocol["models"]):
                results[mode] = {"etat": "terminé", "repris": False}
            elif source.exists() and not (source / "run.json").is_file():
                results[mode] = {"etat": "échec", "erreur": "dossier présent sans run.json"}
            else:
                results[mode] = {"etat": "en attente"}
                queue.append(mode)
        next_report = 0.0
        while queue or children:
            # 1. Lancer les modes en attente tant qu'il y a des emplacements libres
            while queue and slots:
                mode = queue.pop(0)
                kind, token = slots.pop(0)
                source = sources[mode]
                env = os.environ.copy()
                if kind == "cuda":
                    env["CUDA_VISIBLE_DEVICES"] = token
                args_child = command(mode, source, args) + ["--device", kind]
                log = logdir / f"{mode}.log"
                try:
                    with log.open("ab") as stream:
                        stream.write(f"\n--- {mode} · {kind} ---\n".encode())
                        stream.flush()
                        proc = subprocess.Popen(args_child, cwd=ROOT, env=env, stdout=stream,
                                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                                start_new_session=os.name == "posix")
                    children[mode] = (proc, (kind, token))
                    results[mode] = {"etat": "en cours", "appareil": kind, "pid": proc.pid,
                                     "journal": f"journaux/{mode}.log"}
                    print(f"{mode} lancé sur {kind} — {log}", flush=True)
                except OSError as exc:
                    slots.append((kind, token))
                    results[mode] = {"etat": "échec", "erreur": str(exc)}
            # 2. Vérifier les processus terminés et libérer leur emplacement
            for mode, (proc, slot) in list(children.items()):
                code = proc.poll()
                if code is not None:
                    del children[mode]
                    slots.insert(0, slot)
                    try:
                        success = code == 0 and complete(read_run(sources[mode]), protocol["models"])
                    except (OSError, ValueError, RuntimeError) as exc:
                        results[mode]["erreur"] = str(exc)
                        success = False
                    results[mode].update(etat="terminé" if success else "échec", code_retour=code)
                    print(f"{mode} : {results[mode]['etat']} (code {code})", flush=True)
                    next_report = 0.0 # rendu immédiat
            # 3. Rendu intermédiaire toutes les 60 s
            now = time.monotonic()
            if now >= next_report:
                report()
                next_report = now + 60
            if children:
                time.sleep(2)
    except KeyboardInterrupt:
        print("Arrêt des entraînements ; export des époques sauvegardées.", flush=True)
        for mode in [*queue, *children]:
            results[mode]["etat"] = "interrompu"
    finally:
        try:
            stop(children)
            exported = output_ready and report()
            if exported:
                archive = output.with_name(output.name + ".zip")
                temp = shutil.make_archive(str(output.with_name("." + output.name + ".temp")), "zip",
                                          root_dir=output.parent, base_dir=output.name)
                Path(temp).replace(archive)
                print(f"Archive pour le rendu : {archive}", flush=True)
        finally:
            lock.unlink()
    return 0 if exported and all(state["etat"] == "terminé" for state in results.values()) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=tuple(MODES), default=list(MODES))
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/food-11")
    parser.add_argument("--runs", type=Path, default=KIT / "executions/benchmark_nuit")
    parser.add_argument("--output", type=Path, default=KIT / "resultats/nuit")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if len(set(args.modes)) != len(args.modes) or min(args.jobs, args.threads) < 1 or args.workers < 0:
        parser.error("Indiquez des modes distincts et des nombres de jobs, threads et workers valides.")
    if not args.data_dir.is_dir():
        parser.error(f"Dossier Food-11 absent : {args.data_dir}.")
    try:
        code = run(args)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
