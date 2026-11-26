import csv
import json
import subprocess
import sys
from pathlib import Path

VIT = "vit_base_patch16_224"
RESNET = "resnet50"
RUNS = Path("runs")
SWEEP_EPOCHS = 3    # kurze Läufe für die Lernraten-Suche
FINAL_EPOCHS = 10   # Endläufe

# Methode -> feste Argumente und Lernraten für die Suche
FAMILIES = {
    "vit_head":    (["--model", VIT, "--mode", "head"], [1e-2, 1e-3, 1e-4]),
    "vit_lora":    (["--model", VIT, "--mode", "lora", "--rank", "8", "--targets", "qkv"], [2e-3, 1e-3, 3e-4]),
    "vit_full":    (["--model", VIT, "--mode", "full"], [1e-4, 3e-5, 1e-5]),
    "resnet_head": (["--model", RESNET, "--mode", "head"], [1e-2, 1e-3, 1e-4]),
    "resnet_full": (["--model", RESNET, "--mode", "full"], [1e-3, 3e-4, 1e-4]),
}


def run(name: str, extra: list[str]) -> dict:
    """Startet einen Lauf (falls noch nicht fertig) und gibt results.json zurück."""
    result_file = RUNS / name / "results.json"
    # Fertige Läufe werden übersprungen, das Skript kann also jederzeit neu gestartet werden.
    if not result_file.exists():
        command = [sys.executable, "train.py", "--name", name, "--out", str(RUNS)] + extra
        print(f"\n=== {name} ===", flush=True)
        (RUNS / name).mkdir(parents=True, exist_ok=True)
        with open(RUNS / name / "log.txt", "w") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            # Alles in die Logdatei, Warnungen aber nicht auf den Bildschirm
            for line in process.stdout:
                log.write(line)
                if "Warning" not in line and "warn" not in line:
                    print(line, end="", flush=True)
            if process.wait() != 0:
                sys.exit(f"Lauf {name} ist fehlgeschlagen, siehe {RUNS / name / 'log.txt'}")
    return json.loads(result_file.read_text())


def main() -> None:
    # Phase 1: Lernraten-Suche. Jede Methode mit drei Lernraten, je 3 Epochen, ohne Testset.
    best_lr = {}
    sweep_rows = []
    for family, (extra, lrs) in FAMILIES.items():
        scores = {}
        for lr in lrs:
            result = run(f"sweep/{family}_lr{lr:g}", extra + ["--lr", str(lr), "--epochs", str(SWEEP_EPOCHS), "--no-test"])
            scores[lr] = result["best_val_acc"]
            sweep_rows.append([family, lr, result["best_val_acc"], result["best_val_macro_f1"]])
        # Die Lernrate mit der besten Val-Accuracy gewinnt.
        best_lr[family] = max(scores, key=scores.get)
        print(f">>> {family}: beste Lernrate {best_lr[family]:g} ({ {f'{k:g}': round(100 * v, 2) for k, v in scores.items()} })", flush=True)

    (RUNS / "sweep").mkdir(parents=True, exist_ok=True)
    (RUNS / "sweep" / "best_lr.json").write_text(json.dumps(best_lr, indent=2))
    with open(RUNS / "sweep" / "sweep.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["family", "lr", "val_acc", "val_macro_f1"])
        writer.writerows(sweep_rows)

    # Phase 2: Endläufe mit der gewählten Lernrate, 10 Epochen, danach einmal Test
    def final(name: str, extra: list[str], family: str) -> dict:
        return run(name, extra + ["--lr", str(best_lr[family]), "--epochs", str(FINAL_EPOCHS)])

    results = [final("vit_head", FAMILIES["vit_head"][0], "vit_head")]

    # LoRA: zuerst der Rang (nur qkv), dann beim besten Rang weitere Zielschichten
    by_rank = {}
    for rank in (2, 4, 8, 16):
        by_rank[rank] = final(f"vit_lora_r{rank}_qkv",
                              ["--model", VIT, "--mode", "lora", "--rank", str(rank), "--targets", "qkv"], "vit_lora")
        results.append(by_rank[rank])
    best_rank = max(by_rank, key=lambda r: by_rank[r]["best_val_acc"])
    print(f">>> bester Rang nach Val-Accuracy: r = {best_rank}", flush=True)
    for label, targets in (("qkv_proj", "qkv,attn.proj"), ("qkv_proj_mlp", "qkv,attn.proj,mlp.fc1,mlp.fc2")):
        results.append(final(f"vit_lora_r{best_rank}_{label}",
                             ["--model", VIT, "--mode", "lora", "--rank", str(best_rank), "--targets", targets], "vit_lora"))

    results.append(final("vit_full", FAMILIES["vit_full"][0], "vit_full"))
    results.append(final("resnet_head", FAMILIES["resnet_head"][0], "resnet_head"))
    results.append(final("resnet_full", FAMILIES["resnet_full"][0], "resnet_full"))

    # Zusammenfassung aller Endläufe in runs/summary.csv
    columns = ["name", "lr", "trainable_params", "total_params", "best_val_acc", "test_acc", "test_macro_f1",
               "train_seconds_per_epoch", "train_seconds_total", "peak_gpu_gb", "test_images_per_second", "best_epoch"]
    with open(RUNS / "summary.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for r in results:
            writer.writerow([r["args"]["lr"] if c == "lr" else r.get(c) for c in columns])

    print("\n=== ZUSAMMENFASSUNG ===")
    print(f"{'Lauf':28s} {'LR':>7s} {'trainierbar':>12s} {'Val':>7s} {'Test':>7s} {'F1':>7s} {'s/Ep.':>6s} {'GPU GB':>7s}")
    for r in results:
        print(f"{r['name']:28s} {r['args']['lr']:7g} {r['trainable_params']:12,d} {100 * r['best_val_acc']:6.2f}% "
              f"{100 * r['test_acc']:6.2f}% {100 * r['test_macro_f1']:6.2f}% {r['train_seconds_per_epoch']:6.0f} {r['peak_gpu_gb']:7.2f}")
    print("FERTIG", flush=True)


if __name__ == "__main__":
    main()
