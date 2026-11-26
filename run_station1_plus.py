from run_station1 import RESNET, VIT, run

# Dasselbe ViT, aber nur auf ImageNet-1k vortrainiert
VIT_IN1K = "vit_base_patch16_224.augreg_in1k"

# Feste Argumente der Hauptmodelle, Lernraten wie in den Endläufen
HEAD = ["--model", VIT, "--mode", "head", "--lr", "0.01"]
LORA = ["--model", VIT, "--mode", "lora", "--rank", "8", "--targets", "qkv"]
FULL = ["--model", VIT, "--mode", "full"]
RESNET_HEAD = ["--model", RESNET, "--mode", "head", "--lr", "0.03"]
RESNET_FULL = ["--model", RESNET, "--mode", "full"]

# Kontrollläufe mit den vier Hauptmodellen, aber mit anderen Lernraten, Rängen oder Trainingszeiten. 
PLAN = []  # (Name, Argumente)

# 1. Startwerte: die vier Hauptmodelle mit Seed 1 und 2 (Seed 0 gibt es schon), für die Streuung zwischen Läufen
for seed in (1, 2):
    PLAN += [
        (f"seeds/vit_head_s{seed}", HEAD + ["--seed", str(seed)]),
        (f"seeds/vit_lora_r8_qkv_s{seed}", LORA + ["--lr", "0.002", "--seed", str(seed)]),
        (f"seeds/vit_full_s{seed}", FULL + ["--lr", "0.0001", "--seed", str(seed)]),
        (f"seeds/resnet_full_s{seed}", RESNET_FULL + ["--lr", "0.001", "--seed", str(seed)]),
    ]

# 2. Länger trainieren: beide Kopf-Varianten über 30 statt 10 Epochen. Liegt die Schwäche am kurzen Training?
PLAN += [
    ("vit_head_ep30", HEAD + ["--epochs", "30"]),
    ("resnet_head_ep30", RESNET_HEAD + ["--epochs", "30"]),
]

# 3. Rang 1: Untergrenze des Rangs
PLAN.append(("vit_lora_r1_qkv", ["--model", VIT, "--mode", "lora", "--rank", "1", "--targets", "qkv", "--lr", "0.002"]))

# 4. Höhere Lernraten: ein Wert oberhalb des bisher höchsten, wo dieser gewonnen hatte
PLAN += [
    ("vit_lora_r8_qkv_lr0.005", LORA + ["--lr", "0.005"]),
    ("vit_full_lr0.0003", FULL + ["--lr", "0.0003"]),
    ("resnet_full_lr0.003", RESNET_FULL + ["--lr", "0.003"]),
]

# 5. Wenig Daten: 10 % der Serien, dafür 100 Epochen (gleich viele Schritte wie sonst), Validierung jede 5. Epoche
FRACTION = ["--train-fraction", "0.1", "--epochs", "100", "--eval-every", "5"]
PLAN += [
    ("frac10/vit_lora_r8_qkv", LORA + ["--lr", "0.002"] + FRACTION),
    ("frac10/vit_full", FULL + ["--lr", "0.0001"] + FRACTION),
]

# 6. ViT mit reinem ImageNet-1k-Vortraining (wie das ResNet-50), je zwei Lernraten für LoRA und volles Fine-Tuning
PLAN.append(("in1k/vit_head", ["--model", VIT_IN1K, "--mode", "head", "--lr", "0.01"]))
for lr in ("0.002", "0.001"):
    PLAN.append((f"in1k/vit_lora_r8_qkv_lr{lr}",
                 ["--model", VIT_IN1K, "--mode", "lora", "--rank", "8", "--targets", "qkv", "--lr", lr]))
for lr in ("0.0001", "0.00003"):
    PLAN.append((f"in1k/vit_full_lr{lr}", ["--model", VIT_IN1K, "--mode", "full", "--lr", lr]))

if __name__ == "__main__":
    failed = []
    for name, extra in PLAN:
        # Ein fehlgeschlagener Lauf bricht die Reihe nicht ab.
        try:
            result = run(name, extra)
        except SystemExit as error:
            print(f">>> {name}: FEHLER ({error})", flush=True)
            failed.append(name)
            continue
        print(f">>> {name}: Val {100 * result['best_val_acc']:.2f} % | Test {100 * result['test_acc']:.2f} %"
              f" | F1 {100 * result['test_macro_f1']:.2f} | beste Epoche {result['best_epoch']}", flush=True)
    print(f"FERTIG ({len(PLAN) - len(failed)} von {len(PLAN)} Läufen, fehlgeschlagen: {failed or 'keiner'})", flush=True)
