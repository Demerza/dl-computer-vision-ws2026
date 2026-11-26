from run_station1 import FINAL_EPOCHS, RESNET, VIT, run


# Zusatzläufe nach den ersten Ergebnissen, alle über 10 Epochen.
# Welche Lernrate je Konfiguration zählt, entscheidet die Val-Accuracy.
EXTRA = []

# LoRA mit mehr Zielschichten: mit der für qkv gesuchten Lernrate 2e-3 bricht das Training ab Epoche 2 zusammen.
# Deshalb beide Varianten zusätzlich mit 1e-3 und 3e-4.
for label, targets in (("qkv_proj", "qkv,attn.proj"), ("qkv_proj_mlp", "qkv,attn.proj,mlp.fc1,mlp.fc2")):
    for lr in (1e-3, 3e-4):
        EXTRA.append((f"vit_lora_r8_{label}_lr{lr:g}",
                      ["--model", VIT, "--mode", "lora", "--rank", "8", "--targets", targets, "--lr", str(lr)]))

# Kopf-Varianten: in der Lernraten-Suche gewann der höchste Wert (1e-2) mit großem Abstand.
# Deshalb zusätzlich 3e-2 und 1e-1.
for family, model in (("vit_head", VIT), ("resnet_head", RESNET)):
    for lr in (3e-2, 1e-1):
        EXTRA.append((f"{family}_lr{lr:g}", ["--model", model, "--mode", "head", "--lr", str(lr)]))

if __name__ == "__main__":
    for name, extra in EXTRA:
        result = run(name, extra + ["--epochs", str(FINAL_EPOCHS)])
        print(f">>> {name}: Val {100 * result['best_val_acc']:.2f} % | Test {100 * result['test_acc']:.2f} %", flush=True)
    print("FERTIG", flush=True)
