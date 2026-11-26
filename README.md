# ViT-Anpassung auf GTSRB: Linear Probing, LoRA und CNN

Kursprojekt im Modul „Deep Learning for Computer Vision“ (Prof. Dr.-Ing. Jan Rexilius, Hochschule Bielefeld, WS 2026/27), Station 1 „Adaptation“. Autor: Alexej Demerza.

Leitfrage: Wie effizient lässt sich ein vortrainierter Vision Transformer an eine neue Aufgabe anpassen? Verglichen werden auf dem Verkehrszeichen-Datensatz GTSRB (43 Klassen):

- ViT-B/16 mit Linear Probing (nur der Klassifikationskopf wird trainiert),
- ViT-B/16 mit LoRA in verschiedenen Rängen und Zielschichten,
- ViT-B/16 mit vollständigem Fine-Tuning,
- ResNet-50 als CNN, nur Kopf und vollständig nachtrainiert.

Das Paper liegt als `Meilenstein1_Demerza.pdf` im Repository.

## Ergebnisse

Zehn Epochen, Batch-Größe 128, Bilder in 224×224. Ausgewertet wird der Stand mit der besten Validierungsgenauigkeit, einmal auf dem Testset (12.630 Bilder).

| Modell | Anpassung | Lernrate | Trainierbar | Val [%] | Test [%] | Macro-F1 [%] | s/Epoche | GPU [GB] |
|---|---|---|---|---|---|---|---|---|
| ViT-B/16 | Linear Probing | 1e-2 | 33.067 | 74,64 | 76,33 | 66,58 | 23 | 1,7 |
| ViT-B/16 | LoRA r=1, qkv | 2e-3 | 69.931 | 98,86 | 98,73 | 98,20 | 56 | 6,7 |
| ViT-B/16 | LoRA r=2, qkv | 2e-3 | 106.795 | 99,11 | 98,76 | 98,21 | 56 | 6,7 |
| ViT-B/16 | LoRA r=4, qkv | 2e-3 | 180.523 | 99,23 | 98,94 | 98,33 | 56 | 6,7 |
| ViT-B/16 | LoRA r=8, qkv | 2e-3 | 327.979 | 99,39 | 99,01 | 98,58 | 56 | 6,7 |
| ViT-B/16 | LoRA r=16, qkv | 2e-3 | 622.891 | 99,30 | 99,16 | 98,65 | 56 | 6,7 |
| ViT-B/16 | LoRA r=8, qkv+proj | 1e-3 | 475.435 | 99,20 | 99,11 | 98,75 | 61 | 7,3 |
| ViT-B/16 | LoRA r=8, qkv+proj+MLP | 3e-4 | 1.212.715 | 99,28 | 98,77 | 98,06 | 85 | 11,2 |
| ViT-B/16 | Volles Fine-Tuning | 1e-4 | 85.831.723 | 99,03 | 99,33 | 99,02 | 69 | 9,5 |
| ViT-B/16 (1k) | Linear Probing | 1e-2 | 33.067 | 65,43 | 66,28 | 57,01 | 23 | 1,7 |
| ViT-B/16 (1k) | LoRA r=8, qkv | 1e-3 | 327.979 | 98,84 | 98,36 | 97,40 | 56 | 6,7 |
| ViT-B/16 (1k) | Volles Fine-Tuning | 1e-4 | 85.831.723 | 99,46 | 98,87 | 98,26 | 69 | 9,5 |
| ResNet-50 | Nur Kopf | 3e-2 | 88.107 | 78,94 | 80,51 | 72,65 | 13 | 1,6 |
| ResNet-50 | Volles Fine-Tuning | 1e-3 | 23.596.139 | 98,67 | 99,00 | 98,58 | 37 | 5,7 |

Gemessen auf einer RTX 3090 Ti mit gemischter Genauigkeit (bfloat16). „ViT-B/16 (1k)“ ist dasselbe Modell mit Vortraining nur auf ImageNet-1k.

Mittelwert und Standardabweichung über drei Startwerte für die vier Hauptmodelle:

| Konfiguration | Val [%] | Test [%] | Macro-F1 [%] |
|---|---|---|---|
| ViT Linear Probing | 74,34 ± 0,31 | 76,34 ± 0,04 | 66,35 ± 0,26 |
| ViT LoRA r=8 | 99,14 ± 0,23 | 98,99 ± 0,08 | 98,44 ± 0,15 |
| ViT voll | 99,23 ± 0,18 | 99,31 ± 0,07 | 98,91 ± 0,19 |
| ResNet-50 voll | 98,62 ± 0,17 | 98,89 ± 0,16 | 98,09 ± 0,57 |

Weitere Kontrollläufe (längeres Training, höhere Lernraten, 10 % der Daten) stehen im Paper und in `runs/`.

## Dateien

| Datei | Aufgabe |
|---|---|
| `prepare_gtsrb.py` | lädt GTSRB, entpackt es und teilt die Trainingsdaten nach Aufnahmeserien in Training und Validierung (Seed 42, 20 % der Serien je Klasse) |
| `resize_224.py` | skaliert alle Bilder bikubisch auf 224×224 und speichert sie als PNG |
| `train.py` | ein Trainingslauf: Modell, Anpassung (`head`, `lora`, `full`), Lernrate, Rang, Zielschichten |
| `run_station1.py` | Lernraten-Suche (drei Werte je Methode, drei Epochen) und die zehn Hauptläufe |
| `run_station1_extra.py` | weitere Lernraten für LoRA mit mehr Zielschichten und für die Kopf-Varianten |
| `run_station1_plus.py` | Kontrollläufe: drei Startwerte, 30 Epochen, Rang 1, höhere Lernraten, 10 % der Daten, ViT mit ImageNet-1k-Vortraining |
| `runs/` | Ergebnisse aller Läufe: Kennzahlen, Verlauf je Epoche, Konfusionsmatrix (ohne Modellgewichte) |
| `Meilenstein1_Demerza.pdf` | das Paper zu Station 1 |

## Nachrechnen

Voraussetzungen: Python 3.12, eine NVIDIA-Grafikkarte mit mindestens 12 GB Speicher, etwa 4 GB Plattenplatz für die Daten.

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt

python prepare_gtsrb.py data/gtsrb                    # lädt etwa 360 MB, 1 Minute
python resize_224.py data/gtsrb data/gtsrb_224 224    # erzeugt 2,1 GB

python run_station1.py                                # Lernraten-Suche und Hauptläufe, etwa 2,5 Stunden
python run_station1_extra.py                          # etwa 1 Stunde
python run_station1_plus.py                           # etwa 3,5 Stunden
```

Ein einzelner Lauf, zum Beispiel LoRA mit Rang 8:

```
python train.py --name vit_lora_r8_qkv --model vit_base_patch16_224 --mode lora --rank 8 --targets qkv --lr 2e-3
```

Jeder Lauf schreibt nach `runs/<name>/`: `results.json` (alle Kennzahlen und Argumente), `history.csv` (Verlauf je Epoche), `confusion_test.csv`, `best.pt` (trainierte Gewichte) und `log.txt`. Im Repository liegen davon die ersten drei.

Zu manchen Konfigurationen gibt es Läufe mit mehreren Lernraten, zum Beispiel `vit_head`, `vit_head_lr0.03` und `vit_head_lr0.1`. Im Paper zählt jeweils der Lauf mit der besten Validierungsgenauigkeit (`best_val_acc` in `results.json`). Beim vollständigen Fine-Tuning sind es die Läufe `vit_full` und `resnet_full`; die Läufe mit höherer Lernrate werden im Paper getrennt genannt.

## Daten

GTSRB wird nicht mit ins Repository gelegt, `prepare_gtsrb.py` lädt den offiziellen Datensatz (39.209 Trainings- und 12.630 Testbilder). Nach dem Split: 31.379 Bilder im Training, 7.830 in der Validierung, 12.630 im Test. Der Split ist durch den festen Seed reproduzierbar.

Vortrainierte Gewichte kommen aus `timm`: `vit_base_patch16_224.augreg2_in21k_ft_in1k`, `vit_base_patch16_224.augreg_in1k` und `resnet50.a1_in1k`.
