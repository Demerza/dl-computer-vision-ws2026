import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import timm
import torch
from peft import LoraConfig, get_peft_model
from sklearn.metrics import confusion_matrix, f1_score
from timm.data import resolve_data_config
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

NUM_CLASSES = 43


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="Name des Laufs, Ergebnisordner runs/<name>")
    p.add_argument("--model", default="vit_base_patch16_224")
    # head = nur der Kopf (Linear Probing), lora = LoRA-Matrizen und Kopf (nur ViT), full = alle Gewichte
    p.add_argument("--mode", choices=["head", "lora", "full"], required=True)
    p.add_argument("--rank", type=int, default=8, help="LoRA-Rang r")
    p.add_argument("--alpha", type=int, default=None, help="LoRA-Skalierung, Standard 2 * r")
    p.add_argument("--targets", default="qkv", help="LoRA-Zielschichten, mit Komma getrennt (qkv, attn.proj, mlp.fc1, mlp.fc2)")
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default="data/gtsrb_224")
    p.add_argument("--out", default="runs")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--subset", type=int, default=0, help="nur N Trainingsbilder, ohne Augmentierung (Test: Überanpassen)")
    p.add_argument("--train-fraction", type=float, default=1.0,
                   help="nur dieser Anteil der Trainings-Serien je Klasse (ganze Serien, mit Augmentierung)")
    p.add_argument("--eval-every", type=int, default=1, help="Validierung nur jede N-te Epoche (und nach der letzten)")
    p.add_argument("--no-test", action="store_true", help="Testset nicht auswerten (für die Lernraten-Suche)")
    return p.parse_args()


def set_seed(seed: int) -> None:
    # Alle Zufallsquellen auf denselben Startwert setzen, damit ein Lauf wiederholbar ist.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args: argparse.Namespace) -> nn.Module:
    # Vortrainiertes Modell aus timm, der Kopf wird durch eine neue Schicht mit 43 Ausgängen ersetzt.
    model = timm.create_model(args.model, pretrained=True, num_classes=NUM_CLASSES)
    if args.mode == "head":
        # Linear Probing: alles einfrieren, nur der Kopf bleibt trainierbar.
        for param in model.parameters():
            param.requires_grad = False
        for param in model.get_classifier().parameters():
            param.requires_grad = True
    elif args.mode == "lora":
        # LoRA: PEFT friert das Modell ein und hängt die kleinen Matrizen A und B an die Zielschichten.
        head_name = model.default_cfg["classifier"]  # "head" beim ViT
        config = LoraConfig(
            r=args.rank,
            # alpha = 2 * r, so bleibt der Faktor alpha / r bei jedem Rang gleich
            lora_alpha=args.alpha if args.alpha is not None else 2 * args.rank,
            target_modules=args.targets.split(","),
            lora_dropout=args.lora_dropout,
            bias="none",
            modules_to_save=[head_name],  # der Kopf wird normal trainiert, ohne LoRA
        )
        model = get_peft_model(model, config)
    return model


def build_loaders(args: argparse.Namespace, mean, std):
    normalize = transforms.Normalize(mean, std)
    # Die Bilder liegen schon in 224x224 vor. Kein Spiegeln: links/rechts ändert die Klasse.
    augment = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1),
                                interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        normalize,
    ])
    # val und test: keine Augmentierung, nur Normalisierung
    plain = transforms.Compose([transforms.ToTensor(), normalize])

    train = datasets.ImageFolder(f"{args.data}/train", transform=plain if args.subset else augment)
    val = datasets.ImageFolder(f"{args.data}/val", transform=plain)
    test = datasets.ImageFolder(f"{args.data}/test", transform=plain)
    assert train.classes == val.classes == test.classes and len(train.classes) == NUM_CLASSES

    # Nur zum Testen: wenige Bilder ohne Augmentierung, das Modell muss sie auswendig lernen können.
    if args.subset:
        picked = random.Random(args.seed).sample(range(len(train)), args.subset)
        train = Subset(train, picked)
    elif args.train_fraction < 1:
        # Ganze Serien ziehen, je Klasse derselbe Anteil. Dateiname: <Serie>_<Bild>.png
        tracks = {}
        for index, (path, label) in enumerate(train.samples):
            tracks.setdefault((label, Path(path).name.split("_")[0]), []).append(index)
        rng = random.Random(args.seed)
        picked = []
        for label in range(NUM_CLASSES):
            keys = sorted(key for key in tracks if key[0] == label)
            for key in rng.sample(keys, max(1, round(args.train_fraction * len(keys)))):
                picked += tracks[key]
        print(f"Trainingsdaten: {len(picked):,} von {len(train):,} Bildern", flush=True)
        train = Subset(train, picked)

    common = dict(num_workers=args.workers, pin_memory=True, persistent_workers=True)
    # Fester Seed für die Reihenfolge der Bilder
    generator = torch.Generator().manual_seed(args.seed)
    # drop_last: der letzte, unvollständige Batch je Epoche fällt weg, so sind alle Batches gleich groß.
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=not args.subset,
                              generator=generator, **common)
    val_loader = DataLoader(val, batch_size=256, shuffle=False, **common)
    test_loader = DataLoader(test, batch_size=256, shuffle=False, **common)
    return train_loader, val_loader, test_loader


# Beim Auswerten werden keine Gradienten gebraucht, das spart Zeit und Speicher.
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    """Accuracy und Macro-F1 auf einem Datensatz, dazu alle Vorhersagen für die Konfusionsmatrix."""
    model.eval()
    predictions, labels = [], []
    start = time.time()
    for images, target in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(images.cuda(non_blocking=True))
        predictions.append(logits.argmax(1).cpu())
        labels.append(target)
    # Die Grafikkarte rechnet zeitversetzt. Erst warten, bis sie fertig ist, sonst stimmt die Zeitmessung nicht.
    torch.cuda.synchronize()
    seconds = time.time() - start
    predictions, labels = torch.cat(predictions).numpy(), torch.cat(labels).numpy()
    return {
        "accuracy": float((predictions == labels).mean()),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "seconds": seconds,
        "predictions": predictions,
        "labels": labels,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)

    model = build_model(args).cuda()
    # Mittelwert und Standardabweichung für die Normalisierung, passend zum Vortraining des Modells
    cfg = resolve_data_config({}, model=timm.create_model(args.model, pretrained=False))
    train_loader, val_loader, test_loader = build_loaders(args, cfg["mean"], cfg["std"])

    # Nur die trainierbaren Parameter gehen in den Optimierer.
    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"{args.name}: {n_trainable:,} trainierbar von {n_total:,} ({100 * n_trainable / n_total:.3f} %)", flush=True)

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    steps_total = args.epochs * len(train_loader)
    steps_warmup = len(train_loader)  # eine Epoche linear hochfahren, danach Cosine

    def lr_factor(step: int) -> float:
        if step < steps_warmup:
            return (step + 1) / steps_warmup
        progress = (step - steps_warmup) / max(1, steps_total - steps_warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    criterion = nn.CrossEntropyLoss()

    history, best_acc, best_epoch, best_f1 = [], -1.0, -1, -1.0
    train_seconds = 0.0
    # Ab hier den höchsten Speicherbedarf der Grafikkarte messen
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, args.epochs + 1):
        # Beim Linear Probing bleibt das Backbone im Eval-Modus (BatchNorm-Statistik bleibt fest).
        model.eval() if args.mode == "head" else model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        start = time.time()
        for images, target in train_loader:
            # Gemischte Genauigkeit (bfloat16) spart Zeit und Speicher.
            images, target = images.cuda(non_blocking=True), target.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
                loss = criterion(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * len(target)
            correct += (logits.argmax(1) == target).sum().item()
            seen += len(target)
        torch.cuda.synchronize()
        epoch_seconds = time.time() - start
        train_seconds += epoch_seconds

        if epoch % args.eval_every and epoch != args.epochs:  # Epoche ohne Validierung
            history.append({"epoch": epoch, "train_loss": loss_sum / seen, "train_acc": correct / seen,
                            "val_acc": "", "val_macro_f1": "", "epoch_seconds": epoch_seconds})
            print(f"Epoche {epoch:2d} | Loss {loss_sum / seen:.4f} | Train {100 * correct / seen:.2f} % | {epoch_seconds:.0f} s", flush=True)
            continue

        val = evaluate(model, val_loader)
        row = {"epoch": epoch, "train_loss": loss_sum / seen, "train_acc": correct / seen,
               "val_acc": val["accuracy"], "val_macro_f1": val["macro_f1"], "epoch_seconds": epoch_seconds}
        history.append(row)
        print(f"Epoche {epoch:2d} | Loss {row['train_loss']:.4f} | Train {100 * row['train_acc']:.2f} %"
              f" | Val {100 * row['val_acc']:.2f} % | F1 {100 * row['val_macro_f1']:.2f} | {epoch_seconds:.0f} s", flush=True)

        # Besten Stand nach Val-Accuracy speichern. Bei head und lora nur die trainierten Gewichte.
        if val["accuracy"] > best_acc:
            best_acc, best_epoch, best_f1 = val["accuracy"], epoch, val["macro_f1"]
            trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
            state = {k: v.cpu() for k, v in model.state_dict().items()
                     if k in trainable_names or args.mode == "full"}  # "full": auch BatchNorm-Puffer
            torch.save({"args": vars(args), "state": state}, out / "best.pt")

    peak_gb = torch.cuda.max_memory_allocated() / 2**30

    results = {
        "name": args.name, "args": vars(args),
        "trainable_params": n_trainable, "total_params": n_total,
        "train_images": len(train_loader.dataset), "epochs": args.epochs,
        "train_seconds_total": train_seconds, "train_seconds_per_epoch": train_seconds / args.epochs,
        "peak_gpu_gb": peak_gb, "best_epoch": best_epoch, "best_val_acc": best_acc,
        "best_val_macro_f1": best_f1,
    }

    # Das Testset wird nur einmal ausgewertet, mit dem besten Stand nach Val-Accuracy.
    if not args.no_test:
        model.load_state_dict(torch.load(out / "best.pt")["state"], strict=False)
        test = evaluate(model, test_loader)
        matrix = confusion_matrix(test["labels"], test["predictions"], labels=list(range(NUM_CLASSES)))
        np.savetxt(out / "confusion_test.csv", matrix, fmt="%d", delimiter=",")
        results.update({
            "test_acc": test["accuracy"], "test_macro_f1": test["macro_f1"],
            "test_images_per_second": len(test["labels"]) / test["seconds"],
            "test_acc_per_class": (matrix.diagonal() / matrix.sum(1)).tolist(),
        })
        print(f"TEST | Accuracy {100 * test['accuracy']:.2f} % | Macro-F1 {100 * test['macro_f1']:.2f}"
              f" | beste Epoche {best_epoch}", flush=True)

    print(f"Zeit {train_seconds:.0f} s gesamt, {train_seconds / args.epochs:.0f} s je Epoche | Peak-GPU {peak_gb:.2f} GB", flush=True)

    # Alle Kennzahlen und Argumente des Laufs
    with open(out / "results.json", "w") as handle:
        json.dump(results, handle, indent=2)
    # Verlauf je Epoche: Loss, Trainings- und Val-Accuracy, Dauer
    with open(out / "history.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


if __name__ == "__main__":
    main()
