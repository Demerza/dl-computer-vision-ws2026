import csv
import random
import shutil
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

# Offizielles GTSRB-Archiv. Erst zusammen mit einem der Dateinamen darunter ergibt die Adresse einen Download-Link.
BASE_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
TRAIN_ZIP = "GTSRB_Final_Training_Images.zip"
TEST_ZIP = "GTSRB_Final_Test_Images.zip"
TEST_GT_ZIP = "GTSRB_Final_Test_GT.zip"

VAL_FRACTION = 0.2   # Anteil der Tracks je Klasse, der in die Validierung geht
SEED = 42            # fester Startwert, damit der Split bei jedem Aufruf gleich ausfällt

# Klassennamen in der Reihenfolge der Klassen-IDs 0 bis 42
CLASS_NAMES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles over 3.5 metric tons prohibited", "No entry",
    "General caution", "Dangerous curve to the left", "Dangerous curve to the right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left", "Keep right",
    "Keep left", "Roundabout mandatory", "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons",
]


def download(target: Path) -> None:
    """Lädt die drei Archive. Was schon da ist, wird nicht noch einmal geladen."""
    target.mkdir(parents=True, exist_ok=True)
    for name in (TRAIN_ZIP, TEST_ZIP, TEST_GT_ZIP):
        path = target / name
        if path.exists():
            continue
        print("lade", name)
        urllib.request.urlretrieve(BASE_URL + name, path)


def extract(zip_path: Path, target: Path) -> None:
    """Entpackt nur Bilder und CSV-Dateien, ohne die Ordnerstruktur des Archivs."""
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            parts = Path(member.filename).parts
            if member.is_dir() or Path(member.filename).suffix.lower() not in (".ppm", ".csv"):
                continue
            # Sicherheit: kein Eintrag darf aus dem Zielordner herausführen.
            if ".." in parts or Path(member.filename).is_absolute():
                raise ValueError(f"unerwarteter Pfad im Archiv: {member.filename}")
            # Trainingsbilder liegen in Klassenordnern, Testbilder flach.
            sub = parts[-2] if len(parts) > 1 and parts[-2].isdigit() else ""
            out = target / sub / parts[-1]
            out.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)


def read_gt(csv_path: Path) -> list[dict]:
    """Liest eine Ground-Truth-Datei von GTSRB: eine Zeile pro Bild mit Dateiname, Größe und Klasse."""
    with open(csv_path, newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/gtsrb")
    downloads = root / "_download"
    raw = root / "_raw"

    # Nichts überschreiben: Ein vorhandener Split muss von Hand gelöscht werden.
    for split in ("train", "val", "test"):
        if (root / split).exists():
            sys.exit(f"{root / split} existiert schon. Ordner löschen, um neu zu sortieren.")

    download(downloads)
    if raw.exists():
        shutil.rmtree(raw)
    print("entpacke ...")
    extract(downloads / TRAIN_ZIP, raw / "train")
    extract(downloads / TEST_ZIP, raw / "test")
    extract(downloads / TEST_GT_ZIP, raw / "test_gt")

    rows = []  # Zeilen für index.csv
    rng = random.Random(SEED)

    # train / val: Split über ganze Tracks, pro Klasse
    # Ein Track sind 30 Aufnahmen desselben Schildes.Würde man einzelne Bilder zufällig verteilen,
    # lägen fast identische Bilder in train und val, und die Val-Accuracy wäre zu gut.
    for class_dir in sorted(p for p in (raw / "train").iterdir() if p.is_dir()):
        class_id = int(class_dir.name)
        # Dateiname = <Track>_<Bild>.ppm, die Bilder also nach dem Teil vor dem Unterstrich gruppieren.
        by_track = defaultdict(list)
        for entry in read_gt(class_dir / f"GT-{class_dir.name}.csv"):
            by_track[entry["Filename"].split("_")[0]].append(entry)

        # Erst sortieren, dann mischen: so hängt das Ergebnis nur vom Seed ab.
        tracks = sorted(by_track)
        rng.shuffle(tracks)
        # Jede Klasse bekommt mindestens einen Track in der Validierung (die kleinsten haben nur 7 Tracks glaube ich).
        n_val = max(1, round(VAL_FRACTION * len(tracks)))
        val_tracks = set(tracks[:n_val])

        for track, entries in by_track.items():
            split = "val" if track in val_tracks else "train"
            out_dir = root / split / class_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for entry in entries:
                shutil.move(class_dir / entry["Filename"], out_dir / entry["Filename"])
                rows.append([split, class_id, int(track), f"{split}/{class_dir.name}/{entry['Filename']}",
                             entry["Width"], entry["Height"]])

    # test: offizielles Testset (12.630 Bilder),unverändert
    # Die Testbilder liegen im Archiv ohne Klassenordner, die Klasse steht in der Ground-Truth-Datei.
    # Tracks sind für das Testset nicht angegeben, die Spalte bleibt leer.
    for entry in read_gt(raw / "test_gt" / "GT-final_test.csv"):
        class_name = f"{int(entry['ClassId']):05d}"
        out_dir = root / "test" / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(raw / "test" / entry["Filename"], out_dir / entry["Filename"])
        rows.append(["test", int(entry["ClassId"]), "", f"test/{class_name}/{entry['Filename']}",
                     entry["Width"], entry["Height"]])

    shutil.rmtree(raw)

    # index.csv: eine Zeile pro Bild. width/height sind die Originalgröße vor dem Skalieren.
    with open(root / "index.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "class_id", "track", "path", "width", "height"])
        writer.writerows(rows)

    # classes.csv: Klassen-ID, Ordnername und Klassenname
    with open(root / "classes.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "folder", "name"])
        for class_id, name in enumerate(CLASS_NAMES):
            writer.writerow([class_id, f"{class_id:05d}", name])

    for split in ("train", "val", "test"):
        n = sum(1 for row in rows if row[0] == split)
        print(f"{split:5s} {n:6d} Bilder")


if __name__ == "__main__":
    main()
