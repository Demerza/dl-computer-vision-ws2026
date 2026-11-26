import csv
import sys
from multiprocessing import Pool
from pathlib import Path

from PIL import Image


def convert(job: tuple[str, str, int]) -> None:
    """Skaliert ein Bild und speichert es als PNG."""
    src, dst, size = job
    out = Path(dst)
    # Schon vorhandene Bilder überspringen, dann kann man nach einem Abbruch einfach neu starten.
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        # Direkt auf size x size, bikubisch und ohne Zuschnitt. Die meisten Bilder sind fast quadratisch.
        image.convert("RGB").resize((size, size), Image.BICUBIC).save(out, "PNG")


def main() -> None:
    #Aufruf: python resize_224.py [quelle] [ziel] [größe], Standard: data/gtsrb -> data/gtsrb_224 mit 224
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/gtsrb")
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/gtsrb_224")
    size = int(sys.argv[3]) if len(sys.argv) > 3 else 224

    # Die Liste der Bilder kommt aus der index.csv von prepare_gtsrb.py.
    with open(src / "index.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))

    # Gleiche Ordner wie vorher, nur .png statt .ppm
    jobs = []
    for row in rows:
        old = row["path"]
        row["path"] = Path(old).with_suffix(".png").as_posix()
        jobs.append((str(src / old), str(dst / row["path"]), size))

    # Mehrere Prozesse gleichzeitig, bei über 50.000 Bildern dauert es sonst zu lange.
    with Pool() as pool:
        for done, _ in enumerate(pool.imap_unordered(convert, jobs, chunksize=256), 1):
            if done % 10000 == 0 or done == len(jobs):
                print(f"{done} / {len(jobs)}", flush=True)

    # index.csv mit den neuen Pfaden. width und height bleiben die Originalgröße.
    with open(dst / "index.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (dst / "classes.csv").write_bytes((src / "classes.csv").read_bytes())


if __name__ == "__main__":
    main()
