"""Create local GPS/network/reference EDA maps for short, median and long train trips.

The HTML maps contain coordinates and are written under reports/maps, which Git ignores.
"""
import argparse
import os
from pathlib import Path

import folium

from render_trip_map import build_map, read_rows

ROOT = Path(__file__).resolve().parent


def choose_examples(train_dir):
    trips = []
    for folder in sorted(p for p in train_dir.iterdir() if p.is_dir()):
        track_path = folder / f"{folder.name}.track"
        trips.append((len(read_rows(track_path, 3)), folder))
    trips.sort(key=lambda item: item[0])
    if not trips:
        raise SystemExit(f"No train trips found in {train_dir}")
    indexes = sorted({0, len(trips) // 2, len(trips) - 1})
    return [trips[i][1] for i in indexes]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.getenv("PATHSNAP_DATA_ROOT", str(ROOT / "data")))
    parser.add_argument("--output-dir", default=str(ROOT / "reports" / "maps"))
    parser.add_argument("--trip-ids", nargs="*", help="Optional specific training trip IDs")
    args = parser.parse_args()

    train_dir = Path(args.data_root).resolve() / "train"
    folders = choose_examples(train_dir)
    if args.trip_ids:
        by_id = {folder.name: folder for folder in (p for p in train_dir.iterdir() if p.is_dir())}
        missing = [tid for tid in args.trip_ids if tid not in by_id]
        if missing:
            raise SystemExit(f"Unknown training trip IDs: {', '.join(missing)}")
        folders = [by_id[tid] for tid in args.trip_ids]

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for folder in folders:
        tid = folder.name
        track = [[float(x) for x in r] for r in read_rows(folder / f"{tid}.track", 3)]
        nodes = [[float(x) for x in r] for r in read_rows(folder / f"{tid}.nodes", 2)]
        arcs = [[int(x) for x in r] for r in read_rows(folder / f"{tid}.arcs", 2)]
        reference = [int(r[0]) for r in read_rows(folder / f"{tid}.route", 1)]
        fmap = build_map(f"PathSnap EDA train trip: {tid}", track, nodes, arcs, reference)
        destination = output / f"train_eda_{tid}.html"
        fmap.save(str(destination))
        print(f"Saved local GPS/network/reference EDA map: {destination}")


if __name__ == "__main__":
    main()
