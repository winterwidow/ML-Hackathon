"""Render one student-test prediction as an interactive Folium map.

Run hmm_decode.py first. This file reads only GPS points and HMM-predicted arc IDs;
it never reads the reference route labels written to the local prediction table.
"""
import argparse
import csv
import os
from pathlib import Path

import folium

ROOT = Path(__file__).resolve().parent


def read_rows(path, width):
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            cols = line.split()
            if len(cols) != width:
                raise ValueError(f"{path}:{line_no}: expected {width} columns")
            rows.append(cols)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trip-id", required=True, help="Trip ID from student_test")
    parser.add_argument("--data-root", default=os.getenv("PATHSNAP_DATA_ROOT", str(ROOT / "data")))
    parser.add_argument(
        "--predictions",
        default=str(ROOT / "features" / "student_test_hmm_predictions_aligned.csv"),
    )
    parser.add_argument("--output", help="HTML output path; defaults to reports/maps/<trip-id>.html")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    trip_dir = (data_root / "student_test" / args.trip_id).resolve()
    if trip_dir.parent != (data_root / "student_test").resolve() or not trip_dir.is_dir():
        raise SystemExit(f"Unknown student-test trip: {args.trip_id}")

    track = [[float(x) for x in row] for row in read_rows(trip_dir / f"{args.trip_id}.track", 3)]
    nodes = [[float(x) for x in row] for row in read_rows(trip_dir / f"{args.trip_id}.nodes", 2)]
    arcs = [[int(x) for x in row] for row in read_rows(trip_dir / f"{args.trip_id}.arcs", 2)]

    predicted = []
    with Path(args.predictions).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["trip_id"] == args.trip_id and row["hmm_arc_id"] != "":
                aid = int(row["hmm_arc_id"])
                if 0 <= aid < len(arcs):
                    predicted.append(aid)
    if not predicted:
        raise SystemExit(f"No decoded route found for trip {args.trip_id}; run hmm_decode.py first")

    center = [sum(p[1] for p in track) / len(track), sum(p[0] for p in track) / len(track)]
    route_features = []
    for aid in sorted(set(predicted)):
        u, v = arcs[aid]
        route_features.append(
            {
                "type": "Feature",
                "properties": {"arc_id": aid},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[nodes[u][0], nodes[u][1]], [nodes[v][0], nodes[v][1]]],
                },
            }
        )

    fmap = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap", control_scale=True)
    gps = folium.FeatureGroup(name="Noisy GPS trace", show=True)
    folium.PolyLine(
        [[p[1], p[0]] for p in track], color="#2563eb", weight=3, opacity=0.8,
        tooltip="Observed GPS trace",
    ).add_to(gps)
    folium.CircleMarker([track[0][1], track[0][0]], radius=5, color="#16a34a",
                        fill=True, tooltip="Trip start").add_to(gps)
    folium.CircleMarker([track[-1][1], track[-1][0]], radius=5, color="#7c3aed",
                        fill=True, tooltip="Trip end").add_to(gps)
    gps.add_to(fmap)

    matched = folium.FeatureGroup(name="Predicted road segments (HMM)", show=True)
    folium.GeoJson(
        {"type": "FeatureCollection", "features": route_features},
        name="HMM predicted arcs",
        style_function=lambda _feature: {"color": "#dc2626", "weight": 5, "opacity": 0.9},
        tooltip=folium.GeoJsonTooltip(fields=["arc_id"], aliases=["Predicted arc"]),
    ).add_to(matched)
    matched.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    folium.map.Marker(
        center,
        icon=folium.DivIcon(
            html=(
                "<div style='font-size:13px;font-weight:700;color:#111827;"
                "background:white;padding:5px 8px;border:1px solid #cbd5e1;"
                "border-radius:5px'>PathSnap - " + args.trip_id + "</div>"
            )
        ),
    ).add_to(fmap)

    out = Path(args.output) if args.output else ROOT / "reports" / "maps" / f"student_test_{args.trip_id}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out))
    print(f"Saved interactive map to {out}")


if __name__ == "__main__":
    main()
